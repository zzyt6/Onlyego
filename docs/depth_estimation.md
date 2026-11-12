# 深度估计（depth_estimation）：V2D 的方案和我们的迁移

源码：`E:\video_to_data`（NVIDIA Video to Data，Apache-2.0）。我们的实现在 `src\depth_estimation\`。

## 1. V2D 里深度相关的模块

`reconstruction\modules\` 下和深度有关的有五个，但单目第一人称视频（我们的情况）只用其中两个：

| 模块 | 作用 | 单目 ego 流程里用了吗 |
|---|---|---|
| `v2d_moge` | MoGe-2：单张图 → 米制深度 + 相机内参 | **用，唯一的深度来源** |
| `v2d_depth` | `stabilize_intrinsics`（内参取中位数）、`align_depth_sequence`（逐帧尺度对齐） | 前者用；后者写了但**没有任何流程调用** |
| `v2d_unidepth` | UniDepth，另一种单目米制深度 | 否（备选） |
| `v2d_depth_anything` | Depth Anything，相对深度 + 米制缩放 | 否（备选） |
| `v2d_foundation_stereo` | 双目立体深度（TensorRT） | 否，用于多视角/双目采集 |

`run_ego_reconstruction.py` 里 `--depth_source` 已经废弃，注释写明 “this pipeline always uses MoGe”。

## 2. 他们的流程（`v2d_pipelines/run_ego_wilor.py` 第 1–2 步）

```text
视频 ──① 抽帧──► frames/000000.png …
     ──② MoGe-2 逐帧推理──► depth/*.png   intrinsics/*.json
     ──③ 内参取时间中位数──► intrinsics_stable.json
                                   │
后续：SLAM、手/物体对齐、FoundationPose、SAM3D 物体尺度估计都读 depth/ + intrinsics_stable.json
```

可选的焦距先验：如果开了 `--undistort`（AnyCalib 去畸变）或 `--use_geocalib_intrinsics`，
把已知内参换算成水平视场角 `fov_x = 2·atan(W / 2fx)` 交给 MoGe。

V2D 里标定相关的开关和我们的对应关系（顺序都是：先 AnyCalib 去畸变，再在去畸变后的帧上跑 GeoCalib，最后 MoGe）：

| V2D `run_ego_wilor.py` | 这里 `depth_estimation run` | MoGe 拿到的焦距 |
|---|---|---|
| （都不开） | （都不加） | 无，每帧自己猜 |
| `--undistort` | `--undistort`（等同 `--prior anycalib --undistort`） | 去畸变后的相机 |
| `--use_geocalib_intrinsics` | `--prior geocalib` | GeoCalib |
| `--undistort --use_geocalib_intrinsics` | `--undistort --prior geocalib` | GeoCalib（在去畸变后的帧上估计） |
| `--run_geocalib` / `--gravity_align`（只要重力） | `--run-geocalib` | 不变（GeoCalib 只提供重力方向） |
| — | `--prior anycalib`（不去畸变） | AnyCalib 小孔模型（我们新增的用法） |

## 3. 每一步的原理

### ① 抽帧；HDR 视频先转成 SDR（我们新增，2026-11-11）

iPhone 默认录 **HDR 视频**（HEVC 10 位、BT.2020 色域、HLG 曲线；`run.json` 的 `video.color` 里记着）。OpenCV 和
V2D 的 ffmpeg 抽帧都不看色彩标记，把 HLG 信号当成 SDR BT.709 直接读——画面发白、发灰（demo01 饱和度中位数 66，
正确转换后 139），之后所有模型（AnyCalib、MoGe、WiLoR、SAM2、SAM3D 贴图、gsplat 颜色）看到的都是这种画面。

现在检测到 HDR（传输曲线 HLG 或 PQ）时，帧由 ffmpeg（`imageio-ffmpeg` 自带的 gyan.dev 版，有 `zscale`/`tonemap`）
解码并做色调映射：线性光（标称峰值 `--hdr-npl`，默认 100 nit）→ BT.709 色域 → `--hdr-tonemap`（默认 hable，
不降饱和）→ BT.709。同时写一份 `video_sdr.mp4`（H.264，同样的映射），给直接读视频的步骤用（`HOI prepare` 会用它）。
HDR 转 SDR 没有唯一正确答案：demo01 上比较了 hable 203 / hable 100 / mobius 100 / clip 203，hable 100 高光不溢出、
颜色最自然。`--hdr-tonemap off` = 旧行为。SDR 视频不受影响（仍用 OpenCV）。

用旧的发白帧算的 demo01 结果保留为 `runs\*\demo01-hdr-raw`（深度、video_depth、camera_pose；两个 HOI 运行也指向它们）。

### ② MoGe-2：一张图得到米制几何

- 网络（DINOv2-Large 骨干）直接预测每个像素的 **3D 点**（点图），而不是只预测深度。
- 内参是从点图里**解**出来的：找一个焦距（和平移），让点投影回去最贴合像素位置。所以它给出的内参和深度是**自洽**的一对。
- `infer()` 返回 `points`（H×W×3）、`depth`（就是点的 z）、`mask`（有效像素，天空/无穷远为 0）、`normal`、`intrinsics`（**按图像宽高归一化**，需要乘回 W/H）。
- 给了 `fov_x` 时，焦距固定，网络只需要解深度的尺度/平移。
- 每一帧**独立**推理，所以焦距和整体尺度会逐帧抖动——这正是第 ③ 步和 `align` 要处理的问题。

### 可选：用标定器给 MoGe 焦距先验（`--prior anycalib|geocalib`）

MoGe 每帧自己猜焦距，所以焦距会抖（demo01 前 60 帧：fx 1099–1245）。V2D 的办法是先用一个**专门的单图标定模型**
在若干帧上估计相机，取中位数，把 `fov_x = 2·atan(W / 2fx)` 交给 MoGe。MoGe 收到 `fov_x` 后焦距固定，只解深度——
所有帧共用同一个相机。对应 V2D `run_ego_wilor.py`：
`moge_input_intrinsics = geocalib_intr if use_geocalib_intrinsics else (undistorted_intr if undistort else None)`。

| | AnyCalib（`v2d_anycalib`） | GeoCalib（`v2d_geocalib`） |
|---|---|---|
| 原理 | DINOv2 预测每个像素的**光线方向**，再拟合一个相机模型（可含畸变） | 预测每个像素的**"上"方向和纬度**场，联合优化焦距和重力方向 |
| 畸变 | 能拟合鱼眼（Kannala-Brandt `kb:4`）或径向畸变，可以**去畸变** | 只给参数，V2D 不用它去畸变 |
| 额外输出 | — | **重力方向**（相机坐标里的 roll/pitch），以后做"上就是上"的世界坐标用 |
| 采样帧数（默认） | 64（V2D 是 16） | 8 |
| 权重 | `model\anycalib\anycalib_gen.pt`（任意相机模型）、`anycalib_pinhole.pt` | `model\geocalib\geocalib-pinhole.tar`、`geocalib-distorted.tar` |

**AnyCalib 在 V2D 里只和去畸变一起用**：先拟合 `kb:4` 鱼眼模型 → 把每帧重映射成理想小孔相机 → MoGe 在去畸变后的帧上跑，
先验是去畸变后相机的焦距。原因是鱼眼模型的焦距（r = f·θ）不能直接当小孔相机（r = f·tanθ）的焦距用。
所以这里不加 `--undistort` 时，AnyCalib 改用 `anycalib_pinhole` 权重、按无畸变小孔相机拟合，焦距才有意义。

`--balance`：去畸变时 0 = 裁到全是有效像素（视野变小，焦距变短），1 = 保留全部视野（四角有黑边）。

#### 选定：默认用 AnyCalib 小孔模型，不去畸变（2026-11-10）

`depth_estimation run` 不加参数时就是 `--prior anycalib`：AnyCalib（`anycalib_pinhole.pt`）按无畸变小孔相机
在 64 帧上估焦距，取中位数后作为 `fov_x` 交给 MoGe；不去畸变。`--prior none` 回到 V2D 的默认（MoGe 每帧自己猜）。
依据（demo01，iPhone 16 Pro Max 主摄 1x）：

- **MoGe 自己猜不行**：iPhone 16 Pro Max 主摄是 24 mm 等效焦距，16:9 视频不裁切时水平视场角最多 71.6°；
  防抖再裁 1.1–1.2 倍，约 62–66.5°。MoGe 自己估的 78.4° 超出了物理上限，深度因此整体偏近约 25%。
- **不需要去畸变**：radial:1 估出 k1 = 0.037，换算到画面上，中心 0 px、左右边缘约 14 px、四角约 20 px。
  直线检验：取 5 帧里 37 条现实中笔直的边（键盘、纸、烟盒），原图上平均弯曲 0.475 px，
  按 k1 = 0.037 去畸变后 0.453 px，差别远小于约 0.4 px 的测量噪声；|k1| ≥ 0.1 时明显变弯。
  也就是说残留畸变小到量不出来，和手机在机内已做过畸变矫正相符。
  去畸变却要重采样画面、裁掉 1.8%，而且逐帧的 k1 估计不稳（0.009–0.105），所以不做。
  （对比图和检验结果当时在 `runs\depth_estimation\demo01-v2d\compare_undistort\`，已随 runs\ 清掉。）
- **相机模型几乎不影响焦距，权重才影响**：同样 16 帧，`anycalib_gen` 按小孔模型 63.0°、按 radial:1 63.4°；
  `anycalib_pinhole` 65.3°。之前看到的小孔和去畸变相差约 3°，主要来自两套权重不同。
- **选 `anycalib_pinhole`**：它只在无畸变小孔图像上训练，和机内已去畸变的手机画面最对路
  （`anycalib_gen` 还混了畸变和鱼眼图，是 V2D 为鱼眼头戴相机选的）。
  两者在 demo01 上相差 2.3°（约 4% 深度），哪个更接近真值还要拿真值核对：
  用同一部手机、同样的录像设置拍一段棋盘格，或者在点云里量一个已知尺寸的物体。
- **采样 64 帧（V2D 是 16）**：demo01 全部 184 帧逐帧估计，fov 在 60.2–70.2° 之间（标准差 2.0°）。
  均匀取帧后取中位数：16 帧 65.29°，64 帧 65.42°，全部 65.44°。16 帧这次只差 0.15°，但随机抽 16 帧时，
  95% 的结果落在 64.5–67.1°，换了帧可能差 ±1.3°；64 帧基本和全部帧一致。每帧约 0.08 秒，64 帧约 5 秒。
  `--calib-samples` 可改；视频短于 64 帧时就用全部帧。
- 另外：demo01.mp4 最底下 3 行（1077–1079）是黑边，大概是转发时重新编码加上的。
  对 fov 没影响，主点只差 1.5 px。手机原始文件最好用隔空投送或数据线导出，不要经过聊天软件。

#### demo01 前 60 帧的实测（2026-11-09）

| 先验 | 水平视场角 | 逐帧 fx | 深度中位数 | 逐帧深度中位数的波动 |
|---|---|---|---|---|
| 无（MoGe 每帧自己猜） | 78.4° | 1099–1245 | 0.537 m | 18% |
| GeoCalib | 52.9° | 固定 1928 | 0.874 m | 13% |
| AnyCalib 小孔 | 67.6° | 固定 1433 | 0.653 m | 12% |
| AnyCalib `radial:1` + 去畸变 | 64.6° | 固定 1518 | 0.660 m | 11% |

看到的几点：

- **焦距和米制深度绑在一起**：焦距先验从 78° 换到 53°，整个场景的深度从 0.54 m 变成 0.87 m。先验选错，所有尺寸都会按比例错。
- 两种 AnyCalib 互相吻合（65–68°，16 个采样帧之间也只差几度），和常见手机主摄 16:9 视频的视场角相符。
  GeoCalib 在这种俯拍桌面的画面里各帧差很多（48°–65°）——它依赖地平线和竖直线条，俯拍桌面缺少这些线索。
  哪个对，要靠真值确认：手机型号的官方焦距，或者在点云里量一个已知尺寸的东西（如键盘宽度）。
- 固定焦距后，逐帧深度的波动从 18% 降到 11–13%；剩下的部分有多少是相机真在动，要等有了相机位姿才能分清。
- **V2D 默认的 `kb:4`（鱼眼）在这段手机视频上失败**：拟合出的系数是 (−17.6, 147.6, −424, 337)，去畸变后的相机是 NaN。
  V2D 的场景是鱼眼头戴相机，`kb:4` 是为那个准备的。我们的视频来自手机和智能眼镜（非鱼眼），
  所以这里**默认改成 `radial:1`**（一个径向畸变系数）；遇到鱼眼相机再加 `--anycalib-cam-id kb:4`。
  程序现在会在这一步直接报错并给出建议，不会把 NaN 交给 MoGe。
  另外，V2D 对多帧的畸变系数**逐个**取中位数，这几个数可能来自不同帧，拼起来未必是一个合理的模型——系数本身不稳定时这一点会放大问题。

### ③ 内参稳定：取中位数

逐帧的 fx、fy、cx、cy 各取时间中位数，写一个 `intrinsics_stable.json`。中位数对偶尔几帧的离谱估计不敏感。

**值得注意的地方**：深度是在每帧自己的焦距下算出来的，但后续用的是中位数焦距。
焦距和深度在 MoGe 里是耦合的（焦距估大 → 物体推得更远），这样混用会带入一点不一致。
如果在意，可以用 `--fov-from intrinsics_stable.json` 再跑一遍，让所有帧在同一个焦距下解深度（见第 6 节）。

### 可选：逐帧尺度对齐（`align_depth_sequence`）

1. 每帧与参考帧做 SIFT 匹配，**只在背景上**（把物体 mask 抠掉，因为物体本身在动）。
2. 该帧尺度 = 匹配点上 `参考深度 / 当前深度` 的中位数。
3. 匹配不够的帧用相邻帧插值，然后在 **log 空间**做中值滤波（×2 和 ×0.5 离 1 一样远）。
4. 每帧深度乘以平滑后的尺度。

只修尺度不修平移，适合 MoGe 这种米制模型。它假设背景大体静止、相机移动不大；
相机转过大角度后和参考帧没有共同背景，这一步就失效（只能靠插值）。

**用法：`depth_estimation run <视频> --scale-align`**（逐帧尺度对齐），跑完 MoGe 后自动做这一步，写出
`depth_aligned/`、`depth_aligned_scales.json`（每帧的原始/平滑尺度）、`depth_aligned_stats.json`。
查看器（`depth_estimation view`）默认显示 `depth_aligned`。对已有结果也可以单独跑 `depth_estimation align <run>`。

和 V2D 的一处不同：V2D 让所有帧都向参考帧（第 0 帧）看齐，第 0 帧自己的尺度偏了，整段视频就一起偏。
这里默认把所有帧的尺度除以它们的中位数，整体米制尺度取全片的中位数（`--metric-scale reference` 恢复 V2D 的做法）。

**demo01 实测（2026-11-10，相机固定，AnyCalib 小孔先验）**：184/184 帧都估出了尺度。
每个背景像素的深度随时间的晃动（逐像素对时间中位数的相对偏差，再对像素取中位数）：
**1.78%（约 12 mm）→ 0.60%（约 4 mm）**；逐帧深度中位数的波动 5.1% → 1.9%（剩下的部分包括手在画面里移动造成的真实变化）。

前提是每帧都和参考帧有共同的背景，相机动得多的视频不适用，那种情况用多帧联合优化（`video_depth`，见 [video_depth.md](video_depth.md)）。

## 4. 文件格式（和 V2D 完全一致，产物可直接喂给他们的下游模块）

| 文件 | 格式 |
|---|---|
| `depth/NNNNNN.png` | uint16，**逆深度**：`像素 = 65535 / (深度米 + 1)`，向下取整（和 V2D 一样先 clip 再转整数，PNG 逐位相同；NaN 明确写 0）。近处精度高（1 m 处 0.06 mm/级，10 m 处 1.8 mm/级）；0 = 无效 |
| `intrinsics/NNNNNN.json`、`intrinsics_stable.json` | `{"fx","fy","cx","cy","width","height"}`，像素单位 |
| `mask/NNNNNN.png` | MoGe 有效像素，255 = 有效 |

帧号是**原视频的帧号**（用 `--stride` 抽帧时也一样），文件名六位补零。

## 5. 迁移时改了什么

| 地方 | V2D | 这里 | 原因 |
|---|---|---|---|
| 运行方式 | 每个模块一个 Docker 镜像（PyTorch 2.5.1 + CUDA 12.4） | 直接在 `ego` 环境里跑（Windows，Python 3.12，PyTorch 2.11 + CUDA 13.0，复用 `E:\uv-cache`） | 本机没有 Linux 容器流程；复用已有的 torch 缓存 |
| 推理输入 | 重新解码视频 | 读已经抽好的帧 | 保证帧、深度、内参编号一致 |
| 最后一批 | `frame_index == CAP_PROP_FRAME_COUNT - 1` 时才处理 | 按帧列表分批 | 视频元数据里的帧数不准时，V2D 会**丢掉最后不满一批的帧** |
| 断点续跑 | 整步完成才跳过 | 逐帧跳过已完成的 | 长视频中断后不用重来 |
| `align` 的有效性检查 | `d > 0.001` | 另外要求有限值 | 无效像素解码成 inf，V2D 的检查会让 inf 进入比值 |
| AnyCalib 默认相机模型 | `kb:4`（鱼眼） | `radial:1` | 我们的视频来自手机/眼镜；`kb:4` 在 demo01 上拟合出 NaN |
| 去畸变后的帧 | x264 重新编码成视频（crf 17）再解码 | 直接写无损 PNG | 少一次有损压缩 |
| 标定采样帧 | 整段视频里 `linspace(dtype=int)` 取 N 帧 | 同样的取法，但在抽出的帧里取 | `--stride` / `--max-frames` 时只能在已有的帧里取；不加这两个参数时取到的帧相同 |
| 新增 | — | `--prior anycalib` 不去畸变（小孔模型）；`depth_stats.json`、`preview.mp4`、`ply` 命令 | 用来**看见**焦距/尺度的逐帧抖动和深度质量 |

模型权重、MoGe 版本（提交 `925b8ed`）、推理参数、存储格式都没有改。

2026-11-10 又核对了一遍并补齐了几处细节：深度 PNG 改成和 V2D 一样向下取整（之前四舍五入，最多差 1 级）；
标定采样改成向下取整（之前四舍五入，个别采样帧差 1）；GeoCalib 的权重和相机模型分开选（`--geocalib-camera-model`，
默认 `pinhole`，和 V2D 一样；之前选 `distorted` 权重会自动用 `simple_radial`）；可以同时去畸变和跑 GeoCalib，
GeoCalib 在去畸变后的帧上估计；新增 `--run-geocalib`（只要重力方向）；`calibration.json` 每个采样的键和 V2D 相同
（`frame_index`、`roll_rad`/`pitch_rad`、`rotation_matrix`、`uncertainty`、`covariance`）。
**这之前跑出来的结果不受影响，也不用重跑**：深度差别在 1 级以内，旧目录照样能续跑。

## 6. 怎么用

```powershell
conda activate ego
cd E:\ZY_ego

# 先跑 60 帧试试
depth_estimation run demo01 --max-frames 60 --out runs\depth_estimation\demo01-try
# 整段视频
depth_estimation run demo01

# 看结果
#   runs\depth_estimation\demo01\preview.mp4      左原图、右深度（全片同一色标，越暖越近）
#   runs\depth_estimation\demo01\depth_stats.json 每帧的有效比例、深度分位数、fx、fov_x —— 看抖动
depth_estimation ply demo01 --frame 0              # 点云，用 MeshLab / CloudCompare 打开

# 浏览器里交互看 3D 点云（viser，中文界面；装法：uv pip install ... -e .[viz]），打开 http://localhost:8080
#   "深度来源"下拉框：自动找到同名的 depth_estimation 和 video_depth 结果，可在
#     MoGe-2 逐帧（V2D 的深度）/ MoGe-2 + 逐帧尺度对齐 / MoGe-2 + MegaSaM（全分辨率）/ MegaSaM 原始输出 之间切换；
#   "同时显示"：再画一种深度（洋红色），直接看两种哪里不一致；
#   有 video_depth compare 的 compare.json 时，面板显示各来源的稳定性表，每帧显示各来源的静止背景深度；
#   滑块/播放逐帧看；着色可选 图像颜色 / 按深度 / 按帧；"多帧叠加"同时画几帧：
#   相机不动时它们应该重合，分层错开的程度就是 MoGe 逐帧尺度的抖动
depth_estimation view demo01
depth_estimation view runs\video_depth\demo01       # 同样的来源，默认选 MegaSaM
depth_estimation view demo01 --poses demo01        # 按 camera_pose 的轨迹放到世界坐标里，带相机路径
depth_estimation view demo01 --poses demo01 --gravity   # 用重力对齐后的位姿（z 朝上）

# 可选：逐帧尺度对齐（有物体 mask 时加 --masks 文件夹）
depth_estimation align demo01
depth_estimation ply demo01 --frame 0 --depth depth_aligned

# 实验：所有帧用同一个焦距重新解深度，和原版对比
depth_estimation run demo01 --fov-from runs\depth_estimation\demo01\intrinsics_stable.json --out runs\depth_estimation\demo01-fixed-fov

# 可选的焦距先验（每种用单独的 --out，方便对比 depth_stats.json）
depth_estimation run demo01 --prior geocalib --out runs\depth_estimation\demo01-geocalib
depth_estimation run demo01 --prior anycalib --out runs\depth_estimation\demo01-anycalib
depth_estimation run demo01 --prior anycalib --undistort --out runs\depth_estimation\demo01-anycalib-undistort

# 和 V2D 的 ego 流程最接近的组合：去畸变 + GeoCalib 重力（之后 camera_pose run ... --gravity-align 要用）
depth_estimation run demo01 --undistort --run-geocalib --out runs\depth_estimation\demo01-v2d
```

标定结果在 `calibration\<方法>\`：`intrinsics.json`、`distortion.json`、`calibration.json`（每个采样帧的估计），
GeoCalib 另有 `gravity.json`，去畸变时另有 `undistorted_intrinsics.json` 和 `frames_undistorted\`。

## 7. 相机位姿、gsplat、MANO

DROID-SLAM 已经接成下一部分 `camera_pose`，gsplat 已在 WSL 里编译可用，MANO 左手的安装方法——都见 [camera_pose.md](camera_pose.md)。

## 8. 往后可以学的

- ~~V2D 的 DROID-SLAM 步骤~~：已接成 `camera_pose`，包括重力对齐，见 [camera_pose.md](camera_pose.md)。
- SAM3D 物体网格的尺度估计（`mesh_scaled_moge.obj`）用的是 MoGe 深度 + 物体 mask，对应我们以后物体位姿那一部分。
