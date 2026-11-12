# 深度估计（depth_estimation）：MoGe-2 + AnyCalib 焦距先验

代码在 `src\depth_estimation\`，MoGe-2 那部分来自 NVIDIA Video to Data（V2D，Apache-2.0）。
这一步给每帧一张米制深度；之后 `video_depth`（MegaSaM）把它当先验，优化成整段视频一致的深度和相机位姿，
见 [video_depth.md](video_depth.md)。

## 1. 流程

```text
视频 ──① 抽帧（HDR 先转 SDR）──► frames/000000.png …
     ──② AnyCalib 小孔模型，64 帧取中位数 ──► calibration/anycalib/（焦距先验 fov_x）
     ──③ MoGe-2 逐帧推理（焦距固定为先验）──► depth/*.png   intrinsics/*.json   mask/*.png
     ──④ 内参取时间中位数 ──► intrinsics_stable.json
     ──⑤ 检查用 ──► depth_stats.json、preview.mp4
```

每一步的输出已存在就跳过；中断后重跑同一条命令，MoGe 那步逐帧续跑。

## 2. 每一步的原理

### ① 抽帧；HDR 视频先转成 SDR（2026-11-11）

iPhone 默认录 **HDR 视频**（HEVC 10 位、BT.2020 色域、HLG 曲线；`run.json` 的 `video.color` 里记着）。OpenCV 和
V2D 的 ffmpeg 抽帧都不看色彩标记，把 HLG 信号当成 SDR BT.709 直接读——画面发白、发灰（demo01 饱和度中位数 66，
正确转换后 139），之后所有模型（AnyCalib、MoGe、WiLoR、SAM2、SAM3D、Codex 贴图的参考图）看到的都是这种画面。

检测到 HDR（传输曲线 HLG 或 PQ）时，帧由 ffmpeg（`imageio-ffmpeg` 自带的 gyan.dev 版，有 `zscale`/`tonemap`）
解码并做色调映射：线性光（标称峰值 `--hdr-npl`，默认 100 nit）→ BT.709 色域 → `--hdr-tonemap`（默认 hable，
不降饱和）→ BT.709。同时写一份 `video_sdr.mp4`（H.264，同样的映射）。
demo01 上比较了 hable 203 / hable 100 / mobius 100 / clip 203，hable 100 高光不溢出、颜色最自然。SDR 视频不受影响。

### ② AnyCalib：给 MoGe 一个固定焦距

MoGe 每帧自己猜焦距，焦距会抖（demo01 前 60 帧：fx 1099–1245），而且**焦距和米制深度绑在一起**：焦距估大，
整个场景就被推远。所以先用专门的单图标定模型 AnyCalib 估相机：它用 DINOv2 预测每个像素的光线方向，再拟合相机模型。
在 64 帧上各估一次、取中位数，把 `fov_x = 2·atan(W / 2fx)` 交给 MoGe，所有帧共用同一个相机。
权重 `model\anycalib\anycalib_pinhole.pt`，按**无畸变小孔相机**拟合，不去畸变。

为什么这么定（demo01，iPhone 16 Pro Max 主摄 1x；当时比较过的其他方案已从代码里删掉）：

- **MoGe 自己猜不行**：iPhone 16 Pro Max 主摄是 24 mm 等效焦距，16:9 视频不裁切时水平视场角最多 71.6°；
  防抖再裁 1.1–1.2 倍，约 62–66.5°。MoGe 自己估的 78.4° 超出了物理上限，深度因此整体偏近约 25%。
- **不需要去畸变**：带一个径向畸变系数拟合，得到 k1 = 0.037，画面中心 0 px、左右边缘约 14 px、四角约 20 px。
  直线检验：5 帧里 37 条现实中笔直的边（键盘、纸、烟盒），原图上平均弯曲 0.475 px，按 k1 = 0.037 去畸变后 0.453 px，
  差别远小于约 0.4 px 的测量噪声。残留畸变小到量不出来，和手机在机内已做过畸变矫正相符；
  去畸变却要重采样、裁掉 1.8%，逐帧的 k1 估计也不稳（0.009–0.105）。
  V2D 默认的鱼眼模型 `kb:4` 在这段手机视频上直接失败（系数 −17.6, 147.6, −424, 337，去畸变后相机是 NaN）。
- **不用 GeoCalib**：它靠地平线和竖直线条，俯拍桌面缺这些线索，各帧从 48° 到 65°，中位数 52.9°（深度 0.87 m，明显偏远）。
- **选 `anycalib_pinhole` 权重**：它只在无畸变小孔图像上训练，和机内已去畸变的手机画面最对路
  （`anycalib_gen` 还混了畸变和鱼眼图，是 V2D 为鱼眼头戴相机选的）。两者在 demo01 上差 2.3°（约 4% 深度），
  哪个更接近真值还要核对：同一部手机、同样设置拍一段棋盘格，或者在点云里量一个已知尺寸的物体。
- **64 帧（V2D 是 16）**：demo01 全部 184 帧逐帧估计，fov 在 60.2–70.2° 之间（标准差 2.0°）。均匀取帧后取中位数：
  16 帧 65.29°，64 帧 65.42°，全部 65.44°；随机抽 16 帧时 95% 落在 64.5–67.1°，64 帧基本和全部帧一致。
  每帧约 0.08 秒。`--calib-samples` 可改；视频短于 64 帧时用全部帧。

demo01 前 60 帧（2026-11-09）对比，留作参考：

| 先验 | 水平视场角 | 逐帧 fx | 深度中位数 | 逐帧深度中位数的波动 |
|---|---|---|---|---|
| 无（MoGe 每帧自己猜） | 78.4° | 1099–1245 | 0.537 m | 18% |
| GeoCalib | 52.9° | 固定 1928 | 0.874 m | 13% |
| **AnyCalib 小孔（现在用的）** | 67.6° | 固定 1433 | 0.653 m | 12% |
| AnyCalib `radial:1` + 去畸变 | 64.6° | 固定 1518 | 0.660 m | 11% |

另外：demo01.mp4 最底下 3 行（1077–1079）是黑边，大概是转发时重新编码加上的，对 fov 没影响，主点只差 1.5 px。
手机原始文件最好用隔空投送或数据线导出，不要经过聊天软件。

### ③ MoGe-2：一张图得到米制几何

- 网络（DINOv2-Large 骨干）直接预测每个像素的 **3D 点**（点图），而不是只预测深度。
- 内参是从点图里**解**出来的：找一个焦距（和平移），让点投影回去最贴合像素位置。给了 `fov_x` 时焦距固定，只解深度。
- `infer()` 返回 `points`（H×W×3）、`depth`（点的 z）、`mask`（有效像素，天空/无穷远为 0）、`normal`、
  `intrinsics`（按图像宽高归一化，需要乘回 W/H）。
- 每一帧**独立**推理，焦距固定后整体尺度仍会逐帧抖动（demo01 背景像素晃动约 12 mm）——这正是 `video_depth`（MegaSaM）
  要解决的问题。

### ④ 内参稳定：取中位数

逐帧的 fx、fy、cx、cy 各取时间中位数，写 `intrinsics_stable.json`（后面各步都用它）。焦距已经由先验固定，这一步基本不改变数值。

## 3. 文件格式（和 V2D 一致）

| 文件 | 格式 |
|---|---|
| `depth/NNNNNN.png` | uint16，**逆深度**：`像素 = 65535 / (深度米 + 1)`，向下取整（NaN 写 0）。近处精度高（1 m 处 0.06 mm/级，10 m 处 1.8 mm/级）；0 = 无效 |
| `intrinsics/NNNNNN.json`、`intrinsics_stable.json` | `{"fx","fy","cx","cy","width","height"}`，像素单位 |
| `mask/NNNNNN.png` | MoGe 有效像素，255 = 有效 |
| `calibration/anycalib/` | `intrinsics.json`（中位数相机）、`calibration.json`（每个采样帧的估计） |

帧号是**原视频的帧号**（用 `--stride` 抽帧时也一样），文件名六位补零。

## 4. 和 V2D 不一样的地方

| 地方 | V2D | 这里 | 原因 |
|---|---|---|---|
| 焦距先验 | 默认没有（MoGe 每帧自己猜）；可选 AnyCalib 鱼眼去畸变或 GeoCalib | 固定用 AnyCalib 小孔模型，不去畸变 | 见第 2 节② |
| HDR 视频 | 当 SDR 读，画面发白 | 色调映射成 SDR | 见第 2 节① |
| 运行方式 | 每个模块一个 Docker 镜像（PyTorch 2.5.1 + CUDA 12.4） | 直接在 `ego` 环境里跑（Windows，Python 3.12，PyTorch 2.11 + CUDA 13.0） | 本机没有 Linux 容器流程 |
| 推理输入 | 重新解码视频 | 读已经抽好的帧 | 保证帧、深度、内参编号一致 |
| 最后一批 | `frame_index == CAP_PROP_FRAME_COUNT - 1` 时才处理 | 按帧列表分批 | 视频元数据里的帧数不准时，V2D 会**丢掉最后不满一批的帧** |
| 断点续跑 | 整步完成才跳过 | 逐帧跳过已完成的 | 长视频中断后不用重来 |
| 新增 | — | `depth_stats.json`、`preview.mp4`、`view` 查看器 | 用来**看见**深度质量和逐帧抖动 |

MoGe 版本（提交 `925b8ed`）、权重、推理参数、存储格式都没有改。

## 5. 怎么用

```powershell
conda activate ego
cd E:\ZY_ego

depth_estimation run demo01 --max-frames 60 --out runs\depth_estimation\demo01-try   # 先试 60 帧
depth_estimation run demo01                                                          # 整段视频

# 看结果
#   runs\depth_estimation\demo01\preview.mp4      左原图、右深度（全片同一色标，越暖越近）
#   runs\depth_estimation\demo01\depth_stats.json 每帧的有效比例、深度分位数、fx、fov_x
# 浏览器里交互看 3D 点云（viser，中文界面），打开 http://localhost:8080
#   "深度来源"：自动找到同名的 depth_estimation 和 video_depth 结果，
#     在 MoGe-2 逐帧 / MoGe-2 + MegaSaM（全分辨率）/ MegaSaM 原始输出 之间切换；
#   "同时显示"：再画一种深度（洋红色），直接看两种哪里不一致；
#   "多帧叠加"：同时画几帧，相机不动时它们应该重合，分层错开的程度就是逐帧尺度的抖动
depth_estimation view demo01
depth_estimation view runs\video_depth\demo01       # 同样的来源，默认选 MegaSaM
```
