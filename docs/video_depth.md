# 多帧一致深度（video_depth）：MegaSaM + MoGe-2

代码：`src\video_depth\`。MegaSaM：github.com/mega-sam/mega-sam，提交 `a27b4e6`（Apache-2.0，Li 等，CVPR 2025）。

## 1. 为什么要它

MoGe-2 一次只看一张图，每帧的米制尺度由一个"尺度头"从整张图回归出来。手在画面里动、曝光变化、
压缩噪声都会让它变一点，于是整个背景前后伸缩——**相机固定也会抖**（demo01：逐帧尺度 0.84–1.15，
背景像素随时间晃动约 12 mm）。V2D 的 `depth/` 也一样在抖，它靠后面的中位数、EKF、gsplat 精修间接消化；
MINT（无矩科技）的标注流程则是 GeoCalib → MoGe-2 → **MegaSaM**：把 MoGe 当先验，多帧联合优化出一致的深度。
这里走的就是 MINT 这条路。

## 2. MegaSaM 做了什么（三步，按原仓库脚本逐行移植，在一个进程里跑）

```text
depth_estimation 的结果：frames/  depth/（MoGe-2）  intrinsics_stable.json（AnyCalib 定的焦距）
   │
   │ ① 相机跟踪（camera_tracking_scripts/test_demo.py）
   │    改过的 DROID-SLAM：束调整里把单目深度当先验，同时估计每个像素"在动"的概率（手、被拿的物体）；
   │    最后对**所有帧**做一次全局束调整 → 每帧位姿
   │ ② 光流（cvd_opt/preprocess_flow.py）
   │    RAFT，帧间隔 1/2/4/8/15，正反两个方向，前后一致性检查得到可信像素
   │ ③ 一致深度优化（cvd_opt/cvd_opt.py），位姿固定：
   │    (a) 每帧一个尺度 + 平移（100 步）——这一步专门消掉逐帧尺度抖动
   │    (b) 每个像素的视差（400 步）
   │    损失：沿光流对应的两点，按位姿投影过去深度要一致；同时形状贴近先验（尺度无关项、梯度、法向）；
   │    "在动"的像素权重降低
   ▼
米制：MegaSaM 在归一化尺度下工作，最后取一个全局倍数 = 各帧 median(MoGe 深度 / MegaSaM 深度) 的中位数
```

**MoGe-2 怎么接进去**：MegaSaM 原本要两个单目输入——Depth-Anything（相对视差）和 UniDepth（米制深度 + 视场角）。
按 MINT 的做法都换成 MoGe-2：MoGe 的视差当"相对视差"，MoGe 的深度当"米制深度"，视场角用 MoGe 跑的那个
（即 AnyCalib 小孔模型估的）。两者本是同一个东西，所以 MegaSaM 的"相对→米制对齐"得到尺度 1、平移 0。

## 3. 和 MegaSaM 原脚本不同的地方（代码里标了 "ours"）

| 地方 | MegaSaM demo | 这里 | 原因 |
|---|---|---|---|
| 单目输入 | Depth-Anything + UniDepth | MoGe-2（两个角色都是它） | MINT 的做法；已有、米制、焦距已定 |
| 焦距 | `terminate(_opt_intr=True)`，条件好时优化焦距 | 默认固定（`--opt-focal` 恢复） | 焦距已用 AnyCalib 定好，MoGe 深度就是在这个焦距下算的 |
| 光流 | 写到 `cache_flow/` 再读 | 留在内存里 | 一个进程跑完三步 |
| 尺度 | 归一化尺度 | 乘一个全局倍数回到米 | 后续部分都要米制 |
| 输出 | `outputs/*_sgd_cvd_hr.npz`（float16） | 本项目格式：`depth_megasam/` uint16 逆深度 PNG、`poses/` | 和 depth_estimation / camera_pose 一致，查看器直接能看 |
| 分辨率 | 只有跟踪分辨率（1080p → 584×328） | 另外生成全分辨率 `depth/`（第 5 节） | 后续手、物体对齐要原图分辨率 |
| 长视频 | 所有帧一次处理（输出最多 1000 帧） | 分段处理再拼接（`--segment-length`、`--overlap`，第 6 节） | 显存随帧数增长 |

## 4. 环境（WSL）

MegaSaM 自带的改版 DROID 会编译出和上游 DROID-SLAM 同名的 `droid_backends`、`lietorch`，所以单独一个环境
`zyego-megasam`（从 `zyego-cuda` 克隆，PyTorch 2.7.0 + CUDA 11.8）。扩展用 `build_ext --inplace` 编在
`~/zyego-external/mega-sam/base/` 里，运行时通过 PYTHONPATH 找到（它的 setup.py 里有两个 `setup()`，pip 装不了）。

```powershell
wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_cuda_env.sh megasam   # 搭建（已完成；可重复执行）
video_depth check                                                            # 自检
```

搭建时对 MegaSaM 源码做的改动（脚本里自动做）：
- 编译架构从 sm_70–86 改成 sm_89（RTX 4090）；
- PyTorch 2.x 的分发宏要 `scalar_type()`，`tensor.type()` 不再能转换：MegaSaM 自己的 3 处 + 它附带的旧版 lietorch
  的 38 处，改法和上游 lietorch 完全一样。

权重在 `model\megasam\`：`megasam_final.pth`（随 MegaSaM 仓库发布）、`raft-things.pth`（RAFT 官方下载源）。

## 5. 用法

```powershell
depth_estimation run demo01                      # 先要有 MoGe 深度（默认 AnyCalib 小孔先验）
video_depth run demo01                           # → runs\video_depth\demo01\
video_depth run demo01 --max-frames 40 --out runs\video_depth\demo01-try   # 先试几十帧
depth_estimation view runs\video_depth\demo01 --poses runs\video_depth\demo01   # 看结果（带相机轨迹）
```

结果：

| 文件 | 内容 |
|---|---|
| `depth\`、`intrinsics_stable.json`、`mask\` | **全分辨率**一致深度（和原图同尺寸），米；内参就是 depth_estimation 的（焦距没变）。可以像 depth_estimation 的结果一样直接用 |
| `depth_megasam\`、`intrinsics_megasam.json` | MegaSaM 原始输出，跟踪分辨率（1080p → 584×328）；内参的主点不在正中（缩放后从右下裁到 8 的倍数） |
| `poses\`、`trajectory_tum.txt` | 每帧相机→世界，米，格式同 camera_pose |
| `motion_prob.npy` | MegaSaM 估计的"静止"概率（1/8 分辨率；越小越可能在动） |
| `megasam.json` | 米制倍数、MoGe 每帧尺度相对一致深度的偏差、每帧 (a) 步的尺度、耗时、显存、轨迹摘要 |
| `run.json` | 给查看器用：原图在 depth_estimation 结果里 |

### 全分辨率（ours，`fullres.py`）

`depth/ = MoGe 全分辨率深度 × exp(上采样(平滑(log(MegaSaM / MoGe))))`：
修正比例在低分辨率上先做 5×5 中值、再做 σ=2 的高斯平滑（不让 MegaSaM 模糊掉的物体边缘混进来），
然后双线性放大；MoGe 的锐利边缘和细节保留，无效像素保持无效。在 demo01 上比较过三种放大方式（每 4 帧取一帧）：

| 方式 | 背景晃动 | 边缘"光晕" | 和 MegaSaM 的差（低分辨率上） |
|---|---|---|---|
| 每帧一个倍数 | 0.27% | 0 | 0.18% |
| 比例直接双线性放大 | 0.13% | 0.56% | 0.00% |
| **平滑后的比例（采用）** | **0.13%** | **0.034%** | **0.01%** |
| （MoGe 原始） | 1.63% | — | — |

对已有结果重算：`video_depth fullres <run>`。

## 6. 长视频：分段（ours，`stitch.py`）

MegaSaM 一个片段里所有帧都在显存里，一致深度那步一次处理所有帧对（每帧约 5 对），所以显存随长度增长
（184 帧 1080p：11.4 GB）。超过 `--segment-length`（默认 200）帧就切成互相重叠 `--overlap`（默认 30，
要大于 MegaSaM 最大的光流间隔 15）帧的片段，分别跑，再从左到右拼：

1. **深度尺度**：重叠帧上 median(已拼好的深度 / 新片段深度) 的中位数，乘到新片段的深度和相机平移上
   （每段本来都按 MoGe 定了米制，所以接近 1，这一步消掉段与段之间的小跳变）；
2. **世界坐标**：每段的世界 = 它自己的第一帧相机。用重叠帧求一个刚体变换（旋转取弦距平均，平移取平均），
   把新片段放进已拼好的世界里；
3. **重叠区过渡**：深度按权重从旧段线性过渡到新段；位姿和运动概率取权重大的那一边。

最后整段再用一个全局倍数对齐 MoGe（每段已经是米制，这个倍数接近 1）。`megasam.json` 的 `joins` 记录每个接缝的
尺度和"相机位置对不上的程度"。

**demo01 验证**（`--segment-length 100 --overlap 30` → 3 段：0–99、42–141、84–183）：

| | 不分段 | 分 3 段 |
|---|---|---|
| 接缝的深度尺度修正 | — | 1.0047、1.0013 |
| 接缝处相机位置对不上 | — | 中位数 0.3 mm（58 个重叠帧） |
| 背景晃动（全分辨率 `depth/`） | 0.119% | 0.123% |
| 相邻帧深度中位数的最大变化 | 0.34%（第 99 帧，手的动作） | 0.31%（同一帧）——接缝处没有跳变 |
| 两者逐像素差别 | — | 中位数 0.04% |
| 显存峰值 / 耗时 | 11.4 GB / 4.8 分钟 | 8.8 GB / 7.5 分钟（重叠帧算了两遍） |

## 7. demo01 实测（2026-11-10，iPhone 16 Pro Max 1x，相机固定，184 帧）

| | 背景像素随时间的晃动（典型值） | 场景深度中位数 |
|---|---|---|
| MoGe-2 原始 | 1.79%（12.3 mm） | 0.677 m |
| `depth_estimation run --scale-align` | 0.59%（4.1 mm） | 0.679 m |
| **MegaSaM（video_depth）** | **0.12%（0.8 mm）** | 0.679 m |

（同一像素网格 584×328 上比较：每个像素对时间中位数的相对偏差，取中位数。）

- MoGe 每帧尺度相对一致深度在 0.84–1.15 之间（标准差 4.9%）。最明显的是第 92 帧：MoGe 整个场景近了约 15%，
  MegaSaM 里桌面和其他帧一样（对比图 `runs\video_depth\demo01\compare_moge_megasam.jpg`）。
- **手和物体还在**：被拿起的烟盒、手仍然明显高出桌面——"在动"的像素被降权，没有被一致性压平。代价是边缘更软
  （分辨率低 + 平滑）。
- 相机轨迹：范围 ≤ 9 mm、最大转角 0.4°，和"相机固定"一致（DROID-SLAM 之前测的是 ≤ 7 mm）。
- 米制倍数 0.980（MegaSaM 尺度 → 米）；焦距保持 1494.9。
- 耗时 4.8 分钟（跟踪 99 s、光流 74 s、一致深度 118 s），显存峰值 11.4 GB。一致深度那步一次处理所有帧对
  （184 帧 ≈ 890 对），显存随视频长度增长；更长的视频可能要分段。

### 7b. 对比命令 `video_depth compare`（2026-11-11 重跑 demo01）

```powershell
video_depth compare demo01      # 输出到 runs\video_depth\demo01\
```

| 文件 | 内容 |
|---|---|
| `compare.mp4` | 原图 \| MoGe-2 逐帧（= V2D 的深度）\| MoGe-2 + MegaSaM，**整段视频同一个色彩范围**——深度一跳，颜色就闪 |
| `compare_background.png` | 静止背景的深度中位数随帧变化的曲线 |
| `compare.json`、`compare_static_mask.png` | 下面的数字；用来统计的静止背景像素 |

静止背景 = 整段视频里颜色基本不变的像素（手和被拿动的物体自动排除）；只适用于固定机位。demo01 选出了画面的 11.9%
（键盘上方的桌面和纸张；手扫过的下半张桌面被排除了）。全分辨率：

| | 整帧尺度抖动 | 像素晃动 | 像素晃动 (mm) | 相邻帧跳变 |
|---|---|---|---|---|
| MoGe-2 逐帧（V2D） | 4.93% | 4.97% | 35.0 | 1.24% |
| **MoGe-2 + MegaSaM** | **0.36%** | **0.38%** | **2.7** | **0.03%** |

（和第 7 节的数字不同，是因为统计的像素不同：第 7 节用 584×328 网格上所有有效像素对时间中位数的偏差。）
曲线上 MoGe 逐帧在 0.60–0.81 m 之间跳，最大的一次在 68–100 帧（两只手同时入画，整个场景被拉远又拉近）；
MegaSaM 是 0.71–0.72 m 的一条线，只在两只手入画时下沉约 1.4%。

**加上 V2D 的"修正后"深度（2026-11-11）**：V2D 唯一产生修正后场景深度的是最后的 gsplat 精修
（背景 = 20 帧深度融合的一组静态高斯，从每帧相机渲染）。为了公平，另跑了一遍用 V2D 原本输入的 HOI 流程
（`HOI prepare demo01 --depth moge --poses droid --out demo01-v2d --reuse-from demo01`：MoGe-2 逐帧深度 + DROID-SLAM），
精修加了 `--rendered_depth_dir` 写出逐帧渲染深度（`runs\HOI\<run>\depth_refined_simple\`）。`video_depth compare`
自动收进同一视频的这些结果（每种深度只用它自己有效的像素统计，"覆盖"列是用到的比例）：

| | 整帧尺度抖动 | 像素晃动 | 晃动 (mm) | 相邻帧跳变 | 覆盖 |
|---|---|---|---|---|---|
| MoGe-2 逐帧（V2D 的输入） | 4.93% | 4.97% | 35.0 | 1.24% | 100% |
| V2D 修正后（MoGe-2 逐帧 + gsplat 精修） | 1.44% | 2.56% | 17.6 | 1.60% | 100% |
| **MoGe-2 + MegaSaM** | **0.36%** | **0.38%** | **2.7** | **0.03%** | 100% |
| MegaSaM + gsplat 精修 | 1.31% | 1.47% | 10.6 | 0.80% | 72% |

- V2D 精修后整帧尺度稳了很多（68–100 帧的大塌陷没了），但背景比 MegaSaM 近约 3%，渲染出来近、远两层混成噪点：
  20 帧尺度不一的 MoGe 深度融合进同一个世界，桌面成了几层（背景高斯也撑到了 10 万上限，MegaSaM 那组 5.3 万）；
  精修**不优化背景高斯的位置**，这几层修不掉。
- 两组精修的曲线都在 22–25、61、100–103、151–153 帧有尖刺（左右手轨迹的起止帧附近），原因还没查实。
- 精修的渲染深度不是干净的深度图；它的用途是对齐位姿。要用深度，用 MegaSaM。

**毛刺从哪来（对照实验，2026-11-11）**：用 refine_simple 自己的函数和参数（20 帧、5 mm 体素、≤10 万点、
半径 = z/fx×1.5、不透明度 0.5、960×540）重建**未训练**的背景，只改一个因素，在第 0 帧量背景的局部毛刺
（log 深度与 5×5 中值之差的 p90）和覆盖率（alpha ≥ 0.5）：

| 输入深度 | 圆片大小 | 读深度的方式 | 覆盖 | 毛刺 p90 |
|---|---|---|---|---|
| MegaSaM | V2D 原样（半径 1.35 mm） | 高斯中心 z（写出 depth_refined_simple 用的） | 67% | 3.21% |
| MegaSaM | V2D 原样 | gsplat 光线交点期望深度 | 67% | 3.21%（完全相同） |
| MegaSaM | 放大到 5 mm（= 点间距） | 两种都一样 | 100% | 0.13% |
| MoGe 逐帧（V2D） | V2D 原样 | 两种都一样 | 88% | 3.01% |
| MoGe 逐帧（V2D） | 放大到 5 mm | 两种都一样 | 100% | 0.59%（且比当帧深度近约 10%） |

1. **不是读法的问题**：背景圆片的朝向固定为正对相机，光线交点深度 = 圆片中心深度，两种读法逐像素相同。
2. **主因是 V2D 背景的表示**：点按 5 mm 体素去重，圆片半径却按"每像素一个点"设定（≈1.35 mm），圆片之间是空洞，
   空洞边缘就是毛刺；只把圆片放大到点间距，毛刺降 25 倍。精修只训练颜色/不透明度/大小、**不动位置**，学习率 1e-3、
   460 步，只能部分补洞（训练后全分辨率覆盖约 81%）。
3. **背景里烘进了会动的东西**：背景只排除 SAM2 的手和物体 mask（不外扩），**小臂/袖子不在任何 mask 里**，
   20 帧里的袖子成了静态背景（第 0 帧画面没有手，渲染深度下方两角却有两团）；手和盒子 mask 边缘漏掉的一圈像素
   成了飘在桌面前的细线。
4. **V2D 输入的多层**：同样放大圆片后，MoGe 逐帧输入的毛刺仍是 MegaSaM 输入的 4.5 倍，并且偏近（前面那层挡住）。

这些都是 V2D `refine_simple` 本身的设计（它把背景当作位姿精修的陪衬，从不输出深度），不是迁移或读法引入的。

**VS Code 里深度 PNG 是一片灰**：PNG 存的是 uint16 逆深度 `65535 / (米 + 1)`（V2D 的格式，给程序读的），桌面 0.5–1 m
对应 32000–44000，VS Code 把 16 位按 0–65535 映射到灰度，看起来就是一片中灰。要看用 `compare.mp4`、
`depth_estimation` 的 `preview.mp4`，或 `depth_estimation view` 看 3D 点云。

## 8. 还可以做的

- 相机移动的视频上验证（demo01 相机不动，没检验到位姿精度）。
- 和 camera_pose（DROID-SLAM + MoGe 定尺度）对比轨迹。
