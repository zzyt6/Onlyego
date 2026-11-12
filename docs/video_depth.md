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
| 输出 | `outputs/*_sgd_cvd_hr.npz`（float16） | 本项目格式：`depth_megasam/` uint16 逆深度 PNG、`poses/` | 和 depth_estimation 一致，查看器直接能看 |
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
depth_estimation view runs\video_depth\demo01   # 看结果
```

结果：

| 文件 | 内容 |
|---|---|
| `depth\`、`intrinsics_stable.json`、`mask\` | **全分辨率**一致深度（和原图同尺寸），米；内参就是 depth_estimation 的（焦距没变）。可以像 depth_estimation 的结果一样直接用 |
| `depth_megasam\`、`intrinsics_megasam.json` | MegaSaM 原始输出，跟踪分辨率（1080p → 584×328）；内参的主点不在正中（缩放后从右下裁到 8 的倍数） |
| `poses\`、`trajectory_tum.txt` | 每帧相机→世界，米（Transform3d JSON，读写见 `video_depth\poses.py`；object_model / object_pose 用它） |
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
| 逐帧尺度对齐（SIFT 对齐到参考帧，当时的 `--scale-align`，已删） | 0.59%（4.1 mm） | 0.679 m |
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

### 7b. 静止背景上的稳定性（2026-11-11，用当时的对比命令统计；命令已删）

静止背景 = 整段视频里颜色基本不变的像素（手和被拿动的物体自动排除）；只适用于固定机位。demo01 选出了画面的 11.9%
（键盘上方的桌面和纸张；手扫过的下半张桌面被排除了）。全分辨率：

| | 整帧尺度抖动 | 像素晃动 | 像素晃动 (mm) | 相邻帧跳变 |
|---|---|---|---|---|
| MoGe-2 逐帧（V2D） | 4.93% | 4.97% | 35.0 | 1.24% |
| **MoGe-2 + MegaSaM** | **0.36%** | **0.38%** | **2.7** | **0.03%** |

（和第 7 节的数字不同，是因为统计的像素不同：第 7 节用 584×328 网格上所有有效像素对时间中位数的偏差。）
曲线上 MoGe 逐帧在 0.60–0.81 m 之间跳，最大的一次在 68–100 帧（两只手同时入画，整个场景被拉远又拉近）；
MegaSaM 是 0.71–0.72 m 的一条线，只在两只手入画时下沉约 1.4%。

**和 V2D 的"修正后"深度比过（2026-11-11；V2D 的流程和 gsplat 精修代码之后已从项目删掉）**：V2D 唯一产生
修正后场景深度的是最后的 gsplat 精修（背景 = 20 帧深度融合的一组静态高斯，从每帧相机渲染）。同样的统计：

| | 整帧尺度抖动 | 像素晃动 | 晃动 (mm) | 相邻帧跳变 | 覆盖 |
|---|---|---|---|---|---|
| MoGe-2 逐帧（V2D 的输入） | 4.93% | 4.97% | 35.0 | 1.24% | 100% |
| V2D 修正后（MoGe-2 逐帧 + gsplat 精修） | 1.44% | 2.56% | 17.6 | 1.60% | 100% |
| **MoGe-2 + MegaSaM** | **0.36%** | **0.38%** | **2.7** | **0.03%** | 100% |
| MegaSaM + gsplat 精修 | 1.31% | 1.47% | 10.6 | 0.80% | 72% |

精修把整帧尺度稳住了一些，但渲染出的深度满是毛刺：背景圆片比点间距小得多（中间是空洞），精修又不动高斯的位置；
小臂/袖子不在任何 mask 里，被烘进了静态背景；MoGe 逐帧输入融合后桌面成了几层。精修的渲染深度是给位姿对齐用的，
不是干净的深度图。要用深度，用 MegaSaM。

**VS Code 里深度 PNG 是一片灰**：PNG 存的是 uint16 逆深度 `65535 / (米 + 1)`（V2D 的格式，给程序读的），桌面 0.5–1 m
对应 32000–44000，VS Code 把 16 位按 0–65535 映射到灰度，看起来就是一片中灰。要看用
`depth_estimation` 的 `preview.mp4`，或 `depth_estimation view` 看 3D 点云。

## 8. 还可以做的

- 相机移动的视频上验证（demo01 相机不动，没检验到位姿精度）。
