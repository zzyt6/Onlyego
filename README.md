# ZY_ego

从一段手持物体的视频得到：被操作物体的参考裁剪图和描述（第一步），以及由 Codex 为拟合好的物体网格生成的贴图（第二步）。
第一步代码来自 AGILE `generation/`，精简后在 Windows 上独立运行。

项目按部分组织，每个部分是 `src\` 下以该部分命名的一个包：

| 包 | 命令 | 内容 |
|---|---|---|
| `src\four_stage_view` | `four_stage_view` | 视频 → 参考裁剪图和描述（第一步）→ Codex 用参考图 + object_model 拟合好的网格生成贴图 texture.png（第二步） |
| `src\depth_estimation` | `depth_estimation` | 视频 → 逐帧米制深度 + 相机内参（MoGe-2，迁移自 NVIDIA V2D）。原理和改动见 [docs/depth_estimation.md](docs/depth_estimation.md) |
| `src\camera_pose` | `camera_pose` | 深度结果 → 每帧相机在米制世界坐标里的位姿（DROID-SLAM + MoGe 定尺度，在 WSL 里跑）。见 [docs/camera_pose.md](docs/camera_pose.md) |
| `src\hand_model` | `hand_model` | MANO 手模型加载（smplx），按 Wuji MINT 的方法修正官方左手模型的形状缺陷；`hand_model check` 自检。见 [docs/hand_model.md](docs/hand_model.md) |
| `src\video_depth` | `video_depth` | MoGe 深度 → 多帧一致的米制深度（全分辨率）+ 相机位姿（MegaSaM，MoGe-2 当先验，MINT 的做法；长视频分段；在 WSL 环境 zyego-megasam 里跑）。见 [docs/video_depth.md](docs/video_depth.md) |
| `src\HOI` | `HOI` | 手和物体：从 V2D 迁移过来的 WiLoR/HaMeR/HaWoR、GroundingDINO、SAM2、SAM3D、FoundationPose、gsplat 精修代码和 ego 流程编排。`HOI run <视频名> --object-prompt ... --reference-frame N` 在 7 个 WSL conda 环境里跑完整的 V2D ego 流程，深度和相机来自 `video_depth`（`tools\wsl\setup_hoi_env.sh` 搭环境，`check_hoi.sh` 自检）。见 [docs/HOI.md](docs/HOI.md) |
| `src\object_model` | `object_model` | 被拿的物体的 3D 网格：WiLoR + Grounding DINO + SAM2 分割 → 自动选手碰到物体前的最后一帧 → 之前静止帧的 MegaSaM 深度融合成点图 → SAM3D 生成网格 → 拟合尺度和位姿（形状不动）→ 世界坐标。`object_model view` 在 Viser 里和点云一起看。见 [docs/object_model.md](docs/object_model.md) |
| `src\object_pose` | `object_pose` | 被拿的物体每帧的位姿（网格来自 object_model）。`object_pose run` 用 FoundationPose 跟踪（起点用 object_model 拟合的位姿，网格带 Codex 贴图），跟丢后按 SAM2 轮廓 + FoundationPose 打分重新注册；`object_pose baseline` 是 V2D 原来的跟丢处理，做对照；按"可见部分 IoU / 静止段漂移 / 抖动"打分并出叠加视频。见 [docs/object_pose.md](docs/object_pose.md) |

## 环境

```powershell
conda activate ego          # E:\envs\ego，Python 3.12，已设置 PYTHONUTF8=1
cd E:\ZY_ego

# 已装好；重建环境时按顺序执行。用 uv + E:\uv-cache，torch 直接从缓存硬链接，不重新下载
uv pip install --python E:\envs\ego\python.exe --cache-dir E:\uv-cache --link-mode hardlink `
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python E:\envs\ego\python.exe --cache-dir E:\uv-cache --link-mode hardlink -e .[dev,depth,hand,viz]
```

OpenCV 只装 `opencv-python` 一个（MoGe 依赖它）；不要再装 `opencv-python-headless`，两者共用 `cv2` 目录会互相覆盖。

需要编译 CUDA 扩展的部分（DROID-SLAM、gsplat）放在 WSL 发行版 `V2D-Ubuntu` 的 conda 环境 `zyego-cuda` 里
（Python 3.10、PyTorch 2.7.0 + CUDA 11.8），由 Windows 上的命令自动调用：

```powershell
wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_cuda_env.sh all   # 搭建（已完成；可重复执行）
camera_pose check                                                        # 自检
```

- 密钥放在 `secrets.env`（不进 git，格式见 `secrets.example.env`）。已有同名环境变量时以环境变量为准。
  - `AGILE_OPENAI_API_KEY`：第一步视频分析（Responses API）
  - `AGILE_IMAGE_API_KEY`：第二步图像模型（`/images/edits`）
- 两步都调用本机的 Codex CLI，沿用 Codex 自己的登录和服务商配置（`%USERPROFILE%\.codex\config.toml`）。
  第二步里 Codex 用自带 imagegen 技能的命令行脚本生图，模型和接口见 `configs/ego.json` 的 `image`
  （密钥变量名 `api_key_env`，值在 `secrets.env`，只放进 Codex 的运行环境）。
  程序会自动找到 npm 包里的原生 `codex.exe`；找不到时在 `configs/ego.json` 的 `agent.executable` 写完整路径。

## 运行

```powershell
# 第一步：识别被操作的物体，裁剪参考图，生成物体描述
# 视频放在 data\videos\ 下，直接写文件名（也可以给完整路径）
four_stage_view analyze-video demo01
# 另开一个终端查看进度
four_stage_view video-status --watch

# 第二步：Codex 给拟合好的物体网格生成贴图（先跑 object_model run demo01；用第一步打印的任务编号）
four_stage_view generate-texture job-0001 --object demo01
four_stage_view status job-0001
```

输入视频在 `dataideos\`（不进 git），结果在 `runs\job-XXXX\`：

| 文件 | 内容 |
|---|---|
| `references\`、`references.png` | 第一步选出的参考裁剪图 |
| `original-frames\` | 裁剪所在的原始视频帧 |
| `object-description.json` | 物体描述、每张参考图对应的面、看到/没看到的面 |
| `texture-NN\` | 第二步第 NN 次 Codex 会话的工作目录：交给它的材料（`model\` 网格、`references\`、`reference_frame\` 照片 + 相机 + 位姿、`TASK.md` 任务说明）、它写的脚本和生成的图、`texture.png`、`result.json`（它的说明和自查）、`codex_events.jsonl`（全过程） |
| `job.json` | 任务状态，`texture.attempts` 记录每次会话 |

第二步我们的程序只做三件事：把网格（只有 UV，不带 SAM3D 的贴图）、参考图、参考帧照片和位姿交给 Codex；
在工作目录里运行 `codex exec`（只能写这个目录、可以联网生图、生图次数上限见 `texture.max_image_calls`）；
收回 `texture.png`，原样贴到同一个网格上得到 `mesh_textured.glb`，连同贴图复制到 `runs\object_model\<视频>\`。
贴图的内容全部由 Codex 和它的生图产生，自查也由它自己做。`object_model texture demo01` 在 Viser 里看结果。

一次会话会花生图次数，不会自动重来；要新开一次会话用 `--retry`。

## 深度估计（depth_estimation）

```powershell
depth_estimation run demo01 --max-frames 60   # 先试 60 帧；去掉 --max-frames 跑整段
depth_estimation ply demo01 --frame 0          # 某一帧的彩色点云
depth_estimation view demo01                   # 浏览器里看 3D 点云（viser），打开 http://localhost:8080
depth_estimation align demo01                  # 可选：逐帧尺度对齐（也可直接 run 时加 --scale-align）

# 焦距先验：默认先用 AnyCalib 按无畸变小孔相机估计焦距，交给 MoGe（= --prior anycalib）
# iPhone 1x 等手机主摄在机内已去畸变，实测不需要再去畸变（见 docs/depth_estimation.md）
depth_estimation run demo01 --prior none                   # 不给先验，MoGe 每帧自己猜焦距（V2D 默认）
depth_estimation run demo01 --prior geocalib               # GeoCalib：焦距 + 重力方向
depth_estimation run demo01 --undistort                    # AnyCalib 拟合镜头畸变，先去畸变再算深度（V2D 的用法）
                                                           # 按手机/智能眼镜镜头拟合（radial:1）；鱼眼相机加 --anycalib-cam-id kb:4
depth_estimation run demo01 --run-geocalib                 # 另跑 GeoCalib 取重力方向（camera_pose --gravity-align 用）

# 相机位姿（DROID-SLAM，在 WSL 里跑）；--gravity-align 用 GeoCalib 重力把世界转成 z 朝上
camera_pose run demo01 --gravity-align

# 多帧一致深度（MegaSaM + MoGe-2，在 WSL 里跑）：消掉 MoGe 逐帧尺度抖动，同时给出相机位姿
video_depth run demo01
depth_estimation view runs\video_depth\demo01 --poses runs\video_depth\demo01
```

结果在 `runs\depth_estimation\<视频名>\`：

| 文件 | 内容 |
|---|---|
| `frames\` | 抽出的帧（文件名 = 原视频帧号）；HDR 视频（iPhone 默认）会先转成 SDR，`--hdr-tonemap off` 关闭 |
| `video_sdr.mp4` | 只有 HDR 视频才有：转成 SDR 的整段视频，给直接读视频的步骤用（如 `HOI`） |
| `depth\` | uint16 逆深度 PNG，`像素 = 65535 / (米 + 1)`，0 = 无效 |
| `intrinsics\`、`intrinsics_stable.json` | 逐帧内参、取中位数后的稳定内参 |
| `mask\` | MoGe 有效像素 |
| `depth_stats.json`、`preview.mp4` | 每帧深度分位数和焦距；原图/深度并排预览 |
| `depth_aligned\`、`pointclouds\` | `align`、`ply` 命令的输出 |
| `calibration\<方法>\`、`frames_undistorted\` | 用了 `--prior` / `--undistort` 时的标定结果和去畸变后的帧 |

中断后重跑同一条命令会从断点继续。

## 相机位姿（camera_pose）

```powershell
camera_pose run demo01-anycalib     # 参数是 runs\depth_estimation\ 下的一个深度结果
```

结果在 `runs\camera_pose\<深度结果名>\`：`poses\`（每帧相机→世界 4×4，米）、`trajectory_tum.txt`（evo 可读）、
`trajectory.png`（俯视/侧视轨迹）、`summary.json`（移动距离、转角）、`slam.json`（关键帧、尺度）、`pointcloud.ply`。
相机几乎不动时只有 1 个关键帧，会输出单位位姿并标记 `static_camera`。

## 测试

```powershell
pytest
```

## 模型（`model\`，供后续深度、分割、手和物体位姿使用）

| 目录 | 内容 |
|---|---|
| `moge-2-vitl-normal` | MoGe-2 单目米制深度 |
| `sam2.1-hiera-large` | SAM2.1 分割（`.pt` 与 `.safetensors` 两种格式） |
| `dinov3-vitl16-pretrain-lvd1689m` | DINOv3 图像特征 |
| `foundationpose` | FoundationPose refiner / scorer 权重 |
| `mano` | MANO 左右手模型（v1.2；WSL 里另有一份在 `~/models/MANO/`） |
| `wilor` | WiLoR 手部重建权重和手部检测器 |
| `anycalib` | AnyCalib 单图标定（`anycalib_gen.pt` 任意相机模型、`anycalib_pinhole.pt`） |
| `geocalib` | GeoCalib 单图标定 + 重力方向（`geocalib-pinhole.tar`、`geocalib-distorted.tar`） |
| `droid_slam` | DROID-SLAM 相机位姿 `droid.pth`（`camera_pose` 使用） |
| `gsplat_refinement` | gsplat 精修用的 VGG16 感知损失权重（gsplat 已在 WSL 编译可用；精修代码待有手/物体位姿后迁移） |
