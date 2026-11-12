# 相机位姿（camera_pose）：V2D 的 DROID-SLAM 方案和我们的迁移

源码：`E:\video_to_data\reconstruction\modules\v2d_droid_slam\`（Apache-2.0）。我们的实现在 `src\camera_pose\`。

## 1. 它解决什么问题

`depth_estimation` 给出的是**每帧在自己相机坐标系里**的深度。相机一动，同一张桌子在不同帧里的坐标就不一样。
这一部分算出每一帧相机在一个**固定世界坐标系**里的位置和朝向（4×4 相机→世界矩阵），单位是米。
世界坐标系 = 第一帧的相机坐标系（x 右、y 下、z 向前，OpenCV 约定）。

## 2. 流程

```text
depth_estimation 的结果
  frames/  intrinsics_stable.json  depth/
        │
        ▼  （在 WSL 里）
  ① DROID-SLAM 逐帧 track：网络预测帧间稠密光流 → 可微的稠密束调整解出关键帧位姿和逆深度
     （前端：滑动窗口；后端：全局）。光流平均 < 4 px 的帧不建关键帧。
  ② terminate()：两轮全局 BA，再给非关键帧补上位姿 → 每帧相机→世界位姿，但尺度未知
  ③ 定米制尺度：每个关键帧算 median(MoGe 深度 / DROID 深度)，再对所有关键帧取中位数
     → 一个全局倍数，乘到所有相机平移上
        │
        ▼  （回到 Windows）
  poses/NNNNNN.json  trajectory_tum.txt  trajectory.png  pointcloud.ply  slam.json  summary.json
        │
        ▼  [--gravity-align] ④ 用 GeoCalib 的重力方向把世界坐标转成 z 轴朝上（第 6 节）
  poses_gravity/NNNNNN.json  gravity_alignment.json  trajectory_gravity.png
```

**分工**：形状（相机怎么动、场景前后一致）由 SLAM 决定；**尺度**由 MoGe 决定。
MoGe 每帧的尺度抖动在两次中位数里被平均掉了，SLAM 的"不知道真实大小"被 MoGe 补上。

可选 `--depth-prior`：把 MoGe 深度也直接喂进束调整当先验（V2D 支持，但它的 ego 流程只用第 ③ 步的事后对齐）。

## 3. 为什么要在 WSL 里跑

DROID-SLAM 依赖两个必须编译的 CUDA 扩展（`lietorch`、`droid_backends`），gsplat 也是。
V2D 每个模块用一个 Linux Docker 镜像；这台机器的 WSL 发行版 `V2D-Ubuntu` 已经有 CUDA 11.8 / 13.0 编译器和 gcc，
所以在里面建了一个 conda 环境 `zyego-cuda`（Python 3.10、PyTorch 2.7.0 + CUDA 11.8，torch 来自 WSL 的 pip 缓存）。
Windows 上的 `camera_pose` 命令把路径换成 `/mnt/e/...`，调 `wsl.exe` 在那个环境里跑 `camera_pose.slam`，
就像 V2D 的 `run_in_container`。源代码是同一份（WSL 通过 `/mnt/e/ZY_ego/src` 导入）。

环境由 `tools\wsl\setup_cuda_env.sh` 搭建，可重复执行：

```powershell
wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_cuda_env.sh all   # 搭建（已完成）
camera_pose check                                                        # 自检：torch/GPU、lietorch、DROID、gsplat 渲染、manotorch
```

| 组件 | 版本 | 和 V2D 的区别 |
|---|---|---|
| PyTorch | 2.7.0 + CUDA 11.8 | V2D 镜像是 2.5.1 + CUDA 12.4；WSL 里只有 11.8 / 13.0 编译器 |
| DROID-SLAM | 上游提交 `2dfd39f` | V2D 用最新 main，未固定 |
| gsplat | 1.4.0（本机编译） | 同 V2D；GLM 头文件从 conda-forge 装进环境，V2D 用 apt |
| 编译架构 | 只编 8.9（RTX 4090） | V2D 编 8.0/8.6/8.9/9.0 |

## 4. 迁移时改了什么

| 地方 | V2D | 这里 |
|---|---|---|
| 输入 | 视频 + 单独给的内参/深度文件夹 | 一个 depth_estimation 结果目录（帧、稳定内参、深度编号天然一致；去畸变时自动用去畸变的帧） |
| 关键帧不足 2 个 | 上游在后端崩溃（`not enough values to unpack`） | 判定为相机静止，写单位位姿并在 `slam.json` 标 `static_camera` |
| 尺度 | 只保留中位数 | 每个关键帧的尺度都写进 `slam.json`，看分散程度 |
| 位姿文件 | V2D `Transform3d`（四元数 wxyz + 平移 + 缩放） | 同样的键，另加 `matrix`（V2D 的 `from_dict` 会忽略它）；V2D 写的文件也能读 |
| 轨迹文本 | `slam_trajectory.txt`：`帧号 tx ty tz qx qy qz qw` | 同样格式，文件名 `trajectory_tum.txt`（2026-11-10 起第一列也是帧号，之前是秒） |
| 目录名 | `slam_poses/` | `poses/`（内容相同） |
| 重力对齐 | 在最终结果包 `result.npz` 里原地改写所有世界坐标变换 | `poses/` 不动，对齐后的另写到 `poses_gravity/`；以后手、物体用同一个变换 |
| 新增 | — | `summary.json`（路径长度、位移、范围、最大转角）、`trajectory.png`（俯视/侧视图） |

## 5. 实测（2026-11-10）

| 视频 | 帧数 | 关键帧 | 相机位置范围 | 最大转角 | 结论 |
|---|---|---|---|---|---|
| demo01（我们的） | 184 | 6 | ≤ 7 mm | 0.47° | 相机基本固定 |
| demo01，去畸变 + GeoCalib（`demo01-v2d`） | 184 | 6 | ≤ 9 mm | 0.64° | 同上；尺度 0.110 |
| tissue_box（V2D 示例） | 390 | 7 | ≤ 4 mm | 0.15° | 相机基本固定 |
| 合成平移测试 | 61 | 9 | 期望 +x 0.183 m → **测得 +x 0.1834 m** | — | 方向、米制尺度都对 |

- 合成测试：把 demo01 的一帧裁成 960×540 窗口，60 帧内向右平移 400 像素。整幅画面等量平移，几何上就是相机
  在一个平面前横移，期望位移 = 400 px × 场景深度 0.684 m ÷ fx 1498 = 0.183 m。这验证了整条链路（取帧、内参、
  位姿方向、MoGe 定尺度）。
- demo01 前 60 帧只有 1 个关键帧——之前在深度部分看到的 11–18% 逐帧深度波动，**全是 MoGe 自己的抖动**，
  不是相机运动。
- 相机静止时，DROID 的点云几乎没有视差可用，`pointcloud.ply` 点很少（demo01 只有 42 个点）——正常。
- 手在画面里大幅运动也会触发关键帧；DROID 默认把场景当静止的，动的手会带来一点误差（这里是毫米级）。

## 6. 重力对齐（`--gravity-align`，V2D `--gravity_align`）

SLAM 的世界坐标 = 第一帧相机坐标，"上"是哪个方向取决于第一帧怎么拿的手机。V2D 用 GeoCalib 的重力方向把世界转正，
代码在 `v2d_common/result_bundle.py`（`gravity_align_result_bundle`），我们原样移植到 `src\camera_pose\gravity.py`：

1. GeoCalib 每个采样帧给出相机坐标里的**图像"上"方向**（OpenCV 轴：x 右、y 下、z 前；不歪不仰时是 `[0,-1,0]`），
   取反就是重力。用该帧的相机→世界旋转转到世界里；同时取该帧相机的前方（+z）。
2. 所有采样帧取平均、归一化，得到世界里的重力方向和平均前方。
3. 目标 `z_up`（默认）：新 +z = 重力的反方向，新 +y = 平均前方投影到水平面，+x = y × z（右手系）。
   其他目标（`y_down`、`x_up` …）只把重力转到那根轴上（最短旋转），不管水平朝向。
4. 每个相机→世界位姿左乘这个旋转。原点（第一帧相机位置）和尺度不变。

`gravity_alignment.json` 里的字段和 V2D 结果包 manifest 的 `gravity_alignment` 一样（变换矩阵、对齐前的重力、
每个采样帧在世界里的重力和前方、方法名、坐标约定）。

需要深度那一步跑过 GeoCalib：`depth_estimation run ... --run-geocalib`（或 `--prior geocalib`）。

### demo01 实测（2026-11-10）：方法对，但 GeoCalib 的重力在这种画面上不准

`depth_estimation run demo01 --undistort --run-geocalib` → `camera_pose run demo01-v2d --gravity-align`：

- GeoCalib 8 个采样帧的俯仰角在 −10° 到 −38° 之间，平均约 −28°（相机朝下 28°）；而 SLAM 显示相机几乎没动（≤ 0.64°），
  所以这 28° 的散布全是 GeoCalib 自己的误差。
- **独立核对**：在 MoGe 点云里拟合桌面平面（占 50–73% 的点），桌面法线和相机 −y 轴夹约 70°，
  也就是相机实际朝下约 70°。对齐后桌面法线和世界 z 轴还差 **42–46°**。
- 对比 GeoCalib 自己的约定（`from_rp`：`[-sin r·cos p, -cos r·cos p, sin p]`，俯仰为负 = 向下看），
  我们算出的重力方向符号完全正确，差的只是角度大小。这是 GeoCalib 的局限：它靠地平线和竖直线条，
  近距离俯拍一块素色桌面时没有这些线索（同一视频上它的焦距也偏，见 depth_estimation.md）。V2D 跑这段视频会得到同样的结果。
- 移植本身的正确性由单元测试保证：合成的倾斜、偏航相机，对齐后每帧重力都正好是 −z（误差 < 1e-9）。

所以俯拍桌面这类视频，`--gravity-align` 先别当真。可以考虑的改进（V2D 没有）：用 MoGe 深度里的桌面平面法线定"上"。

## 7. 用法

```powershell
depth_estimation run demo01 --prior anycalib        # 先要有深度结果（建议用焦距先验，SLAM 用的就是它的稳定内参）
camera_pose run demo01                              # 默认读 runs\depth_estimation\demo01
camera_pose run demo01-anycalib --depth-prior       # 把 MoGe 深度也放进束调整

# 带重力对齐（深度那一步要加 --run-geocalib）
depth_estimation run demo01 --undistort --run-geocalib --out runs\depth_estimation\demo01-v2d
camera_pose run demo01-v2d --gravity-align
camera_pose gravity-align demo01-v2d --gravity-target y_up   # 对已有结果重做，或换目标轴
```

## 8. V2D 最终的相机轨迹：gsplat 精修（还没迁移，等物体部分）

V2D 最终输出的相机轨迹**优先用 gsplat 精修后的** `camera_refined_simple/`（注意是世界→相机），
没有它时才用 DROID 的 `slam_poses/`（`run_ego_wilor.py` 第 2071 行）。
精修是 `v2d_gsplat_refinement/lib/refine_simple.py`：把静态背景（从 MoGe 深度初始化的 2D 高斯）、
物体网格、两只手的 MANO 放在一起渲染，和原图比较，**同时**优化每帧相机位姿、物体位姿、手腕位姿；
DROID 的位姿是相机的初始值（`background_pose_init_dir`）。

它的接口里**物体网格、物体逐帧位姿、物体 mask 是必填的**（手是可选的）：动的物体和手要被单独渲染，
否则它们会被当成背景把相机拉偏。这些来自后面的部分（SAM2 mask → SAM3D 网格 → FoundationPose 位姿；
WiLoR/HaMeR 手），所以精修和那几部分一起迁移。到那时：相机初始值用 `poses/`，
最终结果包同样再做一次重力对齐（V2D 就是对最终结果包做的）。

## 9. gsplat 和 MANO

gsplat 库已编译、能在 GPU 上渲染，manotorch 也装好了。它们用于第 8 节的联合精修。

左右手模型都已就位：`model\mano\MANO_{LEFT,RIGHT}.pkl`，WSL 里在 `~/models/MANO/`。
左手（2026-11-10）取自 Hugging Face `camenduru/HandRefiner` 的转存，核对过：

- 同一仓库里的 `MANO_RIGHT.pkl` 与我们手上的官方文件逐字节相同（sha256 `45d60aa3…`），
  左手 sha256 `c4022f70…` 与仓库的 LFS 记录一致；
- pickle 里只引用 numpy / scipy.sparse / chumpy 的数据类，没有可执行的内容；
- 用 manotorch 加载：778 个顶点、21 个关节，零姿态下左手正好是右手沿 x 镜像（最大差 0.0 m）。

MANO 是马普所的非商业研究许可，使用前应在 https://mano.is.tue.mpg.de 注册并同意许可。
以后要从官方 `mano_v1_2.zip` 重装：`python tools\install_mano.py <zip> --force`。

manotorch 要求目录结构是 `<root>/models/MANO_LEFT.pkl`，加载时 `mano_assets_root` 指向 `models` 的上一层。

**官方 v1.2 左手的已知缺陷**（官方文件本身如此，不是转存的问题）：`MANO_LEFT.pkl` 的形状基 `shapedirs`
x 分量和右手**相同**，正确的应该是取反（实测 Σ|L−R| = 0.0004，Σ|L+R| = 8.0）。零形状参数时看不出来；
betas ~ N(0, 2) 时左手和镜像右手最大差 **3.3 cm**。manotorch 不会修正。两种做法：

- V2D 的做法（`v2d_hand_alignment/lib/recover_mano_params.py`）：左手 = `diag(-1,1,1)` × 右手 MANO，
  根旋转和关节轴角按 `(rx, -ry, -rz)` 共轭，完全不用左手文件；
- 或沿用 HaMeR / ARCTIC 的补丁：加载左手后若 `shapedirs[:,0,:]` 和右手几乎相同，就把它乘 −1。

我们按 Wuji MINT 的做法（第二种）在 `src\hand_model` 里修正了：用 `hand_model.create_mano("left")` 加载即可，
详见 [hand_model.md](hand_model.md)。
