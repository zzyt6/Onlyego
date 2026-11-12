# 物体模型（object_model）：在手碰到物体之前，把它建出来

代码：`src\object_model\`。命令：`object_model run`、`object_model view`。

## 1. 为什么单独做这一步

手拿着物体移动时，物体的逐帧深度不可信：单目深度在手指和物体的遮挡边界上平滑过渡，运动模糊又让物体表面
没有纹理可依；MegaSaM 只约束静止的场景，运动像素被降权，物体区域的形状基本就是 MoGe 单帧的先验。demo01 实测
（2026-11-11）：盒子静止时盒面起伏 0.4–0.8%，手持移动时 1.4–8.6%，盒面出现折角、靠近手指的一块被拉向手指深度。

V2D 用一个"两只手和物体都看得见"的参考帧给所有东西做提示，SAM3D 也用这一帧建网格。demo01 那一帧（66）正在
运动模糊里、盒子侧对相机，生成的网格是一块扁板。

这里的做法：**物体当刚体，形状只在一个好帧上定一次**。好帧 = 手还没碰到物体、物体完整可见的最后一帧；
米制尺度来自这之前静止的一段帧的融合深度（这段时间物体对 MegaSaM 来说也是静止的，深度可信）。

## 2. 流程

```text
video_depth 的结果：depth/（全分辨率一致深度）  poses/（相机→世界）  intrinsics_stable.json
depth_estimation 的结果：frames/
   │
   │ ① 分割（segment.py，WSL）
   │    WiLoR 每帧找手 → Grounding DINO 在第 0 帧框物体 → SAM2 传播
   │    物体在第 0 帧提示；每只手在它"整只手在画面里、不压在物体框上、框最大"的那一帧提示
   │    → masks/1 物体、masks/2 右手、masks/3 左手
   │ ② 选参考帧（reference.py）
   │    接触帧 = 物体 mask 周围 30 像素内第一次出现手（>50 像素），或物体 mask 和第 0 帧的 IoU < 0.95
   │    参考帧 = 接触前的最后一帧；融合窗口 = 参考帧往前最多 20 帧（物体 mask 不变、手不在旁边）
   │ ③ 融合点图（pointmap.py）
   │    窗口里每帧的物体像素（mask 腐蚀 3 像素：边缘像素混着桌面深度）→ 按位姿搬到参考相机 →
   │    投到参考帧像素上，每个像素取各帧深度的中位数（至少一半的帧有值）；物体外 = NaN
   │ ④ SAM3D（model.py，WSL 环境 hoi-sam3d）
   │    输入：参考帧 RGB + 物体 mask + 融合点图 + 内参
   │    阶段 1：点图归一化后和图像一起 → 64³ 粗体素 + 位姿（旋转、平移、尺度，按点图还原成米制）
   │    阶段 2：只看 RGB + mask → 细节几何 + 3D 高斯 → 网格后处理 → 纹理烘焙
   │ ⑤ 放置和检查（model.py）
   ▼
mesh_object.glb（物体坐标系，米）  mesh_world.glb（世界坐标）  object_pose.json  model.json
```

## 3. 结果文件（`runs\object_model\<视频>\`）

| 文件 | 内容 |
|---|---|
| `wilor\`、`dino.json`、`sam2_prompts.json`、`segment.json` | 分割的中间结果和提示 |
| `masks\1\`、`masks\2\`、`masks\3\` | 物体、右手、左手的逐帧 mask |
| `pointmap_reference.npy`、`intrinsics.json` | 给 SAM3D 的融合点图（参考相机坐标，OpenCV 轴，米） |
| `sam3d\` | SAM3D 原始输出：`mesh.glb`（它自己的标准坐标系）、`transform.json`、`intrinsics.json` |
| `mesh_object.glb` | 网格，米制，物体坐标系 = SAM3D 标准坐标系乘尺度（后面估物体位姿用这个） |
| `mesh_world.glb` | 网格放进 video_depth 的世界坐标（查看用） |
| `object_pose.json` | 物体→参考相机、物体→世界，4×4 |
| `model.json` | 参考帧、接触帧、融合窗口、每帧统计、点图统计、网格尺寸、拟合检查 |

拟合检查（`model.json` 的 `fit`）：
- **轮廓 IoU**：网格投到参考帧上的轮廓和 SAM2 mask；
- **网格深度 − 融合深度**：两者都有值的像素上，中位数 = 整体前后偏移，p10–p90 = 形状差异；
- **点到网格表面**：每个融合点到最近网格表面的距离。

**质量检查（自动）**：拟合后 **轮廓 IoU ≥ 0.95** 且 **点到网格表面中位数 ≤ 3 mm** 才算通过
（`--min-iou`、`--max-point-mm` 可改）。不通过时日志里用一行 `!!!` 框出原因，`model.json` 的 `quality` 记
`passed: false` 和原因，结果文件照样写出，命令以退出码 2 结束（脚本据此判断）；查看器顶部也显示通过/不合格。

## 4. 用法

```powershell
depth_estimation run demo01
video_depth run demo01
object_model run demo01 --prompt "a light blue box"     # → runs\object_model\demo01\
object_model view demo01                                 # http://localhost:8080，默认显示第 0 帧到参考帧
object_model view demo01 --frames 0-60                   # 也看手拿起物体之后
```

查看器（Viser）：网格（SAM3D 贴图 / 单色半透明 / 线框）放在世界坐标里，每帧的点云（MegaSaM 深度）按相机位姿
放进同一个世界；可以只看物体点、去掉手、叠加融合窗口里所有帧（按帧着色），或显示 SAM3D 输入的融合点图（红色）。

## 5. demo01 实测（2026-11-11，`--prompt "a light blue box"`）

| | 结果 |
|---|---|
| 分割 | 物体在第 0 帧提示；右手第 22 帧入画，左手第 61 帧 |
| 接触帧 / 参考帧 | 第 24 帧手到盒子旁 → **参考帧 23**，融合窗口 4–23（20 帧） |
| 融合点图 | 8.1 万像素；同一像素 20 帧深度的偏差中位数 **0.09 mm**（p90 0.16 mm）——静止段 MegaSaM 对物体也很稳 |
| 网格 | 4468 顶点、7306 面，SAM3D 自带烘焙贴图 |

SAM3D 放置 vs 拟合后：

| | 轮廓 IoU | 网格深度 − 融合深度（中位数） | 点到表面 中位数 / p90 | 尺寸（有向包围盒） |
|---|---|---|---|---|
| SAM3D 原样 | 0.557 | −2.1 mm | 3.7 / 14.3 mm | 1.9 × 7.5 × 10.3 cm |
| **拟合后**（尺度 ×1.358，转 3.8°，移 8 mm） | **0.980** | **−0.5 mm** | **0.68 / 1.7 mm** | **2.5 × 10.2 × 14.0 cm** |

SAM3D 的形状和长宽比是对的（融合点图量到盒面 10.1 × 13.8 cm，比例和网格一致），距离也对，只是整体做小了；
V2D 为此在 SAM3D 后面接了 FoundationPose 尺度估计。这里改为直接拟合一个统一尺度 + 位姿（Powell，约 2400 次评估，
约 1.5 分钟），网格形状不动。

注意：米制尺度最终来自 MoGe-2（+ AnyCalib 焦距）定的深度，用尺子量一下实物可以检验整条链的米制精度。

整个 `object_model run`（分割已缓存时）约 4 分钟，其中 SAM3D 加载模型约 3 分钟。

## 6. 下一步

- **贴图**：由 Codex 生成（`four_stage_view generate-texture <job> --object <视频>`）：把这里的网格（只有 UV）、
  参考图和参考帧照片 + 位姿交给它，它用生图直接做出 `texture.png`；结果写回本目录的 `texture.png`、`mesh_textured.glb`，
  `object_model texture <视频>` 里看。
- 以后把选帧接到 `four_stage_view` 的流程里。
