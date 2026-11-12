# 手模型（hand_model）：MANO 加载和左手修正

后面做手部的各个部分（手部重建、手物对齐）都要用 MANO。这里统一提供加载入口，
并按 **Wuji MINT** 的方法修正官方 v1.2 左手模型的缺陷。代码在 `src\hand_model\mano.py`。

## 1. 问题

官方 `mano_v1_2.zip` 里的 `MANO_LEFT.pkl`，形状基 `shapedirs`（10 个形状参数各自让 778 个顶点怎么移动）
的 **x 分量和右手同号**。左手是右手的镜像，x 分量应该取反。

| 检查 | 结果 |
|---|---|
| Σ\|L<sub>x</sub> − R<sub>x</sub>\| / Σ\|L<sub>x</sub> + R<sub>x</sub>\| | 0.0004 / 7.99（说明 L<sub>x</sub> = R<sub>x</sub>，错了） |
| 零形状参数 | 看不出问题（只有形状基有错） |
| 随机手（betas 标准差 2、姿态标准差 0.5），左手 vs 镜像右手 | 最大差 **65 mm** |

## 2. 各家的做法

| 方案 | 做法 |
|---|---|
| **Wuji MINT**（我们采用） | smplx 建好左手层后 `model.shapedirs[:, 0, :] *= -1`（`mint/visualization/mano.py`、训练损失 `mano_geometry.py`） |
| HaMeR / ARCTIC | 同样取反，但先判断左手 x 分量是否和右手几乎相同 |
| V2D | 不读左手文件：左手 = `diag(-1,1,1)` × 右手模型，姿态按 `(rx, -ry, -rz)` 共轭 |

我们用 Wuji 的取反，加上 HaMeR 的判断：取反只在左手 x 分量仍和右手相同时做，
以后换成已修正的文件也不会被翻回去。取反放在建层之前（改加载出的数组），效果和 Wuji 在建层后改完全一样（实测差 0.0）。

## 3. 用法

```python
from hand_model import create_mano, mirror_axis_angle

right = create_mano("right", device="cuda")
left = create_mano("left", device="cuda")        # 已修正；left.left_shapedirs_fixed == True
out = left(betas=betas, global_orient=orient, hand_pose=pose)   # pose: 15 关节 × 3 轴角 = 45 维
out.vertices, out.joints
```

- 默认 `use_pca=False, flat_hand_mean=True`：`hand_pose` 就是 45 维轴角，零姿态是平伸的手（和 manotorch 默认一致）。
  需要 PCA 或平均手势时传参数改。
- 同一只手的镜像姿态：`mirror_axis_angle(pose)`，即每个轴角 `(rx, ry, rz) → (rx, -ry, -rz)`。
- 已经用别的库建好的左手层（smplx 或 manotorch）：`fix_left_layer_(layer)` 原地取反。
- 模型目录默认 `model\mano\`，可用环境变量 `ZYEGO_MANO_DIR` 改；WSL 里通过 `/mnt/e/ZY_ego/model/mano` 读同一份。

读 `.pkl` 不需要 chumpy：用一个受限的反序列化器，只允许 MANO 文件里那 7 种类
（numpy 数组、scipy 稀疏矩阵、chumpy 数组），遇到别的类直接拒绝。读出的数值和装了 chumpy 的标准读法逐位相同。

## 4. 自检

```powershell
hand_model check
```

```text
left shapedirs x: sum|L-R| = 0.0004, sum|L+R| = 7.9903  [v1.2 bug (same sign as right) -> will be fixed]
create_mano('left') applied the fix: True
left vs mirrored right, 64 random hands (betas std 2.0, pose std 0.5):
  unfixed file : max error   65.456 mm
  fixed        : max error    0.013 mm
OK
```

修正后剩下的 0.013 mm 是官方左右手文件本身的微小数值差，可以忽略。Windows（`ego`）和 WSL（`zyego-cuda`）里结果相同。

## 5. 模型文件从哪来

`model\mano\MANO_{LEFT,RIGHT}.pkl`（v1.2），WSL 里另有一份在 `~/models/MANO/`。左手（2026-11-10）取自 Hugging Face
`camenduru/HandRefiner` 的转存，核对过：同一仓库里的 `MANO_RIGHT.pkl` 与官方文件逐字节相同（sha256 `45d60aa3…`），
左手 sha256 `c4022f70…` 与仓库的 LFS 记录一致；pickle 里只引用 numpy / scipy.sparse / chumpy 的数据类，没有可执行的内容。
上面的缺陷是官方文件本身就有的，不是转存的问题。

MANO 是马普所的非商业研究许可，使用前应在 https://mano.is.tue.mpg.de 注册并同意许可。
以后要从官方 `mano_v1_2.zip` 重装：`python tools\install_mano.py <zip> --force`。

## 6. 和 Wuji 不同的地方

Wuji 还把网络输出的左手 `hand_pose` 做一次镜像（`apply_left_hand_pose_flip_111`），
因为它的网络是按"把左手镜像成右手"来预测的。这是**它的网络的输出约定**，和模型文件的缺陷无关，
所以这里不做。以后接入某个手部重建网络时，按那个网络的约定决定要不要镜像姿态。
