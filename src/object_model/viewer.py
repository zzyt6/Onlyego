"""Viser viewer for object_model: the generated mesh in the world, with each frame's point cloud (MegaSaM depth).

    object_model view demo01                 # frames 0..reference; open http://localhost:8080
    object_model view demo01 --frames 0-60   # also look at frames after the hand touches the object

Everything is in the world of the video_depth camera poses (camera->world), so a still object should sit on
its points in every frame of the window. The fused point map that SAM3D was given can be shown in red.
"""

import json
import threading
import time
from functools import lru_cache

import cv2
import numpy as np
import trimesh

from video_depth.poses import load_trajectory, matrix_to_quat_wxyz
from depth_estimation.datatypes import CameraIntrinsics, load_depth

from .segment import LEFT_ID, OBJECT_ID, RIGHT_ID, load_mask

FRAME_COLOURS = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                          (70, 240, 240), (240, 50, 230), (210, 245, 60)], dtype=np.uint8)
MESH_COLOUR = (90, 200, 255)


class Data:
    def __init__(self, p: dict, frames: str | None):
        self.out, self.vd = p["out"], p["video_depth"]
        self.frames_dir = p["depth_run"] / "frames"
        self.record = json.loads((self.out / "model.json").read_text(encoding="utf-8"))
        self.k = CameraIntrinsics.load(self.vd / "intrinsics_stable.json")
        idx, mats = load_trajectory(self.vd / "poses")
        self.poses = dict(zip(idx, mats))
        ref = self.record["reference_frame"]
        if frames:
            a, b = (int(x) for x in frames.split("-"))
        else:
            a, b = 0, ref
        self.indices = [i for i in range(a, b + 1) if i in self.poses]
        self.reference = ref
        self.window = self.record["window"]
        self.mesh_world = trimesh.load(str(self.out / "mesh_world.glb"), force="mesh")
        pose = json.loads((self.out / "object_pose.json").read_text())
        self.obj2world = np.array(pose["object_to_world"])
        pm = np.load(self.out / "pointmap_reference.npy")
        pts = pm[np.isfinite(pm[..., 2])]
        T = self.poses[ref]
        self.fused = (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    @lru_cache(maxsize=64)
    def image(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.frames_dir / f"{i:06d}.png")), cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=256)
    def cloud(self, i: int, stride: int, part: str):
        """World points + colours of frame i. part: all / object / no_hands."""
        depth = load_depth(self.vd / "depth" / f"{i:06d}.png")
        rgb = self.image(i)
        keep = np.isfinite(depth) & (depth > 0)
        obj = load_mask(self.out, OBJECT_ID, i)
        if part == "object" and obj is not None:
            keep &= obj
        if part == "no_hands":
            for hid in (RIGHT_ID, LEFT_ID):
                m = load_mask(self.out, hid, i)
                if m is not None:
                    keep &= ~m
        sub = np.zeros_like(keep)
        sub[::stride, ::stride] = True
        v, u = np.nonzero(keep & sub)
        z = depth[v, u]
        k = self.k
        p = np.stack([(u + 0.5 - k.cx) * z / k.fx, (v + 0.5 - k.cy) * z / k.fy, z], 1)
        T = self.poses[i]
        return (p @ T[:3, :3].T + T[:3, 3]).astype(np.float32), rgb[v, u]


def _info(d: Data) -> str:
    r = d.record
    fit, mesh, pm = r["fit"], r["mesh"], r["pointmap"]
    ext = " × ".join(f"{e * 100:.1f}" for e in mesh["obb_extents_m"])
    q = r.get("quality")
    status = ("" if q is None else
              "✅ **质量检查通过**" if q["passed"] else "⚠️ **质量检查不合格**：" + "；".join(q["failures"]))
    return "\n".join([
        status,
        f"**参考帧** {r['reference_frame']}（手第一次碰到物体：第 {r['contact_frame']} 帧）",
        f"**融合窗口** {r['window'][0]}–{r['window'][-1]}（{len(r['window'])} 帧，"
        f"同一像素随时间的偏差中位数 {pm['temporal_spread_mm_median']:.2f} mm）",
        f"**网格尺寸**（有向包围盒）{ext} cm",
        f"**轮廓 IoU** {fit['silhouette_iou']:.3f}",
        f"**网格深度 − 融合深度** 中位数 {fit['depth_diff_mm']['median']:.1f} mm"
        f"（p10 {fit['depth_diff_mm']['p10']:.1f}，p90 {fit['depth_diff_mm']['p90']:.1f}）",
        f"**点到网格表面** 中位数 {fit['point_to_surface_mm']['median']:.1f} mm，"
        f"p90 {fit['point_to_surface_mm']['p90']:.1f} mm",
    ])


def serve(p: dict, frames: str | None, host: str, port: int):
    import viser

    d = Data(p, frames)
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("-y")
    n = len(d.indices)
    start = d.indices.index(d.reference) if d.reference in d.indices else 0

    server.gui.add_markdown(_info(d))
    with server.gui.add_folder("帧"):
        g_frame = server.gui.add_slider("帧号", d.indices[0], d.indices[-1], 1, d.indices[start])
        g_play = server.gui.add_checkbox("播放", False)
        g_fps = server.gui.add_slider("播放帧率", 1, 30, 1, 8)
    with server.gui.add_folder("点云"):
        g_part = server.gui.add_dropdown("显示哪些点", ["整个画面", "只有物体", "去掉手"], initial_value="整个画面")
        g_stride = server.gui.add_dropdown("像素步长", ["1", "2", "4", "8"], initial_value="2")
        g_size = server.gui.add_slider("点大小 (mm)", 0.5, 6.0, 0.5, 1.5)
        g_stack = server.gui.add_checkbox("叠加融合窗口里所有帧（物体点，按帧着色）", False)
        g_fused = server.gui.add_checkbox("SAM3D 输入的融合点图（红色）", False)
    with server.gui.add_folder("网格"):
        g_mesh = server.gui.add_checkbox("显示网格", True)
        g_style = server.gui.add_dropdown("样式", ["SAM3D 贴图", "单色半透明", "线框"], initial_value="SAM3D 贴图")
        g_opacity = server.gui.add_slider("半透明度", 0.1, 1.0, 0.05, 0.5)
    with server.gui.add_folder("相机"):
        g_frustum = server.gui.add_checkbox("视锥和图像", True)
    g_line = server.gui.add_markdown("")

    handles: dict[str, object] = {}
    lock = threading.Lock()

    def put(name, h):
        old = handles.pop(name, None)
        if old is not None:
            old.remove()
        if h is not None:
            handles[name] = h

    def draw_mesh(_=None):
        with lock:
            if not g_mesh.value:
                put("mesh", None)
                return
            m = d.mesh_world
            if g_style.value == "SAM3D 贴图":
                put("mesh", server.scene.add_mesh_trimesh("/mesh", m))
            else:
                put("mesh", server.scene.add_mesh_simple(
                    "/mesh", m.vertices.astype(np.float32), m.faces.astype(np.uint32), color=MESH_COLOUR,
                    wireframe=g_style.value == "线框", opacity=None if g_style.value == "线框" else g_opacity.value,
                    side="double"))

    def draw(_=None):
        with lock:
            i = int(g_frame.value)
            part = {"整个画面": "all", "只有物体": "object", "去掉手": "no_hands"}[g_part.value]
            size = g_size.value / 1000
            pts, col = d.cloud(i, int(g_stride.value), part)
            put("cloud", server.scene.add_point_cloud("/cloud", pts, col, point_size=size, point_shape="circle"))
            if g_stack.value:
                for j, f in enumerate(d.window):
                    q, _ = d.cloud(f, int(g_stride.value), "object")
                    put(f"stack{j}", server.scene.add_point_cloud(
                        f"/stack/{f}", q, FRAME_COLOURS[j % len(FRAME_COLOURS)], point_size=size))
            else:
                for name in [k for k in handles if k.startswith("stack")]:
                    put(name, None)
            put("fused", server.scene.add_point_cloud("/fused", d.fused, (255, 0, 0), point_size=size)
                if g_fused.value else None)
            if g_frustum.value:
                T = d.poses[i]
                k = d.k
                img = cv2.resize(d.image(i), (k.width // 4, k.height // 4), interpolation=cv2.INTER_AREA)
                put("frustum", server.scene.add_camera_frustum(
                    "/camera", fov=float(2 * np.arctan(k.height / (2 * k.fy))), aspect=k.width / k.height,
                    scale=0.08, image=img, wxyz=matrix_to_quat_wxyz(T[:3, :3]), position=T[:3, 3]))
            else:
                put("frustum", None)
            tags = []
            if i == d.reference:
                tags.append("参考帧（SAM3D 用的这帧）")
            if i in d.window:
                tags.append("在融合窗口里")
            if d.record["contact_frame"] is not None and i >= d.record["contact_frame"]:
                tags.append("手已碰到物体")
            g_line.content = f"第 {i} 帧：" + ("，".join(tags) or "—")

    for g in (g_frame, g_part, g_stride, g_size, g_stack, g_fused, g_frustum):
        g.on_update(draw)
    for g in (g_mesh, g_style, g_opacity):
        g.on_update(draw_mesh)

    @server.on_client_connect
    def _(client):
        T = d.poses[d.reference]
        centre = d.obj2world[:3, 3]
        eye = T[:3, 3] + 0.5 * (centre - T[:3, 3])  # halfway from the camera to the object
        client.camera.position = eye
        client.camera.look_at = centre
        client.camera.up_direction = -T[:3, 1]

    draw_mesh()
    draw()
    print(f"object_model viewer: http://{host}:{port}  (frames {d.indices[0]}-{d.indices[-1]}, "
          f"reference {d.reference})", flush=True)
    while True:
        time.sleep(1.0 / max(1, g_fps.value))
        if g_play.value:
            pos = d.indices.index(int(g_frame.value)) if int(g_frame.value) in d.indices else 0
            g_frame.value = d.indices[(pos + 1) % n]
