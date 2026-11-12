"""Viser viewer for object_pose results: the object mesh moving with each frame's pose, in the world of the
video_depth camera poses, with that frame's point cloud (MegaSaM depth) and camera.

    object_pose view demo01                  # all frames; open http://localhost:8080
    object_pose view demo01 --frames 50-110

"带贴图" shows the mesh with Codex's texture (the one it was tracked with).
object->world = camera->world(i) @ object->camera(i). The pose object_model fitted while the box lay still is shown
as a grey wireframe (where the box should stay until the hand touches it).
"""

import json
import threading
import time

import numpy as np

from camera_pose.poses import matrix_to_quat_wxyz
from object_model.cli import paths as om_paths
from object_model.model import load_mesh
from object_model.viewer import Data

from .cli import MESH, out_dir
from .evaluate import load_poses

KEY, LABEL, COLOUR = "fp", "FoundationPose", (0, 200, 120)


def serve(name: str, frames: str | None, host: str, port: int):
    import viser

    d = Data(om_paths(name), frames or "0-100000")
    contact = d.record["contact_frame"]
    base = out_dir(name)
    if not (base / "metrics.json").exists():
        raise FileNotFoundError(f"run `object_pose run {name}` first")
    cam = load_poses(base / "poses")
    metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))["frames"]
    seqs = {KEY: ({i: d.poses[i] @ T for i, T in cam.items() if i in d.poses}, metrics)}
    labels, colours = {KEY: LABEL}, {KEY: COLOUR}
    mesh = load_mesh(d.out / MESH)

    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("-y")
    n = len(d.indices)
    with server.gui.add_folder("帧"):
        g_frame = server.gui.add_slider("帧号", d.indices[0], d.indices[-1], 1, d.indices[0])
        g_play = server.gui.add_checkbox("播放", False)
        g_fps = server.gui.add_slider("播放帧率", 1, 30, 1, 8)
    with server.gui.add_folder("物体网格"):
        g_seq = {k: server.gui.add_checkbox(labels[k], True) for k in seqs}
        g_style = server.gui.add_dropdown("网格样式", ["带贴图", "单色半透明", "线框"], initial_value="带贴图")
        g_opacity = server.gui.add_slider("半透明度", 0.1, 1.0, 0.05, 0.6)
        g_trail = server.gui.add_checkbox("物体中心轨迹", True)
        g_fit = server.gui.add_checkbox("拟合位姿（静止时盒子应在的位置，灰色线框）", True)
    with server.gui.add_folder("点云"):
        g_part = server.gui.add_dropdown("显示哪些点", ["整个画面", "只有物体", "去掉手", "不显示"],
                                         initial_value="整个画面")
        g_stride = server.gui.add_dropdown("像素步长", ["1", "2", "4", "8"], initial_value="2")
        g_size = server.gui.add_slider("点大小 (mm)", 0.5, 6.0, 0.5, 1.5)
    with server.gui.add_folder("相机"):
        g_frustum = server.gui.add_checkbox("视锥和图像", True)
    g_info = server.gui.add_markdown("")

    lock = threading.Lock()
    meshes, trails, handles = {}, {}, {}

    def put(store, name_, h):
        old = store.pop(name_, None)
        if old is not None:
            old.remove()
        if h is not None:
            store[name_] = h

    def build_meshes(_=None):
        with lock:
            for key in seqs:
                if not g_seq[key].value:
                    put(meshes, key, None)
                    continue
                if g_style.value == "带贴图":
                    h = server.scene.add_mesh_trimesh(f"/seq/{key}", mesh)
                else:
                    h = server.scene.add_mesh_simple(
                        f"/seq/{key}", mesh.vertices.astype(np.float32),
                        mesh.faces.astype(np.uint32), color=colours[key], wireframe=g_style.value == "线框",
                        opacity=None if g_style.value == "线框" else g_opacity.value, side="double")
                put(meshes, key, h)
            for key, (world, _m) in seqs.items():
                if g_trail.value and g_seq[key].value:
                    c = np.array([world[i][:3, :3] @ mesh.centroid + world[i][:3, 3]
                                  for i in d.indices if i in world], np.float32)
                    put(trails, key, server.scene.add_point_cloud(f"/trail/{key}", c, colours[key],
                                                                  point_size=0.003))
                else:
                    put(trails, key, None)
            if g_fit.value:
                put(handles, "fit", server.scene.add_mesh_simple(
                    "/fit", mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32), color=(150, 150, 150),
                    wireframe=True, wxyz=matrix_to_quat_wxyz(d.obj2world[:3, :3]), position=d.obj2world[:3, 3]))
            else:
                put(handles, "fit", None)
        draw()

    def draw(_=None):
        with lock:
            i = int(g_frame.value)
            lines = [f"### 第 {i} 帧　" + ("盒子静止（手还没碰）" if contact is None or i < contact else "手已碰到盒子")]
            for key, (world, metrics) in seqs.items():
                h = meshes.get(key)
                if h is not None and i in world:
                    h.visible = True
                    h.wxyz = matrix_to_quat_wxyz(world[i][:3, :3])
                    h.position = world[i][:3, 3]
                elif h is not None:
                    h.visible = False
                if g_seq[key].value:
                    m = metrics.get(str(i), {})
                    r, g, b = colours[key]
                    s = f"<span style='color:rgb({r},{g},{b})'>■</span> {labels[key]}："
                    s += ("—" if m.get("iou_visible") is None else
                          f"可见 IoU **{m['iou_visible']:.3f}**，整体 {m['iou_full']:.3f}")
                    if m.get("depth_mm") is not None:
                        s += f"，网格−深度 {m['depth_mm']:+.0f} mm"
                    if "static_drift_mm" in m:
                        s += f"，离拟合位姿 {m['static_drift_mm']:.1f} mm / {m['static_drift_deg']:.0f}°"
                    lines.append(s)
            g_info.content = "\n\n".join(lines)
            part = {"整个画面": "all", "只有物体": "object", "去掉手": "no_hands", "不显示": None}[g_part.value]
            if part:
                pts, col = d.cloud(i, int(g_stride.value), part)
                put(handles, "cloud", server.scene.add_point_cloud("/cloud", pts, col, point_size=g_size.value / 1000,
                                                                   point_shape="circle"))
            else:
                put(handles, "cloud", None)
            if g_frustum.value:
                import cv2
                T, k = d.poses[i], d.k
                img = cv2.resize(d.image(i), (k.width // 4, k.height // 4), interpolation=cv2.INTER_AREA)
                put(handles, "frustum", server.scene.add_camera_frustum(
                    "/camera", fov=float(2 * np.arctan(k.height / (2 * k.fy))), aspect=k.width / k.height,
                    scale=0.08, image=img, wxyz=matrix_to_quat_wxyz(T[:3, :3]), position=T[:3, 3]))
            else:
                put(handles, "frustum", None)

    for g in (*g_seq.values(), g_style, g_opacity, g_trail, g_fit):
        g.on_update(build_meshes)
    for g in (g_frame, g_part, g_stride, g_size, g_frustum):
        g.on_update(draw)

    @server.on_client_connect
    def _(client):
        T = d.poses[d.reference]
        centre = d.obj2world[:3, 3]
        client.camera.position = T[:3, 3] + 0.3 * (centre - T[:3, 3])
        client.camera.look_at = centre
        client.camera.up_direction = -T[:3, 1]

    build_meshes()
    print(f"object_pose viewer: http://{host}:{port}  (frames {d.indices[0]}-{d.indices[-1]})", flush=True)
    while True:
        time.sleep(1.0 / max(1, g_fps.value))
        if g_play.value:
            pos = d.indices.index(int(g_frame.value)) if int(g_frame.value) in d.indices else 0
            g_frame.value = d.indices[(pos + 1) % n]
