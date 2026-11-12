"""Viser page for the object's texture: the texture atlas itself and the textured mesh.

    object_model texture demo01      # http://localhost:8080

The texture is Codex's (`four_stage_view generate-texture`, written to runs\object_model\<video>\mesh_textured.glb).

Two ways to look:
  物体单独看   the textured mesh alone at the origin, with its object axes, to turn around freely
  放回场景     the textured mesh at the fitted pose in the world, with the reference frame's point cloud (real
               colours) and camera: compare the texture with the photo
"""

import json
import time

import cv2
import numpy as np
import trimesh

from camera_pose.poses import matrix_to_quat_wxyz

from .model import load_mesh, obb_extents_m
from .segment import OBJECT_ID, load_mask
from .viewer import Data


def _texture(mesh: trimesh.Trimesh) -> np.ndarray | None:
    mat = getattr(mesh.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    return None if img is None else np.asarray(img.convert("RGB"))


def _uv_overlay(tex: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """The atlas with each triangle's UV outline drawn on it (glTF UV: v up -> image row = (1 - v) * H)."""
    h, w = tex.shape[:2]
    uv = np.asarray(mesh.visual.uv)
    px = np.stack([uv[:, 0] * w, (1 - uv[:, 1]) * h], 1)
    out = tex.copy()
    cv2.polylines(out, [np.round(px[f] * 4).astype(np.int32) for f in mesh.faces], True, (255, 0, 80), 1,
                  cv2.LINE_AA, shift=2)
    return out


def _photo_crop(d: Data, frame: int, pad: int = 60) -> np.ndarray:
    img = d.image(frame)
    m = load_mask(d.out, OBJECT_ID, frame)
    if m is None or not m.any():
        return img
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def serve(p: dict, host: str, port: int):
    import viser

    d = Data(p, None)
    ref = d.reference
    if not (d.out / "mesh_textured.glb").exists():
        raise FileNotFoundError(f"{d.out / 'mesh_textured.glb'} not found: run "
                                f"`four_stage_view generate-texture <job> --object {d.out.name}`")
    variants = {"Codex 生成的贴图": load_mesh(d.out / "mesh_textured.glb")}
    meshes = {}
    for label, m in variants.items():
        c = m.copy()
        c.apply_translation(-m.bounding_box.centroid)
        meshes[label] = (m, c, _texture(m))
    mesh = next(iter(variants.values()))
    ext = obb_extents_m(mesh)

    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("-y")
    g_mode = server.gui.add_dropdown("看什么", ["物体单独看", "放回场景"], initial_value="物体单独看")
    g_which = server.gui.add_dropdown("贴图", list(meshes), initial_value=next(iter(meshes)))
    with server.gui.add_folder("物体"):
        g_wire = server.gui.add_checkbox("叠加网格线（看三角形怎么分布）", False)
        g_axes = server.gui.add_checkbox("物体坐标轴（红 x 绿 y 蓝 z）", True)
    with server.gui.add_folder("放回场景"):
        g_cloud = server.gui.add_dropdown("点云（真实颜色）", ["整个画面", "只有物体", "不显示"], initial_value="整个画面")
        g_size = server.gui.add_slider("点大小 (mm)", 0.5, 6.0, 0.5, 1.5)
        g_frustum = server.gui.add_checkbox("相机视锥和图像", True)
    with server.gui.add_folder("贴图图片（1024×1024，网格的 UV）", expand_by_default=True):
        g_uv = server.gui.add_checkbox("贴图上画出三角形（UV 展开）", False)
        g_tex = server.gui.add_image(meshes[g_which.value][2], label="贴图")
    with server.gui.add_folder("参考帧照片里的盒子", expand_by_default=True):
        server.gui.add_image(_photo_crop(d, ref), label=f"第 {ref} 帧")
    server.gui.add_markdown(
        f"**网格** {len(mesh.vertices)} 顶点、{len(mesh.faces)} 面；尺寸（有向包围盒）"
        f"{' × '.join(f'{e * 100:.1f}' for e in ext)} cm\n\n"
        "两种贴图用的是同一个网格（顶点、面、UV 都一样），只是贴图图片不同。")

    handles: dict[str, object] = {}

    def put(name, h):
        old = handles.pop(name, None)
        if old is not None:
            old.remove()
        if h is not None:
            handles[name] = h

    def wire(path, m, **kw):
        return server.scene.add_mesh_simple(path, m.vertices.astype(np.float32), m.faces.astype(np.uint32),
                                            color=(20, 20, 20), wireframe=True, **kw)

    def draw(_=None):
        for name in list(handles):
            put(name, None)
        mesh, centred, tex = meshes[g_which.value]
        if g_mode.value == "物体单独看":
            put("mesh", server.scene.add_mesh_trimesh("/alone/mesh", centred))
            if g_wire.value:
                put("wire", wire("/alone/wire", centred))
            if g_axes.value:
                put("axes", server.scene.add_frame("/alone/axes", axes_length=0.08, axes_radius=0.0015,
                                                   position=-mesh.bounding_box.centroid))
            return
        W = d.obj2world
        put("mesh", server.scene.add_mesh_trimesh("/scene/mesh", mesh, wxyz=matrix_to_quat_wxyz(W[:3, :3]),
                                                  position=W[:3, 3]))
        if g_wire.value:
            put("wire", wire("/scene/wire", mesh, wxyz=matrix_to_quat_wxyz(W[:3, :3]), position=W[:3, 3]))
        if g_axes.value:
            put("axes", server.scene.add_frame("/scene/axes", axes_length=0.08, axes_radius=0.0015,
                                               wxyz=matrix_to_quat_wxyz(W[:3, :3]), position=W[:3, 3]))
        part = {"整个画面": "all", "只有物体": "object", "不显示": None}[g_cloud.value]
        if part:
            pts, col = d.cloud(ref, 2, part)
            put("cloud", server.scene.add_point_cloud("/scene/cloud", pts, col, point_size=g_size.value / 1000,
                                                      point_shape="circle"))
        if g_frustum.value:
            T, k = d.poses[ref], d.k
            img = cv2.resize(d.image(ref), (k.width // 4, k.height // 4), interpolation=cv2.INTER_AREA)
            put("frustum", server.scene.add_camera_frustum(
                "/scene/camera", fov=float(2 * np.arctan(k.height / (2 * k.fy))), aspect=k.width / k.height,
                scale=0.08, image=img, wxyz=matrix_to_quat_wxyz(T[:3, :3]), position=T[:3, 3]))

    def look(client):
        if g_mode.value == "物体单独看":
            client.camera.look_at = (0.0, 0.0, 0.0)
            client.camera.position = (0.0, -0.12, -0.25)
            client.camera.up_direction = (0.0, -1.0, 0.0)
        else:
            T = d.poses[ref]
            c = d.obj2world[:3, 3]
            client.camera.position = T[:3, 3] + 0.4 * (c - T[:3, 3])
            client.camera.look_at = c
            client.camera.up_direction = -T[:3, 1]

    def on_mode(_):
        draw()
        for client in server.get_clients().values():
            look(client)

    g_mode.on_update(on_mode)
    for g in (g_wire, g_axes, g_cloud, g_size, g_frustum):
        g.on_update(draw)
    def show_texture(_=None):
        m, _c, tex = meshes[g_which.value]
        g_tex.image = _uv_overlay(tex, m) if g_uv.value else tex

    g_uv.on_update(show_texture)
    g_which.on_update(lambda _: (draw(), show_texture()))
    server.on_client_connect(look)

    draw()
    print(f"object_model texture viewer: http://{host}:{port}", flush=True)
    while True:
        time.sleep(1.0)
