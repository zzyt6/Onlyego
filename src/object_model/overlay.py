"""Overlay videos: the object mesh drawn on the original video, SAM3D's own size/placement vs the fitted one.

    object_model overlay demo01     # -> runs\\object_model\\demo01\\overlay_sam3d.mp4, overlay_fitted.mp4

The object pose is known only while it lies still before the hand touches it, so the mesh stays where it was in
the world (each frame projected with its own camera pose): until the contact frame the overlay compares sizes;
after it the box is carried away and the mesh stays behind (object tracking is the next part).
Drawing: faces painted far-to-near, shaded by the angle to the view, blended at 45%, silhouette outlined;
green = the SAM2 object mask; per-frame silhouette IoU against it while the object is still.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera_pose.poses import load_trajectory
from depth_estimation.datatypes import CameraIntrinsics

from . import model
from .segment import OBJECT_ID, load_mask

FONT = next((p for p in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"),
                         Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")) if p.exists()), None)


def draw_mesh(image: np.ndarray, mesh, T: np.ndarray, k: CameraIntrinsics, colour, alpha: float = 0.45):
    """Returns (image with the shaded mesh blended in and outlined, silhouette bool)."""
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    uv = np.stack([v[:, 0] / v[:, 2] * k.fx + k.cx, v[:, 1] / v[:, 2] * k.fy + k.cy], 1)
    centres = v[mesh.faces].mean(1)
    normals = mesh.face_normals @ T[:3, :3].T
    shade = 0.35 + 0.65 * np.abs((normals * -centres).sum(1) / np.linalg.norm(centres, axis=1))
    layer = image.copy()
    sil = np.zeros(image.shape[:2], np.uint8)
    tris = np.round(uv[mesh.faces] * 4).astype(np.int32)
    base = np.array(colour, float)
    for f in np.argsort(-centres[:, 2]):  # far to near
        c = tuple(int(x) for x in base * shade[f])
        cv2.fillConvexPoly(layer, tris[f], c, lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(sil, tris[f], 1, shift=2)
    s = sil > 0
    out = image.copy()
    out[s] = (alpha * layer[s] + (1 - alpha) * image[s]).astype(np.uint8)
    cnt, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, tuple(int(x) for x in base), 3, cv2.LINE_AA)
    return out, s


def _text(image: np.ndarray, lines: list[tuple[str, tuple]], size: int = 34) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(FONT), size) if FONT else ImageFont.load_default()
    y = 18
    for text, rgb in lines:
        d.text((22, y), text, font=font, fill=rgb, stroke_width=3, stroke_fill=(0, 0, 0))
        y += size + 14
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def versions(out: Path, c2w_ref: np.ndarray) -> dict:
    """name -> (mesh in its object frame, object->world, title, BGR colour)."""
    rec = json.loads((out / "model.json").read_text(encoding="utf-8"))
    mesh_s, T_s, _ = model.sam3d_placement(out / "sam3d")
    pose = json.loads((out / "object_pose.json").read_text())
    mesh_f = model.load_mesh(out / "mesh_object.glb")

    def size(e):
        return " × ".join(f"{x * 100:.2f}" for x in e) + " cm"

    return {
        "sam3d": (mesh_s, c2w_ref @ T_s, f"SAM3D 原始尺寸和位置：{size(rec['mesh']['obb_extents_m_sam3d'])}",
                  (0, 140, 255)),
        "fitted": (mesh_f, np.array(pose["object_to_world"]),
                   f"拟合后（尺度 ×{pose['refine_scale']:.3f}）：{size(rec['mesh']['obb_extents_m'])}",
                   (255, 200, 0)),
    }


def render_videos(p: dict, log=print) -> list[Path]:
    out, vd = p["out"], p["video_depth"]
    rec = json.loads((out / "model.json").read_text(encoding="utf-8"))
    k = CameraIntrinsics.load(vd / "intrinsics_stable.json")
    idx, mats = load_trajectory(vd / "poses")
    poses = dict(zip(idx, mats))
    contact, ref = rec["contact_frame"], rec["reference_frame"]
    fps = json.loads((vd / "run.json").read_text()).get("video", {}).get("fps", 30.0)
    frames_dir = p["depth_run"] / "frames"
    written = []
    for name, (mesh, obj2world, title, colour) in versions(out, poses[ref]).items():
        path = out / f"overlay_{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (k.width, k.height))
        ious = []
        for i in idx:
            img = cv2.imread(str(frames_dir / f"{i:06d}.png"))
            T = np.linalg.inv(poses[i]) @ obj2world
            img, sil = draw_mesh(img, mesh, T, k, colour)
            mask = load_mask(out, OBJECT_ID, i)
            if mask is not None:
                cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnt, -1, (0, 255, 0), 2, cv2.LINE_AA)
            rgb = tuple(int(c) for c in colour[::-1])
            lines = [(title, rgb)]
            if contact is None or i < contact:
                iou = float((sil & mask).sum() / max(1, (sil | mask).sum())) if mask is not None else float("nan")
                ious.append(iou)
                lines.append((f"第 {i} 帧　盒子静止　轮廓 IoU {iou:.3f}（绿线 = SAM2 的盒子 mask）", (255, 255, 255)))
            else:
                lines.append((f"第 {i} 帧　手已碰到盒子：物体位姿还没跟踪，网格停在原来的位置", (255, 120, 120)))
            writer.write(_text(img, lines))
        writer.release()
        log(f"[overlay] {path.name}: static frames IoU median {np.median(ious):.3f} "
            f"(min {np.min(ious):.3f}, max {np.max(ious):.3f})")
        written.append(path)
    return written
