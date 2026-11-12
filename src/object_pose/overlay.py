"""Overlay video of a pose sequence: the mesh drawn at each frame's pose, SAM2 masks outlined, per-frame scores."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from depth_estimation.datatypes import CameraIntrinsics
from object_model.segment import LEFT_ID, OBJECT_ID, RIGHT_ID, load_mask

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


def _outline(img, mask, colour, width=2):
    if mask is not None:
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, colour, width, cv2.LINE_AA)


def render_video(path: Path, title: str, poses: dict, metrics: dict, mesh, k, frames_dir: Path, om_out: Path,
                 contact: int, rereg: set, fps: float, colour=(255, 200, 0)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (k.width, k.height))
    rgb = tuple(int(c) for c in colour[::-1])
    for i in sorted(poses):
        img = cv2.imread(str(frames_dir / f"{i:06d}.png"))
        img, _ = draw_mesh(img, mesh, poses[i], k, colour)
        _outline(img, load_mask(om_out, OBJECT_ID, i), (0, 255, 0))
        for hid in (RIGHT_ID, LEFT_ID):
            _outline(img, load_mask(om_out, hid, i), (255, 0, 255))
        m = metrics.get(str(i), {})
        state = "盒子静止" if i < contact else "手拿着"
        line = f"第 {i} 帧　{state}"
        if m.get("iou_visible") is not None:
            line += f"　可见部分 IoU {m['iou_visible']:.3f}　整体 IoU {m['iou_full']:.3f}"
        if m.get("depth_mm") is not None:
            line += f"　网格−深度 {m['depth_mm']:+.0f} mm"
        lines = [(title, rgb), (line, (255, 255, 255))]
        if "static_drift_mm" in m:
            lines.append((f"静止段漂移 {m['static_drift_mm']:.1f} mm / {m['static_drift_deg']:.1f}°", (255, 255, 255)))
        if i in rereg:
            lines.append(("跟丢了：这一帧重新注册", (255, 80, 80)))
        lines.append(("绿线 = SAM2 盒子 mask　紫线 = 手 mask", (200, 200, 200)))
        writer.write(_text(img, lines, size=30))
    writer.release()
    return path
