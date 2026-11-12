"""Overlay video of a pose sequence: the mesh drawn at each frame's pose, SAM2 masks outlined, per-frame scores."""

from pathlib import Path

import cv2
import numpy as np

from object_model.overlay import _text, draw_mesh
from object_model.segment import LEFT_ID, OBJECT_ID, RIGHT_ID, load_mask


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
            lines.append(("IoU < 0.3：这一帧重新注册", (255, 80, 80)))
        lines.append(("绿线 = SAM2 盒子 mask　紫线 = 手 mask", (200, 200, 200)))
        writer.write(_text(img, lines, size=30))
    writer.release()
    return path
