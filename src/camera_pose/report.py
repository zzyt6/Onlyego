"""After SLAM (Windows side): trajectory summary, TUM file, and a picture of the camera path."""

import json
from pathlib import Path

import cv2
import numpy as np

from .poses import load_trajectory, trajectory_summary, write_tum


CAMERA_WORLD_VIEWS = (("top view: x right, z (forward) up", (0, 2, True)),
                      ("side view: z (forward) right, y (down) down", (2, 1, False)))


def draw_trajectory(poses: np.ndarray, path: Path, size: int = 900, views=CAMERA_WORLD_VIEWS):
    """Two views of the camera centres, in metres; default for the first camera's frame: top view (x right,
    z forward) and side view (z forward, y down). Small arrows show the viewing direction every few frames.
    views: ((title, (horizontal axis, vertical axis, flip vertical so larger values are higher)), ...)."""
    centers = poses[:, :3, 3]
    forward = poses[:, :3, 2]  # camera z axis in world
    panels = []
    for name, (a, b, flip) in views:
        sign = np.array([1.0, -1.0 if flip else 1.0])
        pts = centers[:, [a, b]] * sign
        lo, hi = pts.min(0), pts.max(0)
        span = max(float((hi - lo).max()), 0.05)
        pad = 60
        scale = (size - 2 * pad) / span
        mid = (lo + hi) / 2

        def px(p):
            q = (p - mid) * scale + size / 2
            return int(round(q[0])), int(round(q[1]))

        img = np.full((size, size, 3), 255, np.uint8)
        n = len(pts)
        for i in range(n - 1):
            color = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[255 * i / max(n - 1, 1)]]), cv2.COLORMAP_VIRIDIS)[0, 0])
            cv2.line(img, px(pts[i]), px(pts[i + 1]), color, 2, cv2.LINE_AA)
        arrow = 0.08 * span
        for i in range(0, n, max(1, n // 15)):
            f = forward[i, [a, b]] * sign
            tip = pts[i] + arrow * f / max(np.linalg.norm(f), 1e-9)
            cv2.arrowedLine(img, px(pts[i]), px(tip), (60, 60, 200), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.circle(img, px(pts[0]), 7, (0, 160, 0), -1)
        cv2.circle(img, px(pts[-1]), 7, (0, 0, 200), -1)
        bar = 10 ** np.floor(np.log10(span / 2))
        cv2.line(img, (pad, size - 30), (pad + int(bar * scale), size - 30), (0, 0, 0), 3)
        cv2.putText(img, f"{bar * 100:g} cm", (pad, size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{name}   green start, red end, arrows = viewing direction", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(img)
    cv2.imwrite(str(path), np.hstack(panels))


def write_report(depth_run: Path, out: Path) -> dict:
    record = json.loads((depth_run / "run.json").read_text(encoding="utf-8"))
    fps = record["video"]["fps"]  # frame indices are source-video frame numbers, whatever the stride
    indices, poses = load_trajectory(out / "poses")
    slam = json.loads((out / "slam.json").read_text(encoding="utf-8"))
    summary = {"depth_run": str(depth_run), "slam": {k: v for k, v in slam.items() if k != "scale_per_keyframe"},
               "trajectory": trajectory_summary(indices, poses, fps)}
    scales = [r["scale"] for r in slam.get("scale_per_keyframe", [])]
    if scales:
        summary["slam"]["scale_spread"] = {"min": min(scales), "median": float(np.median(scales)), "max": max(scales)}
    write_tum(out / "trajectory_tum.txt", indices, poses)
    draw_trajectory(poses, out / "trajectory.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
