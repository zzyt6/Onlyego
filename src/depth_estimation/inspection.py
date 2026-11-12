"""Ways to look at the result: per-frame statistics and a side-by-side preview video."""

from pathlib import Path

import cv2
import numpy as np

from .datatypes import CameraIntrinsics, frame_indices, frame_name, load_depth


def depth_stats(depth_dir: Path, intrinsics_dir: Path) -> list[dict]:
    rows = []
    for i in frame_indices(depth_dir, ".png"):
        depth = load_depth(depth_dir / frame_name(i, ".png"))
        valid = depth[np.isfinite(depth) & (depth > 0)]
        k_path = intrinsics_dir / frame_name(i, ".json")
        k = CameraIntrinsics.load(k_path) if k_path.exists() else None
        rows.append({
            "frame": i,
            "valid_fraction": round(float(valid.size / depth.size), 4),
            "depth_p05": round(float(np.percentile(valid, 5)), 4) if valid.size else None,
            "depth_median": round(float(np.median(valid)), 4) if valid.size else None,
            "depth_p95": round(float(np.percentile(valid, 95)), 4) if valid.size else None,
            "fx": round(k.fx, 2) if k else None,
            "fov_x": round(k.fov_x_degrees, 2) if k else None,
        })
    return rows


def colorize(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Inverse-depth TURBO colouring with one fixed range for the whole video (so colours are comparable)."""
    with np.errstate(divide="ignore"):
        inv = 1.0 / depth
    t = (inv - 1.0 / far) / (1.0 / near - 1.0 / far)
    t = np.where(np.isfinite(t), t, 0.0).clip(0, 1)
    image = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~(np.isfinite(depth) & (depth > 0))] = 0
    return image


def preview_video(frames_dir: Path, depth_dir: Path, output: Path, fps: float, max_width: int = 1600) -> Path:
    indices = frame_indices(depth_dir, ".png")
    if not indices:
        raise FileNotFoundError(f"No depth PNGs in {depth_dir}")
    sample = indices[:: max(1, len(indices) // 30)]
    values = np.concatenate([d[np.isfinite(d) & (d > 0)].ravel()[::50]
                             for d in (load_depth(depth_dir / frame_name(i, ".png")) for i in sample)])
    near, far = (float(np.percentile(values, 2)), float(np.percentile(values, 98))) if values.size else (0.1, 10.0)
    far = max(far, near * 1.01)
    writer = None
    try:
        for i in indices:
            rgb = cv2.imread(str(frames_dir / frame_name(i, ".png")))
            depth_image = colorize(load_depth(depth_dir / frame_name(i, ".png")), near, far)
            if rgb is None or rgb.shape[:2] != depth_image.shape[:2]:
                rgb = np.zeros_like(depth_image)
            panel = np.hstack([rgb, depth_image])
            scale = min(1.0, max_width / panel.shape[1])
            if scale < 1.0:
                panel = cv2.resize(panel, (int(panel.shape[1] * scale) // 2 * 2, int(panel.shape[0] * scale) // 2 * 2))
            cv2.putText(panel, f"{i:06d}  near {near:.2f} m  far {far:.2f} m", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (panel.shape[1], panel.shape[0]))
            writer.write(panel)
    finally:
        if writer is not None:
            writer.release()
    return output
