"""The object's point map in the reference camera, fused from the window of static, untouched frames.

Each window frame's object pixels (mask eroded by `erode_px`: edge pixels mix object and table depth) are
back-projected with its depth, moved to the world by its camera pose (camera->world), into the reference camera,
and projected onto the reference pixel grid. Per reference pixel the median depth of all samples is kept when at
least `min_count` frames hit it; elsewhere NaN. With a still camera this is the per-pixel median over time; with
a moving camera the poses line the frames up.

SAM3D (stage 1) reads this as the object's 3D shape and position in the camera; its scale is what makes the
generated mesh metric. Pixels outside the object are NaN, as in V2D's depth input to SAM3D.
"""

import cv2
import numpy as np

from depth_estimation.datatypes import CameraIntrinsics, load_depth

from .segment import OBJECT_ID, load_mask


def _pixels(depth: np.ndarray, mask: np.ndarray, k: CameraIntrinsics):
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u]
    return np.stack([(u + 0.5 - k.cx) * z / k.fx, (v + 0.5 - k.cy) * z / k.fy, z], 1)


def fuse(depth_dir, seg_dir, poses: dict[int, np.ndarray], k: CameraIntrinsics, reference: int, window: list[int],
         erode_px: int = 3, min_count: int | None = None) -> tuple[np.ndarray, dict]:
    """Returns (pointmap (H,W,3) float32 in the reference camera, OpenCV axes, NaN = no data; stats)."""
    kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    H, W = k.height, k.width
    w2c_ref = np.linalg.inv(poses[reference])
    keys, zs = [], []
    for f in window:
        depth = load_depth(depth_dir / f"{f:06d}.png")
        mask = cv2.erode(load_mask(seg_dir, OBJECT_ID, f).astype(np.uint8), kernel) > 0
        p = _pixels(depth, mask, k)
        T = w2c_ref @ poses[f]
        q = p @ T[:3, :3].T + T[:3, 3]
        u = np.floor(q[:, 0] / q[:, 2] * k.fx + k.cx).astype(np.int64)
        v = np.floor(q[:, 1] / q[:, 2] * k.fy + k.cy).astype(np.int64)
        ok = (q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        keys.append(v[ok] * W + u[ok])
        zs.append(q[ok, 2])
    keys, zs = np.concatenate(keys), np.concatenate(zs)
    order = np.lexsort((zs, keys))
    keys, zs = keys[order], zs[order]
    uniq, start, count = np.unique(keys, return_index=True, return_counts=True)
    lo = zs[start + (count - 1) // 2]
    hi = zs[start + count // 2]
    med = 0.5 * (lo + hi)
    # spread across frames: median absolute deviation per pixel (how consistent the depth is over time)
    dev = np.abs(zs - np.repeat(med, count))
    mad = np.array([np.median(dev[s:s + c]) for s, c in zip(start, count)]) if len(uniq) < 400_000 else None

    min_count = min_count or max(3, int(0.5 * len(window)))
    ref_mask = cv2.erode(load_mask(seg_dir, OBJECT_ID, reference).astype(np.uint8), kernel) > 0
    keep = (count >= min_count) & ref_mask.reshape(-1)[uniq]
    depth = np.full(H * W, np.nan, np.float32)
    depth[uniq[keep]] = med[keep]
    depth = depth.reshape(H, W)
    v, u = np.mgrid[0:H, 0:W]
    pointmap = np.stack([(u + 0.5 - k.cx) * depth / k.fx, (v + 0.5 - k.cy) * depth / k.fy, depth], -1)

    ref_depth = load_depth(depth_dir / f"{reference:06d}.png")
    both = np.isfinite(depth) & np.isfinite(ref_depth)
    stats = {
        "frames": len(window), "min_count": min_count, "erode_px": erode_px,
        "pixels": int(np.isfinite(depth).sum()), "reference_mask_px": int(ref_mask.sum()),
        "median_depth_m": float(np.nanmedian(depth)),
        "temporal_spread_mm_median": None if mad is None else float(np.median(mad[keep]) * 1000),
        "temporal_spread_mm_p90": None if mad is None else float(np.percentile(mad[keep], 90) * 1000),
        "fused_vs_reference_frame_mm_median": float(np.median(np.abs(depth[both] - ref_depth[both])) * 1000),
    }
    return pointmap.astype(np.float32), stats
