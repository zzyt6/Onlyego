"""Temporal scale alignment of a monocular depth sequence (V2D v2d_depth/lib/align_depth_sequence.py).

Each frame's depth comes out with its own global scale. To make the sequence consistent:
  1. match SIFT features between each frame and a reference frame, background only
     (object pixels are masked out, because the object moves relative to the scene);
  2. per-frame scale = median(ref_depth / frame_depth) over the matched pixels;
  3. fill frames without enough matches by interpolation, then median-filter the scales
     in log space (a ratio of 2 and 0.5 are equally far from 1);
  4. multiply each depth map by its smoothed scale.
Only scale is corrected, not shift -- right for a metric model like MoGe-2.

In V2D this utility exists but none of their ego pipelines call it (they use raw MoGe depth
with stabilised intrinsics). It is kept here as a separate, optional step.

Changes from V2D: matched pixels with invalid (inf/0) depth are rejected -- V2D's `d > 0.001`
test lets inf through; per-frame raw and smoothed scales are returned and saved for inspection;
by default the overall metric scale is the median over all frames instead of the reference frame's own
(metric_scale="reference" restores V2D). Intended for a still (or nearly still) camera: every frame must
share background with the reference frame. `depth_estimation run --scale-align` runs it automatically.
"""

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from .datatypes import frame_indices, frame_name, load_depth, load_mask, save_depth

log = logging.getLogger(__name__)
MIN_MATCHES = 10


def estimate_scale(ref_gray, ref_depth, ref_bg, gray, depth, bg, ratio_test=0.75) -> tuple[float | None, int]:
    """Scale that maps this frame's depth onto the reference's: median(d_ref / d) at matched features."""
    sift = cv2.SIFT_create()
    kp_ref, desc_ref = sift.detectAndCompute(ref_gray, ref_bg.astype(np.uint8) * 255)
    kp, desc = sift.detectAndCompute(gray, bg.astype(np.uint8) * 255)
    if desc_ref is None or desc is None or len(kp_ref) < 4 or len(kp) < 4:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_ref, desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio_test * p[1].distance]
    h, w = ref_depth.shape
    ratios = []
    for m in good:
        xr, yr = (int(round(v)) for v in kp_ref[m.queryIdx].pt)
        xf, yf = (int(round(v)) for v in kp[m.trainIdx].pt)
        if not (0 <= xr < w and 0 <= yr < h and 0 <= xf < w and 0 <= yf < h):
            continue
        d_ref, d = ref_depth[yr, xr], depth[yf, xf]
        if np.isfinite(d_ref) and np.isfinite(d) and d_ref > 1e-3 and d > 1e-3:
            ratios.append(d_ref / d)
    if len(ratios) < MIN_MATCHES:
        return None, len(ratios)
    return float(np.median(ratios)), len(ratios)


def smooth_scales(raw: dict[int, float], indices: list[int], window: int) -> dict[int, float]:
    """Interpolate missing frames, then median-filter log(scale) with the given (odd) window."""
    indices = sorted(indices)
    log_raw = np.full(len(indices), np.nan)
    for pos, idx in enumerate(indices):
        if raw.get(idx) and raw[idx] > 0:
            log_raw[pos] = np.log(raw[idx])
    valid = np.flatnonzero(~np.isnan(log_raw))
    if len(valid) == 0:
        return {idx: 1.0 for idx in indices}
    filled = np.interp(np.arange(len(indices)), valid, log_raw[valid])
    half = window // 2
    smoothed = [np.median(filled[max(0, i - half):i + half + 1]) for i in range(len(indices))]
    return {idx: float(np.exp(s)) for idx, s in zip(indices, smoothed)}


def _background(masks: Path | None, index: int, shape) -> np.ndarray:
    if masks is not None and (masks / frame_name(index, ".png")).exists():
        return ~load_mask(masks / frame_name(index, ".png"))
    return np.ones(shape, dtype=bool)


def align_depth_sequence(depth_dir: Path, frames_dir: Path, output_dir: Path, *, masks_dir: Path | None = None,
                         reference_frame: int | None = None, window: int = 11, metric_scale: str = "median") -> dict:
    """metric_scale: which overall metric scale the aligned sequence keeps.
      "median"    (ours, default): divide all scales by their median, so the result has the median of all frames'
                  MoGe scales -- one unlucky reference frame no longer biases the whole video;
      "reference" (V2D): keep the reference frame's own scale (its scale is 1, everything follows it)."""
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    if metric_scale not in ("median", "reference"):
        raise ValueError("metric_scale must be 'median' or 'reference'")
    indices = frame_indices(depth_dir, ".png")
    if not indices:
        raise FileNotFoundError(f"No depth PNGs in {depth_dir}")
    reference = indices[0] if reference_frame is None else reference_frame
    if reference not in indices:
        raise ValueError(f"Reference frame {reference} has no depth map")
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_depth = load_depth(depth_dir / frame_name(reference, ".png"))
    ref_gray = cv2.imread(str(frames_dir / frame_name(reference, ".png")), cv2.IMREAD_GRAYSCALE)
    if ref_gray is None:
        raise FileNotFoundError(f"Missing reference frame image {reference}")
    ref_bg = _background(masks_dir, reference, ref_depth.shape)

    raw, matches = {reference: 1.0}, {reference: None}
    for idx in indices:
        if idx == reference:
            continue
        gray = cv2.imread(str(frames_dir / frame_name(idx, ".png")), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            log.warning("frame %d: missing RGB, scale will be interpolated", idx)
            continue
        depth = load_depth(depth_dir / frame_name(idx, ".png"))
        scale, n = estimate_scale(ref_gray, ref_depth, ref_bg, gray, depth, _background(masks_dir, idx, depth.shape))
        matches[idx] = n
        if scale is None:
            log.warning("frame %d: only %d usable matches, scale will be interpolated", idx, n)
        else:
            raw[idx] = scale

    smoothed = smooth_scales(raw, indices, window)
    if metric_scale == "median":
        centre = float(np.median(list(smoothed.values())))
        smoothed = {i: s / centre for i, s in smoothed.items()}
    for idx in indices:
        src, dst = depth_dir / frame_name(idx, ".png"), output_dir / frame_name(idx, ".png")
        if abs(smoothed[idx] - 1.0) < 1e-6:
            shutil.copy2(src, dst)
        else:
            save_depth(dst, load_depth(src) * smoothed[idx])
    return {"reference_frame": reference, "window": window, "metric_scale": metric_scale,
            "estimated": len(raw), "frames": len(indices),
            "per_frame": [{"frame": i, "raw_scale": raw.get(i), "matches": matches.get(i),
                           "scale": smoothed[i]} for i in indices]}
