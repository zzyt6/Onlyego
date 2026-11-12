"""Long videos: MegaSaM on overlapping segments, chained into one sequence (ours; MegaSaM has no such step).

MegaSaM keeps every frame of a segment in GPU memory and its consistent-depth step processes all frame
pairs at once (~5 per frame), so memory grows with length (184 frames at 1080p: 11.4 GB). Segments of
`length` frames that overlap by `overlap` frames are processed separately, then joined left to right:

  1. depth scale: s = median over overlap frames of median(merged depth / segment depth); the segment's
     depths and camera translations are multiplied by s (each segment is already metric via MoGe, so s ~ 1;
     this removes the small jump between them);
  2. world frame: each segment's world is its own first camera. T = average over overlap frames of
     merged_c2w @ inv(segment_c2w) (rotations: chordal mean; translations: mean) maps the segment's world
     into the merged one;
  3. overlap frames are cross-faded: depth = w * merged + (1 - w) * segment with w going 1 -> 0 across the
     overlap; pose and motion probability come from whichever side has the larger weight.
The overlap should exceed MegaSaM's largest flow step (15 frames) so both sides constrain those frames fully.
"""

import math

import numpy as np


def plan_segments(n: int, length: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) windows covering 0..n, each `length` long (the last ones may share more frames)."""
    if length <= overlap:
        raise ValueError("segment length must exceed the overlap")
    if n <= length:
        return [(0, n)]
    count = math.ceil((n - overlap) / (length - overlap))
    starts = np.round(np.linspace(0, n - length, count)).astype(int)
    return [(int(s), int(s) + length) for s in starts]


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (projection of their sum onto SO(3))."""
    U, _, Vt = np.linalg.svd(rotations.sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def depth_ratio(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.median(a[valid] / b[valid])) if valid.sum() > 64 else np.nan


def chain(segments: list[dict]) -> dict:
    """segments: dicts with 'frames' (global positions, ascending), 'c2w' (T,4,4), 'depth' (T,h,w),
    'motion_prob' (T,...). Returns the merged dict plus per-segment 'join' diagnostics."""
    merged = {k: (list(v) if k == "frames" else v.copy()) for k, v in segments[0].items()
              if k in ("frames", "c2w", "depth", "motion_prob")}
    joins = []
    for seg in segments[1:]:
        pos = {f: j for j, f in enumerate(merged["frames"])}
        common = [f for f in seg["frames"] if f in pos]
        if len(common) < 2:
            raise ValueError("segments must overlap by at least 2 frames")
        si = {f: j for j, f in enumerate(seg["frames"])}
        scale = float(np.nanmedian([depth_ratio(merged["depth"][pos[f]], seg["depth"][si[f]]) for f in common]))
        c2w = seg["c2w"].copy()
        c2w[:, :3, 3] *= scale
        depth = seg["depth"] * scale
        rel = np.stack([merged["c2w"][pos[f]] @ np.linalg.inv(c2w[si[f]]) for f in common])
        T = np.eye(4)
        T[:3, :3] = mean_rotation(rel[:, :3, :3])
        T[:3, 3] = rel[:, :3, 3].mean(axis=0)
        c2w = T @ c2w
        resid = [np.linalg.norm(merged["c2w"][pos[f]][:3, 3] - c2w[si[f]][:3, 3]) for f in common]
        joins.append({"overlap_frames": len(common), "depth_scale": scale,
                      "pose_residual_m": {"median": float(np.median(resid)), "max": float(np.max(resid))}})
        for j, f in enumerate(common):
            w = 1.0 - (j + 0.5) / len(common)  # 1 -> 0 across the overlap
            a, b = pos[f], si[f]
            merged["depth"][a] = w * merged["depth"][a] + (1 - w) * depth[b]
            if w < 0.5:
                merged["c2w"][a] = c2w[b]
                merged["motion_prob"][a] = seg["motion_prob"][b]
        new = [f for f in seg["frames"] if f not in pos]
        if new:
            take = [si[f] for f in new]
            merged["frames"] += new
            merged["c2w"] = np.concatenate([merged["c2w"], c2w[take]])
            merged["depth"] = np.concatenate([merged["depth"], depth[take]])
            merged["motion_prob"] = np.concatenate([merged["motion_prob"], seg["motion_prob"][take]])
    merged["joins"] = joins
    return merged
