"""Compare depth sources over a video: MoGe-2 per frame (depth_estimation = what V2D uses) vs MoGe-2 + MegaSaM.

Outputs, in the video_depth run folder:
  compare.mp4             frame | one column per source; one fixed colour range for the whole video, so depth that
                          moves from frame to frame shows up as flicker (per-frame normalisation would hide it)
  compare_background.png  median depth of the static background per frame, one line per source
  compare.json            the numbers below

Static background = pixels whose colour never changes much over the video (hands and the moved object drop out).
This only means "static" for a fixed camera; with a moving camera every pixel changes and the mask is empty.

Numbers on the static background (pixels valid in every source):
  scale_jitter_pct     std / mean of the per-frame median depth (whole-frame scale flicker)
  pixel_wobble_pct     median over pixels of (temporal std / mean)  (per-pixel depth wobble)
  pixel_wobble_mm      median over pixels of temporal std, in mm
  frame_to_frame_pct   mean over frames of median |d_t - d_(t-1)| / d   (how much depth jumps between frames)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from depth_estimation.datatypes import frame_indices, frame_name, load_depth
from depth_estimation.inspection import colorize

COLORS = [(60, 60, 230), (230, 140, 40), (60, 170, 60), (180, 80, 200)]  # BGR line colours per source


def static_mask(frames_dir: Path, indices: list[int], shape: tuple[int, int], scale: int = 4,
                threshold: float = 20.0) -> np.ndarray:
    """Pixels whose grey value stays within `threshold` of its temporal median in every frame (fixed camera)."""
    H, W = shape
    small = (W // scale, H // scale)
    stack = np.stack([cv2.resize(cv2.imread(str(frames_dir / frame_name(i, ".png")), cv2.IMREAD_GRAYSCALE), small,
                                 interpolation=cv2.INTER_AREA).astype(np.float32) for i in indices])
    deviation = np.abs(stack - np.median(stack, axis=0)).max(axis=0)
    mask = (deviation < threshold).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # keep away from the edges of whatever moved
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def stability(depths: np.ndarray, mask: np.ndarray, min_valid: float = 0.8) -> dict:
    """depths (T,H,W) metric (inf/0 = invalid) on the pixels of `mask` -> the numbers in the module docstring.

    Each source is measured on its own valid pixels: a pixel of `mask` counts if it is valid in at least `min_valid`
    of the frames (rendered depth, e.g. the gsplat refinement's, has holes), and invalid samples are skipped.
    `coverage` = fraction of `mask` that was used."""
    d = depths[:, mask].astype(np.float64)                  # (T, N)
    ok = np.isfinite(d) & (d > 0)
    keep = ok.mean(axis=0) >= min_valid
    d = np.where(ok, d, np.nan)[:, keep]
    if d.shape[1] == 0:
        raise ValueError("no pixel of the static background is valid often enough in this source")
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(d, axis=1)
        std = np.nanstd(d, axis=0)
        mean = np.nanmean(d, axis=0)
        jumps = np.nanmedian(np.abs(np.diff(d, axis=0)) / d[1:], axis=1) if len(d) > 1 else np.zeros(1)
    medians = np.where(np.isfinite(medians), medians, np.nanmedian(medians))
    return {
        "coverage": round(float(keep.mean()), 4),
        "scale_jitter_pct": round(float(medians.std() / medians.mean() * 100), 3),
        "scale_range_pct": round(float((medians.max() - medians.min()) / medians.mean() * 100), 3),
        "pixel_wobble_pct": round(float(np.median(std / mean) * 100), 3),
        "pixel_wobble_mm": round(float(np.median(std) * 1000), 2),
        "frame_to_frame_pct": round(float(np.nanmean(jumps) * 100), 3),
        "median_depth_m": round(float(medians.mean()), 4),
        "per_frame_median_m": [round(float(m), 5) for m in medians],
    }


def plot_series(series: dict[str, list[float]], indices: list[int], output: Path, size=(1400, 600)):
    W, H = size
    left, right, top, bottom = 90, 30, 50, 60
    img = np.full((H, W, 3), 255, np.uint8)
    values = np.concatenate([np.asarray(v) for v in series.values()])
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    x = lambda i: int(left + (i - indices[0]) / max(1, indices[-1] - indices[0]) * (W - left - right))
    y = lambda v: int(top + (hi - v) / (hi - lo) * (H - top - bottom))
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        cv2.line(img, (left, y(v)), (W - right, y(v)), (225, 225, 225), 1)
        cv2.putText(img, f"{v:.3f}", (8, y(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    for k in range(0, indices[-1] + 1, max(10, (indices[-1] // 10) // 10 * 10)):
        cv2.putText(img, str(k), (x(k) - 10, H - bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    cv2.putText(img, "frame", (W // 2 - 20, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(img, "static background: median depth per frame (m)", (left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (40, 40, 40), 2)
    for n, (name, vals) in enumerate(series.items()):
        pts = np.array([(x(i), y(v)) for i, v in zip(indices, vals)], np.int32)
        cv2.polylines(img, [pts], False, COLORS[n % len(COLORS)], 2, cv2.LINE_AA)
        cv2.putText(img, name, (W - right - 420, top + 25 + 24 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS[n % len(COLORS)], 2)
    cv2.imwrite(str(output), img)


def compare(sources: dict[str, Path], frames_dir: Path, out: Path, fps: float = 30.0, width: int = 640,
            log=print) -> dict:
    """sources: label -> depth folder (uint16 inverse-depth PNGs, same frames and size)."""
    indices = sorted(set.intersection(*(set(frame_indices(p, ".png")) for p in sources.values())))
    if not indices:
        raise FileNotFoundError("the depth folders have no frames in common")
    depths = {name: np.stack([load_depth(p / frame_name(i, ".png")) for i in indices]) for name, p in sources.items()}
    shapes = {d.shape[1:] for d in depths.values()}
    if len(shapes) != 1:
        raise ValueError(f"depth sizes differ: { {n: d.shape[1:] for n, d in depths.items()} }")
    H, W = shapes.pop()
    mask = static_mask(frames_dir, indices, (H, W))
    log(f"static background: {mask.mean() * 100:.1f}% of the image")
    results = {name: stability(d, mask) for name, d in depths.items()}

    # One colour range for every source and frame (from the first source's static-free percentiles)
    ref = next(iter(depths.values()))
    valid = ref[np.isfinite(ref) & (ref > 0)]
    near, far = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
    h = int(round(width * H / W))
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width * (1 + len(depths)), h + 34))
    for t, i in enumerate(indices):
        tiles = [cv2.resize(cv2.imread(str(frames_dir / frame_name(i, ".png"))), (width, h))]
        tiles += [cv2.resize(colorize(d[t], near, far), (width, h), interpolation=cv2.INTER_NEAREST)
                  for d in depths.values()]
        frame = np.vstack([np.full((34, width * len(tiles), 3), 30, np.uint8), np.hstack(tiles)])
        for n, label in enumerate(["frame %d" % i, *depths]):
            cv2.putText(frame, label, (8 + n * width, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    plot_series({n: r["per_frame_median_m"] for n, r in results.items()}, indices, out / "compare_background.png")
    summary = {
        "sources": {n: str(p) for n, p in sources.items()},
        "frames": len(indices),
        "indices": indices,  # per_frame_median_m[k] belongs to video frame indices[k]
        "static_background_fraction": round(float(mask.mean()), 4),
        "colour_range_m": [round(near, 3), round(far, 3)],
        "stability": results,
    }
    (out / "compare.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cv2.imwrite(str(out / "compare_static_mask.png"), (mask * 255).astype(np.uint8))
    return summary
