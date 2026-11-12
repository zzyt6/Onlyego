"""Pick the reference frame (object seen clearly, before the hand touches it) and the window of frames before it.

For every frame, from the SAM2 masks:
  - near_hand: object pixels within `contact_px` of a hand mask (dilated hand mask ∩ object mask);
  - iou:       object mask IoU with the detection frame's mask (the object as first seen, lying untouched);
  - sharpness: variance of the Laplacian inside the eroded object mask (motion blur drops it ~10x).
contact frame = first frame where near_hand > `contact_min_px`, or the object mask changes (iou < `min_iou`).
reference     = the last frame before contact.
window        = up to `window` frames ending at the reference whose object mask still matches the reference's
                (iou >= min_iou) and that have no hand near the object: the object is static and untouched, so
                its multi-frame depth can be fused (pointmap.py).
"""

import cv2
import numpy as np

from .segment import LEFT_ID, OBJECT_ID, RIGHT_ID, load_mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def frame_stats(seg_dir, frames_dir, indices: list[int], detect_frame: int, contact_px: int = 30) -> list[dict]:
    ref_mask = load_mask(seg_dir, OBJECT_ID, detect_frame)
    kernel = np.ones((2 * contact_px + 1, 2 * contact_px + 1), np.uint8)
    rows = []
    for i in indices:
        obj = load_mask(seg_dir, OBJECT_ID, i)
        if obj is None:
            rows.append({"frame": i, "object_px": 0})
            continue
        hands = np.zeros_like(obj)
        for hid in (RIGHT_ID, LEFT_ID):
            m = load_mask(seg_dir, hid, i)
            if m is not None:
                hands |= m
        near = int(((cv2.dilate(hands.astype(np.uint8), kernel) > 0) & obj).sum())
        gray = cv2.imread(str(frames_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        inner = cv2.erode(obj.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F)[inner].var()) if inner.sum() > 100 else 0.0
        rows.append({"frame": i, "object_px": int(obj.sum()), "hand_px": int(hands.sum()), "near_hand_px": near,
                     "iou_detect": round(_iou(obj, ref_mask), 4), "sharpness": round(sharp, 1)})
    return rows


def choose(rows: list[dict], seg_dir, detect_frame: int, window: int = 20, min_iou: float = 0.95,
           contact_min_px: int = 50) -> dict:
    by = {r["frame"]: r for r in rows}
    frames = [r["frame"] for r in rows if r["frame"] >= detect_frame]
    contact = next((f for f in frames if by[f]["object_px"] == 0 or by[f]["near_hand_px"] > contact_min_px
                    or by[f]["iou_detect"] < min_iou), None)
    candidates = [f for f in frames if contact is None or f < contact]
    if not candidates:
        raise RuntimeError(f"The hand is already at the object on the detection frame {detect_frame}; "
                           "pick an earlier --detect-frame")
    reference = candidates[-1]
    ref_mask = load_mask(seg_dir, OBJECT_ID, reference)
    win = []
    for f in reversed(candidates):
        if len(win) == window:
            break
        if by[f]["near_hand_px"] <= contact_min_px and _iou(load_mask(seg_dir, OBJECT_ID, f), ref_mask) >= min_iou:
            win.append(f)
    win.sort()
    return {"reference_frame": reference, "contact_frame": contact, "window": win,
            "rule": {"window": window, "min_iou": min_iou, "contact_min_px": contact_min_px}}
