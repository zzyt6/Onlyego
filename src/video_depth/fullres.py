"""MegaSaM depth (tracking resolution, e.g. 584x328) -> full resolution, keeping MoGe-2's detail (ours).

full = MoGe_full * exp(lift(smooth(log(MegaSaM / MoGe))))

The ratio field carries what MegaSaM changed -- mostly each frame's scale, plus slow per-pixel corrections.
It is smoothed at low resolution (5x5 median, then Gaussian sigma 2 px) so MegaSaM's blurred object edges do
not leak into it, then bilinearly lifted to full size (columns/rows cropped by MegaSaM are filled by edge
replication). MoGe's sharp edges and fine shape stay; invalid MoGe pixels stay invalid.

demo01 (every 4th frame), full-res background wobble / edge halo / difference from MegaSaM at low res:
  one scale per frame      0.27% / 0      / 0.18%
  raw bilinear ratio       0.13% / 0.56%  / 0.00%
  smoothed ratio (this)    0.13% / 0.034% / 0.01%      (MoGe alone: 1.63% wobble)
"""

import cv2
import numpy as np


def lift_frame(moge_full: np.ndarray, megasam_low: np.ndarray, resized_hw: tuple[int, int],
               median_ksize: int = 5, sigma: float = 2.0) -> np.ndarray:
    """moge_full (H,W) metric depth (inf = invalid); megasam_low (hc,wc) on MegaSaM's grid, i.e. the frame
    resized to resized_hw = (h1, w1) and cropped to (hc, wc) from the top-left."""
    H, W = moge_full.shape
    h1, w1 = resized_hw
    hc, wc = megasam_low.shape
    m = cv2.resize(moge_full, (w1, h1), interpolation=cv2.INTER_NEAREST_EXACT)[:hc, :wc]
    valid = np.isfinite(m) & (m > 0) & np.isfinite(megasam_low) & (megasam_low > 0)
    if valid.sum() < 64:
        return moge_full.copy()
    logr = np.zeros((hc, wc), np.float32)
    logr[valid] = np.log(megasam_low[valid] / m[valid])
    logr[~valid] = np.median(logr[valid])
    logr = cv2.GaussianBlur(cv2.medianBlur(logr, median_ksize), (0, 0), sigma)
    padded = cv2.copyMakeBorder(logr, 0, h1 - hc, 0, w1 - wc, cv2.BORDER_REPLICATE)
    ratio = np.exp(cv2.resize(padded, (W, H), interpolation=cv2.INTER_LINEAR))
    return (moge_full * ratio).astype(np.float32)
