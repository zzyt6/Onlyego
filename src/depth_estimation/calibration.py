"""AnyCalib focal-length prior for MoGe-2.

AnyCalib (learned single-image calibration) predicts a per-pixel ray field and fits a camera model to it. We fit a
distortion-free pinhole camera with the `anycalib_pinhole` weights: our cameras (iPhone main camera, smart glasses)
are already undistorted in-camera. It runs on N frames spread over the video and the results are median-pooled;
fov_x = 2 atan(W / 2 fx) is then held fixed by MoGe, so every frame gets the same camera instead of its own guess.

From V2D reconstruction/modules/v2d_anycalib/lib/{_anycalib,video_to_calibration}.py; changes: frames are read
from the extracted PNGs, weights come from model/anycalib, pinhole model only.
"""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .datatypes import CameraIntrinsics, frame_indices, frame_name


def sample_indices(indices: list[int], n: int) -> list[int]:
    """n frames evenly spread over the list; floors like np.linspace(..., dtype=int)."""
    if n >= len(indices):
        return list(indices)
    return [indices[i] for i in np.linspace(0, len(indices) - 1, n, dtype=int)]


def _load_rgb_tensor(path: Path, device: str):
    import torch
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)


def _median_intrinsics(ks: list[CameraIntrinsics]) -> CameraIntrinsics:
    return CameraIntrinsics(fx=float(np.median([k.fx for k in ks])), fy=float(np.median([k.fy for k in ks])),
                            cx=float(np.median([k.cx for k in ks])), cy=float(np.median([k.cy for k in ks])),
                            width=ks[0].width, height=ks[0].height)


def valid_intrinsics(k: CameraIntrinsics) -> bool:
    values = np.array([k.fx, k.fy, k.cx, k.cy], dtype=np.float64)
    return bool(np.all(np.isfinite(values)) and k.fx > 0 and k.fy > 0
                and 0 <= k.cx <= k.width and 0 <= k.cy <= k.height)


def run_anycalib(frames: Path, indices: list[int], weights: Path, device: str = "cuda", log=print):
    import torch
    from anycalib import AnyCalib

    if not weights.is_file():
        raise FileNotFoundError(f"AnyCalib weights not found: {weights}")
    model = AnyCalib(model_id=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    samples = []
    for n, i in enumerate(indices, 1):
        image = _load_rgb_tensor(frames / frame_name(i, ".png"), device)
        with torch.no_grad():
            params = model.predict(image, cam_id="pinhole")["intrinsics"].detach().cpu().double().numpy().reshape(-1)
        h, w = image.shape[-2:]
        k = CameraIntrinsics(*(float(v) for v in params[:4]), width=int(w), height=int(h))
        samples.append({"frame_index": i, "intrinsics": asdict(k)})
        log(f"         AnyCalib frame {i} ({n}/{len(indices)}): fx={k.fx:.1f} fov_x={k.fov_x_degrees:.1f} deg")
    return _median_intrinsics([CameraIntrinsics(**s["intrinsics"]) for s in samples]), samples


def calibrate(frames: Path, out: Path, weights: Path, *, samples: int, device: str = "cuda", log=print) -> dict:
    """AnyCalib (pinhole) on `samples` frames; writes out/{intrinsics,calibration}.json."""
    indices = sample_indices(frame_indices(frames, ".png"), samples)
    if not indices:
        raise FileNotFoundError(f"No frames in {frames}")
    k, records = run_anycalib(frames, indices, weights, device, log)
    if not valid_intrinsics(k):
        raise RuntimeError(f"AnyCalib returned an invalid calibration: {k}")
    out.mkdir(parents=True, exist_ok=True)
    k.save(out / "intrinsics.json")
    summary = {"method": "anycalib", "cam_id": "pinhole", "weights": weights.name, "num_samples": len(records),
               "intrinsics": asdict(k), "fov_x_degrees": k.fov_x_degrees, "samples": records}
    (out / "calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
