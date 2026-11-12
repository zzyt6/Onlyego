"""Optional camera calibration used as a focal-length prior for MoGe-2.

From V2D:
  reconstruction/modules/v2d_anycalib/lib/{_anycalib,video_to_calibration,_undistort}.py
  reconstruction/modules/v2d_geocalib/lib/{_geocalib,video_to_calibration}.py
  wiring in v2d_pipelines/run_ego_wilor.py: moge_input_intrinsics = geocalib_intr if use_geocalib_intrinsics
                                            else (undistorted_intr if undistort else None)

Both are learned single-image calibrators. Each is run on N frames spread over the video and the
results are median-pooled. The pooled fx becomes fov_x = 2 atan(W / 2 fx), which MoGe then holds
fixed while it solves depth -- every frame gets the same camera instead of its own guess.

  AnyCalib  predicts a per-pixel ray field and fits a camera model to it, including lens distortion
            (default here "radial:1" for phone / smart-glasses lenses; V2D defaults to the Kannala-Brandt
            fisheye model "kb:4" for its fisheye head cameras). With --undistort the frames are remapped to an
            ideal pinhole camera first and MoGe runs on those -- this is V2D's only AnyCalib path.
            Without --undistort we fit a pure pinhole camera (anycalib_pinhole weights), because a
            fisheye focal length is not a valid pinhole fov prior for distorted frames.
  GeoCalib  optimises focal length and gravity direction together from learned up-vector and
            latitude fields. Besides intrinsics it gives gravity in camera coordinates (roll/pitch),
            which V2D later uses to rotate the world so that "up" is up.

Changes from V2D: frames are read from the extracted PNGs; undistortion writes PNG frames instead
of an ffmpeg re-encode; weights are loaded from files in model/ instead of a TORCH_HOME cache.
Kept as V2D: sample frames np.linspace(..., dtype=int); per-parameter medians; GeoCalib weights and
camera model chosen separately; per-sample records with V2D's keys (frame_index, gravity roll/pitch in
rad and deg, rotation_matrix, uncertainty, covariance), which camera_pose.gravity reads.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .datatypes import CameraIntrinsics, frame_indices, frame_name

ANYCALIB_CAM_IDS = ("pinhole", "radial:1", "radial:2", "radial:4", "kb:4")
# AnyCalib cam_id -> (distortion model in OpenCV terms, number of coefficients)
_ANYCALIB_DISTORTION = {"pinhole": ("pinhole", 0), "radial:1": ("opencv", 1), "radial:2": ("opencv", 2),
                        "radial:4": ("opencv", 4), "kb:4": ("opencv_fisheye", 4)}
# GeoCalib camera models. V2D picks weights and camera model separately (both default "pinhole");
# a distorted camera model needs the "distorted" weights.
GEOCALIB_CAMERA_MODELS = ("pinhole", "simple_radial", "radial", "simple_divisional")
GEOCALIB_DISTORTION_NAME = {"simple_radial": "geocalib_simple_radial", "radial": "geocalib_radial",
                            "simple_divisional": "geocalib_simple_divisional"}


@dataclass
class CameraDistortion:
    model: str  # pinhole | opencv | opencv_fisheye | geocalib_<model>
    params: list[float] = field(default_factory=list)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=4), encoding="utf-8")


def sample_indices(indices: list[int], n: int) -> list[int]:
    """n frames evenly spread over the list; floors like V2D's np.linspace(..., dtype=int)."""
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


def run_anycalib(frames: Path, indices: list[int], weights: Path, cam_id: str, device: str = "cuda", log=print):
    import torch
    from anycalib import AnyCalib

    if cam_id not in _ANYCALIB_DISTORTION:
        raise ValueError(f"Unsupported AnyCalib cam_id {cam_id!r}; choose from {ANYCALIB_CAM_IDS}")
    if not weights.is_file():
        raise FileNotFoundError(f"AnyCalib weights not found: {weights}")
    model = AnyCalib(model_id=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    dist_model, n_dist = _ANYCALIB_DISTORTION[cam_id]
    samples = []
    for n, i in enumerate(indices, 1):
        image = _load_rgb_tensor(frames / frame_name(i, ".png"), device)
        with torch.no_grad():
            params = model.predict(image, cam_id=cam_id)["intrinsics"].detach().cpu().double().numpy().reshape(-1)
        h, w = image.shape[-2:]
        k = CameraIntrinsics(*(float(v) for v in params[:4]), width=int(w), height=int(h))
        samples.append({"frame_index": i, "intrinsics": asdict(k),
                        "distortion": [float(v) for v in params[4:4 + n_dist]]})
        log(f"         AnyCalib frame {i} ({n}/{len(indices)}): fx={k.fx:.1f} fov_x={k.fov_x_degrees:.1f} deg")
    k = _median_intrinsics([CameraIntrinsics(**s["intrinsics"]) for s in samples])
    params = np.median(np.array([s["distortion"] for s in samples]), axis=0).tolist() if n_dist else []
    return k, CameraDistortion(dist_model, [float(v) for v in params]), samples, None


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _to_builtin(value):
    """Tensors/arrays -> JSON values (V2D _geocalib._to_builtin)."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0]) if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    return value


def run_geocalib(frames: Path, indices: list[int], weights: Path, camera_model: str, device: str = "cuda", log=print):
    import torch
    from geocalib import GeoCalib

    if not weights.is_file():
        raise FileNotFoundError(f"GeoCalib weights not found: {weights}")
    model = GeoCalib(weights=str(weights)).to(device).eval()
    samples = []
    for n, i in enumerate(indices, 1):
        image = _load_rgb_tensor(frames / frame_name(i, ".png"), device)
        with torch.no_grad():
            result = model.calibrate(image, camera_model=camera_model)
        camera, gravity = result["camera"], result["gravity"]
        size = _numpy(camera.size).reshape(-1, 2)[0]
        focal = _numpy(camera.f).reshape(-1, 2)[0]
        center = _numpy(camera.c).reshape(-1, 2)[0]
        k = CameraIntrinsics(fx=float(focal[0]), fy=float(focal[1]), cx=float(center[0]), cy=float(center[1]),
                             width=int(round(float(size[0]))), height=int(round(float(size[1]))))
        dist = []
        if camera_model != "pinhole" and hasattr(camera, "dist"):
            dist = _numpy(camera.dist).reshape(-1)
            if hasattr(camera, "num_dist_params"):
                dist = dist[: int(camera.num_dist_params())]
            dist = [float(v) for v in dist]
        roll, pitch = float(_numpy(gravity.roll).reshape(-1)[0]), float(_numpy(gravity.pitch).reshape(-1)[0])
        g = {"vector_camera": [float(v) for v in _numpy(gravity.vec3d).reshape(-1, 3)[0]],
             "roll_rad": roll, "pitch_rad": pitch,
             "roll_deg": float(np.degrees(roll)), "pitch_deg": float(np.degrees(pitch)),
             "rotation_matrix": _numpy(gravity.R).reshape(-1, 3, 3)[0].tolist()}
        samples.append({"frame_index": i, "intrinsics": asdict(k), "distortion": dist, "gravity": g,
                        "uncertainty": {key: _to_builtin(v) for key, v in result.items() if "uncertainty" in key},
                        "covariance": _to_builtin(result.get("covariance"))})
        log(f"         GeoCalib frame {i} ({n}/{len(indices)}): fx={k.fx:.1f} fov_x={k.fov_x_degrees:.1f} deg "
            f"roll={g['roll_deg']:.1f} pitch={g['pitch_deg']:.1f}")
    k = _median_intrinsics([CameraIntrinsics(**s["intrinsics"]) for s in samples])
    params = np.median(np.array([s["distortion"] for s in samples]), axis=0).tolist() if samples[0]["distortion"] else []
    model_name = "pinhole" if camera_model == "pinhole" else GEOCALIB_DISTORTION_NAME.get(camera_model,
                                                                                         f"geocalib_{camera_model}")
    # Gravity: mean of unit vectors, renormalised (V2D aggregate_calibrations)
    vectors = np.array([s["gravity"]["vector_camera"] for s in samples], dtype=np.float64)
    mean = vectors.mean(axis=0)
    if np.linalg.norm(mean) < 1e-8:
        raise RuntimeError("GeoCalib gravity estimates cancelled out during aggregation")
    gravity = {"model": "GeoCalib", "weights": weights.name, "camera_model": camera_model,
               "vector_camera": (mean / np.linalg.norm(mean)).tolist(), "num_samples": len(samples)}
    return k, CameraDistortion(model_name, [float(v) for v in params]), samples, gravity


def undistort_maps(k: CameraIntrinsics, distortion: CameraDistortion, balance: float = 0.0):
    """cv2.remap maps to an ideal pinhole camera, and that camera's intrinsics (V2D _undistort.py).
    balance (alpha): 0 = crop to valid pixels only, 1 = keep the full field of view (black corners)."""
    K = k.matrix().astype(np.float64)
    size = (k.width, k.height)
    if distortion.model == "opencv_fisheye":
        D = np.asarray(distortion.params, dtype=np.float64).reshape(-1, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    elif distortion.model == "opencv":
        D = np.asarray(list(distortion.params) + [0.0] * max(0, 4 - len(distortion.params)), dtype=np.float64)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=balance, newImgSize=size)
        map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    else:
        raise ValueError(f"Cannot undistort distortion model {distortion.model!r}")
    new_k = CameraIntrinsics(fx=float(new_K[0, 0]), fy=float(new_K[1, 1]), cx=float(new_K[0, 2]),
                             cy=float(new_K[1, 2]), width=k.width, height=k.height)
    if not valid_intrinsics(new_k):
        raise ValueError(f"Undistortion with {distortion.model} {distortion.params} gives an invalid camera {new_k}. "
                         "The distortion fit is unusable for this video: try another --anycalib-cam-id "
                         "(radial:1 suits ordinary phone/webcam lenses; kb:4 is for fisheye) or drop --undistort.")
    return map1, map2, new_k


def valid_intrinsics(k: CameraIntrinsics) -> bool:
    values = np.array([k.fx, k.fy, k.cx, k.cy], dtype=np.float64)
    return bool(np.all(np.isfinite(values)) and k.fx > 0 and k.fy > 0
                and 0 <= k.cx <= k.width and 0 <= k.cy <= k.height)


def undistort_frames(frames: Path, out: Path, map1, map2) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in frame_indices(frames, ".png"):
        target = out / frame_name(i, ".png")
        if target.exists():
            continue
        image = cv2.imread(str(frames / frame_name(i, ".png")))
        cv2.imwrite(str(target), cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        n += 1
    return n


def calibrate(method: str, frames: Path, out: Path, weights: Path, *, samples: int, cam_id: str = "pinhole",
              geocalib_camera_model: str = "pinhole", device: str = "cuda", log=print) -> dict:
    """Run one calibrator on `samples` frames; write out/{intrinsics,distortion,calibration}.json (+ gravity.json)."""
    indices = sample_indices(frame_indices(frames, ".png"), samples)
    if not indices:
        raise FileNotFoundError(f"No frames in {frames}")
    if method == "anycalib":
        k, dist, records, gravity = run_anycalib(frames, indices, weights, cam_id, device, log)
        details = {"cam_id": cam_id, "weights": weights.name}
    elif method == "geocalib":
        k, dist, records, gravity = run_geocalib(frames, indices, weights, geocalib_camera_model, device, log)
        details = {"camera_model": geocalib_camera_model, "weights": weights.name}
    else:
        raise ValueError(f"Unknown calibration method {method!r}")
    if not valid_intrinsics(k) or not np.all(np.isfinite(dist.params)):
        raise RuntimeError(f"{method} returned an invalid calibration: {k}, {dist}")
    out.mkdir(parents=True, exist_ok=True)
    k.save(out / "intrinsics.json")
    dist.save(out / "distortion.json")
    if gravity is not None:
        (out / "gravity.json").write_text(json.dumps(gravity, indent=2), encoding="utf-8")
    summary = {"method": method, **details, "num_samples": len(records), "intrinsics": asdict(k),
               "fov_x_degrees": k.fov_x_degrees, "distortion": asdict(dist), "gravity": gravity, "samples": records}
    (out / "calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
