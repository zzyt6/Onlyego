"""Rotate the camera trajectory so the world's "up" is up, using GeoCalib gravity.

Port of V2D v2d_common/result_bundle.py (gravity_align_result_bundle and its helpers), which V2D applies
to the final result bundle when run with --gravity_align. The maths is unchanged:

  1. For every GeoCalib sample frame with a camera pose: GeoCalib's vector is the image *up* direction in
     OpenCV camera axes (+x right, +y down, +z forward), so physical gravity = -vector. Rotate it into the
     world with that frame's camera->world rotation; also take the camera's forward axis (+z) in the world.
  2. gravity_world = normalised mean over samples; forward_world likewise.
  3. target z_up: new world +z = -gravity, +y = mean camera forward projected onto the horizontal plane,
     +x = y x z (right-handed). Other targets (z_down, y_up, ...) only rotate gravity onto the target axis
     (shortest rotation), leaving yaw as it is.
  4. Every camera->world pose is left-multiplied by A = [R_align 0; 0 1]. Translations rotate too; scale and
     the origin (first camera centre) are unchanged.

V2D overwrites the world transforms inside result.npz. Here poses/ (the SLAM output, V2D's slam_poses/) stays
as it is and the aligned copy goes to poses_gravity/, with the transform and V2D's manifest fields in
gravity_alignment.json. When hands and objects exist, the same A applies to their world transforms.
"""

import json
from pathlib import Path

import numpy as np

from .poses import load_trajectory, save_pose

GRAVITY_TARGETS = {
    # Name means "this world axis points up"; gravity points opposite it.
    "z_up": np.array([0.0, 0.0, -1.0]),
    "z_down": np.array([0.0, 0.0, 1.0]),
    "y_up": np.array([0.0, -1.0, 0.0]),
    "y_down": np.array([0.0, 1.0, 0.0]),
    "x_up": np.array([-1.0, 0.0, 0.0]),
    "x_down": np.array([1.0, 0.0, 0.0]),
}
METHOD_Z_UP = "geocalib_opencv_up_to_down_zup_frame"
METHOD_VECTOR = "geocalib_opencv_up_to_down_vector_align"


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def proper_rotation(R: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 block onto SO(3)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1.0
        out = U @ Vt
    return out


def rotation_between_vectors(source, target) -> np.ndarray:
    """R with R @ source ~= target, for unit 3-vectors (Rodrigues; 180 deg handled separately)."""
    a, b = normalize(source), normalize(target)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        axis = normalize(np.cross(a, np.eye(3)[int(np.argmin(np.abs(a)))]))
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def geocalib_up_to_gravity(up_camera) -> np.ndarray:
    """GeoCalib's zero-roll/pitch vector is image-up [0,-1,0] in OpenCV axes; gravity is its negation.
    (Flipping only y would be right for level cameras but wrong in z for pitched ones.)"""
    return normalize(-np.asarray(up_camera, dtype=np.float64))


def geocalib_samples(calibration_path: Path, fallback_frame: int) -> list[tuple[int, np.ndarray]]:
    """(frame, up vector in camera) per GeoCalib sample; the pooled vector if no per-sample ones exist.
    V2D's fallback frame is 0 (its first frame); ours is the first frame that has a pose."""
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    samples = []
    for sample in calibration.get("samples", []):
        vec = sample.get("gravity", {}).get("vector_camera")
        if vec is not None:
            frame = sample.get("frame_index", sample.get("frame", 0))  # "frame" in runs made before 2026-11-10
            samples.append((int(frame), normalize(vec)))
    if not samples:
        vec = (calibration.get("gravity") or {}).get("vector_camera")
        if vec is None:
            raise ValueError(f"No GeoCalib gravity vector in {calibration_path}")
        samples.append((fallback_frame, normalize(vec)))
    return samples


def world_gravity_and_forward(poses: dict[int, np.ndarray], samples):
    frames, gravity, forward = [], [], []
    for frame, up_camera in samples:
        if frame not in poses:
            continue
        R = proper_rotation(poses[frame][:3, :3])
        gravity.append(normalize(R @ geocalib_up_to_gravity(up_camera)))
        forward.append(normalize(R @ np.array([0.0, 0.0, 1.0])))
        frames.append(frame)
    if not gravity:
        raise RuntimeError("No GeoCalib gravity samples overlap frames that have camera poses")
    return normalize(np.mean(gravity, axis=0)), normalize(np.mean(forward, axis=0)), frames, gravity, forward


def z_up_alignment(gravity_world, forward_world):
    z_up = normalize(-gravity_world)
    y = np.asarray(forward_world, dtype=np.float64)
    y = y - float(np.dot(y, z_up)) * z_up
    if np.linalg.norm(y) < 1e-8:  # camera looks straight up or down: any horizontal forward will do
        y = np.array([0.0, 1.0, 0.0]) - z_up[1] * z_up
    y = normalize(y)
    x = normalize(np.cross(y, z_up))
    y = normalize(np.cross(z_up, x))
    basis = np.stack([x, y, z_up], axis=1)
    return basis.T, {"x_right_axis_in_pre_alignment_world": x.tolist(),
                     "y_forward_axis_in_pre_alignment_world": y.tolist(),
                     "z_up_axis_in_pre_alignment_world": z_up.tolist(),
                     "basis_pre_alignment_world_columns": basis.tolist()}


def alignment_transform(poses: dict[int, np.ndarray], samples, target: str = "z_up") -> tuple[np.ndarray, dict]:
    """4x4 A (world -> gravity-aligned world) and the V2D manifest fields describing it."""
    if target not in GRAVITY_TARGETS:
        raise ValueError(f"Unsupported gravity target {target!r}. Choices: {sorted(GRAVITY_TARGETS)}")
    gravity_world, forward_world, frames, g_samples, f_samples = world_gravity_and_forward(poses, samples)
    target_gravity = GRAVITY_TARGETS[target]
    details = {}
    if target == "z_up":
        R_align, details = z_up_alignment(gravity_world, forward_world)
    else:
        R_align = rotation_between_vectors(gravity_world, target_gravity)
    A = np.eye(4)
    A[:3, :3] = R_align
    info = {
        "enabled": True, "target": target, "target_gravity_world": target_gravity.tolist(),
        "gravity_in_world_before_alignment": gravity_world.tolist(), "alignment_transform": A.tolist(),
        "sample_frame_indices": frames, "sample_gravity_vectors_world": [g.tolist() for g in g_samples],
        "sample_camera_forward_vectors_world": [f.tolist() for f in f_samples],
        "method": METHOD_Z_UP if target == "z_up" else METHOD_VECTOR,
        "geocalib_up_vector_negated_to_gravity": True, **details,
    }
    if target == "z_up":
        info["world_coordinate_convention"] = {"handedness": "right_handed", "x_axis": "right",
                                               "y_axis": "average_horizontal_camera_forward", "z_axis": "up",
                                               "gravity_axis": "negative_z"}
    return A, info


def gravity_align_run(camera_run: Path, depth_run: Path, target: str = "z_up") -> dict:
    """poses/ -> poses_gravity/ + gravity_alignment.json, using depth_run/calibration/geocalib/calibration.json."""
    calibration = depth_run / "calibration" / "geocalib" / "calibration.json"
    if not calibration.exists():
        raise FileNotFoundError(f"No GeoCalib result in {depth_run} (needed for the gravity direction); rerun "
                                f"depth_estimation with --run-geocalib (or --prior geocalib) into a new --out")
    indices, mats = load_trajectory(camera_run / "poses")
    if not indices:
        raise FileNotFoundError(f"No camera poses in {camera_run / 'poses'}")
    poses = dict(zip(indices, mats))
    A, info = alignment_transform(poses, geocalib_samples(calibration, indices[0]), target)
    info["source_calibration_path"] = str(calibration)
    out = camera_run / "poses_gravity"
    out.mkdir(exist_ok=True)
    for i, M in poses.items():
        save_pose(out / f"{i:06d}.json", A @ M)
    (camera_run / "gravity_alignment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
