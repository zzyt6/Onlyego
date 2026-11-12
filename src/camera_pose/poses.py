"""Pose files and trajectory maths (pure numpy; used on both the Windows and the WSL side).

Pose JSON per frame, camera-to-world, compatible with V2D's Transform3d:
  {"rotation": [w, x, y, z], "translation": [x, y, z], "scale": [1, 1, 1], "matrix": 4x4}
Camera axes follow OpenCV / DROID-SLAM: x right, y down, z forward. The world frame is the first
frame's camera frame (DROID fixes the first keyframe at the identity).
"""

import json
from pathlib import Path

import numpy as np


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Shepperd's method; returns a unit quaternion with w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def save_pose(path: Path, M: np.ndarray):
    Path(path).write_text(json.dumps({
        "rotation": matrix_to_quat_wxyz(M[:3, :3]), "translation": M[:3, 3].tolist(),
        "scale": [1.0, 1.0, 1.0], "matrix": np.asarray(M).tolist()}, indent=2), encoding="utf-8")


def load_pose(path: Path) -> np.ndarray:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "matrix" in d:
        return np.array(d["matrix"], dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_matrix(d["rotation"]) @ np.diag(d.get("scale", [1, 1, 1]))
    M[:3, 3] = d["translation"]
    return M


def load_trajectory(folder: Path) -> tuple[list[int], np.ndarray]:
    files = sorted(p for p in Path(folder).glob("*.json") if p.stem.isdigit())
    return [int(p.stem) for p in files], np.stack([load_pose(p) for p in files]) if files else np.zeros((0, 4, 4))


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def trajectory_summary(indices: list[int], poses: np.ndarray, fps: float | None = None) -> dict:
    """How much the camera moved: path length, net displacement, extent, rotation from the first frame."""
    if len(poses) == 0:
        return {"frames": 0}
    centers = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    R0 = poses[0, :3, :3]
    angles = [rotation_angle_deg(R0.T @ P[:3, :3]) for P in poses]
    summary = {
        "frames": len(poses), "first_frame": indices[0], "last_frame": indices[-1],
        "path_length_m": round(float(steps.sum()), 4),
        "net_displacement_m": round(float(np.linalg.norm(centers[-1] - centers[0])), 4),
        "extent_m": [round(float(v), 4) for v in centers.max(0) - centers.min(0)],
        "max_rotation_from_first_deg": round(max(angles), 2),
        "final_rotation_from_first_deg": round(angles[-1], 2),
    }
    if fps and len(indices) > 1:
        seconds = (indices[-1] - indices[0]) / fps
        summary["mean_speed_m_per_s"] = round(float(steps.sum() / max(seconds, 1e-9)), 4)
    return summary


def write_tum(path: Path, indices: list[int], poses: np.ndarray):
    """TUM-style lines "frame tx ty tz qx qy qz qw", as V2D writes slam_trajectory.txt:
    the first column is the source-video frame number, not seconds."""
    with open(path, "w", encoding="utf-8") as f:
        for i, M in zip(indices, poses):
            w, x, y, z = matrix_to_quat_wxyz(M[:3, :3])
            tx, ty, tz = (float(v) for v in M[:3, 3])
            f.write(f"{i} {tx} {ty} {tz} {x} {y} {z} {w}\n")
