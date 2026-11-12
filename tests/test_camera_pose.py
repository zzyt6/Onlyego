import json

import numpy as np
import pytest

from camera_pose import wsl
from camera_pose.poses import (load_pose, load_trajectory, matrix_to_quat_wxyz, quat_wxyz_to_matrix, save_pose,
                               trajectory_summary, write_tum)


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()


@pytest.mark.parametrize("axis,deg", [((0, 0, 1), 0), ((1, 0, 0), 90), ((0, 1, 0), 179.9), ((1, 1, 1), -120)])
def test_quaternion_roundtrip(axis, deg):
    R = _rot(np.array(axis) / np.linalg.norm(axis), deg)
    q = matrix_to_quat_wxyz(R)
    assert q[0] >= 0 and np.linalg.norm(q) == pytest.approx(1)
    assert np.allclose(quat_wxyz_to_matrix(q), R, atol=1e-9)


def test_pose_file_is_v2d_compatible(tmp_path):
    M = np.eye(4)
    M[:3, :3] = _rot((0, 1, 0), 30)
    M[:3, 3] = [0.1, -0.2, 0.3]
    save_pose(tmp_path / "000000.json", M)
    d = json.loads((tmp_path / "000000.json").read_text())
    assert set(d) >= {"rotation", "translation", "scale"}  # V2D Transform3d keys
    del d["matrix"]  # a V2D-written file has no matrix; must still load
    (tmp_path / "000001.json").write_text(json.dumps(d))
    assert np.allclose(load_pose(tmp_path / "000001.json"), M)


def test_trajectory_summary_and_tum(tmp_path):
    for i in range(5):  # camera slides 10 cm along x per frame and turns 5 deg about y
        M = np.eye(4)
        M[:3, :3] = _rot((0, 1, 0), 5 * i)
        M[:3, 3] = [0.1 * i, 0, 0]
        save_pose(tmp_path / f"{i * 2:06d}.json", M)
    indices, poses = load_trajectory(tmp_path)
    assert indices == [0, 2, 4, 6, 8]
    s = trajectory_summary(indices, poses, fps=10)
    assert s["path_length_m"] == pytest.approx(0.4) and s["net_displacement_m"] == pytest.approx(0.4)
    assert s["max_rotation_from_first_deg"] == pytest.approx(20)
    assert s["mean_speed_m_per_s"] == pytest.approx(0.5)  # 0.4 m over frames 0..8 at 10 fps
    write_tum(tmp_path / "t.txt", indices, poses)
    second = (tmp_path / "t.txt").read_text().splitlines()[1].split()
    assert second[0] == "2" and float(second[1]) == pytest.approx(0.1)  # frame number first, as V2D


def test_windows_paths_map_into_wsl():
    assert wsl.to_wsl(r"E:\ZY_ego\runs\x") == "/mnt/e/ZY_ego/runs/x"
    cmd = wsl.command("camera_pose.slam", ["--out", "/mnt/e/a b/c"])
    assert cmd[:4] == ["wsl.exe", "-d", wsl.DISTRO, "--"]
    assert "'/mnt/e/a b/c'" in cmd[-1] and "conda activate" in cmd[-1]


def test_report_draws_trajectory(tmp_path):
    from camera_pose.report import write_report
    depth_run, out = tmp_path / "depth", tmp_path / "pose"
    depth_run.mkdir()
    (out / "poses").mkdir(parents=True)
    (depth_run / "run.json").write_text(json.dumps({"video": {"fps": 30.0}, "settings": {"stride": 1}}))
    for i in range(10):
        M = np.eye(4)
        M[:3, 3] = [0.01 * i, 0, 0.02 * i]
        save_pose(out / "poses" / f"{i:06d}.json", M)
    (out / "slam.json").write_text(json.dumps({"scale": 0.5, "scale_per_keyframe": [{"scale": 0.4}, {"scale": 0.6}]}))
    summary = write_report(depth_run, out)
    assert summary["slam"]["scale_spread"]["median"] == pytest.approx(0.5)
    assert (out / "trajectory.png").stat().st_size > 0 and (out / "trajectory_tum.txt").exists()


# --- gravity alignment (port of V2D result_bundle.gravity_align_result_bundle) -------------------------------

def _camera_in_true_world(yaw_deg, pitch_down_deg):
    """OpenCV camera (x right, y down, z forward) in a z-up world: level camera looks along +y, then
    pitches down about its own x axis, then yaws about world z."""
    level = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # columns: camera x, y, z in the world
    return _rot((0, 0, 1), yaw_deg) @ level @ _rot((1, 0, 0), -pitch_down_deg)


def _slam_and_geocalib(frames):
    """SLAM world = first camera's frame. GeoCalib's vector = image-up in camera axes = R^T @ world up."""
    true = {i: _camera_in_true_world(*yp) for i, yp in frames.items()}
    first = true[min(true)]
    poses = {}
    for i, R in true.items():
        M = np.eye(4)
        M[:3, :3] = first.T @ R
        M[:3, 3] = first.T @ np.array([0.05 * i, 0.0, 0.0])
        poses[i] = M
    samples = [(i, R.T @ np.array([0.0, 0.0, 1.0])) for i, R in true.items()]
    return poses, samples


def test_gravity_z_up_recovers_true_up_and_level_forward():
    from camera_pose.gravity import alignment_transform
    # looking down at a table; the +-10 deg yaws share a pitch so their horizontal forwards cancel exactly
    poses, samples = _slam_and_geocalib({0: (0, 40), 3: (10, 30), 6: (-10, 30)})
    A, info = alignment_transform(poses, samples, "z_up")
    assert np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-9) and np.linalg.det(A[:3, :3]) > 0
    for i, M in poses.items():
        R = (A @ M)[:3, :3]
        down_in_camera = samples[[s[0] for s in samples].index(i)][1] * -1
        assert np.allclose(R @ down_in_camera, [0, 0, -1], atol=1e-9)  # gravity is -z in every frame
    # +y = mean forward made horizontal: the yaw of the first and the last cancel, so +y is frame 0's heading
    fwd0 = (A @ poses[0])[:3, 2]
    assert fwd0[0] == pytest.approx(0, abs=1e-9) and fwd0[1] > 0 and fwd0[2] < 0
    assert np.degrees(np.arcsin(-fwd0[2])) == pytest.approx(40)
    assert info["method"] == "geocalib_opencv_up_to_down_zup_frame"
    assert info["world_coordinate_convention"]["z_axis"] == "up"


def test_gravity_vector_targets_and_antiparallel():
    from camera_pose.gravity import GRAVITY_TARGETS, alignment_transform, rotation_between_vectors
    poses, samples = _slam_and_geocalib({0: (0, 20), 1: (5, 25)})
    for target in ("y_down", "x_up", "z_down"):
        A, info = alignment_transform(poses, samples, target)
        g = np.array(info["gravity_in_world_before_alignment"])
        assert np.allclose(A[:3, :3] @ g, GRAVITY_TARGETS[target], atol=1e-9)
        assert info["method"] == "geocalib_opencv_up_to_down_vector_align"
    R = rotation_between_vectors(np.array([0, 1.0, 0]), np.array([0, -1.0, 0]))  # 180 deg branch
    assert np.allclose(R @ [0, 1, 0], [0, -1, 0]) and np.linalg.det(R) == pytest.approx(1)


def test_level_camera_needs_no_rotation_for_y_down():
    """An untilted first camera already has gravity along +y (image down): y_down is the identity."""
    from camera_pose.gravity import alignment_transform
    poses, samples = _slam_and_geocalib({0: (0, 0), 1: (0, 0)})
    A, _ = alignment_transform(poses, samples, "y_down")
    assert np.allclose(A, np.eye(4), atol=1e-9)


def test_gravity_align_run_files(tmp_path):
    from camera_pose.gravity import gravity_align_run
    poses, samples = _slam_and_geocalib({0: (0, 30), 1: (2, 30), 2: (4, 30)})
    run, depth = tmp_path / "pose", tmp_path / "depth"
    (run / "poses").mkdir(parents=True)
    for i, M in poses.items():
        save_pose(run / "poses" / f"{i:06d}.json", M)
    calib = depth / "calibration" / "geocalib"
    with pytest.raises(FileNotFoundError, match="--run-geocalib"):
        gravity_align_run(run, depth)
    calib.mkdir(parents=True)
    # older runs wrote "frame" instead of V2D's "frame_index"; both are read
    (calib / "calibration.json").write_text(json.dumps({"samples": [
        {"frame_index" if i else "frame": i, "gravity": {"vector_camera": v.tolist()}} for i, v in samples]}))
    info = gravity_align_run(run, depth)
    assert info["sample_frame_indices"] == [0, 1, 2]
    indices, aligned = load_trajectory(run / "poses_gravity")
    assert indices == [0, 1, 2]
    A = np.array(json.loads((run / "gravity_alignment.json").read_text())["alignment_transform"])
    assert np.allclose(aligned[2], A @ poses[2])
    assert np.allclose(load_trajectory(run / "poses")[1][2], poses[2])  # SLAM output untouched
    # distances are preserved (pure rotation about the world origin)
    assert np.linalg.norm(aligned[2][:3, 3]) == pytest.approx(np.linalg.norm(poses[2][:3, 3]))
