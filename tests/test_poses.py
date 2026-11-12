import json

import numpy as np
import pytest

from video_depth import wsl
from video_depth.poses import (load_pose, load_trajectory, matrix_to_quat_wxyz, quat_wxyz_to_matrix, save_pose,
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
    cmd = wsl.command("video_depth.megasam", ["--out", "/mnt/e/a b/c"])
    assert cmd[:4] == ["wsl.exe", "-d", wsl.DISTRO, "--"]
    assert "'/mnt/e/a b/c'" in cmd[-1] and "conda activate zyego-megasam" in cmd[-1]
