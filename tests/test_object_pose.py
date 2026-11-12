import json

import numpy as np
from scipy.spatial.transform import Rotation

from object_pose.evaluate import load_poses, rot_deg, summarize


def test_load_poses_reads_transform3d(tmp_path):
    R = Rotation.from_euler("xyz", [10, -20, 30], degrees=True)
    x, y, z, w = R.as_quat()
    (tmp_path / "000007.json").write_text(json.dumps(
        {"rotation": [w, x, y, z], "translation": [0.1, -0.2, 0.6], "scale": [1, 1, 1]}))
    T = load_poses(tmp_path)[7]
    assert np.allclose(T[:3, :3], R.as_matrix())
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.6])


def test_rot_deg():
    assert abs(rot_deg(Rotation.from_euler("z", 25, degrees=True).as_matrix()) - 25) < 1e-6
    assert rot_deg(np.eye(3)) == 0


def test_summarize_splits_at_contact():
    rows = {i: {"iou_full": 0.9, "iou_visible": 0.95 if i < 3 else 0.8, "depth_mm": 1.0, "accel_mm": 0.5,
                "static_drift_mm": 2.0, "static_drift_deg": 0.1} for i in range(6)}
    s = summarize(rows, contact=3)
    assert s["static (before contact)"]["frames"] == 3
    assert s["static (before contact)"]["iou_visible"]["median"] == 0.95
    assert s["held (from contact)"]["iou_visible_below_0.9"] == 3
    assert "drift_mm" not in s["held (from contact)"]
