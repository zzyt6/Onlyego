import numpy as np
import pytest

from video_depth import wsl
from video_depth import cli
from video_depth.megasam import align_mono_to_metric, tracking_size


def test_tracking_size_matches_megasam():
    # 1920x1080 -> ~384*512 pixels keeping aspect, cropped to a multiple of 8
    assert tracking_size(1080, 1920) == (332, 591, 328, 584)
    assert tracking_size(480, 640) == (384, 512, 384, 512)  # 4:3 lands exactly on 384x512


def test_moge_in_both_roles_gives_identity_alignment():
    rng = np.random.default_rng(0)
    depths = [rng.uniform(0.4, 3.0, (40, 60)).astype(np.float32) * s for s in (1.0, 1.05, 0.97)]
    depths[0][0, :5] = np.inf  # invalid MoGe pixels: disparity 0
    disps = [np.where(np.isfinite(d), 1.0 / d, 0.0).astype(np.float32) for d in depths]
    scale, shift, norm = align_mono_to_metric(disps, depths)
    assert scale == pytest.approx(1.0, abs=1e-4) and abs(shift) < 1e-6
    # normaliser puts the 98th-percentile disparity at 2
    assert norm == pytest.approx(np.percentile(np.array(disps), 98) / 2)


def test_wsl_command_uses_megasam_env_and_paths():
    cmd = wsl.command("video_depth.megasam", ["--out", "/mnt/e/x"], env=cli.ENV, pythonpath=cli.PYTHONPATH)
    script = cmd[-1]
    assert "conda activate zyego-megasam" in script
    assert '"$HOME/zyego-external/mega-sam/base/droid_slam"' in script and "DROID-SLAM" not in script


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()


def test_plan_segments():
    from video_depth.stitch import plan_segments
    assert plan_segments(184, 200, 30) == [(0, 184)]
    segs = plan_segments(500, 200, 30)
    assert segs[0][0] == 0 and segs[-1][1] == 500 and all(b - a == 200 for a, b in segs)
    assert all(segs[k][1] - segs[k + 1][0] >= 30 for k in range(len(segs) - 1))  # overlaps at least 30


def test_chain_recovers_world_frame_and_scale():
    """Two segments of one camera path; the second has its own world (its first camera) and a 5% scale error."""
    from video_depth.stitch import chain
    n = 60
    truth = []
    for t in range(n):  # camera moves along x and turns slowly
        M = np.eye(4)
        M[:3, :3] = _rot((0, 1, 0), 0.5 * t)
        M[:3, 3] = [0.01 * t, 0.002 * t, 0.0]
        truth.append(M)
    truth = np.stack(truth)
    depth = np.stack([np.full((8, 10), 1.0 + 0.01 * t, np.float32) for t in range(n)])
    mp = np.ones((n, 2, 2), np.float32)
    a, b = list(range(0, 40)), list(range(20, 60))
    seg1 = {"frames": a, "c2w": truth[a], "depth": depth[a], "motion_prob": mp[a]}
    base = np.linalg.inv(truth[20])  # second segment's world = its first camera
    c2w2 = np.stack([base @ truth[t] for t in b])
    c2w2[:, :3, 3] *= 1.05
    seg2 = {"frames": b, "c2w": c2w2, "depth": depth[b] * 1.05, "motion_prob": mp[b]}
    m = chain([seg1, seg2])
    assert m["frames"] == list(range(60))
    assert m["joins"][0]["depth_scale"] == pytest.approx(1 / 1.05, rel=1e-4)
    assert np.allclose(m["c2w"], truth, atol=1e-6)
    assert np.allclose(m["depth"], depth, rtol=1e-5)


def test_fullres_keeps_moge_detail_and_takes_megasam_scale():
    from video_depth.fullres import lift_frame
    H, W = 108, 192
    h1, w1 = 33, 59
    hc, wc = 32, 56
    yy, xx = np.mgrid[0:H, 0:W]
    moge = (1.0 + 0.3 * (xx > W // 2)).astype(np.float32)  # a sharp depth step
    moge[0, 0] = np.inf
    megasam = np.full((hc, wc), 1.0, np.float32) * 0.9  # MegaSaM: same scene, 10% closer
    megasam[:, wc // 2 + 1:] = 1.3 * 0.9
    full = lift_frame(moge, megasam, (h1, w1))
    assert np.isinf(full[0, 0])
    assert np.median(full[5:-5, 5:W // 2 - 10] / moge[5:-5, 5:W // 2 - 10]) == pytest.approx(0.9, abs=0.01)
    # the step stays sharp: one column apart, the full-resolution depth jumps by ~0.3*0.9
    row = full[H // 2]
    assert row[W // 2 + 1] - row[W // 2] == pytest.approx(0.27, abs=0.03)
