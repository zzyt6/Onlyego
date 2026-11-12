import json

import cv2
import numpy as np
import pytest

from depth_estimation import pipeline
from depth_estimation.align import align_depth_sequence, smooth_scales
from depth_estimation.datatypes import CameraIntrinsics, decode_depth, encode_depth, load_depth, save_depth
from depth_estimation.inspection import unproject
from depth_estimation.intrinsics import stabilize_intrinsics


def test_depth_png_roundtrip_and_invalid(tmp_path):
    depth = np.array([[0.0, 0.5, 1.0], [3.0, 10.0, np.inf]], dtype=np.float32)
    save_depth(tmp_path / "d.png", depth)
    back = load_depth(tmp_path / "d.png")
    assert np.isinf(back[1, 2])
    finite = np.isfinite(depth)
    # one uint16 step of inverse depth is (d+1)^2/65535 m
    assert np.all(np.abs(back[finite] - depth[finite]) <= (depth[finite] + 1) ** 2 / 65535 + 1e-6)
    assert encode_depth(np.array([np.nan]))[0] == 0
    assert decode_depth(np.array([65535], dtype=np.uint16))[0] == 0
    # truncates like V2D (65535/2 = 32767.5 -> 32767; rounding would give 32768)
    assert encode_depth(np.array([1.0], dtype=np.float32))[0] == 32767


def test_stabilize_intrinsics_is_median(tmp_path):
    folder = tmp_path / "intrinsics"
    folder.mkdir()
    for i, fx in enumerate([500, 510, 900, 505, 495]):  # 900 is an outlier frame
        CameraIntrinsics(fx, fx, 320 + i, 240, 640, 480).save(folder / f"{i:06d}.json")
    stable = stabilize_intrinsics(folder, tmp_path / "stable.json")
    assert stable.fx == 505 and stable.cx == 322
    fixed = stabilize_intrinsics(folder, tmp_path / "fixed.json", fix_principal_point=True)
    assert (fixed.cx, fixed.cy) == (320, 240)


def test_smooth_scales_interpolates_and_rejects_outlier():
    raw = {0: 1.0, 1: 1.0, 2: 5.0, 3: 1.0, 5: 1.0}  # frame 2 outlier, frame 4 missing
    out = smooth_scales(raw, [0, 1, 2, 3, 4, 5], window=3)
    assert out[2] == pytest.approx(1.0) and out[4] == pytest.approx(1.0)


def _textured(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    image = cv2.resize(rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_align_recovers_per_frame_scale(tmp_path):
    frames, depth = tmp_path / "frames", tmp_path / "depth"
    frames.mkdir()
    depth.mkdir()
    gray = _textured()
    truth = np.linspace(1.0, 3.0, gray.shape[1], dtype=np.float32)[None].repeat(gray.shape[0], 0)
    wrong = {0: 1.0, 1: 1.25, 2: 0.8}  # each frame's depth came out with a different global scale
    for i, s in wrong.items():
        cv2.imwrite(str(frames / f"{i:06d}.png"), gray)
        save_depth(depth / f"{i:06d}.png", truth * s)
    report = align_depth_sequence(depth, frames, tmp_path / "aligned", window=1)
    assert report["estimated"] == 3
    for i, s in wrong.items():
        aligned = load_depth(tmp_path / "aligned" / f"{i:06d}.png")
        assert np.median(aligned / truth) == pytest.approx(1.0, abs=0.01)


def test_unproject_pinhole():
    k = CameraIntrinsics(fx=100, fy=100, cx=1, cy=1, width=3, height=3)
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    depth[0, 0] = np.inf
    points, _ = unproject(depth, k)
    assert len(points) == 8
    assert points.tolist()[-1] == pytest.approx([0.02, 0.02, 2.0])  # pixel (2,2): (2-1)*2/100


class FakeMoGe:
    """Stands in for MoGeModel.infer: constant 2 m depth, fx = 0.8 W."""

    def infer(self, tensor, fov_x=None):
        import torch
        b, _, h, w = tensor.shape
        k = torch.tensor([[0.8, 0, 0.5], [0, 0.8 * w / h, 0.5], [0, 0, 1]]).repeat(b, 1, 1)
        return {"depth": torch.full((b, h, w), 2.0), "intrinsics": k, "mask": torch.ones(b, h, w, dtype=torch.bool)}


def test_pipeline_end_to_end_with_fake_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(7):
        writer.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    writer.release()
    import depth_estimation.moge_depth as moge_depth
    monkeypatch.setattr(moge_depth, "load_model", lambda weights, device: FakeMoGe())
    out = tmp_path / "run"
    record = pipeline.run_depth(video, out, tmp_path / "weights", batch_size=3, device="cpu", log=lambda m: None)
    assert len(list((out / "depth").glob("*.png"))) == 7  # last partial batch is not dropped
    assert record["intrinsics_stable"]["fx"] == pytest.approx(51.2)
    stats = json.loads((out / "depth_stats.json").read_text())
    assert stats[0]["depth_median"] == pytest.approx(2.0, abs=1e-3)
    assert (out / "preview.mp4").stat().st_size > 0
    with pytest.raises(ValueError, match="different settings"):
        pipeline.run_depth(video, out, tmp_path / "weights", stride=2, device="cpu", log=lambda m: None)


def test_sample_indices_spread_and_small():
    from depth_estimation.calibration import sample_indices
    assert sample_indices(list(range(100)), 5) == [0, 24, 49, 74, 99]  # floors, as V2D's linspace(dtype=int)
    assert sample_indices([3, 7], 8) == [3, 7]


def test_undistort_maps_zero_distortion_is_identity():
    from depth_estimation.calibration import CameraDistortion, undistort_maps
    k = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    # Brown-Conrady with zero coefficients is already a pinhole camera: nothing changes
    map1, _, new_k = undistort_maps(k, CameraDistortion("opencv", [0.0]), balance=0.0)
    assert new_k.fx == pytest.approx(500, rel=0.01)
    assert map1.shape[:2] == (480, 640)
    assert tuple(map1[240, 320]) == pytest.approx((320, 240), abs=1)
    # Kannala-Brandt with zero coefficients is an equidistant fisheye (r = f*theta), not a pinhole
    # (r = f*tan theta); the pinhole that keeps only valid pixels has a shorter focal length
    map1, _, new_k = undistort_maps(k, CameraDistortion("opencv_fisheye", [0.0] * 4), balance=0.0)
    assert new_k.fx < 500
    assert tuple(map1[240, 320]) == pytest.approx((320, 240), abs=1)
    with pytest.raises(ValueError):
        undistort_maps(k, CameraDistortion("pinhole"))


def _write_video(path, n=5, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), 40 * i, dtype=np.uint8))
    writer.release()


class RecordingMoGe(FakeMoGe):
    fov_seen = []

    def infer(self, tensor, fov_x=None):
        RecordingMoGe.fov_seen.append(fov_x)
        return super().infer(tensor, fov_x)


FOCAL = {"anycalib": 32.0, "geocalib": 64 / (2 * np.tan(np.radians(30)))}  # fov_x 90 deg / 60 deg on 64 px


def _fake_calibration(monkeypatch, calls, anycalib_dist=("opencv_fisheye", [0.01, 0, 0, 0])):
    import depth_estimation.calibration as calib
    import depth_estimation.moge_depth as moge_depth

    def fake_calibrate(method, frames, out, weights, **kw):
        calls.append({"method": method, "weights": weights.name, "cam_id": kw["cam_id"], "frames": frames.name,
                      "camera_model": kw["geocalib_camera_model"]})
        out.mkdir(parents=True, exist_ok=True)
        f = FOCAL[method]
        CameraIntrinsics(f, f, 32, 24, 64, 48).save(out / "intrinsics.json")
        dist = anycalib_dist if method == "anycalib" else ("pinhole", [])
        calib.CameraDistortion(*dist).save(out / "distortion.json")
        (out / "calibration.json").write_text(json.dumps(
            {"method": method, "distortion": {"model": dist[0], "params": dist[1]}, "samples": []}))

    monkeypatch.setattr(calib, "calibrate", fake_calibrate)
    monkeypatch.setattr(moge_depth, "load_model", lambda weights, device: RecordingMoGe())
    RecordingMoGe.fov_seen = []


@pytest.mark.parametrize("kwargs,expect_calls,expect_images,expect_fov", [
    # V2D --use_geocalib_intrinsics
    ({"prior": "geocalib"}, [("geocalib", "geocalib-pinhole.tar", "frames")], "frames", 60.0),
    # V2D --undistort (our --prior anycalib --undistort is the same thing)
    ({"undistort": True}, [("anycalib", "anycalib_gen.pt", "frames")], "frames_undistorted", None),
    ({"prior": "anycalib", "undistort": True}, [("anycalib", "anycalib_gen.pt", "frames")], "frames_undistorted",
     None),
    # ours: AnyCalib focal without undistortion -> pinhole model
    ({"prior": "anycalib"}, [("anycalib", "anycalib_pinhole.pt", "frames")], "frames", 90.0),
    # V2D --undistort --use_geocalib_intrinsics: GeoCalib runs on the undistorted frames, its focal wins
    ({"prior": "geocalib", "undistort": True},
     [("anycalib", "anycalib_gen.pt", "frames"), ("geocalib", "geocalib-pinhole.tar", "frames_undistorted")],
     "frames_undistorted", 60.0),
    # V2D --undistort --run_geocalib: GeoCalib only for gravity, MoGe keeps the undistorted camera
    ({"undistort": True, "run_geocalib": True},
     [("anycalib", "anycalib_gen.pt", "frames"), ("geocalib", "geocalib-pinhole.tar", "frames_undistorted")],
     "frames_undistorted", None),
    # V2D --run_geocalib alone: gravity only, MoGe guesses the focal per frame
    ({"run_geocalib": True}, [("geocalib", "geocalib-pinhole.tar", "frames")], "frames", "none"),
])
def test_pipeline_calibration_wiring_matches_v2d(tmp_path, monkeypatch, kwargs, expect_calls, expect_images,
                                                 expect_fov):
    pytest.importorskip("torch")
    calls = []
    _fake_calibration(monkeypatch, calls)
    video = tmp_path / "clip.mp4"
    _write_video(video)
    out = tmp_path / "run"
    record = pipeline.run_depth(video, out, tmp_path / "model" / "moge", device="cpu", preview=False,
                                log=lambda m: None, **kwargs)
    assert [(c["method"], c["weights"], c["frames"]) for c in calls] == expect_calls
    assert record["image_dir"] == expect_images
    fov = RecordingMoGe.fov_seen[0]
    if expect_fov == "none":
        assert fov is None
    elif expect_fov is None:  # undistorted camera: whatever cv2 made of the fit, but a prior is given
        new_k = CameraIntrinsics.load(out / "calibration" / "anycalib" / "undistorted_intrinsics.json")
        assert fov == pytest.approx(new_k.fov_x_degrees)
    else:
        assert fov == pytest.approx(expect_fov)
    if kwargs.get("undistort"):
        assert len(list((out / "frames_undistorted").glob("*.png"))) == 5
        assert next(c for c in calls if c["method"] == "anycalib")["cam_id"] == "radial:1"


def test_old_geocalib_run_still_resumes(tmp_path, monkeypatch):
    """A --prior geocalib run recorded before run_geocalib / geocalib_camera_model existed is not 'different'."""
    pytest.importorskip("torch")
    _fake_calibration(monkeypatch, [])
    video = tmp_path / "clip.mp4"
    _write_video(video)
    out = tmp_path / "run"
    pipeline.run_depth(video, out, tmp_path / "w", prior="geocalib", device="cpu", preview=False, log=lambda m: None)
    record = json.loads((out / "run.json").read_text())
    for key in ("run_geocalib", "geocalib_camera_model"):
        del record["settings"][key]
    (out / "run.json").write_text(json.dumps(record))
    pipeline.run_depth(video, out, tmp_path / "w", prior="geocalib", device="cpu", preview=False, log=lambda m: None)


def test_prior_option_conflicts(tmp_path):
    with pytest.raises(ValueError, match="either --fov-from or --prior"):
        pipeline.run_depth(tmp_path / "v.mp4", tmp_path / "o", tmp_path / "w", prior="anycalib",
                           fov_from=tmp_path / "k.json")
    with pytest.raises(ValueError, match="needs --geocalib-weights distorted"):
        pipeline.run_depth(tmp_path / "v.mp4", tmp_path / "o", tmp_path / "w", prior="geocalib",
                           geocalib_camera_model="simple_radial")


def test_unusable_distortion_fit_is_rejected():
    from depth_estimation.calibration import CameraDistortion, undistort_maps
    k = CameraIntrinsics(2227, 2227, 960, 540, 1920, 1080)
    # the kb:4 coefficients AnyCalib returned for demo01 (a non-fisheye phone video)
    bad = CameraDistortion("opencv_fisheye", [-17.63, 147.56, -424.09, 336.49])
    with pytest.raises(ValueError, match="unusable"):
        undistort_maps(k, bad)


@pytest.mark.parametrize("argv,expected", [([], "anycalib"), (["--prior", "none"], "none"),
                                           (["--fov-from", "k.json"], "none"), (["--prior", "geocalib"], "geocalib")])
def test_cli_default_prior_is_anycalib_pinhole(tmp_path, monkeypatch, argv, expected):
    from depth_estimation import cli
    seen = {}
    monkeypatch.setattr(pipeline, "run_depth", lambda *a, **kw: seen.update(kw))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    cli.main(["run", str(video), "--out", str(tmp_path / "o")] + argv)
    assert seen["prior"] == expected and seen["undistort"] is False


def test_default_calibration_sample_counts(tmp_path, monkeypatch):
    """AnyCalib calibrates on 64 frames by default (V2D: 16), GeoCalib on 8 (as V2D); a short video uses all."""
    pytest.importorskip("torch")
    calls = []
    _fake_calibration(monkeypatch, calls)
    import depth_estimation.calibration as calib
    fake = calib.calibrate
    samples = []
    monkeypatch.setattr(calib, "calibrate", lambda method, *a, **kw: (samples.append((method, kw["samples"])),
                                                                       fake(method, *a, **kw))[1])
    video = tmp_path / "clip.mp4"
    _write_video(video)
    pipeline.run_depth(video, tmp_path / "run", tmp_path / "w", prior="anycalib", run_geocalib=True,
                       device="cpu", preview=False, log=lambda m: None)
    assert samples == [("anycalib", 64), ("geocalib", 8)]
    assert calib.sample_indices(list(range(5)), 64) == [0, 1, 2, 3, 4]


def test_backproject_uses_pixel_centres():
    from depth_estimation.viewer import backproject
    k = CameraIntrinsics(fx=100, fy=100, cx=2, cy=1, width=4, height=2)  # MoGe: centre = (W/2, H/2)
    depth = np.full((2, 4), 2.0, dtype=np.float32)
    depth[0, 0] = np.inf
    rgb = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    pts, cols = backproject(depth, rgb, k)
    assert len(pts) == 7
    # pixel (u=1, v=0): centre (1.5, 0.5) -> x = (1.5-2)*2/100, y = (0.5-1)*2/100
    assert pts[0] == pytest.approx([-0.01, -0.01, 2.0]) and (cols[0] == rgb[0, 1]).all()
    mask = np.zeros((2, 4), bool)
    mask[1, 3] = True
    assert len(backproject(depth, rgb, k, mask=mask)[0]) == 1


def _fake_run(root, n=4):
    for d in ("depth", "intrinsics", "frames", "mask"):
        (root / d).mkdir(parents=True)
    for i in range(n):
        save_depth(root / "depth" / f"{i:06d}.png", np.full((48, 64), 1.0 + 0.1 * i, np.float32))
        CameraIntrinsics(60, 60, 32, 24, 64, 48).save(root / "intrinsics" / f"{i:06d}.json")
        cv2.imwrite(str(root / "frames" / f"{i:06d}.png"), np.full((48, 64, 3), 40 * i, np.uint8))
        cv2.imwrite(str(root / "mask" / f"{i:06d}.png"), np.full((48, 64), 255, np.uint8))
    (root / "run.json").write_text(json.dumps({"image_dir": "frames", "video": {"fps": 30}, "settings": {"stride": 1}}))


@pytest.mark.parametrize("with_poses", [False, True])
def test_viewer_starts(tmp_path, with_poses):
    pytest.importorskip("viser")
    import socket
    from depth_estimation.viewer import RunData, serve
    run = tmp_path / "run"
    _fake_run(run)
    poses = None
    if with_poses:
        from camera_pose.poses import save_pose
        poses = tmp_path / "pose" / "poses"
        poses.mkdir(parents=True)
        for i in range(4):
            M = np.eye(4)
            M[0, 3] = 0.1 * i
            save_pose(poses / f"{i:06d}.json", M)
    data = RunData(run, poses)
    assert list(data.sources) == ["MoGe-2 逐帧（V2D 的深度）"]
    pts, _ = data.cloud(2, data.default_source, 4, True)
    assert pts[:, 2] == pytest.approx(1.2, abs=1e-3) and len(pts) == 12 * 16
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server, ui = serve(run, poses, port=port, block=False)
    try:
        assert len(ui.handles) == 2  # one point cloud + one frustum
        ui.overlay.value = 3  # setting a control from Python redraws (the Play loop relies on this)
        assert len(ui.handles) == 6
        ui.colour.value = "按帧"
        ui.frame.value = 3
        assert "第 **3** 帧" in ui.info.content and "第 **1** 帧" in ui.info.content  # wraps around
        assert "video_depth compare" in ui.stats.content  # no compare.json yet
    finally:
        server.stop()


def test_align_metric_scale_median_ignores_unlucky_reference(tmp_path):
    """Frame 0 (the reference) came out 1.25x too far; the other frames are right. 'median' keeps the true
    scale, V2D's 'reference' would stretch every frame to frame 0's error."""
    frames, depth = tmp_path / "frames", tmp_path / "depth"
    frames.mkdir()
    depth.mkdir()
    gray = _textured()
    truth = np.linspace(1.0, 3.0, gray.shape[1], dtype=np.float32)[None].repeat(gray.shape[0], 0)
    wrong = {0: 1.25, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    for i, s in wrong.items():
        cv2.imwrite(str(frames / f"{i:06d}.png"), gray)
        save_depth(depth / f"{i:06d}.png", truth * s)
    for mode, expected in (("median", 1.0), ("reference", 1.25)):
        align_depth_sequence(depth, frames, tmp_path / mode, window=1, metric_scale=mode)
        for i in wrong:
            ratio = np.median(load_depth(tmp_path / mode / f"{i:06d}.png") / truth)
            assert ratio == pytest.approx(expected, abs=0.01), (mode, i)


def test_pipeline_scale_align_writes_aligned_depth(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import depth_estimation.moge_depth as moge_depth
    monkeypatch.setattr(moge_depth, "load_model", lambda weights, device: FakeMoGe())
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        writer.write(cv2.cvtColor(_textured(48, 64), cv2.COLOR_GRAY2BGR))
    writer.release()
    out = tmp_path / "run"
    record = pipeline.run_depth(video, out, tmp_path / "w", prior="none", device="cpu", preview=False,
                                scale_align=True, log=lambda m: None)
    assert len(list((out / "depth_aligned").glob("*.png"))) == 5
    assert (out / "depth_aligned_scales.json").exists() and (out / "depth_aligned_stats.json").exists()
    assert record["scale_align"]["frames"] == 5


def test_viewer_offers_both_runs_and_their_stability(tmp_path):
    """A video_depth run and its depth_estimation run are found from either side; compare.json fills the table."""
    pytest.importorskip("viser")
    import socket
    from depth_estimation.viewer import RunData, serve
    depth_run, vd_run = tmp_path / "depth_estimation" / "demo", tmp_path / "video_depth" / "demo"
    _fake_run(depth_run)
    _fake_run(vd_run)
    (vd_run / "megasam.json").write_text("{}")
    (vd_run / "run.json").write_text(json.dumps({"image_dir": str(depth_run / "frames"), "depth_run": str(depth_run),
                                                 "video": {"fps": 30}, "settings": {"stride": 1}}))
    row = {"scale_jitter_pct": 4.93, "pixel_wobble_pct": 4.97, "pixel_wobble_mm": 35.0, "frame_to_frame_pct": 1.24,
           "per_frame_median_m": [0.7, 0.6, 0.8, 0.7]}
    (vd_run / "compare.json").write_text(json.dumps({
        "sources": {"MoGe-2 per frame (V2D)": str(depth_run / "depth"), "MoGe-2 + MegaSaM": str(vd_run / "depth")},
        "indices": [0, 1, 2, 3], "static_background_fraction": 0.12,
        "stability": {"MoGe-2 per frame (V2D)": row,
                      "MoGe-2 + MegaSaM": {**row, "scale_jitter_pct": 0.36, "per_frame_median_m": [0.71] * 4}}}))
    data = RunData(vd_run)
    assert list(data.sources) == ["MoGe-2 逐帧（V2D 的深度）", "MoGe-2 + MegaSaM（全分辨率）"]
    assert data.default_source == "MoGe-2 + MegaSaM（全分辨率）"
    assert data.background_depth(1) == {"MoGe-2 逐帧（V2D 的深度）": 0.6, "MoGe-2 + MegaSaM（全分辨率）": 0.71}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server, ui = serve(vd_run, port=port, block=False)
    try:
        assert "| MoGe-2 逐帧（V2D 的深度） | 4.93% |" in ui.stats.content and "0.36%" in ui.stats.content
        assert "本帧静止背景深度" in ui.info.content and "MegaSaM **0.710 m**" in ui.info.content
        ui.source.value = "MoGe-2 逐帧（V2D 的深度）"
        assert "V2D **0.700 m**" in ui.info.content
        n = len(ui.handles)
        ui.second.value = "MoGe-2 + MegaSaM（全分辨率）"  # second source drawn on top
        assert len(ui.handles) == n + 1 and "洋红色" in ui.info.content
    finally:
        server.stop()


def test_viewer_finds_hoi_refined_depth_of_the_same_video(tmp_path, monkeypatch):
    """HOI runs name their depth_estimation run in zyego_sources.json; their rendered refined depth becomes a source,
    labelled by the depth the HOI run was given (MoGe per frame = V2D, or MegaSaM)."""
    from depth_estimation import viewer
    monkeypatch.setattr(viewer, "PROJECT", tmp_path)
    depth_run = tmp_path / "runs" / "depth_estimation" / "demo"
    _fake_run(depth_run)
    for name, depth in (("demo-v2d", "depth_estimation (MoGe-2 per frame, as V2D)"),
                        ("other", "depth_estimation (MoGe-2 per frame, as V2D)")):
        hoi = tmp_path / "runs" / "HOI" / name
        (hoi / "depth_refined_simple").mkdir(parents=True)
        save_depth(hoi / "depth_refined_simple" / "000000.png", np.full((48, 64), 1.0, np.float32))
        target = depth_run if name == "demo-v2d" else tmp_path / "elsewhere"
        (hoi / "zyego_sources.json").write_text(json.dumps({"depth_estimation": str(target), "depth": depth}))
    found = viewer.find_sources(depth_run)
    assert list(found) == [("depth_estimation", "depth"), ("hoi_moge", "depth_refined_simple")]
    assert found[("hoi_moge", "depth_refined_simple")][0].name == "demo-v2d"  # not the other video's run
    data = viewer.RunData(depth_run)
    assert "V2D 修正后（MoGe-2 逐帧 + gsplat 精修）" in data.sources


def test_hdr_video_is_recognised_and_tone_mapped_to_sdr():
    from depth_estimation.frames import parse_color_info, tonemap_filter
    iphone = ("  Stream #0:0[0x1](und): Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, "
              "bt2020nc/bt2020/arib-std-b67), 1920x1080, 9802 kb/s, 30 fps, 30 tbr, 600 tbn (default)")
    info = parse_color_info(iphone)
    assert info["hdr"] == "HLG" and (info["width"], info["height"]) == (1920, 1080)
    assert (info["matrix"], info["primaries"], info["range"]) == ("bt2020nc", "bt2020", "tv")
    f = tonemap_filter(info)
    assert f.startswith("zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=100")
    assert "tonemap=tonemap=hable" in f and f.endswith("format=bgr24")
    sdr = parse_color_info("  Stream #0:0: Video: h264 (High) (avc1), yuv420p(tv, bt709), 1280x720, 30 fps")
    assert sdr["hdr"] is None and sdr["transfer"] == "bt709"
    rotated = parse_color_info(iphone + "\n      displaymatrix: rotation of -90.00 degrees")
    assert (rotated["width"], rotated["height"]) == (1080, 1920)  # ffmpeg auto-rotates
