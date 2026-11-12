import json

import cv2
import numpy as np
import pytest

from depth_estimation import pipeline
from depth_estimation.datatypes import CameraIntrinsics, decode_depth, encode_depth, load_depth, save_depth
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
    _fake_calibration(monkeypatch, [])
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


def _fake_calibration(monkeypatch, calls):
    """AnyCalib stand-in: fx = 32 on 64 px, i.e. fov_x 90 deg."""
    import depth_estimation.calibration as calib
    import depth_estimation.moge_depth as moge_depth

    def fake_calibrate(frames, out, weights, *, samples, device="cuda", log=print):
        calls.append({"weights": weights.name, "frames": frames.name, "samples": samples})
        out.mkdir(parents=True, exist_ok=True)
        CameraIntrinsics(32.0, 32.0, 32, 24, 64, 48).save(out / "intrinsics.json")
        (out / "calibration.json").write_text(json.dumps({"method": "anycalib", "cam_id": "pinhole", "samples": []}))

    monkeypatch.setattr(calib, "calibrate", fake_calibrate)
    monkeypatch.setattr(moge_depth, "load_model", lambda weights, device: RecordingMoGe())
    RecordingMoGe.fov_seen = []


def test_anycalib_pinhole_focal_is_moges_prior(tmp_path, monkeypatch):
    """AnyCalib (pinhole weights, 64 frames by default; V2D: 16) on the extracted frames; its fov goes to MoGe."""
    pytest.importorskip("torch")
    calls = []
    _fake_calibration(monkeypatch, calls)
    video = tmp_path / "clip.mp4"
    _write_video(video)
    out = tmp_path / "run"
    record = pipeline.run_depth(video, out, tmp_path / "model" / "moge", device="cpu", preview=False,
                                log=lambda m: None)
    assert calls == [{"weights": "anycalib_pinhole.pt", "frames": "frames", "samples": 64}]
    assert record["image_dir"] == "frames" and record["prior"]["method"] == "anycalib"
    assert RecordingMoGe.fov_seen[0] == pytest.approx(90.0)
    from depth_estimation.calibration import sample_indices
    assert sample_indices(list(range(5)), 64) == [0, 1, 2, 3, 4]


def test_old_run_with_extra_settings_still_resumes(tmp_path, monkeypatch):
    """Runs recorded with the removed options (undistort, run_geocalib, ...) at their defaults are not 'different'."""
    pytest.importorskip("torch")
    _fake_calibration(monkeypatch, [])
    video = tmp_path / "clip.mp4"
    _write_video(video)
    out = tmp_path / "run"
    pipeline.run_depth(video, out, tmp_path / "w", device="cpu", preview=False, log=lambda m: None)
    record = json.loads((out / "run.json").read_text())
    record["settings"] |= {"fov_from": None, "undistort": False, "run_geocalib": False, "balance": None}
    (out / "run.json").write_text(json.dumps(record))
    pipeline.run_depth(video, out, tmp_path / "w", device="cpu", preview=False, log=lambda m: None)


def test_cli_passes_calibration_samples(tmp_path, monkeypatch):
    from depth_estimation import cli
    seen = {}
    monkeypatch.setattr(pipeline, "run_depth", lambda *a, **kw: seen.update(kw))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    cli.main(["run", str(video), "--out", str(tmp_path / "o"), "--calib-samples", "16"])
    assert seen["calib_samples"] == 16 and "prior" not in seen


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


def test_viewer_starts(tmp_path):
    pytest.importorskip("viser")
    import socket
    from depth_estimation.viewer import RunData, serve
    run = tmp_path / "run"
    _fake_run(run)
    data = RunData(run)
    assert list(data.sources) == ["MoGe-2 逐帧"]
    pts, _ = data.cloud(2, data.default_source, 4, True)
    assert pts[:, 2] == pytest.approx(1.2, abs=1e-3) and len(pts) == 12 * 16
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server, ui = serve(run, port=port, block=False)
    try:
        assert len(ui.handles) == 2  # one point cloud + one frustum
        ui.overlay.value = 3  # setting a control from Python redraws (the Play loop relies on this)
        assert len(ui.handles) == 6
        ui.colour.value = "按帧"
        ui.frame.value = 3
        assert "第 **3** 帧" in ui.info.content and "第 **1** 帧" in ui.info.content  # wraps around
    finally:
        server.stop()


def test_viewer_offers_both_runs(tmp_path):
    """A video_depth run and its depth_estimation run are found from either side."""
    pytest.importorskip("viser")
    import socket
    from depth_estimation.viewer import RunData, serve
    depth_run, vd_run = tmp_path / "depth_estimation" / "demo", tmp_path / "video_depth" / "demo"
    _fake_run(depth_run)
    _fake_run(vd_run)
    (vd_run / "megasam.json").write_text("{}")
    (vd_run / "run.json").write_text(json.dumps({"image_dir": str(depth_run / "frames"), "depth_run": str(depth_run),
                                                 "video": {"fps": 30}, "settings": {"stride": 1}}))
    data = RunData(vd_run)
    assert list(data.sources) == ["MoGe-2 逐帧", "MoGe-2 + MegaSaM（全分辨率）"]
    assert data.default_source == "MoGe-2 + MegaSaM（全分辨率）"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server, ui = serve(vd_run, port=port, block=False)
    try:
        ui.source.value = "MoGe-2 逐帧"
        assert "第 **0** 帧：深度中位数 **1.000 m**" in ui.info.content
        n = len(ui.handles)
        ui.second.value = "MoGe-2 + MegaSaM（全分辨率）"  # second source drawn on top
        assert len(ui.handles) == n + 1 and "洋红色" in ui.info.content
    finally:
        server.stop()


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
