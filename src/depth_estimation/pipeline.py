"""Depth stage as resumable steps:

  1. frames/                   extract frames (HDR video tone-mapped to SDR)
  2. calibration/anycalib/     AnyCalib pinhole fit, median over 64 frames -> MoGe's focal prior
  3. depth/ intrinsics/ mask/  MoGe-2 per frame with that focal   (resumes per frame)
  4. intrinsics_stable.json    temporal median of the per-frame camera
  5. depth_stats.json, preview.mp4   inspection
A step whose output exists is skipped (as V2D's _step()); delete the output to redo it.
Then video_depth (MegaSaM) makes the depth consistent over the video.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .datatypes import CameraIntrinsics, frame_indices
from .frames import DEFAULT_NPL, DEFAULT_TONEMAP, extract_frames, video_info, write_sdr_video
from .inspection import depth_stats, preview_video
from .intrinsics import stabilize_intrinsics

# Frames calibrated (evenly spread, median-pooled): 64, not V2D's 16 -- on demo01 a 16-frame median lands anywhere
# in about +-1.3 deg depending on which frames, 64 gets within 0.1 deg of all frames, at 0.08 s/frame.
ANYCALIB_SAMPLES = 64


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def _step(label, done, log):
    log(f"  [{'skip' if done else 'run '}] {label}")
    return done


def _anycalib_prior(out: Path, frames: Path, *, samples: int, weights: Path, device: str, log) -> tuple:
    """AnyCalib pinhole calibration (skipped when its calibration.json exists); returns (intrinsics, summary)."""
    from . import calibration as calib

    method_dir = out / "calibration" / "anycalib"
    if not _step(f"AnyCalib calibration ({samples} frames)", (method_dir / "calibration.json").exists(), log):
        calib.calibrate(frames, method_dir, weights, samples=samples, device=device, log=log)
    summary = json.loads((method_dir / "calibration.json").read_text(encoding="utf-8"))
    summary.pop("samples", None)
    k = CameraIntrinsics.load(method_dir / "intrinsics.json")
    log(f"         AnyCalib: fx={k.fx:.1f} fov_x={k.fov_x_degrees:.1f} deg")
    return k, summary


def run_depth(video: Path, out: Path, weights: Path, *, stride: int = 1, max_frames: int | None = None,
              batch_size: int = 8, calib_samples: int | None = None, model_dir: Path | None = None,
              save_points: bool = False, save_normals: bool = False, device: str = "cuda", preview: bool = True,
              hdr_tonemap: str = DEFAULT_TONEMAP, hdr_npl: float = DEFAULT_NPL, log=print) -> dict:
    model_dir = model_dir or Path(weights).parent
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    settings = {"video": str(video.resolve()), "stride": stride, "max_frames": max_frames,
                "weights": str(weights), "prior": "anycalib", "anycalib_cam_id": "pinhole",
                "calib_samples": calib_samples}
    info = video_info(video)
    hdr = (info.get("color") or {}).get("hdr")
    if hdr:  # HDR (iPhone HLG): frames are tone-mapped to SDR, see frames.py
        settings["hdr_tonemap"] = "off" if hdr_tonemap == "off" else f"{hdr_tonemap}, npl {hdr_npl:g}"
    previous = record.get("settings")
    if previous and {k: previous.get(k) for k in settings} != settings:
        raise ValueError(f"{out} was made with different settings {previous}; use another --out or delete the folder")
    record.update(settings=settings, video=info, started_at=record.get("started_at")
                  or datetime.now(timezone.utc).isoformat())
    write_json(record_path, record)

    frames = out / "frames"
    if not _step("Extract frames", bool(frame_indices(frames, ".png")), log):
        record["frames"] = extract_frames(video, frames, stride, max_frames, tonemap=hdr_tonemap, npl=hdr_npl)
        write_json(record_path, record)
    if hdr and hdr_tonemap != "off":
        log(f"         {hdr} HDR video ({info['color'].get('primaries')}): frames tone-mapped to SDR BT.709 "
            f"({settings['hdr_tonemap']})")
        sdr = out / "video_sdr.mp4"
        if not _step("SDR copy of the video (for steps that read the video itself)", sdr.exists(), log):
            write_sdr_video(video, sdr, hdr_tonemap, hdr_npl)
        record["video_sdr"] = str(sdr.resolve())
        write_json(record_path, record)
    n_frames = len(frame_indices(frames, ".png"))
    log(f"         {n_frames} frames, {info['width']}x{info['height']} @ {info['fps']:.2f} fps")

    images = frames
    prior_k, summary = _anycalib_prior(out, frames, samples=calib_samples or ANYCALIB_SAMPLES,
                                       weights=model_dir / "anycalib" / "anycalib_pinhole.pt", device=device, log=log)
    record["prior"] = {"method": "anycalib", "anycalib": summary}
    record["image_dir"] = images.name
    write_json(record_path, record)
    fov_x = prior_k.fov_x_degrees
    if not 1.0 < fov_x < 179.0:  # also catches NaN
        raise ValueError(f"Focal prior gives an impossible field of view ({fov_x} deg); not passing it to MoGe")
    log(f"         MoGe focal prior: fov_x = {fov_x:.2f} deg")

    have = len(frame_indices(out / "depth", ".png"))
    if not _step(f"MoGe-2 depth + intrinsics ({have}/{n_frames} done)", have >= n_frames, log):
        from .moge_depth import frames_to_depth
        t0 = time.time()

        def progress(done, total):
            rate = done / max(time.time() - t0, 1e-6)
            log(f"         {done}/{total} frames  {rate:.1f} fps  eta {(total - done) / max(rate, 1e-6):.0f}s")

        frames_to_depth(images, out, weights, batch_size=batch_size, fov_x=fov_x, save_points=save_points,
                        save_normals=save_normals, device=device, on_batch=progress)
        record["depth_seconds"] = round(time.time() - t0, 1)
        write_json(record_path, record)

    stable_path = out / "intrinsics_stable.json"
    if not _step("Stabilise intrinsics (temporal median)", stable_path.exists(), log):
        stable = stabilize_intrinsics(out / "intrinsics", stable_path)
        log(f"         fx={stable.fx:.1f} fy={stable.fy:.1f} cx={stable.cx:.1f} cy={stable.cy:.1f} "
            f"fov_x={stable.fov_x_degrees:.1f} deg")
    record["intrinsics_stable"] = json.loads(stable_path.read_text(encoding="utf-8"))

    stats_path = out / "depth_stats.json"
    if not _step("Depth statistics", stats_path.exists(), log):
        write_json(stats_path, depth_stats(out / "depth", out / "intrinsics"))
    if preview and not _step("Preview video", (out / "preview.mp4").exists(), log):
        preview_video(images, out / "depth", out / "preview.mp4", info["fps"] / stride)

    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(record_path, record)
    return record
