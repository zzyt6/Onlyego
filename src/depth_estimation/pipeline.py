"""Depth stage as resumable steps, in V2D's order (run_ego_wilor.py steps 0-2):

  1. frames/                   extract frames
  2. calibration/anycalib/     [--undistort, or --prior anycalib] AnyCalib
     frames_undistorted/       [--undistort] frames remapped to the pinhole camera of the AnyCalib fit
     calibration/geocalib/     [--run-geocalib, or --prior geocalib] GeoCalib focal + gravity, on the
                               (undistorted) frames; gravity is used later by camera_pose --gravity-align
     MoGe prior = GeoCalib if --prior geocalib, else the undistorted camera if --undistort,
                  else AnyCalib pinhole if --prior anycalib, else none (MoGe guesses per frame)
  3. depth/ intrinsics/ mask/  MoGe-2 per frame, with the focal prior if any   (resumes per frame)
  4. intrinsics_stable.json    temporal median of the per-frame camera
  5. depth_stats.json, preview.mp4   inspection
  6. depth_aligned/            [--scale-align] per-frame scale alignment (align.py)
Optional afterwards: `align` (depth_aligned/) and `ply` (point cloud of one frame).
A step whose output exists is skipped, like V2D's _step(); delete the output to redo it.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datatypes import CameraIntrinsics, frame_indices
from .frames import DEFAULT_NPL, DEFAULT_TONEMAP, extract_frames, video_info, write_sdr_video
from .inspection import depth_stats, preview_video
from .intrinsics import stabilize_intrinsics

PRIORS = ("none", "anycalib", "geocalib")
# Frames calibrated (evenly spread, median-pooled). AnyCalib: 64, not V2D's 16 -- on demo01 a 16-frame median
# lands anywhere in about +-1.3 deg depending on which frames, 64 gets within 0.1 deg of all frames, at 0.08 s/frame.
ANYCALIB_SAMPLES = 64
GEOCALIB_SAMPLES = 8  # as V2D


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


def _calibration(method, out, frames, *, samples, cam_id, weights, geocalib_camera_model, device, log):
    """Run one calibrator (skipped when its calibration.json exists); return (intrinsics, summary)."""
    from . import calibration as calib

    method_dir = out / "calibration" / method
    if not _step(f"{method} calibration ({samples} frames)", (method_dir / "calibration.json").exists(), log):
        calib.calibrate(method, frames, method_dir, weights, samples=samples, cam_id=cam_id,
                        geocalib_camera_model=geocalib_camera_model, device=device, log=log)
    summary = json.loads((method_dir / "calibration.json").read_text(encoding="utf-8"))
    summary.pop("samples", None)
    k = CameraIntrinsics.load(method_dir / "intrinsics.json")
    log(f"         {method}: fx={k.fx:.1f} fov_x={k.fov_x_degrees:.1f} deg, distortion {summary['distortion']}")
    return k, summary


def _undistort(out, frames, k, summary, balance, log):
    """Remap frames to the pinhole camera of the AnyCalib fit (V2D step 0). Returns (frames, camera)."""
    from . import calibration as calib

    dist = calib.CameraDistortion(**summary["distortion"])
    if dist.model == "pinhole":
        log("         camera fitted as distortion-free; nothing to undistort")
        return frames, k
    method_dir = out / "calibration" / "anycalib"
    undistorted = out / "frames_undistorted"
    map1, map2, new_k = calib.undistort_maps(k, dist, balance)
    new_k.save(method_dir / "undistorted_intrinsics.json")
    if not _step(f"Undistort frames ({dist.model}, balance={balance})",
                 len(frame_indices(undistorted, ".png")) >= len(frame_indices(frames, ".png")), log):
        calib.undistort_frames(frames, undistorted, map1, map2)
    summary["undistorted_intrinsics"] = json.loads((method_dir / "undistorted_intrinsics.json").read_text())
    log(f"         undistorted camera: fx={new_k.fx:.1f} fov_x={new_k.fov_x_degrees:.1f} deg")
    return undistorted, new_k


def _calibrate_and_choose_prior(out, frames, *, prior, undistort, run_geocalib, calib_samples, anycalib_cam_id,
                                geocalib_weights, geocalib_camera_model, balance, model_dir, device, log):
    """V2D run_ego_wilor.py steps 0 / 0b and its choice of MoGe prior:
         0.  AnyCalib undistortion        [--undistort]  -> the frames used by everything after this
         0b. GeoCalib intrinsics+gravity  [--run-geocalib, or --prior geocalib], on those frames
         moge_input_intrinsics = geocalib if use_geocalib_intrinsics else (undistorted if undistort else None)
       Our addition: --prior anycalib without --undistort fits a distortion-free pinhole camera.
    Returns (frames folder for MoGe, prior intrinsics or None, record of what ran)."""
    images, prior_k, record = frames, None, {"method": prior}
    if prior == "anycalib" or undistort:
        cam_id = anycalib_cam_id if undistort else "pinhole"
        weights = model_dir / "anycalib" / ("anycalib_gen.pt" if undistort else "anycalib_pinhole.pt")
        k, summary = _calibration("anycalib", out, frames, samples=calib_samples or ANYCALIB_SAMPLES, cam_id=cam_id,
                                  weights=weights, geocalib_camera_model="pinhole", device=device, log=log)
        if undistort:
            images, k = _undistort(out, frames, k, summary, balance, log)
        record["anycalib"] = summary
        prior_k = k
    if run_geocalib or prior == "geocalib":
        k, summary = _calibration("geocalib", out, images, samples=calib_samples or GEOCALIB_SAMPLES, cam_id="pinhole",
                                  weights=model_dir / "geocalib" / f"geocalib-{geocalib_weights}.tar",
                                  geocalib_camera_model=geocalib_camera_model, device=device, log=log)
        record["geocalib"] = summary
        if prior == "geocalib":
            prior_k = k
    return images, prior_k, record


def run_depth(video: Path, out: Path, weights: Path, *, stride: int = 1, max_frames: int | None = None,
              batch_size: int = 8, fov_from: Path | None = None, prior: str = "none", undistort: bool = False,
              calib_samples: int | None = None, anycalib_cam_id: str = "radial:1", geocalib_weights: str = "pinhole",
              geocalib_camera_model: str = "pinhole", run_geocalib: bool = False,
              balance: float = 0.0, model_dir: Path | None = None, save_points: bool = False,
              save_normals: bool = False, device: str = "cuda", preview: bool = True, scale_align: bool = False,
              hdr_tonemap: str = DEFAULT_TONEMAP, hdr_npl: float = DEFAULT_NPL, log=print) -> dict:
    from .calibration import GEOCALIB_CAMERA_MODELS

    if prior not in PRIORS:
        raise ValueError(f"--prior must be one of {PRIORS}")
    if undistort and prior == "none":
        prior = "anycalib"  # V2D: with --undistort, MoGe gets the undistorted camera unless GeoCalib's is chosen
    if fov_from and (prior != "none" or undistort):
        raise ValueError("Use either --fov-from or --prior/--undistort, not both")
    if geocalib_camera_model not in GEOCALIB_CAMERA_MODELS:
        raise ValueError(f"--geocalib-camera-model must be one of {GEOCALIB_CAMERA_MODELS}")
    if geocalib_camera_model != "pinhole" and geocalib_weights != "distorted":
        raise ValueError("A distorted GeoCalib camera model needs --geocalib-weights distorted")  # as V2D
    run_geocalib = run_geocalib or prior == "geocalib"
    model_dir = model_dir or Path(weights).parent
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    settings = {"video": str(video.resolve()), "stride": stride, "max_frames": max_frames,
                "weights": str(weights), "fov_from": str(fov_from) if fov_from else None,
                "prior": prior, "undistort": undistort, "run_geocalib": run_geocalib}
    if prior == "anycalib" or undistort:
        settings |= {"calib_samples": calib_samples, "anycalib_cam_id": anycalib_cam_id if undistort else "pinhole",
                     "balance": balance if undistort else None}
    if run_geocalib:
        settings |= {"calib_samples": calib_samples, "geocalib_weights": geocalib_weights,
                     "geocalib_camera_model": geocalib_camera_model}
    info = video_info(video)
    hdr = (info.get("color") or {}).get("hdr")
    if hdr:  # HDR (iPhone HLG): frames are tone-mapped to SDR, see frames.py
        settings["hdr_tonemap"] = "off" if hdr_tonemap == "off" else f"{hdr_tonemap}, npl {hdr_npl:g}"
    previous = record.get("settings")
    if previous:  # fill keys that older runs did not record, with what those runs actually did
        previous = {"prior": "none", "undistort": False} | previous
        previous.setdefault("run_geocalib", previous["prior"] == "geocalib")
        if previous["run_geocalib"]:
            previous.setdefault("geocalib_camera_model",
                                "simple_radial" if previous.get("geocalib_weights") == "distorted" else "pinhole")
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

    images, prior_k = frames, None
    if fov_from:
        prior_k = CameraIntrinsics.load(fov_from)
        record["prior"] = {"method": "file", "path": str(fov_from)}
    elif prior != "none" or run_geocalib:
        images, prior_k, record["prior"] = _calibrate_and_choose_prior(
            out, frames, prior=prior, undistort=undistort, run_geocalib=run_geocalib, calib_samples=calib_samples,
            anycalib_cam_id=anycalib_cam_id, geocalib_weights=geocalib_weights,
            geocalib_camera_model=geocalib_camera_model, balance=balance, model_dir=model_dir, device=device, log=log)
    record["image_dir"] = images.name
    write_json(record_path, record)
    fov_x = prior_k.fov_x_degrees if prior_k else None
    if fov_x is not None and not 1.0 < fov_x < 179.0:  # also catches NaN
        raise ValueError(f"Focal prior gives an impossible field of view ({fov_x} deg); not passing it to MoGe")
    if fov_x is not None:
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

    if scale_align:  # optional: per-frame scale alignment -> depth_aligned/
        aligned_report = out / "depth_aligned_scales.json"
        if not _step("Per-frame scale alignment", aligned_report.exists(), log):
            from .align import align_depth_sequence
            report = align_depth_sequence(out / "depth", images, out / "depth_aligned")
            write_json(aligned_report, report)
            write_json(out / "depth_aligned_stats.json", depth_stats(out / "depth_aligned", out / "intrinsics"))
        report = json.loads(aligned_report.read_text(encoding="utf-8"))
        before = [r["depth_median"] for r in json.loads(stats_path.read_text(encoding="utf-8")) if r["depth_median"]]
        after = [r["depth_median"] for r in json.loads((out / "depth_aligned_stats.json").read_text(encoding="utf-8"))
                 if r["depth_median"]]
        spread = lambda v: 100 * float(np.std(v) / np.mean(v))
        record["scale_align"] = {"frames_with_scale": report["estimated"], "frames": report["frames"],
                                   "depth_median_cv_percent_before": round(spread(before), 2),
                                   "depth_median_cv_percent_after": round(spread(after), 2)}
        log(f"         scale estimated on {report['estimated']}/{report['frames']} frames; per-frame median depth "
            f"varies {spread(before):.1f}% -> {spread(after):.1f}% (std/mean)")

    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(record_path, record)
    return record
