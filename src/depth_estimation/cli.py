"""depth_estimation: video -> per-frame metric depth + stable camera intrinsics (MoGe-2, after V2D)."""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIDEOS = PROJECT / "data" / "videos"
RUNS = PROJECT / "runs" / "depth_estimation"
WEIGHTS = PROJECT / "model" / "moge-2-vitl-normal"


def resolve_video(value: Path) -> Path:
    """A plain name (demo01 or demo01.mp4) is looked up in <project>/data/videos; a path is used as given."""
    if value.is_file() or value.parent != Path("."):
        return value
    for candidate in (VIDEOS / value.name, VIDEOS / f"{value.name}.mp4"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found: {value} (looked in {VIDEOS})")


def resolve_run(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        path = RUNS / value
    if not (path / "depth").is_dir():
        raise FileNotFoundError(f"Not a depth run (no depth/ folder): {path}")
    return path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="depth_estimation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Frames -> MoGe-2 depth + intrinsics -> stable intrinsics -> preview")
    run.add_argument("video", type=Path, help="Name in data/videos (e.g. demo01) or a path to a video")
    run.add_argument("--out", type=Path, help="Output folder (default: runs/depth_estimation/<video name>)")
    run.add_argument("--weights", type=Path, default=WEIGHTS)
    run.add_argument("--stride", type=int, default=1, help="Use every N-th frame (default 1 = all)")
    run.add_argument("--max-frames", type=int, help="Stop after this many frames (quick tries)")
    run.add_argument("--batch-size", type=int, default=8)
    prior = run.add_argument_group("focal-length prior for MoGe (default: AnyCalib pinhole fit)")
    prior.add_argument("--prior", choices=["none", "anycalib", "geocalib"],
                       help="Calibrator whose focal length MoGe gets. Default anycalib (pinhole model, no "
                            "undistortion: phone main cameras are already undistorted in-camera); "
                            "none = MoGe guesses the focal per frame (V2D's default)")
    prior.add_argument("--undistort", action="store_true",
                       help="AnyCalib: fit a lens-distortion model and remap frames to a pinhole camera first "
                            "(V2D --undistort); MoGe then gets that camera unless --prior geocalib")
    prior.add_argument("--run-geocalib", action="store_true",
                       help="Also run GeoCalib for the gravity direction (needed by camera_pose --gravity-align) "
                            "without using its focal length (V2D --run_geocalib)")
    prior.add_argument("--calib-samples", type=int, help="Frames to calibrate on, evenly spread (default 64 AnyCalib, 8 GeoCalib)")
    prior.add_argument("--anycalib-cam-id", default="radial:1",
                       choices=["radial:1", "radial:2", "radial:4", "kb:4"],
                       help="Distortion model fitted with --undistort. Default radial:1 (phone / smart-glasses lens); "
                            "kb:4 = Kannala-Brandt fisheye, V2D's default for fisheye head-mounted cameras")
    prior.add_argument("--balance", type=float, default=0.0,
                       help="--undistort: 0 = crop to valid pixels, 1 = keep full FoV with black corners")
    prior.add_argument("--geocalib-weights", choices=["pinhole", "distorted"], default="pinhole")
    prior.add_argument("--geocalib-camera-model", default="pinhole",
                       choices=["pinhole", "simple_radial", "radial", "simple_divisional"],
                       help="Camera model GeoCalib fits; a distorted one needs --geocalib-weights distorted")
    prior.add_argument("--fov-from", type=Path,
                       help="Use the focal length in this intrinsics JSON instead, e.g. a previous intrinsics_stable.json")
    run.add_argument("--model-dir", type=Path, default=PROJECT / "model",
                     help="Folder holding anycalib/ and geocalib/ weights")
    run.add_argument("--save-points", action="store_true", help="Also save MoGe point maps (.npy, large)")
    run.add_argument("--save-normals", action="store_true", help="Also save MoGe normals (.npy, large)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--no-preview", action="store_true")
    run.add_argument("--hdr-tonemap", choices=["hable", "mobius", "reinhard", "clip", "off"], default="hable",
                     help="HDR video (e.g. iPhone HLG): tone mapping to SDR for the frames (off = read the HDR "
                          "signal as SDR, washed out, as OpenCV / V2D do)")
    run.add_argument("--hdr-npl", type=float, default=100.0,
                     help="HDR tone mapping: nominal peak luminance in nits (lower = brighter)")
    run.add_argument("--scale-align", action="store_true",
                     help="Per-frame scale alignment: also write depth_aligned/, each frame's depth rescaled so the "
                          "background matches the reference frame (SIFT); needs shared background, i.e. little "
                          "camera motion. For moving cameras use video_depth (MegaSaM)")

    align = sub.add_parser("align", help="Optional: per-frame scale alignment to a reference frame (SIFT, background)")
    align.add_argument("run", help="Run folder or its name under runs/depth_estimation")
    align.add_argument("--masks", type=Path, help="Object mask PNGs (<frame>.png, non-zero = object) to exclude")
    align.add_argument("--reference-frame", type=int, help="Default: first frame")
    align.add_argument("--window", type=int, default=11, help="Odd median-filter window over log-scale")
    align.add_argument("--metric-scale", choices=["median", "reference"], default="median",
                       help="Overall scale kept: median over all frames (default) or the reference frame's own (V2D)")

    ply = sub.add_parser("ply", help="Back-project one frame to a coloured point cloud (.ply)")
    ply.add_argument("run")
    ply.add_argument("--frame", type=int, default=0)
    ply.add_argument("--depth", default="depth", choices=["depth", "depth_aligned"])
    ply.add_argument("--intrinsics", default="per-frame", choices=["per-frame", "stable"])
    ply.add_argument("--stride", type=int, default=2, help="Pixel step (2 = a quarter of the pixels)")

    view = sub.add_parser("view", help="Interactive 3D point-cloud viewer in the browser (viser)")
    view.add_argument("run", help="Run folder or its name under runs/depth_estimation")
    view.add_argument("--poses", help="camera_pose run (folder or name under runs/camera_pose): show points in "
                                      "world coordinates along the camera trajectory")
    view.add_argument("--gravity", action="store_true", help="With --poses: use the gravity-aligned poses (z up)")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the network")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
                raise ValueError("--stride and --max-frames must be >= 1")
            if args.calib_samples is not None and args.calib_samples < 1:
                raise ValueError("--calib-samples must be >= 1")
            if not 0.0 <= args.balance <= 1.0:
                raise ValueError("--balance must be between 0 and 1")
            from .pipeline import run_depth
            if args.prior is None:  # --fov-from replaces the calibrator
                args.prior = "none" if args.fov_from else "anycalib"
            video = resolve_video(args.video)
            out = args.out or RUNS / video.stem
            print(f"Video: {video}\nOutput: {out}")
            run_depth(video, out, args.weights, stride=args.stride, max_frames=args.max_frames,
                      batch_size=args.batch_size, fov_from=args.fov_from, prior=args.prior,
                      undistort=args.undistort, calib_samples=args.calib_samples,
                      anycalib_cam_id=args.anycalib_cam_id, geocalib_weights=args.geocalib_weights,
                      geocalib_camera_model=args.geocalib_camera_model, run_geocalib=args.run_geocalib,
                      balance=args.balance, model_dir=args.model_dir, save_points=args.save_points,
                      save_normals=args.save_normals, device=args.device, preview=not args.no_preview,
                      scale_align=args.scale_align, hdr_tonemap=args.hdr_tonemap, hdr_npl=args.hdr_npl)
            print(f"Done: {out}")
        elif args.command == "align":
            from .align import align_depth_sequence
            from .inspection import image_dir
            from .pipeline import write_json
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
            out = resolve_run(args.run)
            report = align_depth_sequence(out / "depth", image_dir(out), out / "depth_aligned",
                                          masks_dir=args.masks, reference_frame=args.reference_frame,
                                          window=args.window, metric_scale=args.metric_scale)
            write_json(out / "depth_aligned_scales.json", report)
            scales = [r["scale"] for r in report["per_frame"]]
            print(f"Scale estimated on {report['estimated']}/{report['frames']} frames; "
                  f"smoothed scale range {min(scales):.3f}-{max(scales):.3f}\n"
                  f"Written: {out / 'depth_aligned'}  (details: depth_aligned_scales.json)")
        elif args.command == "ply":
            from .inspection import frame_point_cloud
            out = resolve_run(args.run)
            target = out / "pointclouds" / f"{args.frame:06d}_{args.depth}_{args.intrinsics}.ply"
            n = frame_point_cloud(out, args.frame, target, depth_subdir=args.depth,
                                  intrinsics=args.intrinsics, stride=args.stride)
            print(f"{n} points -> {target}")
        elif args.command == "view":
            from .viewer import serve
            out = resolve_run(args.run)
            poses = None
            if args.poses:
                cam = Path(args.poses)
                if not cam.is_dir():
                    cam = PROJECT / "runs" / "camera_pose" / args.poses
                poses = cam / ("poses_gravity" if args.gravity else "poses")
                if not poses.is_dir():
                    raise FileNotFoundError(f"No {poses.name}/ in {cam}"
                                            + (" (run camera_pose with --gravity-align)" if args.gravity else ""))
            serve(out, poses, host=args.host, port=args.port)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
