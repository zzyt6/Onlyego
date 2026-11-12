"""depth_estimation: video -> per-frame metric depth + stable camera intrinsics (MoGe-2 with an AnyCalib focal prior)."""

import argparse
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
    run.add_argument("--calib-samples", type=int,
                     help="Frames AnyCalib calibrates on, evenly spread (default 64); its focal is MoGe's prior")
    run.add_argument("--model-dir", type=Path, default=PROJECT / "model",
                     help="Folder holding anycalib/ weights")
    run.add_argument("--save-points", action="store_true", help="Also save MoGe point maps (.npy, large)")
    run.add_argument("--save-normals", action="store_true", help="Also save MoGe normals (.npy, large)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--no-preview", action="store_true")
    run.add_argument("--hdr-tonemap", choices=["hable", "mobius", "reinhard", "clip", "off"], default="hable",
                     help="HDR video (e.g. iPhone HLG): tone mapping to SDR for the frames (off = read the HDR "
                          "signal as SDR, washed out, as OpenCV / V2D do)")
    run.add_argument("--hdr-npl", type=float, default=100.0,
                     help="HDR tone mapping: nominal peak luminance in nits (lower = brighter)")

    view = sub.add_parser("view", help="Interactive 3D point-cloud viewer in the browser (viser)")
    view.add_argument("run", help="Run folder or its name under runs/depth_estimation")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the network")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
                raise ValueError("--stride and --max-frames must be >= 1")
            if args.calib_samples is not None and args.calib_samples < 1:
                raise ValueError("--calib-samples must be >= 1")
            from .pipeline import run_depth
            video = resolve_video(args.video)
            out = args.out or RUNS / video.stem
            print(f"Video: {video}\nOutput: {out}")
            run_depth(video, out, args.weights, stride=args.stride, max_frames=args.max_frames,
                      batch_size=args.batch_size, calib_samples=args.calib_samples, model_dir=args.model_dir,
                      save_points=args.save_points, save_normals=args.save_normals, device=args.device,
                      preview=not args.no_preview, hdr_tonemap=args.hdr_tonemap, hdr_npl=args.hdr_npl)
            print(f"Done: {out}")
        elif args.command == "view":
            from .viewer import serve
            out = resolve_run(args.run)
            serve(out, host=args.host, port=args.port)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
