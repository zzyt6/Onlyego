"""camera_pose: per-frame metric camera poses from a depth_estimation run (DROID-SLAM in WSL, after V2D)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEPTH_RUNS = PROJECT / "runs" / "depth_estimation"
RUNS = PROJECT / "runs" / "camera_pose"
WEIGHTS = PROJECT / "model" / "droid_slam" / "droid.pth"
GRAVITY_TARGET_NAMES = ("z_up", "z_down", "y_up", "y_down", "x_up", "x_down")  # = gravity.GRAVITY_TARGETS


def resolve_depth_run(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        path = DEPTH_RUNS / value
    missing = [n for n in ("run.json", "intrinsics_stable.json", "depth") if not (path / n).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a finished depth_estimation run (missing {', '.join(missing)}); "
                                f"run  depth_estimation run <video>  first")
    return path


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="camera_pose", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="DROID-SLAM over the run's frames, scaled to its MoGe depth")
    run.add_argument("depth_run", help="depth_estimation run folder, or its name under runs/depth_estimation")
    run.add_argument("--out", type=Path, help="Default: runs/camera_pose/<depth run name>")
    run.add_argument("--weights", type=Path, default=WEIGHTS)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--depth-prior", action="store_true",
                     help="Also feed MoGe depth into bundle adjustment (V2D's ego pipeline does not)")
    run.add_argument("--aligned-depth", action="store_true", help="Use depth_aligned/ instead of depth/")
    run.add_argument("--keyframe-thresh", type=float, default=4.0)
    run.add_argument("--save-keyframe-depth", action="store_true")
    run.add_argument("--no-pointcloud", action="store_true")
    run.add_argument("--gravity-align", action="store_true",
                     help="Afterwards rotate the world with GeoCalib gravity -> poses_gravity/ "
                          "(needs depth_estimation --run-geocalib or --prior geocalib)")
    gravity = sub.add_parser("gravity-align",
                             help="Rotate an existing run's poses so gravity points down an axis (V2D --gravity_align)")
    gravity.add_argument("camera_run", help="camera_pose run folder, or its name under runs/camera_pose")
    for p in (run, gravity):
        p.add_argument("--gravity-target", default="z_up", choices=GRAVITY_TARGET_NAMES,
                       help="World axis that points up after alignment (V2D --gravity_align_target)")
    sub.add_parser("check", help="Check the WSL CUDA env (DROID-SLAM, lietorch, gsplat, manotorch)")
    args = parser.parse_args(argv)

    from . import wsl
    try:
        if args.command == "check":
            code = subprocess_check()
            sys.exit(code)
        if args.command == "gravity-align":
            camera_run = Path(args.camera_run)
            if not camera_run.is_dir():
                camera_run = RUNS / args.camera_run
            summary_path = camera_run / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"{camera_run} is not a finished camera_pose run (no summary.json)")
            depth_run = Path(json.loads(summary_path.read_text(encoding="utf-8"))["depth_run"])
            print_gravity(gravity_align_and_report(camera_run, depth_run, args.gravity_target))
            return
        depth_run = resolve_depth_run(args.depth_run)
        if args.gravity_align and not (depth_run / "calibration" / "geocalib" / "calibration.json").exists():
            raise FileNotFoundError(f"--gravity-align needs GeoCalib gravity, but {depth_run} has no "
                                    f"calibration/geocalib/; rerun depth_estimation with --run-geocalib")
        out = args.out or RUNS / depth_run.name
        if (out / "poses").is_dir() and any((out / "poses").iterdir()):
            raise ValueError(f"{out} already has poses; delete it or pass another --out")
        if not args.weights.is_file():
            raise FileNotFoundError(f"DROID-SLAM weights not found: {args.weights}")
        out.mkdir(parents=True, exist_ok=True)
        print(f"Depth run: {depth_run}\nOutput: {out}")
        slam_args = ["--depth-run", wsl.to_wsl(depth_run), "--out", wsl.to_wsl(out),
                     "--weights", wsl.to_wsl(args.weights), "--stride", str(args.stride),
                     "--keyframe-thresh", str(args.keyframe_thresh)]
        if args.aligned_depth:
            slam_args += ["--depth-subdir", "depth_aligned"]
        slam_args += [flag for flag, on in (("--depth-prior", args.depth_prior),
                                            ("--save-keyframe-depth", args.save_keyframe_depth),
                                            ("--no-pointcloud", args.no_pointcloud)) if on]
        code = wsl.run_module("camera_pose.slam", slam_args)
        if code != 0:
            raise RuntimeError(f"DROID-SLAM failed in WSL (exit {code})")
        from .report import write_report
        summary = write_report(depth_run, out)
        print(json.dumps(summary["trajectory"], indent=2))
        if args.gravity_align:
            print_gravity(gravity_align_and_report(out, depth_run, args.gravity_target))
        print(f"Done: {out}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


def gravity_align_and_report(camera_run: Path, depth_run: Path, target: str) -> dict:
    from .gravity import gravity_align_run
    from .report import draw_trajectory
    from .poses import load_trajectory
    info = gravity_align_run(camera_run, depth_run, target)
    if target == "z_up":
        views = (("top view: x right, y (horizontal forward) up", (0, 1, True)),
                 ("side view: y (horizontal forward) right, z up", (1, 2, True)))
        draw_trajectory(load_trajectory(camera_run / "poses_gravity")[1], camera_run / "trajectory_gravity.png",
                        views=views)
    return info


def print_gravity(info: dict):
    import numpy as np
    g = np.array(info["gravity_in_world_before_alignment"])
    # angle between gravity and the first camera's +y (image down): how far the camera was tilted
    tilt = np.degrees(np.arccos(np.clip(g[1], -1, 1)))
    print(f"Gravity in the SLAM world (first camera's axes): {np.round(g, 4).tolist()} "
          f"({tilt:.1f} deg from the first camera's image-down axis), from {len(info['sample_frame_indices'])} "
          f"GeoCalib frames\nRotated to {info['target']} -> poses_gravity/, gravity_alignment.json")


def subprocess_check() -> int:
    from . import wsl
    script = PROJECT / "tools" / "wsl" / "setup_cuda_env.sh"
    import os
    import subprocess
    return subprocess.call(["wsl.exe", "-d", wsl.DISTRO, "--", "bash", wsl.to_wsl(script), "check"],
                           env=dict(os.environ, MSYS_NO_PATHCONV="1"))


if __name__ == "__main__":
    main()
