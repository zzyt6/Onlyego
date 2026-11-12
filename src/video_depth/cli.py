"""video_depth: temporally consistent metric depth + camera poses (MegaSaM with MoGe-2 prior, in WSL)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEPTH_RUNS = PROJECT / "runs" / "depth_estimation"
RUNS = PROJECT / "runs" / "video_depth"
MODEL = PROJECT / "model" / "megasam"
ENV = "zyego-megasam"
MEGASAM = "$HOME/zyego-external/mega-sam"
PYTHONPATH = tuple(f"{MEGASAM}/{p}" for p in ("base", "base/thirdparty/lietorch", "base/droid_slam",
                                               "cvd_opt", "cvd_opt/core"))


def resolve_depth_run(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        path = DEPTH_RUNS / value
    missing = [n for n in ("run.json", "intrinsics_stable.json", "depth") if not (path / n).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a finished depth_estimation run (missing {', '.join(missing)})")
    return path


def write_fullres(depth_run: Path, out: Path, log=print) -> int:
    """depth/ at the frames' resolution: MoGe-2 detail x MegaSaM's (smoothed) correction; see fullres.py."""
    import shutil
    from depth_estimation.datatypes import frame_indices, frame_name, load_depth, save_depth
    from .fullres import lift_frame
    info = json.loads((out / "megasam.json").read_text(encoding="utf-8"))
    w1, h1 = info["resized_size"]
    (out / "depth").mkdir(exist_ok=True)
    indices = frame_indices(out / "depth_megasam", ".png")
    for n, i in enumerate(indices, 1):
        full = lift_frame(load_depth(depth_run / "depth" / frame_name(i, ".png")),
                          load_depth(out / "depth_megasam" / frame_name(i, ".png")), (h1, w1))
        save_depth(out / "depth" / frame_name(i, ".png"), full)
        if n % 50 == 0 or n == len(indices):
            log(f"full resolution: {n}/{len(indices)} frames")
    shutil.copy2(depth_run / "intrinsics_stable.json", out / "intrinsics_stable.json")  # same camera, full size
    if (depth_run / "mask").is_dir():  # MoGe-valid pixels are exactly the valid ones here too
        shutil.copytree(depth_run / "mask", out / "mask", dirs_exist_ok=True)
    return len(indices)


def finish(depth_run: Path, out: Path) -> dict:
    """Windows side after WSL: full-resolution depth, run.json for the viewer, trajectory summary."""
    from camera_pose.poses import load_trajectory, trajectory_summary, write_tum
    write_fullres(depth_run, out)
    record = json.loads((depth_run / "run.json").read_text(encoding="utf-8"))
    images = (depth_run / record.get("image_dir", "frames")).resolve()
    (out / "run.json").write_text(json.dumps({"image_dir": str(images), "depth_run": str(depth_run.resolve()),
                                              "video": record.get("video", {}), "settings": record.get("settings", {})},
                                             indent=2), encoding="utf-8")
    indices, poses = load_trajectory(out / "poses")
    write_tum(out / "trajectory_tum.txt", indices, poses)
    fps = record.get("video", {}).get("fps")
    summary = json.loads((out / "megasam.json").read_text(encoding="utf-8"))
    summary["trajectory"] = trajectory_summary(indices, poses, fps)
    (out / "megasam.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="video_depth", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="MegaSaM on a depth_estimation run -> consistent depth + poses")
    run.add_argument("depth_run", help="depth_estimation run folder, or its name under runs/depth_estimation")
    run.add_argument("--out", type=Path, help="Default: runs/video_depth/<depth run name>")
    run.add_argument("--opt-focal", action="store_true",
                     help="Let MegaSaM refine the focal length (its demo does; default keeps the AnyCalib focal)")
    run.add_argument("--max-frames", type=int, help="Only the first N frames (quick tries)")
    run.add_argument("--segment-length", type=int, default=200,
                     help="Longer videos are split into overlapping segments of this many frames, run "
                          "separately and joined (GPU memory grows with segment length; 184 frames ~ 11 GB)")
    run.add_argument("--overlap", type=int, default=30, help="Frames shared by neighbouring segments (> 15)")
    full = sub.add_parser("fullres", help="Recompute the full-resolution depth/ of an existing run")
    full.add_argument("run", help="video_depth run folder, or its name under runs/video_depth")
    sub.add_parser("check", help="Check the WSL MegaSaM env")
    cmp_ = sub.add_parser("compare", help="Side-by-side video + background stability: MoGe-2 per frame (as V2D) "
                                          "vs MoGe-2 + MegaSaM (see compare.py)")
    cmp_.add_argument("run", help="video_depth run folder, or its name under runs/video_depth")
    args = parser.parse_args(argv)

    from camera_pose import wsl
    try:
        if args.command == "check":
            import subprocess, os
            script = PROJECT / "tools" / "wsl" / "setup_cuda_env.sh"
            sys.exit(subprocess.call(["wsl.exe", "-d", wsl.DISTRO, "--", "bash", wsl.to_wsl(script), "megasam_check"],
                                     env=dict(os.environ, MSYS_NO_PATHCONV="1")))
        if args.command == "compare":
            from .compare import compare
            out = Path(args.run) if Path(args.run).is_dir() else RUNS / args.run
            record = json.loads((out / "run.json").read_text(encoding="utf-8"))
            depth_run = Path(record["depth_run"])
            from depth_estimation.viewer import ENGLISH_LABELS, find_sources
            # every full-resolution depth of this video (MegaSaM's native low-res output has another size)
            sources = {ENGLISH_LABELS[key]: folder for key, (_, folder) in find_sources(out).items()
                       if key != ("video_depth", "depth_megasam")}
            fps = record.get("video", {}).get("fps") or 30.0
            summary = compare(sources, Path(record["image_dir"]), out, fps=fps)
            print(f"{'':28s}{'scale jitter':>14s}{'pixel wobble':>14s}{'wobble mm':>11s}{'frame jump':>12s}{'coverage':>10s}")
            for name, r in summary["stability"].items():
                print(f"{name:28s}{r['scale_jitter_pct']:>13.2f}%{r['pixel_wobble_pct']:>13.2f}%"
                      f"{r['pixel_wobble_mm']:>11.2f}{r['frame_to_frame_pct']:>11.3f}%{r['coverage'] * 100:>9.0f}%")
            for name in ("compare.mp4", "compare_background.png", "compare.json"):
                print(f"-> {out / name}")
            return
        if args.command == "fullres":
            out = Path(args.run) if Path(args.run).is_dir() else RUNS / args.run
            depth_run = Path(json.loads((out / "run.json").read_text(encoding="utf-8"))["depth_run"])
            n = write_fullres(depth_run, out)
            print(f"{n} full-resolution depth maps -> {out / 'depth'}")
            return
        if args.overlap <= 15 or args.segment_length <= args.overlap:
            raise ValueError("--overlap must exceed 15 (MegaSaM's largest flow step) and be < --segment-length")
        depth_run = resolve_depth_run(args.depth_run)
        out = args.out or RUNS / depth_run.name
        if (out / "depth_megasam").is_dir() and any((out / "depth_megasam").iterdir()):
            raise ValueError(f"{out} already has results; delete it or pass another --out")
        for name in ("megasam_final.pth", "raft-things.pth"):
            if not (MODEL / name).is_file():
                raise FileNotFoundError(f"{MODEL / name} missing; run  wsl -d {wsl.DISTRO} -- bash "
                                        f"{wsl.to_wsl(PROJECT / 'tools/wsl/setup_cuda_env.sh')} megasam")
        out.mkdir(parents=True, exist_ok=True)
        print(f"Depth run: {depth_run}\nOutput: {out}")
        margs = ["--depth-run", wsl.to_wsl(depth_run), "--out", wsl.to_wsl(out),
                 "--weights", wsl.to_wsl(MODEL / "megasam_final.pth"), "--raft", wsl.to_wsl(MODEL / "raft-things.pth")]
        if args.opt_focal:
            margs.append("--opt-focal")
        if args.max_frames:
            margs += ["--max-frames", str(args.max_frames)]
        margs += ["--segment-length", str(args.segment_length), "--overlap", str(args.overlap)]
        code = wsl.run_module("video_depth.megasam", margs, env=ENV, pythonpath=PYTHONPATH)
        if code != 0:
            raise RuntimeError(f"MegaSaM failed in WSL (exit {code})")
        summary = finish(depth_run, out)
        j = summary["moge_scale_vs_consistent"]
        print(f"MoGe per-frame scale vs consistent depth: {j['min']:.3f}-{j['max']:.3f} (std {j['std_percent']}%)\n"
              f"focal fx {summary['focal_prior_fx_fullres']:.1f} -> {summary['focal_final_fx_fullres']:.1f} (full res)\n"
              f"camera path {summary['trajectory'].get('path_length_m')} m, "
              f"max rotation {summary['trajectory'].get('max_rotation_from_first_deg')} deg\n"
              f"Done: {out}\nView: depth_estimation view {out}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
