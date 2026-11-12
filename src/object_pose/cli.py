"""object_pose: the hand-held object's pose in every frame.

    object_pose run demo01      # FoundationPose tracking -> runs\\object_pose\\demo01\\
    object_pose view demo01     # the moving mesh with the point clouds in the browser

Needs `object_model run` (mesh, masks, reference pose), Codex's texture (`four_stage_view generate-texture <job>
--object <video>` -> mesh_textured.glb) and `video_depth run` (depth, camera poses).

FoundationPose (WSL env hoi-fp, fp_track.py) starts at object_model's reference frame from the pose object_model
fitted there (silhouette IoU 0.98) and tracks forward and backward with RGB + masked depth. When tracking is lost
(visible IoU against the SAM2 mask, hand pixels removed, < 0.8) and the object is fully in view, it registers the
frame again and picks among all of FoundationPose's hypotheses: the silhouette filters, FoundationPose's score
(textured render vs photo) decides. Then every frame is scored and drawn over the video.
"""

import argparse
import json
import time

import numpy as np

from object_model.cli import RUNS, paths as om_paths

FP_NOISE = ("[make_crop_data_batch()]", "[track_one()]", "[register()]", "[guess_translation()]",
            "[find_best_among_pairs()]", "[predict()]", "[__init__()]", "[reset_object()]", "[make_rotation_grid()]",
            "[sample_views_icosphere()]", "[generate_random_pose_hypo()]", "[compute_add_err_matrix()]")
MESH = "mesh_textured.glb"  # object_model's mesh with Codex's texture (four_stage_view generate-texture)


def _quiet(log):
    def f(line):
        if not line.startswith(FP_NOISE):
            log(line)
    return f


def out_dir(name: str):
    return RUNS / "object_pose" / name


def track(name: str, log=print, videos: bool = True) -> dict:
    from video_depth.poses import load_trajectory
    from depth_estimation.datatypes import CameraIntrinsics
    from object_model import model, wsl

    from . import evaluate
    from .overlay import render_video

    p = om_paths(name)
    om, vd, frames_dir = p["out"], p["video_depth"], p["depth_run"] / "frames"
    if not (om / "object_pose.json").exists():
        raise FileNotFoundError(f"run `object_model run {name}` first ({om})")
    if not (om / MESH).exists():
        raise FileNotFoundError(f"{om / MESH} not found: run `four_stage_view generate-texture <job> --object {name}`")
    out = out_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    rec = json.loads((om / "model.json").read_text(encoding="utf-8"))
    ref, contact = rec["reference_frame"], rec["contact_frame"]
    fit = json.loads((om / "object_pose.json").read_text())
    k = CameraIntrinsics.load(vd / "intrinsics_stable.json")
    idx, mats = load_trajectory(vd / "poses")
    c2w = dict(zip(idx, mats))
    mesh = model.load_mesh(om / MESH)
    fps = json.loads((vd / "run.json").read_text()).get("video", {}).get("fps", 30.0)

    if not (out / "track.json").exists():
        log(f"[track] FoundationPose from object_model's pose at frame {ref}, forward and backward")
        t0 = time.time()
        wsl.run("foundation_pose", "object_pose.fp_track", [
            "--frames", wsl.path(frames_dir), "--depth", wsl.path(vd / "depth"),
            "--masks", wsl.path(om / "masks" / "1"),
            "--hand-masks", wsl.path(om / "masks" / "2"), wsl.path(om / "masks" / "3"),
            "--intrinsics", wsl.path(vd / "intrinsics_stable.json"), "--mesh", wsl.path(om / MESH),
            "--weights", wsl.weights("foundation_pose"), "--reference", str(ref),
            "--start-pose", wsl.path(om / "object_pose.json"), "--out", wsl.path(out / "poses"),
            "--info", wsl.path(out / "track.json")], log=_quiet(log))
        log(f"[track] {time.time() - t0:.0f} s")
    track_info = json.loads((out / "track.json").read_text())
    rereg = {int(i) for i, f in track_info["frames"].items() if f["reregistered"]}
    poses = evaluate.load_poses(out / "poses")
    log(f"[evaluate] {len(poses)} frames")
    ev = evaluate.evaluate(poses, mesh, k, c2w, om, vd / "depth", contact, np.array(fit["object_to_world"]))
    ev["reregistered"] = sorted(rereg)
    (out / "metrics.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False), encoding="utf-8")
    if videos:
        render_video(out / "overlay.mp4", "FoundationPose 从拟合位姿出发", poses, ev["frames"], mesh, k, frames_dir,
                     om, contact, rereg, fps)
    s = {**ev["summary"], "reregistered": ev["reregistered"]}
    (out / "summary.json").write_text(json.dumps({"reference_frame": ref, "contact_frame": contact, **s},
                                                 indent=2, ensure_ascii=False), encoding="utf-8")
    st, held = s["static (before contact)"], s["held (from contact)"]
    log(f"static: IoU {st['iou_visible']['median']:.3f}, drift {st['drift_mm']['median']:.1f} mm "
        f"(max {st['drift_mm']['max']:.1f}) | held: visible IoU median {held['iou_visible']['median']:.3f} "
        f"p10 {held['iou_visible']['p10']:.3f}, <0.9 in {held['iou_visible_below_0.9']}/{held['frames']} frames, "
        f"depth diff {held['depth_mm']['median']:+.1f} mm, jitter {held['accel_mm']['median']:.1f} mm")
    log(f"Done: {out}")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="object_pose", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="FoundationPose tracking from object_model's pose, scored and overlaid")
    r.add_argument("name")
    r.add_argument("--no-video", action="store_true")
    v = sub.add_parser("view", help="the moving mesh + point clouds in the browser (viser)")
    v.add_argument("name")
    v.add_argument("--frames", help="frame range a-b (default: all)")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    if a.command == "run":
        track(a.name, videos=not a.no_video)
    else:
        from .viewer import serve
        serve(a.name, a.frames, a.host, a.port)


if __name__ == "__main__":
    main()
