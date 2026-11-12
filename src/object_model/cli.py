"""object_model: the hand-held object's 3D model, from the frames before the hand touches it.

    object_model run demo01 --prompt "a light blue box"     # -> runs\\object_model\\demo01\\
    object_model view demo01                                # viser, http://localhost:8080

Needs `depth_estimation run` (frames, intrinsics) and `video_depth run` (consistent depth, camera poses).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
RUNS = PROJECT / "runs"


def _video(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p.resolve()
    for ext in (".mp4", ".mov", ".MOV", ".MP4"):
        q = PROJECT / "data" / "videos" / (name + ext)
        if q.exists():
            return q
    raise FileNotFoundError(f"video {name!r} not found in data/videos")


def paths(name: str) -> dict:
    return {"depth_run": RUNS / "depth_estimation" / name, "video_depth": RUNS / "video_depth" / name,
            "out": RUNS / "object_model" / name}


def run(name: str, prompt: str, detect_frame: int, window: int, seed: int, log=print,
        min_iou: float | None = None, max_point_mm: float | None = None) -> dict:
    import cv2

    from video_depth.poses import load_trajectory
    from depth_estimation.datatypes import CameraIntrinsics, frame_indices

    from . import model, pointmap, reference, segment

    min_iou = model.MIN_IOU if min_iou is None else min_iou
    max_point_mm = model.MAX_POINT_TO_SURFACE_MM if max_point_mm is None else max_point_mm
    p = paths(name)
    out, frames_dir = p["out"], p["depth_run"] / "frames"
    if not (p["video_depth"] / "depth").is_dir():
        raise FileNotFoundError(f"run `video_depth run {name}` first ({p['video_depth']})")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    seg = segment.segment(_video(name), frames_dir, out, prompt, detect_frame, log)

    indices = frame_indices(p["video_depth"] / "depth", ".png")
    rows = reference.frame_stats(out, frames_dir, indices, detect_frame)
    ref = reference.choose(rows, out, detect_frame, window=window)
    r = ref["reference_frame"]
    log(f"[reference] contact at frame {ref['contact_frame']}; reference frame {r}; "
        f"window {ref['window'][0]}-{ref['window'][-1]} ({len(ref['window'])} frames)")

    k = CameraIntrinsics.load(p["video_depth"] / "intrinsics_stable.json")
    idx, mats = load_trajectory(p["video_depth"] / "poses")
    poses = dict(zip(idx, mats))
    pm, pm_stats = pointmap.fuse(p["video_depth"] / "depth", out, poses, k, r, ref["window"])
    np.save(out / "pointmap_reference.npy", pm)
    k.save(out / "intrinsics.json")
    log(f"[pointmap] {pm_stats['pixels']} px from {pm_stats['frames']} frames; temporal spread median "
        f"{pm_stats['temporal_spread_mm_median']:.2f} mm")

    mask_path = out / "masks" / str(segment.OBJECT_ID) / f"{r:06d}.png"
    if not (out / "sam3d" / "mesh.glb").exists():
        log("[sam3d] generating the mesh (first load of the model ~3 min)")
        model.run_sam3d(frames_dir / f"{r:06d}.png", mask_path, out / "pointmap_reference.npy",
                        out / "intrinsics.json", out / "sam3d", seed=seed, log=log)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    mesh, T_sam3d, sam3d_scale = model.sam3d_placement(out / "sam3d")

    def check(tag, m, T):
        fit = model.fit_checks(m, T, k, mask, pm)
        ext = model.obb_extents_m(m)
        log(f"[check {tag}] IoU {fit['silhouette_iou']}; depth diff median {fit['depth_diff_mm']['median']} mm; "
            f"point->surface median {fit['point_to_surface_mm']['median']} mm, p90 "
            f"{fit['point_to_surface_mm']['p90']} mm; size {' x '.join(f'{e * 100:.1f}' for e in ext)} cm")
        return fit, ext

    fit_sam3d, ext_sam3d = check("SAM3D", mesh, T_sam3d)
    scale, T, refine_info = model.refine(mesh, T_sam3d, k, mask, pm, log=log)
    mesh.apply_scale(scale)
    fit, extents = check("refined", mesh, T)
    model.write(out, mesh, T, poses[r], {"reference_frame": r, "sam3d_scale": sam3d_scale,
                                         "refine_scale": scale})

    record = {"video": name, "prompt": prompt, "segment": seg, **ref, "pointmap": pm_stats,
              "mesh": {"obb_extents_m": extents, "obb_extents_m_sam3d": ext_sam3d, "vertices": len(mesh.vertices),
                       "faces": len(mesh.faces), "seed": seed},
              "fit_sam3d": fit_sam3d, "refine": refine_info, "fit": fit,
              "quality": model.quality(fit, min_iou, max_point_mm),
              "frames": rows, "seconds": round(time.time() - t0, 1),
              "sources": {k_: str(v) for k_, v in p.items()}}
    (out / "model.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    q = record["quality"]
    if q["passed"]:
        log(f"[quality] 通过：IoU {fit['silhouette_iou']:.3f} ≥ {min_iou}，点到表面中位数 "
            f"{fit['point_to_surface_mm']['median']:.2f} mm ≤ {max_point_mm} mm")
    else:
        bar = "!" * 70
        log(f"{bar}\n[quality] 不合格：" + "；".join(q["failures"]) +
            f"\n  结果仍已写出，请用 `object_model view {name}` 检查（网格选“单色半透明”最容易看出哪里没对上）\n{bar}")
    log(f"Done: {out}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(prog="object_model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="masks -> reference frame -> fused point map -> SAM3D mesh in the world")
    r.add_argument("name", help="video name in data/videos (runs/depth_estimation/<name>, runs/video_depth/<name>)")
    r.add_argument("--prompt", required=True, help="Grounding DINO text, e.g. 'a light blue box'")
    r.add_argument("--detect-frame", type=int, default=0, help="frame where the object lies untouched (default 0)")
    r.add_argument("--window", type=int, default=20, help="static frames fused into the point map (default 20)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--min-iou", type=float, default=None,
                   help="quality check: silhouette IoU after the fit must reach this (default 0.95)")
    r.add_argument("--max-point-mm", type=float, default=None,
                   help="quality check: median fused-point to mesh-surface distance at most this (default 3 mm)")
    v = sub.add_parser("view", help="mesh + point clouds in the browser (viser)")
    v.add_argument("name")
    v.add_argument("--frames", help="frame range a-b to load (default: 0 to the reference frame)")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--host", default="127.0.0.1")
    t = sub.add_parser("texture", help="the texture atlas and the textured mesh in the browser (viser)")
    t.add_argument("name")
    t.add_argument("--port", type=int, default=8080)
    t.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    if a.command == "texture":
        from .texture_view import serve as serve_texture
        serve_texture(paths(a.name), a.host, a.port)
    elif a.command == "run":
        record = run(a.name, a.prompt, a.detect_frame, a.window, a.seed,
                     min_iou=a.min_iou, max_point_mm=a.max_point_mm)
        if not record["quality"]["passed"]:
            raise SystemExit(2)  # outputs are written; the non-zero exit tells scripts the fit failed the check
    else:
        from .viewer import serve
        serve(paths(a.name), a.frames, a.host, a.port)


if __name__ == "__main__":
    main()
