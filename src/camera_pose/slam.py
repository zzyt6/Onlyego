"""DROID-SLAM on a depth_estimation run, scale-aligned to its MoGe depth. Runs inside WSL.

From V2D v2d_droid_slam/lib/_slam.py + video_to_slam.py. What DROID-SLAM does:
  - a recurrent network predicts dense optical flow (with confidence) between frame pairs;
  - a differentiable dense bundle adjustment layer turns that flow into camera poses and
    per-pixel inverse depth for keyframes, over a sliding window (frontend), then globally (backend);
  - terminate() runs a final global BA and fills in poses for non-keyframes.
A single monocular video only determines geometry up to one global scale. V2D fixes that scale
afterwards: per keyframe, median(MoGe depth / DROID depth); the median over keyframes is the scale
applied to every camera translation (_solve_global_scale). Optionally MoGe depth can instead be
fed into BA as a per-pixel prior (prior_depth) -- V2D's ego pipeline uses the post-hoc route only.

Changes from V2D: frames, the fixed intrinsics (intrinsics_stable.json) and depth come from a
depth_estimation run folder; per-keyframe scales are written out, not only their median;
camera poses get a "matrix" field; outputs live in one folder with a run.json.

    python -m camera_pose.slam --depth-run <dir> --out <dir> --weights droid.pth [options]
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from depth_estimation.datatypes import CameraIntrinsics, frame_indices, frame_name, load_depth, save_depth

from .poses import save_pose


@dataclass
class Frame:
    index: int
    image_bgr: np.ndarray
    prior_depth: np.ndarray | None = None
    align_depth: np.ndarray | None = None


def resize_for_droid(image_bgr, k: CameraIntrinsics, size, prior_depth=None):
    """Resize to DROID's network input and scale intrinsics to match (V2D _resize_for_droid).
    Image stays BGR uint8 (1,3,H,W): DROID's motion filter reorders channels itself.
    Invalid prior depth is zeroed -- DROID reads 0 as "no prior at this pixel"."""
    import torch
    h, w = size
    src_h, src_w = image_bgr.shape[:2]
    image = torch.from_numpy(cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None]
    sx, sy = w / src_w, h / src_h
    intr = torch.tensor([k.fx * sx, k.fy * sy, k.cx * sx, k.cy * sy], dtype=torch.float32)
    depth = None
    if prior_depth is not None:
        d = prior_depth.astype(np.float32).copy()
        d[~np.isfinite(d) | (d <= 0)] = 0.0
        depth = torch.from_numpy(cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST))
    return image, intr, depth


def keyframe_disparities(droid, upsample: bool) -> np.ndarray:
    video = droid.video
    disps = video.disps_up if upsample and hasattr(video, "disps_up") else video.disps
    return disps.detach().cpu().numpy()


def solve_scale(droid, frames: dict[int, Frame], upsample: bool) -> tuple[float, list[dict]]:
    """Global metric scale = median over keyframes of median(reference depth / DROID depth)."""
    n_kf = int(droid.video.counter.value)
    disps = keyframe_disparities(droid, upsample)
    stamps = droid.video.tstamp.detach().cpu().numpy().astype(np.int64)
    per_kf = []
    for kf in range(n_kf):
        frame = frames.get(int(stamps[kf]))
        if frame is None or frame.align_depth is None:
            continue
        ref, disp = frame.align_depth, disps[kf]
        if disp.shape != ref.shape:
            disp = cv2.resize(disp, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = (disp > 1e-3) & np.isfinite(ref) & (ref > 1e-3)
        if valid.sum() < 64:
            continue
        per_kf.append({"keyframe": kf, "frame": int(stamps[kf]), "pixels": int(valid.sum()),
                       "scale": float(np.median(ref[valid] * disp[valid]))})  # ref / (1/disp)
    if not per_kf:
        return 1.0, per_kf
    return float(np.median([r["scale"] for r in per_kf])), per_kf


def write_pointcloud(droid, scale: float, path: Path, min_views: int = 2, filter_thresh: float = 0.005) -> int:
    """Keyframe points that are depth-consistent across >= min_views keyframes (V2D _write_pointcloud_ply)."""
    import droid_backends
    import torch
    from lietorch import SE3

    from depth_estimation.inspection import write_ply

    video = droid.video
    n_kf = int(video.counter.value)
    if n_kf == 0:
        return 0
    dirty = torch.arange(n_kf, device=video.poses.device)
    thresh = filter_thresh * torch.ones(n_kf, device=video.poses.device)
    count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty, thresh)
    disps = video.disps[:n_kf].detach()
    cam_to_world = SE3(video.poses[:n_kf].detach()).inv()
    points = (droid_backends.iproj(cam_to_world.data, disps, video.intrinsics[0].detach()) * scale).cpu().numpy()
    colors = video.images[:n_kf, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1).cpu().numpy()
    d = disps.cpu().numpy()
    mask = (count.cpu().numpy() >= min_views) & (d > 0.5 * d.reshape(n_kf, -1).mean(1)[:, None, None]) & (d > 1e-3)
    if not mask.any():
        return 0
    write_ply(path, points[mask].astype(np.float32), colors[mask].astype(np.uint8))
    return int(mask.sum())


def run(depth_run: Path, out: Path, weights: Path, *, stride: int = 1, use_depth_prior: bool = False,
        depth_subdir: str = "depth", image_size=(384, 512), buffer: int = 512, upsample: bool = False,
        keyframe_thresh: float = 4.0, frontend_thresh: float = 16.0, frontend_window: int = 25,
        backend_thresh: float = 22.0, filter_thresh: float = 2.4, warmup: int = 8,
        save_keyframe_depth: bool = False, pointcloud: bool = True) -> dict:
    import torch
    from droid import Droid

    record = json.loads((depth_run / "run.json").read_text(encoding="utf-8"))
    images_dir = depth_run / record.get("image_dir", "frames")
    k = CameraIntrinsics.load(depth_run / "intrinsics_stable.json")
    depth_dir = depth_run / depth_subdir
    indices = frame_indices(images_dir, ".png")[::stride]
    if len(indices) < warmup:
        raise ValueError(f"Only {len(indices)} frames; DROID-SLAM needs at least warmup={warmup}")

    def frames():
        for i in indices:
            image = cv2.imread(str(images_dir / frame_name(i, ".png")))
            dpath = depth_dir / frame_name(i, ".png")
            depth = load_depth(dpath) if dpath.exists() else None
            yield Frame(i, image, depth if use_depth_prior else None, depth)

    args = SimpleNamespace(
        weights=str(weights), image_size=list(image_size), buffer=buffer, beta=0.3, filter_thresh=filter_thresh,
        warmup=warmup, keyframe_thresh=keyframe_thresh, frontend_thresh=frontend_thresh,
        frontend_window=frontend_window, frontend_radius=2, frontend_nms=1, backend_thresh=backend_thresh,
        backend_radius=2, backend_nms=3, upsample=upsample, disable_vis=True, stereo=False)
    torch.multiprocessing.set_start_method("spawn", force=True)
    t0 = time.time()
    droid, by_index = None, {}
    for n, f in enumerate(frames(), 1):
        droid = droid or Droid(args)
        image, intr, depth = resize_for_droid(f.image_bgr, k, image_size, f.prior_depth)
        droid.track(f.index, image, depth=depth, intrinsics=intr)
        f.image_bgr = None  # keep memory flat; only depth is needed later
        by_index[f.index] = f
        if n % 50 == 0 or n == len(indices):
            print(f"[slam] tracked {n}/{len(indices)} frames, {int(droid.video.counter.value)} keyframes", flush=True)
    track_seconds = time.time() - t0
    n_kf = int(droid.video.counter.value)
    if n_kf < 2:
        # Too little motion for a new keyframe (mean flow < keyframe_thresh px): DROID's backend has no
        # frame pairs and upstream crashes. Report a static camera instead of failing.
        print(f"[slam] only {n_kf} keyframe: the camera did not move enough for SLAM; writing identity poses", flush=True)
        poses_dir = out / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        for i in indices:
            save_pose(poses_dir / frame_name(i, ".json"), np.eye(4))
        result = {"frames": len(indices), "keyframes": n_kf, "scale": None, "scale_per_keyframe": [],
                  "static_camera": True, "pointcloud_points": 0, "track_seconds": round(track_seconds, 1),
                  "total_seconds": round(time.time() - t0, 1),
                  "intrinsics": json.loads((depth_run / "intrinsics_stable.json").read_text())}
        (out / "slam.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def replay():
        for f in frames():
            image, intr, _ = resize_for_droid(f.image_bgr, k, image_size)
            yield f.index, image, intr

    traj = droid.terminate(replay())  # (N, 7) camera-to-world [tx ty tz qx qy qz qw], unknown scale
    scale, per_kf = solve_scale(droid, by_index, upsample)
    print(f"[slam] {int(droid.video.counter.value)} keyframes; metric scale {scale:.5f} "
          f"from {len(per_kf)} keyframes", flush=True)

    from scipy.spatial.transform import Rotation
    poses_dir = out / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    for row, i in zip(traj, indices):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
        M[:3, 3] = row[:3] * scale
        save_pose(poses_dir / frame_name(i, ".json"), M)

    if save_keyframe_depth:
        kd = out / "keyframe_depth"
        kd.mkdir(exist_ok=True)
        disps = keyframe_disparities(droid, upsample)
        stamps = droid.video.tstamp.detach().cpu().numpy().astype(np.int64)
        for kf in range(int(droid.video.counter.value)):
            disp = cv2.resize(disps[kf], (k.width, k.height), interpolation=cv2.INTER_LINEAR)
            save_depth(kd / frame_name(int(stamps[kf]), ".png"), scale / np.maximum(disp, 1e-6))
    points = write_pointcloud(droid, scale, out / "pointcloud.ply") if pointcloud else 0

    result = {"frames": len(indices), "keyframes": int(droid.video.counter.value), "scale": scale,
              "scale_per_keyframe": per_kf, "pointcloud_points": points,
              "track_seconds": round(track_seconds, 1), "total_seconds": round(time.time() - t0, 1),
              "intrinsics": json.loads((depth_run / "intrinsics_stable.json").read_text())}
    (out / "slam.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--depth-prior", action="store_true", help="Feed MoGe depth into bundle adjustment too")
    p.add_argument("--depth-subdir", default="depth")
    p.add_argument("--image-size", type=int, nargs=2, default=(384, 512), metavar=("H", "W"))
    p.add_argument("--buffer", type=int, default=512, help="Max keyframes held on the GPU")
    p.add_argument("--keyframe-thresh", type=float, default=4.0, help="Mean flow (px) needed for a new keyframe")
    p.add_argument("--save-keyframe-depth", action="store_true")
    p.add_argument("--no-pointcloud", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run(a.depth_run, a.out, a.weights, stride=a.stride, use_depth_prior=a.depth_prior, depth_subdir=a.depth_subdir,
        image_size=tuple(a.image_size), buffer=a.buffer, keyframe_thresh=a.keyframe_thresh,
        save_keyframe_depth=a.save_keyframe_depth, pointcloud=not a.no_pointcloud)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
