"""MegaSaM on a depth_estimation run, with MoGe-2 as the mono-depth prior. Runs inside WSL (env zyego-megasam).

MegaSaM (Li et al., CVPR 2025; github.com/mega-sam/mega-sam @ a27b4e6) turns per-frame monocular depth into
camera poses plus depth that is consistent across frames, in three stages that this file runs in one process,
following the repo's demo scripts line by line:

  1. camera tracking (camera_tracking_scripts/test_demo.py): a DROID-SLAM variant whose bundle adjustment
     uses the mono depth as a prior and estimates a per-pixel motion probability (moving hands/objects);
     a final global BA over all frames gives every frame's pose.
  2. optical flow (cvd_opt/preprocess_flow.py): RAFT between frame pairs 1, 2, 4, 8, 15 apart, forward and
     backward, with a left-right consistency mask.
  3. consistent video depth (cvd_opt/cvd_opt.py): with poses fixed, first a scale and shift per frame, then
     every pixel's disparity, are optimised so that depth warped along the flow agrees with depth
     reprojected through the poses, while staying close to the prior's shape (scale-invariant, gradient
     and normal terms). Step one is exactly what removes MoGe's per-frame scale jitter.

The MegaSaM repo feeds two mono inputs: Depth-Anything (relative disparity) and UniDepth (metric depth +
FoV). As in MINT's EgoPipeline (GeoCalib -> MoGe-2 -> MegaSaM), MoGe-2 replaces both: its disparity is the
"mono disparity", its depth the "metric depth", and the FoV is the one MoGe ran with (our AnyCalib prior).
MegaSaM works in a normalised scale; at the end one global factor -- median over frames of
median(MoGe depth / MegaSaM depth) -- returns depths and camera translations to metres.

Changes from the MegaSaM demo, all marked "ours" below: inputs come from a depth_estimation run; the focal
length stays fixed unless --opt-focal (the demo always lets terminate() optimise it when well conditioned;
we keep the calibrated focal MoGe's depth was computed with); flows stay in memory instead of cache_flow/;
the metric rescale; long videos run as overlapping segments joined by stitch.py; outputs in this project's
formats (native resolution here; the full-resolution depth/ is made on the Windows side by fullres.py).

    python -m video_depth.megasam --depth-run <dir> --out <dir> --weights megasam_final.pth --raft raft-things.pth
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from camera_pose.poses import save_pose
from depth_estimation.datatypes import CameraIntrinsics, frame_indices, frame_name, load_depth, save_depth


def tracking_size(h0: int, w0: int) -> tuple[int, int, int, int]:
    """MegaSaM resizes to ~384*512 pixels keeping the aspect, then crops to a multiple of 8.
    Returns (resized h, w, cropped h, w)."""
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    return h1, w1, h1 - h1 % 8, w1 - w1 % 8


def mono_inputs(depth_run: Path, indices: list[int], h1: int, w1: int):
    """MoGe-2 depth as both MegaSaM mono inputs, at tracking resolution (nearest-exact, like image_stream).
    Returns (mono disparity list, metric depth list); invalid MoGe pixels -> disparity 0 (like sky)."""
    disps, metrics = [], []
    for i in indices:
        d = load_depth(depth_run / "depth" / frame_name(i, ".png"))
        d = cv2.resize(d, (w1, h1), interpolation=cv2.INTER_NEAREST_EXACT)
        with np.errstate(divide="ignore"):
            disp = np.where(np.isfinite(d) & (d > 0), 1.0 / d, 0.0).astype(np.float32)
        disps.append(disp)
        metrics.append(np.where(np.isfinite(d), d, np.inf).astype(np.float32))
    return disps, metrics


def align_mono_to_metric(mono_disp_list, metric_depth_list):
    """test_demo.py: per frame, scale/shift of mono disparity onto metric disparity (median-based); the frame
    whose scale*shift is the median supplies one alignment for the whole video; the normaliser puts the
    98th-percentile disparity at 2. With MoGe in both roles scale ~ 1 and shift ~ 0."""
    scales, shifts = [], []
    for da_disp, metric_depth in zip(mono_disp_list, metric_depth_list):
        gt_disp = 1.0 / (metric_depth + 1e-8)
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)  # "avoid some bug from UniDepth" (kept as is)
        gt_disp[valid_mask] = 1e-2
        gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
        da_disp_ms = da_disp - np.median(da_disp) + 1e-8
        scales.append(np.median(gt_disp_ms / da_disp_ms))
        shifts.append(np.median(gt_disp - scales[-1] * da_disp))
    ss_product = np.array(scales) * np.array(shifts)
    med_idx = int(np.argmin(np.abs(ss_product - np.median(ss_product))))
    align_scale, align_shift = scales[med_idx], shifts[med_idx]
    normalize_scale = np.percentile(align_scale * np.array(mono_disp_list) + align_shift, 98) / 2.0
    return float(align_scale), float(align_shift), float(normalize_scale)


def image_stream(image_paths, mono_disp_list, aligns, K):
    """test_demo.py image_stream (use_depth=True): BGR uint8 image, prior depth, intrinsics at tracking size."""
    import torch
    import torch.nn.functional as F

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for t, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        with np.errstate(divide="ignore"):
            depth = np.clip(1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp_list[t] + aligns[1])), 1e-4, 1e4)
        depth[depth < 1e-2] = 0.0
        h0, w0, _ = image.shape
        h1, w1, hc, wc = tracking_size(h0, w0)
        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)[:hc, :wc]
        image = torch.as_tensor(image).permute(2, 0, 1)
        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()[:hc, :wc]
        mask = torch.ones_like(depth)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0
        yield t, image[None], depth, intrinsics, mask


def track(image_paths, mono_disp_list, aligns, K, weights: Path, opt_focal: bool, log=print):
    """Stage 1. Returns (poses w2c (T,7), sensor depth list, intrinsics at tracking size (4,), motion prob)."""
    import torch
    from droid import Droid
    from argparse import Namespace

    args = Namespace(  # test_demo.py defaults
        weights=str(weights), buffer=max(1024, len(image_paths) + 8), image_size=None, disable_vis=True,
        beta=0.3, filter_thresh=2.0, warmup=8, keyframe_thresh=2.0, frontend_thresh=12.0, frontend_window=25,
        frontend_radius=2, frontend_nms=1, stereo=False, depth=False, upsample=False, scene_name="video",
        backend_thresh=16.0, backend_radius=2, backend_nms=3)
    rgb_list, sensor_depth_list = [], []
    droid = None
    for t, image, depth, intrinsics, mask in image_stream(image_paths, mono_disp_list, aligns, K):
        rgb_list.append(image[0])
        sensor_depth_list.append(depth)
        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)
        if (t + 1) % 50 == 0:
            log(f"[megasam] tracked {t + 1}/{len(image_paths)} frames, {int(droid.video.counter.value)} keyframes")
    droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)
    log(f"[megasam] {int(droid.video.counter.value)} keyframes; global bundle adjustment"
        f"{' (focal optimised if well conditioned)' if opt_focal else ' (focal fixed)'}")
    traj, _depth_est, motion_prob = droid.terminate(
        image_stream(image_paths, mono_disp_list, aligns, K), _opt_intr=opt_focal, full_ba=True, scene_name="video")
    n = traj.shape[0]
    if n != len(image_paths):
        raise RuntimeError(f"MegaSaM returned {n} poses for {len(image_paths)} frames")
    intrinsics = (droid.video.intrinsics[:n].cpu().numpy() * 8.0)[0]  # stored at 1/8 resolution
    images = np.stack([im.numpy() for im in rgb_list[:n]])           # (T,3,h,w) BGR uint8
    disps = 1.0 / (np.stack([d.numpy() for d in sensor_depth_list[:n]]) + 1e-6)  # save_full_reconstruction
    del droid
    torch.cuda.empty_cache()
    return traj, images, disps, intrinsics, motion_prob


def flows(image_paths, raft_weights: Path, log=print):
    """Stage 2, cvd_opt/preprocess_flow.py (default bilinear resize, RGB, mixed precision as in the demo)."""
    import torch
    from argparse import Namespace
    from raft import RAFT
    from core.utils.utils import InputPadder
    from preprocess_flow import resize_flow, warp_flow

    model = torch.nn.DataParallel(RAFT(Namespace(small=False, mixed_precision=True, alternate_corr=False,
                                                 dropout=0)))
    model.load_state_dict(torch.load(str(raft_weights), map_location="cuda"))
    flow_model = model.module.cuda().eval()

    img_data = []
    for path in image_paths:
        image = cv2.imread(str(path))[..., ::-1]
        h0, w0, _ = image.shape
        h1, w1, hc, wc = tracking_size(h0, w0)
        img_data.append(cv2.resize(image, (w1, h1))[:hc, :wc].transpose(2, 0, 1))
    img_data = np.array(img_data)

    ii, jj, flows_up, masks_up = [], [], [], []
    low_fwd, low_bwd = {}, {}
    for step in [1, 2, 4, 8, 15]:
        for i in range(max(0, -step), img_data.shape[0] - max(0, step)):
            image1 = torch.as_tensor(np.ascontiguousarray(img_data[i:i + 1])).float().cuda()
            image2 = torch.as_tensor(np.ascontiguousarray(img_data[i + step:i + step + 1])).float().cuda()
            ii.append(i)
            jj.append(i + step)
            with torch.no_grad():
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_init = None
                if abs(step) > 1:
                    flow_init = torch.as_tensor(np.ascontiguousarray(np.stack([low_fwd[i], low_bwd[i + step]])))
                    flow_init = flow_init.float().cuda().permute(0, 3, 1, 2)
                flow_low, flow_up, _ = flow_model(torch.cat([image1, image2]), torch.cat([image2, image1]),
                                                  iters=22, test_mode=True, flow_init=flow_init)
                fwd_low = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                bwd_low = flow_low[1].cpu().numpy().transpose(1, 2, 0)
                h2, w2 = flow_up.shape[-2] // 2, flow_up.shape[-1] // 2
                fwd_up = resize_flow(flow_up[0].cpu().numpy().transpose(1, 2, 0), h2, w2)
                bwd_up = resize_flow(flow_up[1].cpu().numpy().transpose(1, 2, 0), h2, w2)
                lr_error = np.linalg.norm(fwd_up + warp_flow(bwd_up, fwd_up), axis=-1)
                low_bwd[i + step], low_fwd[i] = bwd_low, fwd_low
                flows_up.append(fwd_up)
                masks_up.append(lr_error < 1.0)
        log(f"[megasam] flow: step {step} done, {len(ii)} pairs")
    del flow_model, model
    torch.cuda.empty_cache()
    return (np.float16(np.array(flows_up).transpose(0, 3, 1, 2)), np.array(masks_up)[:, None],
            np.stack([ii, jj]))


def consistent_depth(images, disps, intrinsics, poses, motion_prob, flow_data, w_grad=2.0, w_normal=5.0,
                     log=print):
    """Stage 3, cvd_opt/cvd_opt.py main (demo weights w_grad 2, w_normal 5). Returns (depth (T,h,w), K, c2w)."""
    import torch
    from lietorch import SE3
    import cvd_opt as cvd

    flows_np, flow_masks_np, iijj_np = flow_data
    img_data = images[:, ::-1]
    disp_data = disps + 1e-6
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (float(v) for v in intrinsics)
    K = K.float().cuda()
    poses_th = torch.as_tensor(poses, device="cpu").float().cuda()
    flows_t = torch.from_numpy(np.ascontiguousarray(flows_np)).float().cuda()
    flow_masks = torch.from_numpy(np.ascontiguousarray(np.float32(flow_masks_np))).float().cuda()
    iijj = torch.from_numpy(np.ascontiguousarray(iijj_np)).float().cuda()
    ii, jj = iijj[0].long(), iijj[1].long()
    img_data_pt = torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
    init_disp = torch.from_numpy(disp_data).float().cuda()
    disp_t = torch.from_numpy(disp_data).float().cuda()
    R = cvd.RESIZE_FACTOR
    init_disp = torch.nn.functional.interpolate(init_disp.unsqueeze(1), scale_factor=(R, R), mode="bilinear").squeeze(1)
    disp_t = torch.nn.functional.interpolate(disp_t.unsqueeze(1), scale_factor=(R, R), mode="bilinear").squeeze(1)
    fg_alpha = cvd.sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
    fg_alpha = fg_alpha.squeeze(1).float() + 0.2
    cvd_prob = torch.nn.functional.interpolate(torch.from_numpy(motion_prob).unsqueeze(1).cuda(),
                                               scale_factor=(4, 4), mode="bilinear")
    cvd_prob[cvd_prob > 0.5] = 0.5
    cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)
    K_o = K.clone()
    K[0:2, ...] *= R
    K_inv = torch.linalg.inv(K)
    uncertainty = cvd_prob
    compute_normals = [cvd.NormalGenerator(disp_t.shape[-2], disp_t.shape[-1])]
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)

    # (a) per-frame scale and shift
    log_scale_ = torch.zeros(init_disp.shape[0], device=disp_t.device, requires_grad=True)
    shift_ = torch.zeros(init_disp.shape[0], device=disp_t.device, requires_grad=True)
    uncertainty.requires_grad = True
    optim = torch.optim.Adam([{"params": log_scale_, "lr": 1e-2}, {"params": shift_, "lr": 1e-2},
                              {"params": uncertainty, "lr": 1e-2}])
    for i in range(100):
        optim.zero_grad()
        cam_c2w = SE3(poses_th).inv().matrix()
        scale_ = torch.exp(log_scale_)
        loss = cvd.consistency_loss(cam_c2w, K, K_inv,
                                    torch.clamp(disp_t * scale_[..., None, None] + shift_[..., None, None], 1e-3, 1e3),
                                    init_disp, torch.clamp(uncertainty, 1e-4, 1e3), flows_t, flow_masks, ii, jj,
                                    compute_normals, fg_alpha)
        loss.backward()
        for p in (uncertainty, log_scale_, shift_):
            p.grad = torch.nan_to_num(p.grad, nan=0.0)
        optim.step()
    log(f"[megasam] consistent depth (a) per-frame scale/shift: loss {loss.item():.4f}; "
        f"scale range {torch.exp(log_scale_).min().item():.3f}-{torch.exp(log_scale_).max().item():.3f}")
    frame_scales = torch.exp(log_scale_).detach().cpu().numpy()

    # (b) per-pixel disparity
    disp_t = (disp_t * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = (init_disp * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)
    disp_t.requires_grad = True
    uncertainty.requires_grad = True
    optim = torch.optim.Adam([{"params": disp_t, "lr": 5e-3}, {"params": uncertainty, "lr": 5e-3}])
    for i in range(400):
        optim.zero_grad()
        cam_c2w = SE3(poses_th).inv().matrix()
        loss = cvd.consistency_loss(cam_c2w, K, K_inv, torch.clamp(disp_t, 1e-3, 1e3), init_disp,
                                    torch.clamp(uncertainty, 1e-4, 1e3), flows_t, flow_masks, ii, jj,
                                    compute_normals, fg_alpha, w_ratio=1.0, w_flow=0.2, w_si=1,
                                    w_grad=w_grad, w_normal=w_normal)
        loss.backward()
        disp_t.grad = torch.nan_to_num(disp_t.grad, nan=0.0)
        uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
        optim.step()
        if (i + 1) % 100 == 0:
            log(f"[megasam] consistent depth (b) per-pixel: step {i + 1}/400 loss {loss.item():.4f}")
    disp_opt = torch.nn.functional.interpolate(disp_t.unsqueeze(1), scale_factor=(2, 2),
                                               mode="bilinear").squeeze(1).detach().cpu().numpy()
    depth = np.clip(1.0 / disp_opt, 1e-3, 1e2).astype(np.float32)  # MegaSaM saves float16; we keep float32
    del img_data_pt
    torch.cuda.empty_cache()
    return depth, K_o.cpu().numpy(), cam_c2w.detach().cpu().numpy(), frame_scales


def metric_ratios(metric_low: list[np.ndarray], depth: np.ndarray) -> np.ndarray:
    """Per frame median(MoGe / MegaSaM) on pixels valid in both."""
    out = []
    for m, d in zip(metric_low, depth):
        valid = np.isfinite(m) & (m > 0) & (d > 1e-3) & (d < 1e2)
        out.append(float(np.median(m[valid] / d[valid])) if valid.sum() > 64 else np.nan)
    return np.array(out)


def run_segment(image_paths, mono_disp, metric, K, weights, raft_weights, opt_focal, log) -> dict:
    """The three MegaSaM stages on one segment; depth and translations returned in metres."""
    hc, wc = mono_disp[0].shape[0] - mono_disp[0].shape[0] % 8, mono_disp[0].shape[1] - mono_disp[0].shape[1] % 8
    t0 = time.time()
    aligns = align_mono_to_metric(mono_disp, metric)
    log(f"[megasam] mono->metric alignment scale {aligns[0]:.4f} shift {aligns[1]:.2e}; normaliser {aligns[2]:.4f}")
    traj, images, disps, intr, motion_prob = track(image_paths, mono_disp, aligns, K, weights, opt_focal, log)
    t_track = time.time() - t0
    flow_data = flows(image_paths, raft_weights, log)
    t_flow = time.time() - t0 - t_track
    depth, K_cvd, c2w, frame_scales = consistent_depth(images, disps, intr, traj, motion_prob, flow_data, log=log)
    t_cvd = time.time() - t0 - t_track - t_flow
    # ours: back to metres with one factor, median over frames of median(MoGe / MegaSaM)
    scale = float(np.nanmedian(metric_ratios([m[:hc, :wc] for m in metric], depth)))
    c2w[:, :3, 3] *= scale
    return {"c2w": c2w.astype(np.float64), "depth": depth * scale, "motion_prob": motion_prob, "K": K_cvd,
            "scale": scale, "aligns": aligns, "frame_scales": frame_scales,
            "seconds": {"tracking": round(t_track, 1), "flow": round(t_flow, 1), "consistent_depth": round(t_cvd, 1)}}


def run(depth_run: Path, out: Path, weights: Path, raft_weights: Path, *, opt_focal: bool = False,
        max_frames: int | None = None, segment_length: int = 200, overlap: int = 30, log=print) -> dict:
    import torch
    from .stitch import chain, plan_segments

    record = json.loads((depth_run / "run.json").read_text(encoding="utf-8"))
    images_dir = depth_run / record.get("image_dir", "frames")
    indices = frame_indices(depth_run / "depth", ".png")[:max_frames]
    image_paths = [images_dir / frame_name(i, ".png") for i in indices]
    k = CameraIntrinsics.load(depth_run / "intrinsics_stable.json")
    K = k.matrix().astype(np.float64)
    h1, w1, hc, wc = tracking_size(k.height, k.width)
    windows = plan_segments(len(indices), segment_length, overlap)
    log(f"[megasam] {len(indices)} frames {k.width}x{k.height} -> tracking {wc}x{hc}; "
        f"focal prior fx={k.fx:.1f} (fov_x {k.fov_x_degrees:.1f} deg); "
        f"{len(windows)} segment(s): {', '.join(f'{indices[a]}-{indices[b - 1]}' for a, b in windows)}")

    mono_disp, metric = mono_inputs(depth_run, indices, h1, w1)
    segments = []
    for n, (a, b) in enumerate(windows, 1):
        if len(windows) > 1:
            log(f"[megasam] segment {n}/{len(windows)}: frames {indices[a]}-{indices[b - 1]}")
        seg = run_segment(image_paths[a:b], mono_disp[a:b], metric[a:b], K, weights, raft_weights, opt_focal, log)
        seg["frames"] = list(range(a, b))
        segments.append(seg)
        torch.cuda.empty_cache()
    merged = chain(segments) if len(segments) > 1 else segments[0]
    for j, info in enumerate(merged.get("joins", []), 2):
        log(f"[megasam] join segment {j}: depth scale {info['depth_scale']:.4f}, camera position mismatch "
            f"median {1000 * info['pose_residual_m']['median']:.1f} mm over {info['overlap_frames']} overlap frames")

    # one global metric factor for the whole (joined) sequence; per segment it is ~1 already
    ratios = metric_ratios([m[:hc, :wc] for m in metric], merged["depth"])
    g = float(np.nanmedian(ratios))
    depth_m = merged["depth"] * g
    c2w = merged["c2w"].copy()
    c2w[:, :3, 3] *= g
    jitter = ratios / g  # >1: MoGe put this frame farther than the consistent depth
    log(f"[megasam] MoGe per-frame scale relative to the consistent depth: "
        f"{np.nanmin(jitter):.3f}-{np.nanmax(jitter):.3f} (std {100 * np.nanstd(jitter):.1f}%)")

    # native outputs (tracking resolution); the full-resolution depth/ is written on the Windows side
    for sub in ("depth_megasam", "poses"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    K_cvd = segments[0]["K"]
    kc = CameraIntrinsics(fx=float(K_cvd[0, 0]), fy=float(K_cvd[1, 1]), cx=float(K_cvd[0, 2]),
                          cy=float(K_cvd[1, 2]), width=int(wc), height=int(hc))
    for t, i in enumerate(indices):
        save_depth(out / "depth_megasam" / frame_name(i, ".png"), depth_m[t])
        save_pose(out / "poses" / frame_name(i, ".json"), c2w[t])
    kc.save(out / "intrinsics_megasam.json")
    np.save(out / "motion_prob.npy", np.float16(merged["motion_prob"]))
    full_fx = kc.fx * k.width / w1  # intrinsics were scaled by the resize (w1), the crop only drops columns
    result = {
        "depth_run": str(depth_run), "frames": len(indices), "tracking_size": [int(wc), int(hc)],
        "resized_size": [int(w1), int(h1)],
        "focal_prior_fx_fullres": k.fx, "focal_final_fx_fullres": round(full_fx, 2), "focal_optimised": opt_focal,
        "segments": [{"frames": [indices[a], indices[b - 1]], "metric_scale": s["scale"],
                      "mono_alignment": dict(zip(("scale", "shift", "normaliser"), s["aligns"])),
                      "seconds": s["seconds"]} for (a, b), s in zip(windows, segments)],
        "joins": merged.get("joins", []), "global_metric_factor": g,
        "moge_scale_vs_consistent": {"min": float(np.nanmin(jitter)), "max": float(np.nanmax(jitter)),
                                     "std_percent": round(100 * float(np.nanstd(jitter)), 2),
                                     "per_frame": [None if np.isnan(v) else round(float(v), 4) for v in jitter]},
        "cvd_frame_scales": [round(float(v), 4) for s in segments for v in s["frame_scales"]],
        "seconds": {key: round(sum(s["seconds"][key] for s in segments), 1)
                    for key in ("tracking", "flow", "consistent_depth")},
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (out / "megasam.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="megasam_final.pth")
    p.add_argument("--raft", type=Path, required=True, help="raft-things.pth")
    p.add_argument("--opt-focal", action="store_true", help="Let MegaSaM refine the focal length (demo default)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--segment-length", type=int, default=200)
    p.add_argument("--overlap", type=int, default=30)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run(a.depth_run, a.out, a.weights, a.raft, opt_focal=a.opt_focal, max_frames=a.max_frames,
        segment_length=a.segment_length, overlap=a.overlap)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
