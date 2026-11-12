"""FoundationPose tracking over a video (runs in the WSL env hoi-fp).

Start at the reference frame, track forward to the end and backward to frame 0, each frame refined from the
previous pose with RGB + depth (depth outside the object mask zeroed). The start pose is the one object_model fitted
at the reference frame (--start-pose; silhouette IoU 0.98). Frames are read from the extracted PNGs.

When tracking goes wrong:
  1. lost = the *visible* IoU (hand pixels removed from both the rendered mesh and the SAM2 mask) below
     --lost-iou on 2 consecutive frames, or below 0.5 once. Hand occlusion no longer looks like "lost",
     and a wrong pose with IoU 0.6 is no longer kept.
  2. only re-register when the object is fully seen: not when its mask touches the image border or suddenly
     shrinks to under half its recent size (cut off / covered); then keep tracking and wait.
  3. re-registration picks the hypothesis itself among all of FoundationPose's refined hypotheses: the
     silhouette filters, the texture decides. Keep those whose visible IoU with the SAM2 mask is within
     --iou-tolerance (0.03) of the best (rejects e.g. 90-degree errors); among them take FoundationPose's
     highest score (its scorer compares the textured render with the photo: tells front from back and 0
     from 180 degrees, whose silhouettes are the same; IoU differences of 0.01 are noise and must not
     decide that). Taken only if its visible IoU beats the tracked pose's by 0.02; at most one attempt per
     5 frames.

Writes <out>/<frame:06d>.json (Transform3d: object->camera, quaternion wxyz + translation) and
<info> = {frame: {"iou": full-mask IoU, "iou_visible", "reregistered": bool, "reliable": mask fit for
re-registration (yes / edge / partial / no mask), "recovery": the chosen hypothesis on re-registrations}}.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from HOI.common.datatypes import CameraIntrinsics, DepthImage, Image, Mask, Transform3d
from HOI.foundation_pose.foundation_pose_tracker import FoundationPoseTracker
from HOI.mesh.mesh import Mesh
from Utils import nvdiffrast_render  # FoundationPose's renderer (on sys.path once the tracker is imported)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fp_track")
MIN_MASK_PIXELS = 200  # below this the object is (almost) hidden: track without the IoU check
EDGE_PX = 4  # mask within this many pixels of the image border = object cut off by the frame
SCALE = 0.5  # silhouettes for the visible IoU are compared at half resolution


class Silhouettes:
    """Render the mesh at many poses at once and compare with SAM2 masks (visible part only)."""

    def __init__(self, tracker: FoundationPoseTracker, k: CameraIntrinsics):
        self.t = tracker
        K = k.to_matrix().astype(np.float64).copy()
        K[:2] *= SCALE
        self.K, self.H, self.W = K, round(k.height * SCALE), round(k.width * SCALE)
        self.tf = tracker._est.get_tf_to_centered_mesh()  # centred-mesh pose @ tf = caller's pose
        self.tf_inv = torch.linalg.inv(self.tf)

    def small(self, m: np.ndarray) -> np.ndarray:
        return cv2.resize(m.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0

    def visible_iou(self, centred: torch.Tensor, mask: np.ndarray, hand: np.ndarray) -> np.ndarray:
        """centred: (N,4,4) poses of the centred mesh (FoundationPose's internal frame)."""
        m, h = self.small(mask), self.small(hand)
        b = m & ~h
        out = []
        for c in range(0, len(centred), 64):
            with torch.no_grad():
                _, d, _ = nvdiffrast_render(self.K, self.H, self.W, centred[c:c + 64], glctx=self.t._glctx,
                                            mesh_tensors=self.t._est.mesh_tensors, get_normal=False)
            a = (d.cpu().numpy() > 0.001) & ~h[None]
            out.append((a & b[None]).sum((1, 2)) / np.maximum(1, (a | b[None]).sum((1, 2))))
        return np.concatenate(out)

    def centred(self, pose: Transform3d) -> torch.Tensor:
        return torch.as_tensor(pose.to_matrix(), device="cuda", dtype=torch.float)[None] @ self.tf_inv

    def caller(self, centred: torch.Tensor) -> Transform3d:
        return Transform3d.from_matrix((centred @ self.tf).cpu().numpy())


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle between rotations; Ra (3,3), Rb (N,3,3)."""
    tr = np.einsum("ij,nij->n", Ra, Rb)  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--hand-masks", nargs="*", default=[], help="SAM2 hand mask folders")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--reference", type=int, required=True)
    ap.add_argument("--start-pose", required=True, help="json with object_to_camera (4x4) at the reference frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--lost-iou", type=float, default=0.8, help="visible IoU below this (2 frames) = lost")
    ap.add_argument("--iou-tolerance", type=float, default=0.03,
                    help="hypotheses within this of the best visible IoU count as silhouette-equal")
    ap.add_argument("--register-iteration", type=int, default=10)
    ap.add_argument("--track-iteration", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = CameraIntrinsics.load(a.intrinsics)
    frames = sorted(int(p.stem) for p in Path(a.frames).glob("*.png"))
    tracker = FoundationPoseTracker(Mesh.load(a.mesh, force_mesh=True), a.weights)
    log.info("FoundationPose backend: %s", tracker.runtime_info.get("backend"))
    sil = Silhouettes(tracker, k)

    def load(i):
        rgb = Image.load(f"{a.frames}/{i:06d}.png")
        depth = DepthImage.load(f"{a.depth}/{i:06d}.png")
        mp = Path(a.masks) / f"{i:06d}.png"
        mask = Mask.load(str(mp)) if mp.exists() else None
        if mask is not None:  # depth outside the object mask -> 0
            d = depth.depth.copy()
            d[~mask.mask.astype(bool)] = 0.0
            depth = DepthImage(depth=d)
        return rgb, depth, mask

    def hands(i, shape):
        h = np.zeros(shape, bool)
        for folder in a.hand_masks:
            p = Path(folder) / f"{i:06d}.png"
            if p.exists():
                h |= Mask.load(str(p)).mask.astype(bool)
        return h

    info = {}

    def save(i, pose: Transform3d, rec: dict):
        pose.save(str(out / f"{i:06d}.json"))
        info[i] = rec

    def reliable(m: np.ndarray, areas: list) -> str:
        """'' if the mask shows the whole object (fit for re-registration), else why not."""
        if m[:EDGE_PX].any() or m[-EDGE_PX:].any() or m[:, :EDGE_PX].any() or m[:, -EDGE_PX:].any():
            return "edge"
        recent = areas[-10:]
        if len(recent) >= 3 and m.sum() < 0.5 * np.median(recent):
            return "partial"
        return ""

    def recover(rgb, depth, mask, hand, last_rot, cur_iou):
        """Register from scratch, then choose among all refined hypotheses (see the module docstring)."""
        tracker.register(rgb, depth, mask, k, iteration=a.register_iteration)
        P = tracker._est.poses  # (N,4,4) centred-mesh poses, sorted by FoundationPose's score (best first)
        n = len(P)
        iou = sil.visible_iou(P, mask.mask.astype(bool), hand)
        rot = rot_deg(last_rot, P[:, :3, :3].cpu().numpy())  # recorded only
        j = int(np.flatnonzero(iou >= iou.max() - a.iou_tolerance)[0])  # sorted by score: first = best score
        choice = {"hypotheses": n, "fp_rank": j, "iou_visible": round(float(iou[j]), 4),
                  "deg_from_last_reliable": round(float(rot[j]), 1),
                  "fp_top1_iou_visible": round(float(iou[0]), 4), "best_iou_visible": round(float(iou.max()), 4)}
        if iou[j] < cur_iou + 0.02:
            return None, choice
        return sil.caller(P[j]), choice

    r = a.reference
    rgb, depth, mask = load(r)
    start = Transform3d.from_matrix(np.array(json.loads(Path(a.start_pose).read_text())["object_to_camera"]))
    m0 = mask.mask.astype(bool)
    rec0 = {"iou": tracker._mask_iou(mask, k, start), "reregistered": False,
            "iou_visible": round(float(sil.visible_iou(sil.centred(start), m0, hands(r, m0.shape))[0]), 4)}
    save(r, start, rec0)
    log.info("start pose at frame %d (object_model fit), IoU %.3f", r, rec0["iou"])

    t0, n_rereg, n_attempt = time.time(), 0, 0
    for direction in (1, -1):
        tracker.reset_to_pose(start)
        seq = [i for i in frames if (i > r if direction > 0 else i < r)]
        last_rot, bad, cooldown, areas = start.to_matrix()[:3, :3], 0, 0, []
        for i in (seq if direction > 0 else seq[::-1]):
            rgb, depth, mask = load(i)
            has_mask = mask is not None and mask.mask.sum() >= MIN_MASK_PIXELS
            pose = tracker.track_one(rgb, depth, k, iteration=a.track_iteration)
            rec = {"reregistered": False}
            if not has_mask:
                rec.update(iou=float("nan"), reliable="no mask")
                save(i, pose, rec)
                cooldown = max(0, cooldown - 1)
                continue
            m = mask.mask.astype(bool)
            hand = hands(i, m.shape)
            v = float(sil.visible_iou(sil.centred(pose), m, hand)[0])
            why = reliable(m, areas)
            if why != "edge":
                areas.append(int(m.sum()))
            rec.update(iou_visible=round(v, 4), reliable=why or "yes")
            if not why:
                bad = bad + 1 if v < a.lost_iou else 0
                if (bad >= 2 or v < 0.5) and cooldown == 0:
                    n_attempt += 1
                    cooldown = 5
                    new, choice = recover(rgb, depth, mask, hand, last_rot, v)
                    rec["recovery"] = choice
                    log.info("frame %d: lost (visible IoU %.3f) -> %s; FoundationPose's top hypothesis %.3f, "
                             "chosen #%d %.3f (%.0f deg from the last reliable pose)", i, v,
                             "re-registered" if new is not None else "kept the tracked pose",
                             choice["fp_top1_iou_visible"], choice["fp_rank"], choice["iou_visible"],
                             choice["deg_from_last_reliable"])
                    if new is not None:
                        pose, v, bad = new, choice["iou_visible"], 0
                        rec.update(reregistered=True, iou_visible=v)
                        n_rereg += 1
                    tracker.reset_to_pose(pose)  # register() moved FoundationPose's state; put it back on our pose
                if v >= a.lost_iou:
                    last_rot = pose.to_matrix()[:3, :3]
            cooldown = max(0, cooldown - 1)
            rec["iou"] = tracker._mask_iou(mask, k, pose)
            save(i, pose, rec)

    Path(a.info).write_text(json.dumps({"reference": r,
                                        "seconds": round(time.time() - t0, 1), "attempts": n_attempt,
                                        "frames": {str(i): info[i] for i in sorted(info)}}, indent=1))
    log.info("tracked %d frames in %.1f s, %d re-registrations (%d attempts)", len(frames), time.time() - t0,
             n_rereg, n_attempt)


if __name__ == "__main__":
    main()
