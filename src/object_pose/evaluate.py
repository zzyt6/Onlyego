"""How well a per-frame object pose sequence fits the video.

Per frame (object->camera T_i, the unchanged object_model mesh):
  iou_full      silhouette vs the SAM2 object mask; fingers in front
                of the box are missing from the mask, so this is < 1 even for a perfect pose while the hand holds it
  iou_visible   the same with the hand pixels (SAM2 hand masks) removed from both: the fair score while held
  depth_mm      rendered mesh depth - MegaSaM depth on the visible object pixels (median): + = mesh behind the depth
  world         object->world = camera->world_i @ T_i; while the object lies still (before contact) it must not
                move: drift = distance / angle from object_model's fitted pose
  speed/accel   world translation per frame and its second difference (jitter), mm
"""

import json
from pathlib import Path

import numpy as np

from depth_estimation.datatypes import load_depth
from object_model import model
from object_model.segment import LEFT_ID, OBJECT_ID, RIGHT_ID, load_mask


def load_poses(folder: Path) -> dict[int, np.ndarray]:
    """Transform3d JSONs (wxyz quaternion, translation; scale 1) -> {frame: 4x4 object->camera}."""
    from scipy.spatial.transform import Rotation
    out = {}
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text())
        w, x, y, z = d["rotation"]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        T[:3, 3] = d["translation"]
        out[int(p.stem)] = T
    return out


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def hands(om_out: Path, i: int, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for hid in (RIGHT_ID, LEFT_ID):
        h = load_mask(om_out, hid, i)
        if h is not None:
            m |= h
    return m


def frame_metrics(mesh, T, k, mask, hand, depth) -> dict:
    sil, rdepth = model.render(mesh, T, k, samples=120_000)
    iou = lambda a, b: float((a & b).sum() / max(1, (a | b).sum()))
    vis = mask & sil & ~hand & np.isfinite(rdepth) & np.isfinite(depth) & (depth > 0)
    d = (rdepth[vis] - depth[vis]) * 1000
    return {"iou_full": round(iou(sil, mask), 4), "iou_visible": round(iou(sil & ~hand, mask & ~hand), 4),
            "depth_mm": round(float(np.median(d)), 1) if d.size > 200 else None,
            "mask_px": int(mask.sum()), "hand_px_on_object": int((sil & hand).sum())}


def evaluate(poses: dict[int, np.ndarray], mesh, k, c2w: dict[int, np.ndarray], om_out: Path, depth_dir: Path,
             contact: int, obj2world_fit: np.ndarray) -> dict:
    rows = {}
    for i in sorted(poses):
        mask = load_mask(om_out, OBJECT_ID, i)
        if mask is None or mask.sum() < 200:
            rows[i] = {"iou_full": None, "iou_visible": None, "depth_mm": None}
        else:
            rows[i] = frame_metrics(mesh, poses[i], k, mask, hands(om_out, i, mask.shape),
                                    load_depth(depth_dir / f"{i:06d}.png"))
        W = c2w[i] @ poses[i]
        rows[i]["world_t"] = W[:3, 3].tolist()
        if i < contact:
            rows[i]["static_drift_mm"] = round(float(np.linalg.norm(W[:3, 3] - obj2world_fit[:3, 3]) * 1000), 1)
            rows[i]["static_drift_deg"] = round(rot_deg(obj2world_fit[:3, :3].T @ W[:3, :3]), 2)
    idx = sorted(rows)
    t = np.array([rows[i]["world_t"] for i in idx]) * 1000
    speed = np.r_[np.nan, np.linalg.norm(np.diff(t, axis=0), axis=1)]
    accel = np.r_[np.nan, np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=1), np.nan]
    for j, i in enumerate(idx):
        rows[i]["speed_mm"] = None if np.isnan(speed[j]) else round(float(speed[j]), 2)
        rows[i]["accel_mm"] = None if np.isnan(accel[j]) else round(float(accel[j]), 2)
    return {"frames": {str(i): rows[i] for i in idx}, "summary": summarize(rows, contact)}


def _stats(v) -> dict | None:
    v = np.array([x for x in v if x is not None], float)
    if not v.size:
        return None
    return {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3)}


def summarize(rows: dict, contact: int) -> dict:
    phases = {"static (before contact)": [i for i in rows if i < contact],
              "held (from contact)": [i for i in rows if i >= contact]}
    out = {}
    for name, ids in phases.items():
        s = {key: _stats([rows[i].get(key) for i in ids])
             for key in ("iou_full", "iou_visible", "depth_mm", "accel_mm")}
        s["frames"] = len(ids)
        s["iou_visible_below_0.9"] = sum(1 for i in ids if (rows[i].get("iou_visible") or 1) < 0.9)
        if name.startswith("static"):
            s["drift_mm"] = _stats([rows[i]["static_drift_mm"] for i in ids])
            s["drift_deg"] = _stats([rows[i]["static_drift_deg"] for i in ids])
        out[name] = s
    return out
