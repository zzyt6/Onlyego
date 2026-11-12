"""Run SAM3D on the reference frame and place the mesh in the camera and the world; check how well it fits.

SAM3D (`HOI.sam3d.image_to_mesh`, WSL env hoi-sam3d) gets the reference RGB, the object mask and the fused
point map (pointmap.py). It returns the mesh in its own canonical frame (`sam3d/mesh.glb`) and a transform
(rotation, translation, per-axis scale) that puts it in the reference camera, OpenCV axes (`sam3d/transform.json`).

SAM3D's placement gets the distance right but not the size (demo01: the mesh came out 0.74x the box, centred
at the right depth). So the placement is refined (`refine`): one uniform scale + rotation + translation of the
unchanged mesh, minimising (1 - silhouette IoU with the SAM2 mask) + 0.5 x mean truncated distance from the
fused points to the mesh surface (in 5 mm units, capped at 20 mm), Powell from scale = sqrt(mask area /
silhouette area). The shape is never changed.

Written here:
  mesh_object.glb      the mesh in metres (after the scale fit), object frame = SAM3D's canonical frame
  mesh_world.glb       the mesh placed in the world of the camera poses (video_depth), for viewing
  object_pose.json     object->camera (reference frame) and object->world, 4x4 rigid, metres
Checks (model.json "fit_sam3d" before and "fit" after the refinement):
  silhouette IoU       mesh projected into the reference frame vs the SAM2 object mask
  depth difference     rendered mesh depth - fused depth, on pixels both cover (median = whole-mesh offset
                       along the view, spread = shape mismatch)
  point -> surface     distance from each fused point to the nearest mesh surface point
"""

import json

import cv2
import numpy as np
import trimesh

from depth_estimation.datatypes import CameraIntrinsics

from . import wsl


def run_sam3d(image, mask, pointmap_npy, intrinsics_json, out_dir, seed: int = 0, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    wsl.run("sam3d", "HOI.sam3d.image_to_mesh",
            ["--image_path", wsl.path(image), "--mask_path", wsl.path(mask),
             "--mesh_path", wsl.path(out_dir / "mesh.glb"), "--transform_path", wsl.path(out_dir / "transform.json"),
             "--intrinsics_path", wsl.path(out_dir / "intrinsics.json"), "--weights_dir", wsl.weights("sam3d"),
             "--seed", str(seed), "--with_mesh_postprocess", "--with_texture_baking",
             "--pointmap_path", wsl.path(pointmap_npy), "--pointmap_intrinsics_path", wsl.path(intrinsics_json)],
            log)


def load_mesh(path) -> trimesh.Trimesh:
    return trimesh.load(str(path), force="mesh")


def transform_matrix(t: dict) -> np.ndarray:
    """SAM3D transform.json (already converted to OpenCV by image_to_mesh) -> 4x4 with scale:
    x_cam = R @ (s * x) + t."""
    from camera_pose.poses import quat_wxyz_to_matrix
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_matrix(t["rotation"]) @ np.diag(t["scale"])
    M[:3, 3] = t["translation"]
    return M


def silhouette(mesh: trimesh.Trimesh, T: np.ndarray, k: CameraIntrinsics, down: int = 1) -> np.ndarray:
    """Mesh silhouette under object->camera T (may include scale), at 1/down resolution. One fillConvexPoly per
    triangle: a single fillPoly over all triangles cancels where they overlap."""
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    uv = np.stack([v[:, 0] / v[:, 2] * k.fx + k.cx, v[:, 1] / v[:, 2] * k.fy + k.cy], 1) / down
    sil = np.zeros((k.height // down, k.width // down), np.uint8)
    for tri in np.round(uv[mesh.faces] * 4).astype(np.int32):  # 2 bits of sub-pixel precision
        cv2.fillConvexPoly(sil, tri, 1, shift=2)
    return sil > 0


def render(mesh: trimesh.Trimesh, T: np.ndarray, k: CameraIntrinsics, samples: int = 400_000):
    """(silhouette bool (H,W), depth (H,W) NaN outside) of the mesh under object->camera T."""
    H, W = k.height, k.width
    sil = silhouette(mesh, T, k)
    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    p = pts @ T[:3, :3].T + T[:3, 3]
    u = np.floor(p[:, 0] / p[:, 2] * k.fx + k.cx).astype(int)
    w = np.floor(p[:, 1] / p[:, 2] * k.fy + k.cy).astype(int)
    ok = (p[:, 2] > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
    depth = np.full((H, W), np.inf)
    np.minimum.at(depth, (w[ok], u[ok]), p[ok, 2])
    depth = cv2.erode(np.where(np.isinf(depth), 1e3, depth).astype(np.float32), np.ones((3, 3), np.uint8))
    depth[(depth > 1e2) | ~sil] = np.nan
    return sil, depth


def fit_checks(mesh: trimesh.Trimesh, T_cam: np.ndarray, k: CameraIntrinsics, mask: np.ndarray,
               pointmap: np.ndarray) -> dict:
    from scipy.spatial import cKDTree
    sil, depth = render(mesh, T_cam, k)
    fused = pointmap[..., 2]
    both = np.isfinite(depth) & np.isfinite(fused)
    diff = (depth[both] - fused[both]) * 1000
    pts = pointmap[np.isfinite(fused)]
    surf, _ = trimesh.sample.sample_surface(mesh, 300_000, seed=1)
    surf = surf @ T_cam[:3, :3].T + T_cam[:3, 3]
    d = cKDTree(surf).query(pts)[0] * 1000
    return {
        "silhouette_iou": round(float((sil & mask).sum() / max(1, (sil | mask).sum())), 4),
        "mask_px": int(mask.sum()), "silhouette_px": int(sil.sum()),
        "depth_diff_mm": {"median": round(float(np.median(diff)), 2),
                          "p10": round(float(np.percentile(diff, 10)), 2),
                          "p90": round(float(np.percentile(diff, 90)), 2), "pixels": int(both.sum())},
        "point_to_surface_mm": {"median": round(float(np.median(d)), 2), "p90": round(float(np.percentile(d, 90)), 2)},
    }


MIN_IOU = 0.95                 # silhouette IoU with the SAM2 mask, after the refinement
MAX_POINT_TO_SURFACE_MM = 3.0  # median distance from the fused points to the mesh surface


def quality(fit: dict, min_iou: float = MIN_IOU, max_point_mm: float = MAX_POINT_TO_SURFACE_MM) -> dict:
    """Pass/fail of the final fit; `failures` says which check missed and by how much."""
    iou, dist = fit["silhouette_iou"], fit["point_to_surface_mm"]["median"]
    failures = []
    if iou < min_iou:
        failures.append(f"轮廓 IoU {iou:.3f} < {min_iou}")
    if dist > max_point_mm:
        failures.append(f"点到网格表面中位数 {dist:.2f} mm > {max_point_mm} mm")
    return {"passed": not failures, "failures": failures,
            "thresholds": {"min_iou": min_iou, "max_point_to_surface_mm": max_point_mm},
            "values": {"silhouette_iou": iou, "point_to_surface_mm_median": dist}}


def obb_extents_m(mesh: trimesh.Trimesh) -> list[float]:
    return sorted(float(x) for x in mesh.bounding_box_oriented.primitive.extents)


def sam3d_placement(sam3d_dir) -> tuple[trimesh.Trimesh, np.ndarray, list[float]]:
    """(mesh scaled by SAM3D's scale, in the object frame; rigid object->camera; SAM3D scale)."""
    t = json.loads((sam3d_dir / "transform.json").read_text())
    S = np.diag([*t["scale"], 1.0])                      # canonical -> object frame (metres, per SAM3D)
    T_obj2cam = transform_matrix(t) @ np.linalg.inv(S)   # rigid
    mesh = load_mesh(sam3d_dir / "mesh.glb")
    mesh.apply_transform(S)
    return mesh, T_obj2cam, t["scale"]


def refine(mesh: trimesh.Trimesh, T0: np.ndarray, k: CameraIntrinsics, mask: np.ndarray, pointmap: np.ndarray,
           down: int = 2, n_points: int = 20_000, max_evals: int = 3000, log=print):
    """Uniform scale + rigid motion of the unchanged mesh (about its vertex centroid) that fits the mask and the
    fused points. Returns (scale, rigid object->camera for the mesh scaled by `scale`, info)."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    pts = pointmap[np.isfinite(pointmap[..., 2])]
    pts = pts[np.random.default_rng(0).choice(len(pts), min(n_points, len(pts)), replace=False)]
    c = mesh.vertices.mean(0)
    surf, _ = trimesh.sample.sample_surface(mesh, 200_000, seed=0)
    tree = cKDTree(surf)
    mask_lo = cv2.resize(mask.astype(np.uint8), (k.width // down, k.height // down),
                         interpolation=cv2.INTER_NEAREST) > 0
    R0, t0 = T0[:3, :3], T0[:3, 3]

    def similarity(x):  # x = (log scale, rotation vector in the object frame, translation in the camera)
        s, R = np.exp(x[0]), Rotation.from_rotvec(x[1:4]).as_matrix()
        A = np.eye(4)
        A[:3, :3] = R0 @ R * s
        A[:3, 3] = R0 @ (c - s * R @ c) + t0 + x[4:7]
        return A

    def terms(x):
        A = similarity(x)
        sil = silhouette(mesh, A, k, down)
        iou = (sil & mask_lo).sum() / max(1, (sil | mask_lo).sum())
        Ai = np.linalg.inv(A)
        d = tree.query(pts @ Ai[:3, :3].T + Ai[:3, 3])[0] * np.exp(x[0]) * 1000  # mm, in the camera
        return iou, d

    def cost(x):
        iou, d = terms(x)
        return (1 - iou) + 0.5 * np.mean(np.minimum(d, 20.0) / 5.0)

    x0 = np.zeros(7)
    x0[0] = 0.5 * np.log(mask_lo.sum() / max(1, silhouette(mesh, T0, k, down).sum()))
    r = minimize(cost, x0, method="Powell", options={"maxfev": max_evals, "xtol": 1e-4, "ftol": 1e-5})
    s = float(np.exp(r.x[0]))
    A = similarity(r.x)
    T = A.copy()
    T[:3, :3] /= s  # rigid, acting on the mesh scaled by s
    iou, d = terms(r.x)
    info = {"scale": round(s, 4), "initial_scale_from_area": round(float(np.exp(x0[0])), 4),
            "rotation_deg": round(float(np.degrees(np.linalg.norm(r.x[1:4]))), 2),
            "translation_mm": [round(float(v) * 1000, 2) for v in r.x[4:7]],
            "evaluations": int(r.nfev), "cost": round(float(r.fun), 5),
            "iou_half_res": round(float(iou), 4), "point_to_surface_mm_median": round(float(np.median(d)), 2)}
    log(f"[refine] scale x{s:.3f}, rotation {info['rotation_deg']} deg, translation {info['translation_mm']} mm "
        f"({r.nfev} evaluations)")
    return s, T, info


def write(out_dir, mesh: trimesh.Trimesh, T_obj2cam: np.ndarray, c2w_ref: np.ndarray, extra: dict):
    mesh.export(out_dir / "mesh_object.glb")
    world = mesh.copy()
    world.apply_transform(c2w_ref @ T_obj2cam)
    world.export(out_dir / "mesh_world.glb")
    pose = {"object_to_camera": T_obj2cam.tolist(), "object_to_world": (c2w_ref @ T_obj2cam).tolist(), **extra,
            "convention": "x_cam = R x_obj + t, OpenCV camera axes, metres; x_obj = vertices of mesh_object.glb"}
    (out_dir / "object_pose.json").write_text(json.dumps(pose, indent=2))
