"""One stable camera from noisy per-frame estimates (V2D v2d_depth/lib/stabilize_intrinsics.py).

A monocular model guesses the focal length again on every frame. V2D takes the temporal median
of fx, fy, cx, cy -- robust to occasional bad frames -- and every downstream stage (hand/object
alignment, SLAM, FoundationPose) uses this single intrinsics_stable.json.

Note: the per-frame depth was computed under each frame's own focal length, not the median one.
To get depth consistent with the stable camera, rerun depth with --fov-from intrinsics_stable.json
(MoGe then keeps the focal fixed and only solves depth).
"""

from pathlib import Path

import numpy as np

from .datatypes import CameraIntrinsics


def load_all(folder: Path) -> dict[int, CameraIntrinsics]:
    files = sorted(p for p in Path(folder).glob("*.json") if p.stem.isdigit())
    if not files:
        raise FileNotFoundError(f"No intrinsics JSON files in {folder}")
    return {int(p.stem): CameraIntrinsics.load(p) for p in files}


def stabilize_intrinsics(folder: Path, output: Path, fix_principal_point: bool = False) -> CameraIntrinsics:
    per_frame = list(load_all(folder).values())
    width = int(np.median([k.width for k in per_frame]))
    height = int(np.median([k.height for k in per_frame]))
    stable = CameraIntrinsics(
        fx=float(np.median([k.fx for k in per_frame])),
        fy=float(np.median([k.fy for k in per_frame])),
        cx=width / 2.0 if fix_principal_point else float(np.median([k.cx for k in per_frame])),
        cy=height / 2.0 if fix_principal_point else float(np.median([k.cy for k in per_frame])),
        width=width, height=height,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    stable.save(output)
    return stable
