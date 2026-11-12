"""MANO hand model with the left-hand shapedirs fix used by Wuji MINT.

The official MANO v1.2 MANO_LEFT.pkl stores the x component of its shape blend shapes with the
same sign as the right hand; a mirrored hand needs the opposite sign. With non-zero betas the left
hand therefore comes out wrong by up to ~3 cm. Wuji MINT (mint/visualization/mano.py,
model_train/losses/mano_geometry.py) builds the left smplx MANO layer and then does

    model.shapedirs[:, 0, :] *= -1

We apply the same negation, but to the loaded arrays before the layer is built, and only when the
left x component still matches the right hand's (so an already-corrected file is not flipped back).

The .pkl files are read with a restricted unpickler: it maps the seven classes a MANO pickle uses
(numpy arrays, a scipy sparse matrix, chumpy arrays) and refuses anything else, so loading needs
neither chumpy nor trust in the file.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SIDES = ("left", "right")


def mano_dir(root: str | Path | None = None) -> Path:
    """Folder holding MANO_LEFT.pkl / MANO_RIGHT.pkl: argument, $ZYEGO_MANO_DIR, else model/mano."""
    return Path(root or os.environ.get("ZYEGO_MANO_DIR") or PROJECT / "model" / "mano")


def mano_path(side: str, root: str | Path | None = None) -> Path:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    path = mano_dir(root) / f"MANO_{side.upper()}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (MANO is licensed: https://mano.is.tue.mpg.de, "
                                f"install with  python tools/install_mano.py <mano_v1_2.zip>)")
    return path


# ---------------------------------------------------------------- loading without chumpy

class _Chumpy:
    """Stand-in for chumpy.Ch / chumpy.reordering.Select: keeps the pickled state, yields an array."""

    def __setstate__(self, state):
        self.state = state if isinstance(state, dict) else {"x": state}

    def array(self) -> np.ndarray:
        s = self.state
        if "x" in s:
            return _to_array(s["x"])
        if "a" in s and "idxs" in s:
            out = _to_array(s["a"]).ravel()[np.asarray(s["idxs"])]
            shape = s.get("preferred_shape")
            return out.reshape(shape) if shape is not None else out
        raise ValueError(f"unsupported chumpy object (state keys {sorted(s)})")


def _to_array(value):
    return value.array() if isinstance(value, _Chumpy) else np.asarray(value)


def _numpy_reconstruct():
    try:
        from numpy._core.multiarray import _reconstruct  # numpy >= 2
    except ImportError:
        from numpy.core.multiarray import _reconstruct
    return _reconstruct


class _MANOUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        from scipy.sparse import csc_matrix
        allowed = {
            ("chumpy.ch", "Ch"): _Chumpy,
            ("chumpy.reordering", "Select"): _Chumpy,
            ("__builtin__", "set"): set,
            ("builtins", "set"): set,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct(),
            ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct(),
            ("scipy.sparse.csc", "csc_matrix"): csc_matrix,
            ("scipy.sparse._csc", "csc_matrix"): csc_matrix,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a MANO file")
        return allowed[(module, name)]


def load_mano_data(side: str, root: str | Path | None = None) -> dict:
    """Raw MANO parameters as numpy arrays (J_regressor stays a scipy sparse matrix)."""
    with open(mano_path(side, root), "rb") as f:
        raw = _MANOUnpickler(f, encoding="latin1").load()
    return {k: (v.array() if isinstance(v, _Chumpy) else v) for k, v in raw.items()}


# ---------------------------------------------------------------- the left-hand fix

def left_shapedirs_mismatch(left: dict, right: dict) -> tuple[float, float]:
    """(sum|L_x - R_x|, sum|L_x + R_x|) over the x component of shapedirs.

    A correctly mirrored left hand has L_x = -R_x (second number ~0); the v1.2 bug gives L_x = R_x.
    """
    lx = np.asarray(left["shapedirs"])[:, 0, :]
    rx = np.asarray(right["shapedirs"])[:, 0, :]
    return float(np.abs(lx - rx).sum()), float(np.abs(lx + rx).sum())


def needs_left_fix(left: dict, right: dict) -> bool:
    same, opposite = left_shapedirs_mismatch(left, right)
    return same < opposite


def fix_left_data(left: dict) -> dict:
    """Copy of the left-hand data with shapedirs[:, 0, :] negated (Wuji MINT's fix)."""
    out = dict(left)
    shapedirs = np.array(left["shapedirs"], dtype=np.float64, copy=True)
    shapedirs[:, 0, :] *= -1
    out["shapedirs"] = shapedirs
    return out


def fix_left_layer_(layer) -> None:
    """In-place fix for an already-built left layer: smplx (`shapedirs`) or manotorch (`th_shapedirs`).

    Use only on layers loaded from the unfixed v1.2 file; create_mano() already fixes its layers.
    """
    name = "shapedirs" if hasattr(layer, "shapedirs") else "th_shapedirs"
    getattr(layer, name)[:, 0, :] *= -1


def load_mano_data_fixed(side: str, root: str | Path | None = None) -> tuple[dict, bool]:
    """MANO data for `side`; for the left hand, with shapedirs corrected if needed. Returns (data, fixed)."""
    data = load_mano_data(side, root)
    if side == "left" and needs_left_fix(data, load_mano_data("right", root)):
        return fix_left_data(data), True
    return data, False


# ---------------------------------------------------------------- smplx layer (as in Wuji MINT)

def create_mano(side: str, *, root: str | Path | None = None, device="cpu", use_pca: bool = False,
                flat_hand_mean: bool = True, batch_size: int = 1, fix_left: bool = True):
    """smplx.MANO for one hand. The left hand gets the shapedirs fix unless fix_left=False.

    Defaults (no PCA, flat hand mean) mean hand_pose is the plain 45-D axis-angle of the 15 joints.
    The returned module has `.left_shapedirs_fixed` telling whether the fix was applied.
    """
    import smplx
    from smplx.utils import Struct

    if fix_left:
        data, fixed = load_mano_data_fixed(side, root)
    else:
        data, fixed = load_mano_data(side, root), False
    with contextlib.redirect_stdout(io.StringIO()):  # smplx prints a note about 10 shape coefficients
        model = smplx.MANO(model_path=str(mano_path(side, root)), is_rhand=side == "right",
                           data_struct=Struct(**data), use_pca=use_pca, flat_hand_mean=flat_hand_mean,
                           batch_size=batch_size)
    model.left_shapedirs_fixed = fixed
    return model.to(device).eval()


def mirror_axis_angle(pose):
    """Axis-angle of the mirror image across the x = 0 plane: (rx, ry, rz) -> (rx, -ry, -rz).

    Works on (..., 3k) arrays or tensors (each consecutive triple is one rotation).
    """
    flip = np.tile(np.array([1.0, -1.0, -1.0]), pose.shape[-1] // 3)
    if hasattr(pose, "new_tensor"):
        flip = pose.new_tensor(flip)
    return pose * flip
