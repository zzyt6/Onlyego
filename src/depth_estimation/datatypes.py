"""On-disk formats, kept identical to V2D so outputs can be fed to their downstream modules.

Depth PNG: uint16, inverse depth  pixel = 65535 / (depth_m + 1)
  - near objects get large values, depth 0 m -> 65535, invalid/infinite depth -> 0
  - resolution is finest up close: one step is (d+1)^2 / 65535 m, i.e. ~0.06 mm at 1 m, ~1.8 mm at 10 m
Intrinsics JSON: {"fx", "fy", "cx", "cy", "width", "height"} in pixels.
Mask PNG: uint8, non-zero = object.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def encode_depth(depth: np.ndarray) -> np.ndarray:
    """Metric depth (m, float; inf/nan = invalid) -> uint16 inverse-depth array.

    Truncates like V2D's DepthImage.to_pil_image (clip, then astype) so the PNGs are bit-identical;
    the only addition is that NaN is written as 0 explicitly (a NaN cast to uint16 is undefined)."""
    depth = np.asarray(depth, dtype=np.float32)  # V2D computes in MoGe's float32
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 65535.0 * (1.0 / (depth + 1.0))
    inverse = np.where(np.isfinite(inverse), inverse, 0.0)
    return inverse.clip(0, 65535).astype(np.uint16)


def decode_depth(array: np.ndarray) -> np.ndarray:
    """uint16 inverse-depth array -> float32 depth in metres; pixel 0 -> inf (invalid)."""
    inverse = np.asarray(array, dtype=np.float32) / 65535.0
    with np.errstate(divide="ignore"):
        return (1.0 / inverse - 1.0).astype(np.float32)


def save_depth(path: Path, depth: np.ndarray):
    Image.fromarray(encode_depth(depth)).save(path)


def load_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return decode_depth(np.array(image))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def fov_x_degrees(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=4), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "CameraIntrinsics":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return CameraIntrinsics(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                                width=int(d["width"]), height=int(d["height"]))

    @staticmethod
    def from_normalized(k: np.ndarray, width: int, height: int) -> "CameraIntrinsics":
        """MoGe returns intrinsics normalised by image size (fx/W, fy/H, cx/W, cy/H)."""
        return CameraIntrinsics(fx=float(k[0, 0] * width), fy=float(k[1, 1] * height),
                                cx=float(k[0, 2] * width), cy=float(k[1, 2] * height),
                                width=int(width), height=int(height))


def frame_name(index: int, suffix: str) -> str:
    return f"{index:06d}{suffix}"


def frame_indices(folder: Path, suffix: str) -> list[int]:
    return sorted(int(p.stem) for p in Path(folder).glob(f"*{suffix}") if p.stem.isdigit())
