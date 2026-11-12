"""Per-frame metric depth and intrinsics with MoGe-2 (V2D v2d_moge/lib/video_to_depth.py).

MoGe-2 predicts, from one RGB image, a metric point map; depth is its z channel and the
intrinsics are solved from the point map. Every frame is independent, so focal length and
scale jitter from frame to frame -- intrinsics.py and align.py deal with that afterwards.

Changes from V2D:
  - reads the extracted frame PNGs instead of re-decoding the video, so depth/frames/intrinsics
    share the same indices by construction (V2D flushes its last batch only when
    frame_index == CAP_PROP_FRAME_COUNT - 1; if that metadata overcounts, the tail is dropped);
  - runs on Windows without Docker; the model is loaded once per call, not kept as a global.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .datatypes import CameraIntrinsics, frame_indices, frame_name, save_depth


def find_checkpoint(weights: Path) -> Path:
    weights = Path(weights)
    for candidate in (weights / "model.pt", weights / "pytorch_model.bin", weights / "model.safetensors", weights):
        if candidate.is_file():
            return candidate
    found = sorted(p.name for p in weights.iterdir()) if weights.is_dir() else []
    raise FileNotFoundError(f"MoGe-2 checkpoint not found in {weights}; files: {found}")


def load_model(weights: Path, device: str = "cuda"):
    import torch
    from moge.model.v2 import MoGeModel

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; install a CUDA build of PyTorch (see README) or pass --device cpu")
    model = MoGeModel.from_pretrained(str(find_checkpoint(weights)))
    return model.to(device).eval()


def frames_to_depth(frames: Path, out: Path, weights: Path, *, batch_size: int = 8, fov_x: float | None = None,
                    save_points: bool = False, save_normals: bool = False, device: str = "cuda",
                    model=None, on_batch=None) -> int:
    """Write depth/<i>.png, intrinsics/<i>.json, mask/<i>.png (+ points/normals .npy) for every frame.

    fov_x: optional horizontal FoV prior in degrees (from known calibration). MoGe then only solves
    depth scale/shift; the intrinsics written are still MoGe's returned values, which are the ones
    geometrically consistent with the depth it produced.
    Frames that already have depth + intrinsics are skipped, so an interrupted run resumes.
    """
    import torch

    dirs = {name: out / name for name in ("depth", "intrinsics", "mask")}
    if save_points:
        dirs["points"] = out / "points"
    if save_normals:
        dirs["normals"] = out / "normals"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    todo = [i for i in frame_indices(frames, ".png")
            if not ((dirs["depth"] / frame_name(i, ".png")).exists()
                    and (dirs["intrinsics"] / frame_name(i, ".json")).exists())]
    if not todo:
        return 0
    model = model or load_model(weights, device)
    done = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        images = []
        for i in batch:
            with Image.open(frames / frame_name(i, ".png")) as image:
                images.append(np.asarray(image.convert("RGB")))
        if len({im.shape for im in images}) != 1:
            raise ValueError(f"Frames {batch[0]}..{batch[-1]} have different sizes")
        height, width = images[0].shape[:2]
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.no_grad():
            pred = model.infer(tensor, fov_x=fov_x)
        for k, i in enumerate(batch):
            save_depth(dirs["depth"] / frame_name(i, ".png"), pred["depth"][k].cpu().numpy())
            CameraIntrinsics.from_normalized(pred["intrinsics"][k].cpu().numpy(), width, height).save(
                dirs["intrinsics"] / frame_name(i, ".json"))
            mask = pred["mask"][k].cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(mask).save(dirs["mask"] / frame_name(i, ".png"))
            if save_points:
                np.save(dirs["points"] / frame_name(i, ".npy"), pred["points"][k].cpu().numpy().astype(np.float32))
            if save_normals and "normal" in pred:
                np.save(dirs["normals"] / frame_name(i, ".npy"), pred["normal"][k].cpu().numpy().astype(np.float32))
        done += len(batch)
        if on_batch:
            on_batch(done, len(todo))
    return done
