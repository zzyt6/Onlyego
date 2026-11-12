"""Object and hand masks for the whole video (WiLoR -> Grounding DINO -> SAM2), each step in its WSL env.

  1. WiLoR on every frame: hand boxes (+ left/right, score)        -> wilor/<frame>.json
  2. Grounding DINO on the detection frame (default 0, where the object lies untouched): object box -> dino.json
  3. SAM2 prompts: the object box on the detection frame; each hand (right / left) its box on the frame where
     WiLoR is most confident about it. SAM2 takes a prompt per object on any frame and propagates both ways.
                                                                  -> masks/{1 object, 2 right hand, 3 left hand}/
Unlike V2D (one reference frame that must show both hands and the object), the object and each hand are
prompted where they are seen best.
"""

import json
from pathlib import Path

import numpy as np

from . import wsl

OBJECT_ID, RIGHT_ID, LEFT_ID = 1, 2, 3
TRACKS = {OBJECT_ID: "object", RIGHT_ID: "right_hand", LEFT_ID: "left_hand"}


def _load_wilor(folder: Path) -> dict[int, list[dict]]:
    return {int(p.stem): json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def _area(b: dict) -> float:
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["y1"] - b["y0"])


def _overlap(a: dict, b: dict) -> float:
    """Intersection area / area of a."""
    w = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    h = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    return w * h / max(1e-6, _area(a))


def hand_prompts(wilor: dict[int, list[dict]], object_box: dict, min_frames: int = 5,
                 edge: float = 5.0) -> list[dict]:
    """Per side, one box prompt on the frame where the hand is seen best. WiLoR's score is always 1, so "best"
    = the box does not touch the image border (whole hand in view) and does not cover the object box (a box
    prompt there could grab the object), then the largest such box. A side seen in fewer than `min_frames`
    frames is treated as a false detection."""
    prompts = []
    for oid, right in ((RIGHT_ID, True), (LEFT_ID, False)):
        cands = [(f, d) for f, dets in sorted(wilor.items()) for d in dets if bool(d["is_right"]) == right]
        if len({f for f, _ in cands}) < min_frames:
            continue

        def inside(d):
            b, (w, h) = d["bbox"], d["image_size"]
            return b["x0"] > edge and b["y0"] > edge and b["x1"] < w - edge and b["y1"] < h - edge

        def rank(fd):
            d = fd[1]
            return (inside(d), _overlap(object_box, d["bbox"]) < 0.05, _area(d["bbox"]))

        f, d = max(cands, key=rank)
        prompts.append({"frame_index": f, "object_id": oid, "points": None, "point_labels": None,
                        "box": d["bbox"], "mask_path": None, "frames_detected": len({f for f, _ in cands})})
    return prompts


def segment(video: Path, frames_dir: Path, out: Path, prompt: str, detect_frame: int = 0, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    wilor_dir = out / "wilor"
    if not any(wilor_dir.glob("*.json")):
        log("[segment] WiLoR hand detection on every frame")
        wsl.run("wilor", "HOI.wilor.video_to_hands",
                ["--video_path", wsl.path(video), "--output_dir", wsl.path(wilor_dir),
                 "--weights_dir", wsl.weights("wilor")], log)
    wilor = _load_wilor(wilor_dir)

    dino = out / "dino.json"
    if not dino.exists():
        log(f"[segment] Grounding DINO on frame {detect_frame}: {prompt!r}")
        wsl.run("grounding_dino", "HOI.grounding_dino.image_to_object_bboxes",
                ["--image_path", wsl.path(frames_dir / f"{detect_frame:06d}.png"), "--output_path", wsl.path(dino),
                 "--prompt", prompt, "--model_dir", wsl.weights("grounding_dino")], log)
    dets = json.loads(dino.read_text())
    if not dets:
        raise RuntimeError(f"Grounding DINO found nothing for {prompt!r} on frame {detect_frame}")
    object_box = dets[0]["box"]

    prompts = [{"frame_index": detect_frame, "object_id": OBJECT_ID, "points": None, "point_labels": None,
                "box": object_box, "mask_path": None}] + hand_prompts(wilor, object_box)
    prompts_path = out / "sam2_prompts.json"
    prompts_path.write_text(json.dumps({"prompts": [{k: v for k, v in p.items() if k != "frames_detected"}
                                                    for p in prompts]}, indent=2))
    masks = out / "masks"
    if not (masks / str(OBJECT_ID)).is_dir():
        log("[segment] SAM2: " + ", ".join(f"{TRACKS[p['object_id']]} @ frame {p['frame_index']}" for p in prompts))
        wsl.run("sam2", "HOI.sam2.video_to_masks",
                ["--video_path", wsl.path(video), "--prompts_path", wsl.path(prompts_path),
                 "--masks_dir", wsl.path(masks), "--weights_dir", wsl.weights("sam2")], log)

    info = {"video": str(video), "prompt": prompt, "detect_frame": detect_frame,
            "object_detection": dets[0],
            "tracks": {str(p["object_id"]): {"name": TRACKS[p["object_id"]], "prompt_frame": p["frame_index"],
                                             "box": p["box"],
                                             **({"wilor_frames": p["frames_detected"]} if "frames_detected" in p
                                                else {})}
                       for p in prompts}}
    (out / "segment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def load_mask(out: Path, track: int, frame: int) -> np.ndarray | None:
    import cv2
    p = out / "masks" / str(track) / f"{frame:06d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return None if m is None else m > 0
