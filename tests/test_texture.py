import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from four_stage_view import texture
from four_stage_view.models import EgoConfig

CONFIG = EgoConfig.model_validate({"image": {"model": "gpt-image-2.6", "base_url": "https://example.com/v1"},
                                   "texture": {"size": 2048, "max_image_calls": 5}})


def test_write_obj_roundtrip(tmp_path):
    v = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.02]])
    uv = np.array([[0, 0], [1, 0], [0, 1.0]])
    f = np.array([[0, 1, 2]])
    texture.write_obj(tmp_path / "m.obj", v, uv, f)
    m = trimesh.load(tmp_path / "m.obj", force="mesh", process=False)
    assert np.allclose(m.vertices, v, atol=1e-6)
    assert np.allclose(m.visual.uv, uv, atol=1e-6)


def test_task_text_states_size_budget_model():
    text = texture.task_text(CONFIG)
    assert "2048 x 2048" in text and "at most 5 image" in text and "--model gpt-image-2.6" in text
    assert "texture.png" in text and "result.json" in text
    assert "OPENAI_API_KEY" in text and "never print" in text  # the key itself is never in the text


def test_config_defaults():
    c = EgoConfig.model_validate({"image": {"model": "x"}})
    assert (c.texture.size, c.texture.max_image_calls) == (2048, 8)
    with pytest.raises(Exception):
        EgoConfig.model_validate({"image": {"model": "x", "base_url": "https://user:pw@host/v1"}})


def test_status_lines():
    job = {"id": "job-0001", "status": "ready", "texture": {"object": "demo01", "attempts": [
        {"folder": "texture-01", "status": "done", "image_calls_seen": 4, "seconds": 812.0}]}}
    out = texture.format_status(job)
    assert "texture-01: done, 4 image calls" in out


def test_prepare_workspace_on_demo01(tmp_path):
    from four_stage_view.store import Store
    runs = Path(texture.PROJECT) / "runs"
    if not (runs / "object_model" / "demo01" / "mesh_object.glb").exists() or not (runs / "job-0001").exists():
        pytest.skip("demo01 results not present")
    store = Store(runs)
    job = store.load("job-0001")
    info = texture.prepare_workspace(store, job, "demo01", CONFIG, tmp_path / "ws")
    ws = tmp_path / "ws"
    for name in ("model/mesh.obj", "model/mesh_object.glb", "model/model.json", "reference_frame/photo.png",
                 "reference_frame/object_mask.png", "reference_frame/camera.json", "TASK.md"):
        assert (ws / name).is_file(), name
    assert len(list((ws / "references").glob("*.png"))) == len(job["references"])
    cam = json.loads((ws / "reference_frame" / "camera.json").read_text())
    assert cam["frame_index"] == info["reference_frame"] and len(cam["object_to_camera"]) == 4
    m = trimesh.load(ws / "model" / "mesh_object.glb", force="mesh", process=False)
    img = getattr(m.visual.material, "baseColorTexture", None)
    assert img is None or max(img.size) <= 4  # SAM3D's 1024 px texture is not handed over (trimesh may add a 2x2)
    assert m.visual.uv.shape == (len(m.vertices), 2)
