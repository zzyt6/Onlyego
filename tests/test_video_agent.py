import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from four_stage_view.video_agent import VideoReader, analyze_video_file, _stream_response
from four_stage_view.models import EgoConfig
from four_stage_view.store import Store
def observed_profile():
    return dict(category="test object", shape="rectangular body", proportions="thin",
                colors=["blue"], material_appearance="glossy appearance",
                visible_parts=["lid seam"], markings=["unreadable print on front"],
                observed_views=["front", "side"], unobserved_or_occluded=["back"])


@pytest.fixture
def video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "input clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    assert writer.isOpened()
    for i in range(12):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 18
        frame[10:30, i * 4:i * 4 + 16, 1] = 200
        writer.write(frame)
    writer.release()
    return path





def test_video_tools_cover_every_frame_and_crop_actual_pixels(video, tmp_path):
    reader = VideoReader(video, tmp_path / "tools")
    try:
        info = reader.build_overview()
        assert info["frame_count"] == 12
        value, images = reader.execute("read_video_overview", {"start_page": 0, "page_count": 6})
        assert value["pages"] == [{"page": 0, "first_frame": 0, "last_frame": 11}]
        assert len(images) == 1 and images[0].suffix == ".jpg"
        value, _ = reader.execute("crop_video_frames", {"crops": [{"frame_index": 5, "bbox": [100, 100, 800, 900]}]})
        record = value["crops"][0]
        with Image.open(record["original_path"]) as full, Image.open(record["crop_path"]) as crop:
            assert np.array_equal(np.asarray(crop), np.asarray(full.crop(record["bbox_pixels"])))
        assert reader.crops["crop-0001"]["frame_index"] == 5
        with pytest.raises(ValueError):
            reader.execute("read_video_frames", {"frame_indices": [999]})
        with pytest.raises(ValueError):
            reader.execute("crop_video_frames", {"crops": [{"frame_index": 1, "bbox": [900, 0, 100, 1000]}]})
    finally:
        reader.close()


def test_batch_crop_returns_all_previews_and_rejects_bad_batch_atomically(video, tmp_path):
    reader = VideoReader(video, tmp_path / "tools")
    try:
        reader.build_overview()
        value, previews = reader.execute("crop_video_frames", {"crops": [
            {"frame_index": 2, "bbox": [0, 0, 500, 500]},
            {"frame_index": 9, "bbox": [500, 500, 1000, 1000]},
        ]})
        assert [c["crop_id"] for c in value["crops"]] == ["crop-0001", "crop-0002"]
        assert [c["frame_index"] for c in value["crops"]] == [2, 9]
        assert len(previews) == 2 and all(p.is_file() for p in previews)
        # A bad box anywhere in the batch writes no crop at all.
        with pytest.raises(ValueError, match="Invalid bbox"):
            reader.execute("crop_video_frames", {"crops": [
                {"frame_index": 3, "bbox": [0, 0, 500, 500]},
                {"frame_index": 4, "bbox": [600, 0, 500, 1000]},
            ]})
        assert len(reader.crops) == 2 and not (tmp_path / "tools" / "crop-0003.png").exists()
    finally:
        reader.close()


def coverage(observed=("front", "left"), missing=("back", "right", "top", "bottom")):
    return {"observed_faces": list(observed), "missing_faces": list(missing),
            "search_notes": "Checked the rotation frames; the back stayed hidden."}


def test_view_coverage_must_account_for_every_face_once():
    from four_stage_view.video_agent import VideoUnderstanding

    base = {
        "interaction_target": "box", "interaction_evidence": "hand lifts it",
        "evidence_frame_indices": [1], "object_description": "box",
        "object_profile": observed_profile(), "limitations": [],
        "references": [{"crop_id": "crop-0001", "view": "front", "reason": "clear"}],
    }
    assert VideoUnderstanding.model_validate({**base, "view_coverage": coverage()})
    for bad in (
        coverage(missing=("back", "right", "top")),                    # bottom missing
        coverage(missing=("front", "back", "right", "top", "bottom")), # front twice
        coverage(observed=("front", "front"), missing=("back", "left", "right", "top", "bottom")),
    ):
        with pytest.raises(ValueError, match="six faces"):
            VideoUnderstanding.model_validate({**base, "view_coverage": bad})
    # A reference labelled with a face reported as missing is contradictory.
    with pytest.raises(ValueError, match="observed_faces"):
        VideoUnderstanding.model_validate({**base, "view_coverage": coverage(), "references": [
            {"crop_id": "crop-0001", "view": "back", "reason": "x"}]})
    oblique = [{"crop_id": "crop-0001", "view": "oblique", "reason": "corner view"}]
    assert VideoUnderstanding.model_validate({**base, "view_coverage": coverage(), "references": oblique})


def test_direct_api_history_includes_tools_images_and_previous_context(video, tmp_path, monkeypatch):
    config = tmp_path / "agent.json"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Inspect video using tools.")
    config.write_text(json.dumps({
        "base_url": "https://example.test/v1", "api_key_env": "TEST_VIDEO_AGENT_KEY",
        "model": "test-model", "prompt_file": "prompt.txt",
    }))
    monkeypatch.setenv("TEST_VIDEO_AGENT_KEY", "fixture-key")
    seen = []

    def response(client, endpoint, headers, body, directory):
        turn = len(seen)
        seen.append(json.loads(json.dumps(body)))
        assert endpoint == "https://example.test/v1/responses"
        if turn == 0:
            name, args = "read_video_overview", {"start_page": 0, "page_count": 6}
        elif turn == 1:
            assert any(i.get("type") == "function_call_output" for i in body["input"])
            assert any(p.get("type") == "input_image" for i in body["input"]
                       for p in i.get("content", []) if isinstance(p, dict))
            name, args = "read_video_frames", {"frame_indices": [3, 5]}
        elif turn == 2:
            name, args = "crop_video_frames", {"crops": [{"frame_index": 5, "bbox": [50, 50, 950, 950]}]}
        else:
            assert turn == 3
            name, args = "finish_video_analysis", {
                "interaction_target": "test object",
                "interaction_evidence": "Hand touches and moves the target.",
                "evidence_frame_indices": [3, 5], "object_description": "test target",
                "object_profile": observed_profile(),
                "references": [{"crop_id": "crop-0001", "view": "front", "reason": "Clear view"}],
                "view_coverage": coverage(),
                "limitations": ["back not visible"],
            }
        return {"id": f"test-response-{turn}", "usage": {}, "output": [
            {"type": "function_call", "call_id": f"call-{turn}", "name": name,
             "arguments": json.dumps(args)},
        ]}

    monkeypatch.setattr("four_stage_view.video_agent._stream_response", response)
    store = Store(tmp_path / "runs")
    job = analyze_video_file(store, video, EgoConfig.model_validate({"image": {"model": "test-image"}}), config)
    assert len(seen) == 4
    assert [len(b["input"]) for b in seen] == [1, 4, 7, 10]
    assert all(b["parallel_tool_calls"] is True for b in seen)
    assert job["status"] == "ready" and "views" not in job
    assert job["video_analysis_mode"] == "direct_api_video_agent"
    directory = Path(job["video_input_directory"])
    summary = json.loads((directory / "session-summary.json").read_text())
    assert summary["overview_frames"] == 12
    assert summary["detail_frames_seen"] == [3, 5]
    document = json.loads((store.directory(job["id"]) / "object-description.json").read_text())
    assert document["object"] == observed_profile()
    assert document["references"][0]["bbox_normalized"] == [50, 50, 950, 950]
    assert document["references"][0]["view"] == "front"
    assert document["view_coverage"] == coverage()
    ref = store.resolve(job["id"], job["references"][0])
    with Image.open(ref) as im:
        assert im.size[0] < 96 and im.size[1] < 64


def test_parallel_tool_calls_put_all_outputs_before_one_image_message(video, tmp_path, monkeypatch):
    config = tmp_path / "agent.json"
    (tmp_path / "prompt.txt").write_text("Inspect video using tools.")
    config.write_text(json.dumps({
        "base_url": "https://example.test/v1", "api_key_env": "TEST_VIDEO_AGENT_KEY",
        "model": "test-model", "prompt_file": "prompt.txt", "max_rounds": 2,
    }))
    monkeypatch.setenv("TEST_VIDEO_AGENT_KEY", "fixture-key")
    seen = []

    def response(client, endpoint, headers, body, directory):
        seen.append(json.loads(json.dumps(body)))
        return {"id": "r", "usage": {}, "output": [
            {"type": "function_call", "call_id": "a", "name": "read_video_overview",
             "arguments": json.dumps({"start_page": 0, "page_count": 1})},
            {"type": "function_call", "call_id": "b", "name": "read_video_frames",
             "arguments": json.dumps({"frame_indices": [1, 2]})},
        ]}

    monkeypatch.setattr("four_stage_view.video_agent._stream_response", response)
    with pytest.raises(ValueError, match="reached 2 rounds"):
        analyze_video_file(Store(tmp_path / "runs"), video, EgoConfig.model_validate({"image": {"model": "test-image"}}), config)
    history = seen[1]["input"]
    kinds = [i.get("type") or i.get("role") for i in history]
    assert kinds == ["user", "function_call", "function_call", "function_call_output",
                     "function_call_output", "user"]
    images = [p for p in history[-1]["content"] if p["type"] == "input_image"]
    assert len(images) == 3  # one overview page + two detail frames, in call order


def test_stream_error_is_saved_and_not_retried(tmp_path):
    import httpx
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(400, json={"error": "unsupported tool"})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="HTTP 400"):
            _stream_response(client, "https://example.test/v1/responses", {}, {}, tmp_path)
    assert len(calls) == 1
    assert json.loads((tmp_path / "error.json").read_text())["http_status"] == 400


def completed_stream():
    import httpx
    event ={"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": []}}
    return httpx.Response(200, content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())


def test_handshake_failure_is_retried_because_nothing_was_sent(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr("four_stage_view.video_agent.time.sleep", lambda s: None)
    attempts = []
    def handle(request):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF")
        return completed_stream()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = _stream_response(client, "https://example.test/v1/responses", {}, {"a": 1}, tmp_path)
    assert result["id"] == "r1" and len(attempts) == 3
    assert (tmp_path / "connect-error-2.json").is_file()


def test_persistent_connect_failure_is_a_clear_error_without_traceback_type(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr("four_stage_view.video_agent.time.sleep", lambda s: None)
    attempts = []
    def handle(request):
        attempts.append(1)
        raise httpx.ConnectError("EOF")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="No request was delivered"):
            _stream_response(client, "https://example.test/v1/responses", {}, {}, tmp_path)
    assert len(attempts) == 3


def test_failure_after_connecting_is_never_retried(tmp_path):
    import httpx
    attempts = []
    def handle(request):
        attempts.append(1)
        raise httpx.ReadError("connection reset mid-response")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.ReadError):
            _stream_response(client, "https://example.test/v1/responses", {}, {}, tmp_path)
    assert len(attempts) == 1
