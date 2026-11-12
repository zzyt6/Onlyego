"""Step 2: Codex generates the texture of the fitted object mesh.

    four_stage_view generate-texture <job> --object demo01

Our code only hands over material and collects the result; all of the texture comes from Codex and its image
generation (the imagegen skill CLI with the image model in configs/ego.json, e.g. gpt-image-2.6):

1. hand over: a workspace <job>/texture-NN/ with
     model/mesh_object.glb, model/mesh.obj   object_model's fitted mesh (metres, with its UVs, no texture)
     references/, object-description.json     step 1's reference photos and description
     reference_frame/                         the photo of the frame the mesh was fitted to, the object mask,
                                              camera intrinsics and the object's pose in that camera
     TASK.md                                  the task (also given to Codex as its prompt)
2. run `codex exec` in the workspace: it may write only there, has network access for image generation, and the
   image API key/endpoint in its environment (never in a file);
3. collect: texture.png (size x size) must exist; mesh_textured.glb = the same mesh with that texture attached;
   both are copied to runs\\object_model\\<video>\\ (texture.png, mesh_textured.glb, texture.json).

A started session is never repeated implicitly (it spends image calls): a new attempt needs --retry.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .codex_backend import resolve_codex
from .models import EgoConfig
from .store import atomic_json, now

PROJECT = Path(__file__).resolve().parents[2]
IMAGEGEN = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "skills" / ".system" / "imagegen" / \
    "scripts" / "image_gen.py"
PYTHON = Path(sys.executable)

TASK = """# Task: generate the texture of a 3D object mesh

You are given the 3D model (mesh) of a real object, fitted to a video of a hand handling it, and reference photos of
that object. Produce its texture: **`texture.png`, {size} x {size} RGB, for the mesh's existing UV coordinates.**
All texture content must come from your image generation (see "Image generation"); you decide how to use it.
Do not change the mesh or its UVs.

## Files (this folder; write only inside it)
- `model/mesh.obj` - the mesh: `v` vertices in metres, `vt` UV coordinates, `f v/vt`. Same data in
  `model/mesh_object.glb`. UV convention: u to the right, v up; texture pixel (x, y) = (u * {size}, (1 - v) * {size}).
- `model/model.json` - size of the object and of the mesh.
- `references/*.png` - reference photos of the real object (crops from the video), described in
  `object-description.json` (which faces each photo shows, what was never seen).
- `reference_frame/photo.png` - the full video frame the mesh was fitted to (the object lies untouched),
  `reference_frame/object_mask.png` - the object's pixels in it,
  `reference_frame/camera.json` - pinhole intrinsics (fx, fy, cx, cy, width, height) and `object_to_camera` (4 x 4):
  a mesh point x maps to the camera as x_cam = R x + t (OpenCV axes: x right, y down, z forward) and to the
  photo at u = fx * x_cam / z_cam + cx, v = fy * y_cam / z_cam + cy. The mesh projected this way covers the object
  in the photo (silhouette IoU about 0.98).

## Image generation
Use the imagegen skill's CLI (this task explicitly asks for the CLI path):
`"{python}" "{imagegen}" generate|edit --model {model} --size 1024x1024|1536x1024|1024x1536 --quality high ...`
(`edit` takes one or more `--image` inputs, e.g. reference photos). The API key and endpoint are already in your
environment (`OPENAI_API_KEY`, `OPENAI_BASE_URL`); never print or write them. Budget: **at most {max_calls} image
generation calls in total** - plan before generating. Images come out at most 1536 px; composing them into the
{size} x {size} texture is up to you.

## Tools
Python with numpy, scipy, trimesh, opencv-python, pillow, moderngl (OpenGL rendering) is at `"{python}"`. Write any
scripts you need into this folder (e.g. to read the UV layout, find which surface each UV island belongs to,
place generated images into the texture, render the textured mesh).

## Quality bar
- Seen from the reference camera, the textured mesh must look like the object in `reference_frame/photo.png`:
  the same artwork and printed text in the same place, orientation and colours.
- No artwork mirrored or upside down on the mesh; artwork crossing an edge lines up across it.
- Surfaces no reference shows: continue the object's design plausibly; do not invent readable text, logos or brands.
- Flat surface colour (albedo): no shadows, glare, reflections, hands or fingers.
- Every UV island gets a colour (no black/empty parts on the mesh).

## Before you finish
Check your result yourself: render the mesh with `texture.png` from the reference camera and compare with the
photo, and from a few other directions; fix what is wrong while the budget allows.
Then write `result.json`: {{"texture": "texture.png", "image_calls": <number>, "generated_images": [...],
"method": "<how you built the texture>", "self_check": "<what you checked and what still looks wrong>"}}.
"""


def write_obj(path: Path, vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray):
    """OBJ with one UV per vertex (as in the glb): v, vt, f v/vt."""
    with open(path, "w", encoding="ascii") as f:
        f.write("# object_model mesh, metres; vt = UV (u right, v up)\n")
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
        f.writelines(f"vt {u:.6f} {v:.6f}\n" for u, v in uv)
        f.writelines(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n" for a, b, c in faces)


def task_text(config: EgoConfig) -> str:
    return TASK.format(size=config.texture.size, max_calls=config.texture.max_image_calls, model=config.image.model,
                       python=PYTHON, imagegen=IMAGEGEN)


def prepare_workspace(store, job: dict, video: str, config: EgoConfig, folder: Path) -> dict:
    import trimesh

    from object_model.cli import paths as om_paths
    om = om_paths(video)["out"]
    frames = om_paths(video)["depth_run"] / "frames"
    if not (om / "mesh_object.glb").exists():
        raise FileNotFoundError(f"run `object_model run {video}` first ({om})")
    rec = json.loads((om / "model.json").read_text(encoding="utf-8"))
    pose = json.loads((om / "object_pose.json").read_text())
    k = json.loads((om / "intrinsics.json").read_text())
    ref = rec["reference_frame"]

    (folder / "model").mkdir(parents=True)
    mesh = trimesh.load(str(om / "mesh_object.glb"), force="mesh", process=False)
    uv = np.asarray(mesh.visual.uv)
    bare = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    bare.visual = trimesh.visual.TextureVisuals(uv=uv)  # UVs only: SAM3D's own texture is not handed over
    bare.export(folder / "model" / "mesh_object.glb")
    write_obj(folder / "model" / "mesh.obj", mesh.vertices, uv, mesh.faces)
    atomic_json(folder / "model" / "model.json", {
        "units": "metres", "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "oriented_bounding_box_cm": [round(e * 100, 2) for e in rec["mesh"]["obb_extents_m"]],
        "source": "object_model (SAM3D shape, scale and pose fitted to the video)",
        "note": "the absolute scale still carries the depth estimate's error; proportions are right"})

    (folder / "references").mkdir()
    for r in job["references"]:
        shutil.copyfile(store.resolve(job["id"], r), folder / "references" / Path(r["path"]).name)
    desc = store.directory(job["id"]) / "object-description.json"
    if desc.exists():
        shutil.copyfile(desc, folder / "object-description.json")

    (folder / "reference_frame").mkdir()
    shutil.copyfile(frames / f"{ref:06d}.png", folder / "reference_frame" / "photo.png")
    shutil.copyfile(om / "masks" / "1" / f"{ref:06d}.png", folder / "reference_frame" / "object_mask.png")
    atomic_json(folder / "reference_frame" / "camera.json", {
        "frame_index": ref, **{key: k[key] for key in ("fx", "fy", "cx", "cy", "width", "height")},
        "object_to_camera": pose["object_to_camera"], "convention": pose["convention"]})
    (folder / "TASK.md").write_text(task_text(config), encoding="utf-8")
    return {"reference_frame": ref, "mesh": str(om / "mesh_object.glb")}


def _summarise(event: dict) -> str | None:
    item = event.get("item") or {}
    kind = item.get("type")
    if event.get("type") == "item.completed" and kind == "agent_message":
        return "[codex] " + item.get("text", "").strip().replace("\n", " ")[:400]
    if event.get("type") == "item.started" and kind == "command_execution":
        return "[codex $] " + str(item.get("command", ""))[:300]
    if event.get("type") == "item.completed" and kind == "error":
        return "[codex !] " + item.get("message", "")[:300]
    if event.get("type") == "turn.completed":
        return f"[codex] turn finished, usage {event.get('usage')}"
    return None


def run_codex(folder: Path, config: EgoConfig, log=print) -> dict:
    """codex exec in the workspace; returns {exit_code, seconds, image_calls_seen}."""
    key = os.environ.get(config.image.api_key_env)
    if not key:
        raise ValueError(f"Set {config.image.api_key_env} (secrets.env) for image generation")
    if not IMAGEGEN.is_file():
        raise FileNotFoundError(f"Codex imagegen skill not found at {IMAGEGEN}")
    agent = config.agent
    argv = [resolve_codex(agent.executable), "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
            "-c", "sandbox_workspace_write.network_access=true",
            # pass OPENAI_API_KEY / OPENAI_BASE_URL through to Codex's commands (excluded by default as *KEY*)
            "-c", "shell_environment_policy.inherit=all",
            "-c", "shell_environment_policy.ignore_default_excludes=true",
            "--json", "--output-last-message", str(folder / "codex_last_message.txt"), "--cd", str(folder)]
    if agent.model:
        argv += ["--model", agent.model]
    if agent.reasoning_effort:
        argv += ["-c", f'model_reasoning_effort="{agent.reasoning_effort}"']
    argv.append("-")  # prompt from stdin
    env = dict(os.environ, OPENAI_API_KEY=key)
    if config.image.base_url:
        env["OPENAI_BASE_URL"] = config.image.base_url
    t0, calls = time.time(), 0
    with open(folder / "codex_events.jsonl", "w", encoding="utf-8") as events, \
            subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env, text=True, encoding="utf-8", errors="replace") as proc:
        proc.stdin.write(task_text(config))
        proc.stdin.close()
        try:
            for line in proc.stdout:
                events.write(line)
                events.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if line.strip():
                        log("[codex] " + line.strip()[:300])
                    continue
                item = event.get("item") or {}
                if (event.get("type") == "item.started" and item.get("type") == "command_execution"
                        and "image_gen.py" in str(item.get("command")) and "--dry-run" not in str(item.get("command"))):
                    calls += 1
                text = _summarise(event)
                if text:
                    log(text)
                if time.time() - t0 > config.texture.timeout_seconds:
                    proc.kill()
                    log(f"[codex] stopped after {config.texture.timeout_seconds} s")
                    break
        finally:
            proc.wait()
    return {"exit_code": proc.returncode, "seconds": round(time.time() - t0, 1), "image_calls_seen": calls}


def collect(folder: Path, video: str, config: EgoConfig) -> dict:
    """texture.png must exist; attach it unchanged to the mesh; copy both next to object_model's results."""
    import trimesh
    from PIL import Image

    from object_model.cli import paths as om_paths
    tex_path = folder / "texture.png"
    if not tex_path.is_file():
        raise FileNotFoundError(f"Codex did not write {tex_path}")
    with Image.open(tex_path) as img:
        img.load()
        texture = img.convert("RGB")
    mesh = trimesh.load(str(folder / "model" / "mesh_object.glb"), force="mesh", process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(mesh.visual.uv), material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture))
    mesh.export(folder / "mesh_textured.glb")
    out = om_paths(video)["out"]
    shutil.copyfile(tex_path, out / "texture.png")
    shutil.copyfile(folder / "mesh_textured.glb", out / "mesh_textured.glb")
    result = {}
    if (folder / "result.json").is_file():
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {"unreadable": True}
    return {"size": list(texture.size), "expected_size": config.texture.size, "result": result}


def generate_texture(store, job_id: str, video: str, *, retry: bool = False, log=print) -> dict:
    job = store.load(job_id)
    config = EgoConfig.model_validate(json.loads((PROJECT / "configs" / "ego.json").read_text(encoding="utf-8")))
    state = job.setdefault("texture", {"object": video, "attempts": []})
    if state.get("object") != video:
        raise ValueError(f"this job's texture is for {state.get('object')!r}, not {video!r}")
    last = state["attempts"][-1] if state["attempts"] else None
    if last and last.get("status") in ("started", "codex_finished") and not retry:
        raise ValueError(f"{last['folder']} was started and has no collected result; inspect it and rerun with "
                         "--retry to start a new Codex session (it spends image calls again)")
    if last and last.get("status") == "done" and not retry:
        log(f"texture already generated in {last['folder']} (use --retry for a new attempt)")
        return job
    number = len(state["attempts"]) + 1
    folder = store.directory(job_id) / f"texture-{number:02d}"
    folder.mkdir()
    attempt = {"number": number, "folder": folder.name, "created_at": now(), "status": "preparing",
               "image_model": config.image.model, "max_image_calls": config.texture.max_image_calls}
    state["attempts"].append(attempt)
    store.save(job)
    attempt.update(prepare_workspace(store, job, video, config, folder))
    attempt["status"] = "started"
    attempt["codex_started_at"] = now()
    store.save(job)
    log(f"[texture] Codex session in {folder}")
    attempt.update(run_codex(folder, config, log))
    attempt["status"] = "codex_finished"
    store.save(job)
    try:
        attempt["collected"] = collect(folder, video, config)
        attempt["status"] = "done"
        atomic_json(Path(attempt["mesh"]).parent / "texture.json",
                    {"job": job_id, "attempt": folder.name, "collected": attempt["collected"], "written_at": now()})
        log(f"[texture] done: {folder / 'texture.png'} -> also in {Path(attempt['mesh']).parent}")
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        log(f"[texture] failed: {attempt['error']}")
    store.save(job)
    return job


def format_status(job: dict) -> str:
    lines = [f"Job: {job['id']}", f"State: {job['status']}"]
    state = job.get("texture")
    if state:
        lines.append(f"Texture for: {state['object']}")
        for a in state["attempts"]:
            text = f"  {a['folder']}: {a['status']}"
            if "image_calls_seen" in a:
                text += f", {a['image_calls_seen']} image calls, {a.get('seconds')} s"
            if a.get("error"):
                text += f" ({a['error'][:200]})"
            lines.append(text)
    return "\n".join(lines)
