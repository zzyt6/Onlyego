"""four_stage_view: video -> object reference crops + description (step 1) -> Codex generates the texture of the
fitted object mesh from them (step 2)."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIDEOS = PROJECT / "data" / "videos"


def load_secrets(path: Path = PROJECT / "secrets.env"):
    """KEY=value lines; existing environment variables win. Values are never printed."""
    if not path.is_file():
        return
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        pairs.append((key.strip(), value.strip().strip('"').strip("'")))
    for key, value in pairs:
        os.environ.setdefault(key, value)
    # values may refer to other variables (AGILE_IMAGE_API_KEY=$AGILE_OPENAI_API_KEY): expand them, without
    # touching variables that already came from the environment
    for key, value in pairs:
        if "$" in value and os.environ.get(key) == value:
            os.environ[key] = os.path.expandvars(value)


def resolve_video(value: Path) -> Path:
    """A plain name (demo01 or demo01.mp4) is looked up in <project>/data/videos; a path is used as given."""
    if value.is_file() or value.parent != Path("."):
        return value
    for candidate in (VIDEOS / value.name, VIDEOS / f"{value.name}.mp4"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video not found: {value} (looked in {VIDEOS})")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    load_secrets()
    parser = argparse.ArgumentParser(prog="four_stage_view", description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "runs", help="Run folder (default: <project>/runs)")
    sub = parser.add_subparsers(dest="command", required=True)
    video = sub.add_parser("analyze-video", help="Step 1: find the manipulated object, crop references, describe it")
    video.add_argument("video", type=Path, help="Name in data/videos (e.g. demo01) or a path to an MP4")
    video.add_argument("--config", type=Path, default=PROJECT / "configs" / "ego.json")
    video.add_argument("--agent-config", type=Path, default=PROJECT / "configs" / "video-agent.json")
    tex = sub.add_parser("generate-texture",
                         help="Step 2: Codex generates the texture of object_model's fitted mesh from the references")
    tex.add_argument("job")
    tex.add_argument("--object", metavar="VIDEO", required=True,
                     help="video whose fitted mesh gets the texture (runs/object_model/VIDEO)")
    tex.add_argument("--retry", action="store_true",
                     help="start a new Codex session although an earlier one exists (spends image calls again)")
    status = sub.add_parser("status", help="Show a job's state")
    status.add_argument("job")
    status.add_argument("--json", action="store_true")
    watch = sub.add_parser("video-status", help="Progress of a running video analysis")
    watch.add_argument("--run", help="video-inputs folder name; default: latest")
    watch.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)

    from .store import Store
    store = Store(args.root)
    try:
        if args.command == "analyze-video":
            from .video_agent import analyze_video_file
            job = analyze_video_file(store, resolve_video(args.video), json.loads(args.config.read_text(encoding="utf-8")),
                                     args.agent_config)
            print(f"Job: {job['id']}\nWorkspace: {store.directory(job['id'])}\n"
                  f"Next: four_stage_view generate-texture {job['id']} --object <video>")
        elif args.command == "generate-texture":
            from .texture import format_status, generate_texture
            job = generate_texture(store, args.job, args.object, retry=args.retry)
            print(format_status(job))
            print(f"Workspace: {store.directory(job['id'])}")
        elif args.command == "status":
            from .texture import format_status
            job = store.load(args.job)
            print(json.dumps(job, ensure_ascii=False, indent=2) if args.json else format_status(job))
        elif args.command == "video-status":
            from .video_progress import show_video_status
            show_video_status(args.root, args.run, args.watch)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
