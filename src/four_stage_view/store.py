"""Job-scoped files, atomic JSON checkpoints and checksummed evidence."""

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def numbered_directory(parent: Path, prefix: str) -> Path:
    """Allocate a readable sequence number without colliding with another launch."""
    parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r"-(\d+)$")
    number = max((int(m[1]) for p in parent.iterdir()
                  if (m := pattern.fullmatch(p.name))), default=0) + 1
    while True:
        path = parent / f"{prefix}-{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def within(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path is outside the configured workspace")
    return path


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"(?:[a-f0-9]{16}|job-[0-9]{4,})", job_id):
            raise ValueError("Invalid job ID")
        return within(self.root / job_id, self.root)

    def lock(self, job_id: str):
        return FileLock(self.directory(job_id) / ".lock", timeout=0)

    def load(self, job_id: str) -> dict:
        return json.loads((self.directory(job_id) / "job.json").read_text(encoding="utf-8"))

    def save(self, job: dict):
        job["updated_at"] = now()
        latest = job.get("attempts", [])[-1] if job.get("attempts") else {}
        event = {
            "status": job["status"],
            "attempt": latest.get("number"),
            "phase": latest.get("phase"),
            "error": latest.get("error"),
        }
        events = job.setdefault("events", [])
        if not events or {k: events[-1].get(k) for k in event} != event:
            events.append({"at": job["updated_at"], **event})
        atomic_json(self.directory(job["id"]) / "job.json", job)

    def artifact(self, job_id: str, path: Path) -> dict:
        directory = self.directory(job_id)
        path = within(path, directory)
        # POSIX separators keep job.json portable between Windows and Linux/WSL.
        return {"path": path.relative_to(directory).as_posix(), "sha256": digest(path)}

    def resolve(self, job_id: str, artifact: dict) -> Path:
        path = within(self.directory(job_id) / artifact["path"], self.directory(job_id))
        if digest(path) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum changed: {artifact['path']}")
        return path
