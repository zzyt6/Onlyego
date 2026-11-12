"""Terminal progress on stderr and read-only inspection of video-selection runs."""

import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .store import atomic_json, now


def format_progress(value):
    current, total = value.get("current"), value.get("total")
    bar = ""
    if total:
        filled = min(20, max(0, int(20 * current / total)))
        bar = f" [{'#' * filled}{'-' * (20 - filled)}] {current}/{total}"
    return f"{value.get('phase', '状态')}{bar} {value.get('message', '')}".rstrip()


class VideoProgress:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.last_write = 0.0
        self.last_phase = None
        self.lock = threading.Lock()

    def emit(self, phase, current=None, total=None, message="", *, force=False):
        with self.lock:
            clock = time.monotonic()
            if not force and phase == self.last_phase and clock - self.last_write < 1:
                return
            self.last_write, self.last_phase = clock, phase
            value = dict(phase=phase, current=current, total=total, message=message, updated_at=now())
            atomic_json(self.directory / "progress.json", value)
            print(format_progress(value), file=sys.stderr, flush=True)

    @contextmanager
    def waiting(self, phase, current, total, message):
        start = time.monotonic()
        stop = threading.Event()
        self.emit(phase, current, total, message + "；等待模型响应…", force=True)

        def heartbeat():
            while not stop.wait(10):
                self.emit(
                    phase, current, total,
                    message + f"；仍在等待模型响应，已等待 {int(time.monotonic() - start)} 秒",
                    force=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def inspect_video_run(root, run=None):
    root = Path(root).expanduser().resolve()
    if run is None:
        directories = [p for p in (root / "video-inputs").glob("*") if p.is_dir()]
        if not directories:
            raise ValueError(f"No video-selection run found under {root / 'video-inputs'}")
        directory = max(directories, key=lambda p: p.stat().st_mtime)
    else:
        if not run or Path(run).name != run or run in (".", ".."):
            raise ValueError("--run must be a video-inputs directory name")
        directory = root / "video-inputs" / run
        if not directory.is_dir():
            raise ValueError(f"Video run not found: {directory}")
    state = _read(directory / "state.json")
    progress = _read(directory / "progress.json")
    status = state.get("status", "unknown")
    if status == "selected":
        progress = {"phase": "完成", "message": f"任务 {state.get('job_id', '')}；参考帧已保存"}
    elif status == "needs_attention":
        progress = {"phase": "需要检查", "message": f"{state.get('error_type', '')}；记录：{state.get('selection_directory', directory)}"}
    elif not progress:
        # Compatibility with an already running process started before progress support.
        decoded = len(list((directory / "frames").glob("*.png")))
        manifest = _read(directory / "frames" / "manifest.json")
        claims = list((directory / "reviews").glob("*/claim.json"))
        completed = sum(_read(p).get("status") == "completed" for p in claims)
        pending = [p for p in claims if _read(p).get("status") != "completed"]
        if pending:
            latest = max(pending, key=lambda p: p.stat().st_mtime)
            claim_status = _read(latest).get("status", "unknown")
            progress = {"phase": "模型审核", "message": f"已完成 {completed} 次调用；当前 {latest.parent.name}（{claim_status}）"}
        elif claims:
            progress = {"phase": "选帧", "message": f"已完成 {completed} 次调用；等待下一阶段记录"}
        elif status == "extracting":
            progress = {"phase": "解码视频", "current": decoded,
                        "total": manifest.get("reported_frame_count"), "message": f"已写出 {decoded} 帧"}
        else:
            count = sum(1 for p in (directory / "review-frames").glob("*")
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            progress = {"phase": "准备审核图片", "current": count, "total": decoded,
                        "message": "尚无模型调用记录"}
    return {"directory": str(directory), "status": status, "progress": progress}


def show_video_status(root, run=None, watch=False):
    last = None
    try:
        while True:
            value = inspect_video_run(root, run)
            # Pin the first selected run so another task cannot switch the watcher.
            run = Path(value["directory"]).name
            message = value["directory"] + "\n" + format_progress(value["progress"])
            if message != last:
                print(message, flush=True)
                last = message
            if not watch or value["status"] in ("selected", "needs_attention"):
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print("已停止查看；原任务继续运行。", flush=True)
