"""Run a Python module of this repo inside a WSL conda env (MegaSaM here; object_model.wsl for the model wrappers).

The repo is reached through /mnt/<drive>/..., so the Linux side imports the same source files.
"""

import os
import re
import subprocess
from pathlib import Path, PureWindowsPath

DISTRO = os.environ.get("ZYEGO_WSL_DISTRO", "V2D-Ubuntu")
ENV_NAME = os.environ.get("ZYEGO_WSL_ENV", "zyego-megasam")
EXTERNAL = "$HOME/zyego-external"
PROJECT = Path(__file__).resolve().parents[2]


def to_wsl(path: Path | str) -> str:
    """E:\\ZY_ego\\runs -> /mnt/e/ZY_ego/runs"""
    p = PureWindowsPath(Path(path).resolve())
    if not re.fullmatch(r"[A-Za-z]:\\?", p.anchor):
        raise ValueError(f"Not a drive path, cannot map into WSL: {path}")
    return "/mnt/" + p.drive[0].lower() + "/" + "/".join(p.parts[1:])


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def command(module: str, args: list[str], env: str = ENV_NAME, pythonpath: tuple[str, ...] = ()) -> list[str]:
    """pythonpath: extra Linux paths after the repo's src/ (may use $HOME, so they are double-quoted)."""
    extra = "".join(f':"{p}"' for p in pythonpath)
    script = (f'source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate {env} && '
              f'export PYTHONPATH={_quote(to_wsl(PROJECT / "src"))}{extra} '
              f'PYTHONUNBUFFERED=1 && exec python -m {module} ' + " ".join(_quote(a) for a in args))
    return ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", script]


def run_module(module: str, args: list[str], log=print, env: str = ENV_NAME,
               pythonpath: tuple[str, ...] = ()) -> int:
    """Stream the Linux process output; returns its exit code."""
    env_vars = dict(os.environ, MSYS_NO_PATHCONV="1")
    with subprocess.Popen(command(module, args, env, pythonpath), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=env_vars, text=True, encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            log(line.rstrip())
    return proc.returncode
