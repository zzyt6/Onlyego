"""Run one HOI model wrapper (WiLoR, Grounding DINO, SAM2, SAM3D, FoundationPose) in its WSL conda env.

Envs built by tools/wsl/setup_hoi_env.sh; each module is called directly with `python -m`.
"""

import os
import subprocess
from pathlib import Path

from video_depth.wsl import DISTRO, PROJECT, _quote, to_wsl

MODEL = to_wsl(PROJECT / "model")
_EGL = {"PYOPENGL_PLATFORM": "egl"}

# module group -> (conda env, extra environment variables)
ENVS = {
    "wilor": ("hoi-hands", _EGL),
    "grounding_dino": ("hoi-seg", {"HF_HOME": f"{MODEL}/huggingface", "HF_HUB_OFFLINE": "1",
                                   "TRANSFORMERS_OFFLINE": "1"}),
    "sam2": ("hoi-seg", {}),
    "sam3d": ("hoi-sam3d", {**_EGL, "HYDRA_FULL_ERROR": "1", "LIDRA_SKIP_INIT": "1"}),
    "foundation_pose": ("hoi-fp", {"OPENCV_IO_ENABLE_OPENEXR": "1",
                                   "TRTEXEC_PATH": "$HOME/zyego-external/tensorrt/bin/trtexec"}),
}
# extra import paths after src/ (FoundationPose imports its own `learning`, `mycpp`, ... packages)
EXTRA_PATH = {"foundation_pose": [to_wsl(PROJECT / "src" / "HOI" / "foundation_pose" / "FoundationPose")]}


def _value(v: str) -> str:
    """Single-quoted, except a leading $HOME, which must expand on the Linux side."""
    return '"$HOME"' + _quote(v[5:]) if v.startswith("$HOME") else _quote(v)


def weights(name: str) -> str:
    return f"{MODEL}/{name}"


def run(group: str, module: str, args: list[str], log=print) -> None:
    """python -m <module> <args> in the env of `group`; raises on a non-zero exit."""
    env, variables = ENVS[group]
    pythonpath = ":".join([to_wsl(PROJECT / "src"), *EXTRA_PATH.get(group, [])])
    exports = " ".join(f"{k}={_value(v)}" for k, v in
                       {**variables, "PYTHONPATH": pythonpath, "PYTHONUNBUFFERED": "1"}.items())
    # cd $HOME: wsl.exe starts in the Windows folder, where e.g. model/sam2 could shadow the sam2 package
    script = (f'source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate {env} && cd "$HOME" && '
              f'export {exports} && exec python -m {module} ' + " ".join(_quote(a) for a in args))
    cmd = ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", script]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", env=dict(os.environ, MSYS_NO_PATHCONV="1")) as proc:
        for line in proc.stdout:
            line = line.rstrip()
            if line and "%|" not in line:  # drop tqdm redraws
                log(line)
    if proc.returncode:
        raise RuntimeError(f"{module} failed in WSL env {env} (exit {proc.returncode})")


def path(p: Path | str) -> str:
    return to_wsl(Path(p))
