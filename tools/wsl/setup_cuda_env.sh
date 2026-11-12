#!/usr/bin/env bash
# Linux side of ZY_ego: a conda env inside WSL (distro V2D-Ubuntu) holding the parts that need
# compiled CUDA extensions -- DROID-SLAM (lietorch, droid_backends) and gsplat -- plus a second env for
# MegaSaM, whose modified DROID compiles modules with the same names (droid_backends, lietorch).
# Mirrors V2D's Docker images (reconstruction/modules/v2d_droid_slam/docker/Dockerfile and
# v2d_gsplat_refinement/docker/Dockerfile) but with torch 2.7.0 + CUDA 11.8, the toolkit this
# WSL distro already has. Re-running is safe: finished steps are skipped.
#
#   wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_cuda_env.sh [all|base|droid|gsplat|check|megasam]
set -euo pipefail

ENV_NAME=zyego-cuda
EXTERNAL=$HOME/zyego-external
DROID_COMMIT=2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99
GSPLAT_VERSION=1.4.0
MEGASAM_ENV=zyego-megasam
MEGASAM_COMMIT=a27b4e633c5cc0828a62ed943ef9f6505705fd3f
MODEL_DIR=/mnt/e/ZY_ego/model/megasam
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090; V2D builds 8.0 8.6 8.9 9.0
export MAX_JOBS=${MAX_JOBS:-8}
STEP=${1:-all}

source "$HOME/miniforge3/etc/profile.d/conda.sh"

base() {
  if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n "$ENV_NAME" python=3.10
  fi
  conda activate "$ENV_NAME"
  python -c "import torch; assert torch.__version__.startswith('2.7.0')" 2>/dev/null || \
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
  python -c "import torch_scatter" 2>/dev/null || \
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
  pip install "numpy<2" scipy opencv-python-headless pillow pyyaml tqdm matplotlib einops ninja
}

droid() {
  conda activate "$ENV_NAME"
  mkdir -p "$EXTERNAL"
  if [ ! -d "$EXTERNAL/DROID-SLAM/.git" ]; then
    git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git "$EXTERNAL/DROID-SLAM"
  fi
  cd "$EXTERNAL/DROID-SLAM"
  git checkout -q "$DROID_COMMIT" && git submodule update --init --recursive -q
  python -c "import lietorch" 2>/dev/null || (cd thirdparty/lietorch && pip install --no-build-isolation -v -e .)
  python -c "import droid_backends" 2>/dev/null || pip install --no-build-isolation -v -e .
}

gsplat() {
  conda activate "$ENV_NAME"
  # gsplat's CUDA code includes GLM headers (V2D's Dockerfile: apt-get install libglm-dev);
  # take them from conda-forge into the env instead of touching the system
  [ -f "$CONDA_PREFIX/include/glm/glm.hpp" ] || conda install -y -c conda-forge glm
  export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
  python -c "import gsplat, gsplat.csrc" 2>/dev/null || \
    pip install --no-build-isolation --no-binary gsplat "gsplat==$GSPLAT_VERSION"
  # V2D's refinement also needs these (MANO layer for hands, mesh I/O)
  pip install trimesh rtree imageio imageio-ffmpeg roma "smplx==0.1.28"
  python -c "import manotorch" 2>/dev/null || pip install --no-build-isolation \
    git+https://github.com/lixiny/manotorch.git git+https://github.com/mattloper/chumpy.git
}

megasam() {
  # MegaSaM (Li et al., CVPR 2025): DROID-SLAM variant with mono-depth priors + consistent video depth.
  # Separate env cloned from $ENV_NAME (same torch 2.7.0+cu118), because MegaSaM's base/ builds its own
  # droid_backends and lietorch. Extensions are built in place and found through PYTHONPATH (see
  # src/video_depth/megasam.py): base/setup.py holds two setup() calls, which pip cannot install.
  conda env list | grep -q "^$MEGASAM_ENV " || conda create -y -n "$MEGASAM_ENV" --clone "$ENV_NAME"
  conda activate "$MEGASAM_ENV"
  pip uninstall -y droid_backends lietorch 2>/dev/null || true
  python -c "import kornia" 2>/dev/null || pip install "kornia==0.7.4"
  mkdir -p "$EXTERNAL"
  if [ ! -d "$EXTERNAL/mega-sam/.git" ]; then
    git -c http.version=HTTP/1.1 clone --recursive https://github.com/mega-sam/mega-sam.git "$EXTERNAL/mega-sam"
  fi
  cd "$EXTERNAL/mega-sam"
  git checkout -q "$MEGASAM_COMMIT" && git -c http.version=HTTP/1.1 submodule update --init --recursive -q
  # base/setup.py hard-codes sm_70..sm_86 for both extensions; build for this GPU (sm_89) only
  sed -i -E "/'-gencode=arch=compute_(70|75|80),code=sm_(70|75|80)',/d; s/compute_86,code=sm_86/compute_89,code=sm_89/" base/setup.py
  # PyTorch >= 2.x: the dispatch macros need a ScalarType, tensor.type() no longer converts. Same fix as
  # upstream lietorch (3 calls in MegaSaM's kernels, 19 + 19 in its bundled, older lietorch)
  sed -i -E 's/(AT_DISPATCH_[A-Z_]+\()([A-Za-z0-9_]+)\.type\(\)/\1\2.scalar_type()/' \
    base/src/correlation_kernels.cu base/src/altcorr_kernel.cu
  sed -i -E 's/(DISPATCH_GROUP_AND_FLOATING_TYPES\(group_id, *)([A-Za-z0-9_]+)\.type\(\)/\1\2.scalar_type()/' \
    base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu base/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp
  cd base
  if ! PYTHONPATH="$PWD:$PWD/thirdparty/lietorch" python -c "import droid_backends, lietorch_backends" 2>/dev/null; then
    python setup.py build_ext --inplace
  fi
  mkdir -p "$MODEL_DIR"
  [ -f "$MODEL_DIR/megasam_final.pth" ] || cp "$EXTERNAL/mega-sam/checkpoints/megasam_final.pth" "$MODEL_DIR/"
  if [ ! -f "$MODEL_DIR/raft-things.pth" ]; then   # the source of RAFT's own download_models.sh
    local tmp
    tmp=$(mktemp -d)
    curl -fL --retry 5 -o "$tmp/models.zip" https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
    # no unzip in this distro; python's zipfile does the same
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extract('models/raft-things.pth', sys.argv[2])" \
      "$tmp/models.zip" "$tmp"
    cp "$tmp/models/raft-things.pth" "$MODEL_DIR/"
    rm -rf "$tmp"
  fi
  [ -f "$MODEL_DIR/raft-things.pth" ] || { echo "raft-things.pth missing" >&2; exit 1; }
  megasam_check
}

megasam_check() {
  conda activate "$MEGASAM_ENV"
  local M="$EXTERNAL/mega-sam"
  PYTHONPATH="$M/base:$M/base/thirdparty/lietorch:$M/base/droid_slam:$M/cvd_opt:$M/cvd_opt/core" python - <<'PY'
import torch, kornia
import droid_backends, lietorch
from lietorch import SE3
from droid import Droid
from raft import RAFT
x = SE3.Random(4, device="cuda")
print("megasam env: torch", torch.__version__, "kornia", kornia.__version__,
      "| lietorch from", lietorch.__file__.split("zyego-external/")[-1],
      "| SE3 round trip", float((x * x.inv()).log().abs().max()))
PY
}

check() {
  conda activate "$ENV_NAME"
  PYTHONPATH="$EXTERNAL/DROID-SLAM/droid_slam:${PYTHONPATH:-}" python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
import lietorch, droid_backends, torch_scatter
from droid import Droid
from lietorch import SE3
x = SE3.Random(4, device="cuda"); print("lietorch SE3 ok", tuple((x * x.inv()).log().abs().max().shape))
import gsplat
from gsplat.rendering import rasterization
means = torch.randn(100, 3, device="cuda") + torch.tensor([0, 0, 3.0], device="cuda")
quats = torch.nn.functional.normalize(torch.randn(100, 4, device="cuda"), dim=-1)
scales = torch.full((100, 3), 0.05, device="cuda"); opac = torch.ones(100, device="cuda")
colors = torch.rand(100, 3, device="cuda")
K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]], device="cuda")
img, alpha, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device="cuda")[None], K, 128, 96)
print("gsplat", gsplat.__version__, "render", tuple(img.shape), "alpha max %.2f" % alpha.max().item())
from manotorch.manolayer import ManoLayer
print("manotorch ok")
EOF
}

case "$STEP" in
  all) base; droid; gsplat; check; megasam ;;
  base|droid|gsplat|check|megasam|megasam_check) "$STEP" ;;
  *) echo "usage: $0 [all|base|droid|gsplat|check|megasam|megasam_check]"; exit 2 ;;
esac
