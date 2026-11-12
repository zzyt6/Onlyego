#!/usr/bin/env bash
# Linux side of video_depth: MegaSaM (compiled CUDA extensions droid_backends, lietorch) in a conda env inside WSL
# (distro V2D-Ubuntu). `base` = env zyego-cuda with torch 2.7.0 + CUDA 11.8 (the toolkit this distro already has);
# `megasam` clones it into zyego-megasam and builds MegaSaM in place. Re-running is safe: finished steps are skipped.
#
#   wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_cuda_env.sh [all|base|megasam|megasam_check]
set -euo pipefail

ENV_NAME=zyego-cuda
EXTERNAL=$HOME/zyego-external
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

case "$STEP" in
  all) base; megasam ;;
  base|megasam|megasam_check) "$STEP" ;;
  *) echo "usage: $0 [all|base|megasam|megasam_check]"; exit 2 ;;
esac
