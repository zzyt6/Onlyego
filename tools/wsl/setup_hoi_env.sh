#!/usr/bin/env bash
# Environments + weights for src/HOI (V2D's hand + object modules), inside WSL distro V2D-Ubuntu.
# Each V2D module ran in its own Docker image (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel); here each group
# of modules gets one conda env with the same PyTorch 2.5.1 + CUDA 12.4 wheels, built from that module's
# Dockerfile (copied to src/HOI/<module>/docker/Dockerfile). Re-running is safe: finished steps are skipped.
#
#   wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/setup_hoi_env.sh <step>
#   steps: system cuda124 wheels hands mediapipe hawor seg sam3d fp trtexec | weights_hands weights_mediapipe
#          weights_hawor weights_seg weights_sam3d weights_fp | check | all
#
#   env            modules (src/HOI/...)                         Dockerfile(s)
#   hoi-hands      wilor hamer hand_alignment mesh               wilor, hamer, hand_alignment, mesh
#   hoi-mediapipe  mediapipe (no torch; own opencv-contrib)      mediapipe
#   hoi-hawor      hawor (builds HaWoR's own DROID-SLAM)          hawor
#   hoi-seg        sam2 grounding_dino                            sam2, grounding_dino
#   hoi-sam3d      sam3d                                          sam3d
#   hoi-fp         foundation_pose mesh                           foundation_pose
#   zyego-cuda     gsplat_refinement (already built by setup_cuda_env.sh: gsplat 1.4.0, manotorch, roma)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"

EXTERNAL=$HOME/zyego-external
WHEELS=$EXTERNAL/wheels
DOWNLOADS=$EXTERNAL/downloads
PROJECT=/mnt/e/ZY_ego
SRC=$PROJECT/src
MODEL=$PROJECT/model
VERSIONS=$PROJECT/tools/wsl/hoi_versions.txt
CUDA124=/usr/local/cuda-12.4
HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905   # pinned in V2D's Dockerfile
MOGE_SAM3D_COMMIT=a8c37341bc0325ca99b9d57981cc3bb2bd3e255b  # pinned in V2D's sam3d Dockerfile
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090 (V2D builds 8.0 8.6 8.9 9.0)
export MAX_JOBS=${MAX_JOBS:-8}
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$EXTERNAL" "$WHEELS" "$DOWNLOADS"
STEP=${1:-}

use_cuda124() { export CUDA_HOME=$CUDA124 PATH=$CUDA124/bin:$PATH LD_LIBRARY_PATH=$CUDA124/lib64:${LD_LIBRARY_PATH:-}; }
record() { grep -v "^$1 " "$VERSIONS" 2>/dev/null > "$VERSIONS.tmp" || true; echo "$1 $2" >> "$VERSIONS.tmp"; sort -o "$VERSIONS" "$VERSIONS.tmp"; rm -f "$VERSIONS.tmp"; }
clone() {  # clone <url> <dir> [commit]: pinned commit if given, else current HEAD (recorded)
  local url=$1 dir=$2 commit=${3:-}
  [ -d "$dir/.git" ] || git -c http.version=HTTP/1.1 clone --recursive "$url" "$dir"
  if [ -n "$commit" ]; then git -C "$dir" checkout -q "$commit"; fi
  git -C "$dir" -c http.version=HTTP/1.1 submodule update --init --recursive -q
  record "$(basename "$dir")" "$(git -C "$dir" rev-parse HEAD) $url"
}
mkenv() {  # conda env with python 3.11 (as the pytorch:2.5.1 image) and torch 2.5.1+cu124
  local name=$1 torch=${2:-yes}
  conda env list | grep -q "^$name " || conda create -y -q -n "$name" python=3.11
  conda activate "$name"
  pip install -q "setuptools<75" wheel ninja packaging
  if [ "$torch" = yes ]; then
    python -c "import torch; assert torch.__version__.startswith('2.5.1')" 2>/dev/null || \
      pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    # v2d_common[io] (frame I/O in HOI.common.video), installed in V2D's images that use it
    pip install -q "imageio[pyav]" "av==18.0.0" h5py
  fi
  # numpy<2 as V2D's stack; pandas 3.x does not load with numpy 1.x, so keep 2.x wherever something pulls pandas in
  pip install -q "numpy<2" "pandas<3" scipy Pillow opencv-python tqdm pyyaml
}
hf_token() {  # HF_TOKEN from the environment or secrets.env (never printed)
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$PROJECT/secrets.env" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "$PROJECT/secrets.env" | head -1 | cut -d= -f2- | tr -d '\r"' || true)
  fi
  export HF_TOKEN=${HF_TOKEN:-}
}

system() {
  # archive.ubuntu.com does not answer from this network (not even through the Clash proxy), so this one install
  # reads a temporary sources list pointing at a domestic mirror; /etc/apt is left untouched.
  local mirror=${APT_MIRROR:-https://mirrors.aliyun.com/ubuntu/} list=/tmp/zyego-apt-sources.list
  local codename; codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  : > "$list"
  for suite in "$codename" "$codename-updates" "$codename-security"; do
    echo "deb $mirror $suite main restricted universe multiverse" >> "$list"
  done
  local opts=(-o "Dir::Etc::sourcelist=$list" -o "Dir::Etc::sourceparts=-" -o "APT::Get::List-Cleanup=0")
  # sudo drops http(s)_proxy; Clash runs in fake-IP mode here (names resolve to 198.18.x.x), so apt must use the
  # same proxy as curl/wget/pip or nothing connects
  local proxy=${https_proxy:-${http_proxy:-}}
  [ -n "$proxy" ] && opts+=(-o "Acquire::http::Proxy=$proxy" -o "Acquire::https::Proxy=$proxy")
  sudo DEBIAN_FRONTEND=noninteractive apt-get "${opts[@]}" update -q
  sudo DEBIAN_FRONTEND=noninteractive apt-get "${opts[@]}" install -y -q --no-install-recommends \
    libosmesa6 libosmesa6-dev libgl1 libegl1 libgles2 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
    libgomp1 libglfw3 freeglut3-dev cmake ninja-build libeigen3-dev libboost-all-dev libglm-dev \
    wget zip unzip git-lfs ffmpeg build-essential
}

cuda124() {  # the toolkit of V2D's base image, for compiling extensions against torch cu124
  if [ ! -x "$CUDA124/bin/nvcc" ]; then
    local run=$DOWNLOADS/cuda_12.4.1_550.54.15_linux.run prev
    wget -q -c -O "$run" https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    prev=$(readlink -f /usr/local/cuda || true)
    sudo sh "$run" --silent --toolkit --toolkitpath="$CUDA124" --no-man-page --override
    [ -n "$prev" ] && sudo ln -sfn "$prev" /usr/local/cuda   # leave the existing default CUDA as it was
  fi
  "$CUDA124/bin/nvcc" --version | tail -1
}

wheels() {  # pytorch3d + nvdiffrast compiled once for torch 2.5.1/cu124/py3.11, used by hoi-sam3d and hoi-fp
  mkenv hoi-build
  use_cuda124
  clone https://github.com/facebookresearch/pytorch3d.git "$EXTERNAL/pytorch3d"
  clone https://github.com/NVlabs/nvdiffrast.git "$EXTERNAL/nvdiffrast"
  ls "$WHEELS"/pytorch3d-*.whl >/dev/null 2>&1 || pip wheel --no-build-isolation --no-deps -w "$WHEELS" "$EXTERNAL/pytorch3d"
  ls "$WHEELS"/nvdiffrast-*.whl >/dev/null 2>&1 || pip wheel --no-build-isolation --no-deps -w "$WHEELS" "$EXTERNAL/nvdiffrast"
  ls -1 "$WHEELS"
}

hands() {  # wilor + hamer + hand_alignment + mesh Dockerfiles
  mkenv hoi-hands
  use_cuda124
  pip install -q pyrender trimesh smplx yacs hydra-core omegaconf einops timm huggingface_hub "transformers<5" \
    safetensors ultralytics dill pytorch-lightning webdataset braceexpand av joblib loguru scikit-image \
    pyopengl fast-simplification rtree imageio imageio-ffmpeg
  python -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy.git
  python -c "import manotorch" 2>/dev/null || pip install -q --no-build-isolation git+https://github.com/lixiny/manotorch.git
  clone https://github.com/warmshao/WiLoR-mini.git "$EXTERNAL/WiLoR-mini"
  # WiLoR-mini pins torch<=2.5 (pip would swap 2.5.1 for 2.5.0), so --no-deps + its other requirements at its pins
  pip install -q --no-deps -e "$EXTERNAL/WiLoR-mini"
  pip install -q roma "smplx==0.1.28" "ultralytics==8.1.34"
  clone https://github.com/geopavlakos/hamer.git "$EXTERNAL/hamer" "$HAMER_COMMIT"
  pip install -q --no-deps -e "$EXTERNAL/hamer"
  python -c "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline; \
from hamer.models import load_hamer; from hamer.datasets.utils import generate_image_patch_cv2; \
import manotorch, pyrender, smplx; print('hoi-hands: wilor_mini, hamer, manotorch, pyrender OK')"
}

mediapipe() {  # mediapipe Dockerfile (python:3.11-slim, no torch; mediapipe brings opencv-contrib)
  mkenv hoi-mediapipe no
  pip install -q "mediapipe==0.10.18" opencv-python-headless
  python -c "import mediapipe as mp; from mediapipe.tasks.python import vision; print('hoi-mediapipe', mp.__version__, 'OK')"
}

hawor() {  # hawor Dockerfile
  mkenv hoi-hawor
  use_cuda124
  pip install -q gdown joblib natsort einops timm yacs omegaconf hydra-core smplx pyrender trimesh \
    "pytorch-lightning==2.2.4" lightning-utilities "torchmetrics==1.4.0" imageio imageio-ffmpeg av
  clone https://github.com/ThunderVVV/HaWoR.git "$EXTERNAL/HaWoR"
  if [ -f "$EXTERNAL/HaWoR/requirements.txt" ]; then
    # V2D runs `pip install -r requirements.txt || true`: all-or-nothing, so its unbuildable pytorch3d@stable line
    # leaves every package uninstalled. Here the lines that need special handling get it, the rest go one by one.
    python -c "import pytorch3d" 2>/dev/null || pip install -q "$WHEELS"/pytorch3d-*.whl
    python -c "import chumpy" 2>/dev/null || pip install -q --no-build-isolation git+https://github.com/mattloper/chumpy.git
    python -c "import torch_scatter" 2>/dev/null || pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
    # mmcv 1.3.9's setup.py needs pkg_resources, gone from the latest setuptools in pip's isolated build env
    python -c "import mmcv" 2>/dev/null || pip install -q --no-build-isolation "mmcv==1.3.9"
    local failed=()
    while read -r req; do
      req=${req%%#*}; req=$(echo "$req" | tr -d '\r' | xargs)
      [ -z "$req" ] && continue
      echo "$req" | grep -qiE '^(torch|torchvision|torchaudio|torch-scatter|mmcv)([=<> ]|$)|pytorch3d|chumpy' && continue
      pip install -q "$req" || failed+=("$req")
    done < "$EXTERNAL/HaWoR/requirements.txt"
    [ ${#failed[@]} -gt 0 ] && echo "HaWoR requirements that did not install: ${failed[*]}"
  fi
  python -c "import droid_backends" 2>/dev/null || (cd "$EXTERNAL/HaWoR/thirdparty/DROID-SLAM" && python setup.py install)
  pip install -q "matplotlib<3.9" "numpy<2"
  (cd "$EXTERNAL/HaWoR" && PYTHONPATH=.:thirdparty/DROID-SLAM:thirdparty/DROID-SLAM/droid_slam python -c "import demo; print('hoi-hawor: HaWoR import OK')")
}

seg() {  # sam2 + grounding_dino Dockerfiles
  mkenv hoi-seg
  use_cuda124
  pip install -q flask decord huggingface-hub omegaconf requests
  clone https://github.com/facebookresearch/sam2.git "$EXTERNAL/sam2"
  python -c "import sam2.build_sam" 2>/dev/null || pip install -q --no-build-isolation -e "$EXTERNAL/sam2"
  clone https://github.com/IDEA-Research/GroundingDINO.git "$EXTERNAL/GroundingDINO"
  # compiled ops load torch's libraries (libc10.so), so torch must be imported first
  python -c "import torch, groundingdino._C" 2>/dev/null || (cd "$EXTERNAL/GroundingDINO" && FORCE_CUDA=1 pip install -q --no-build-isolation -e .)
  pip install -q "transformers==4.27.4"
  python -c "import torch, sam2, groundingdino, groundingdino._C, transformers; from sam2.build_sam import build_sam2_video_predictor; \
print('hoi-seg: sam2, groundingdino (+CUDA ops), transformers', transformers.__version__, 'OK')"
}

sam3d() {  # sam3d Dockerfile
  mkenv hoi-sam3d
  use_cuda124
  pip install -q gsplat
  pip install -q kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
  pip install -q "$WHEELS"/pytorch3d-*.whl "$WHEELS"/nvdiffrast-*.whl
  python -c "import moge" 2>/dev/null || pip install -q "git+https://github.com/microsoft/MoGe.git@$MOGE_SAM3D_COMMIT#egg=moge"
  clone https://github.com/facebookresearch/sam-3d-objects.git "$EXTERNAL/sam-3d-objects"
  pip install -q --no-deps "$EXTERNAL/sam-3d-objects"
  pip install -q loguru timm open3d astor easydict networkx cumm-cu124 spconv-cu124 lightning xatlas pyvista \
    pymeshfix igraph imageio trimesh omegaconf hydra-core seaborn huggingface-hub
  # imported by sam3d_objects' inference path but not in V2D's list (its pytorch base image shipped it);
  # version as pinned in sam-3d-objects/requirements.txt
  pip install -q "optree==0.14.1"
  python -c "import flash_attn" 2>/dev/null || MAX_JOBS=4 pip install -q flash-attn==2.7.4.post1 --no-build-isolation
  pip install -q "pyrender==0.1.45" "pyglet==2.1.15" "numpy<2"
  python -c "import kaolin, pytorch3d, nvdiffrast.torch, flash_attn, spconv, gsplat, moge, networkx, pyrender; \
print('hoi-sam3d: kaolin', kaolin.__version__, 'pytorch3d', pytorch3d.__version__, 'flash_attn', flash_attn.__version__, 'OK')"
}

fp() {  # foundation_pose (+ mesh) Dockerfile
  mkenv hoi-fp
  use_cuda124
  pip install -q joblib scikit-learn ruamel.yaml trimesh imageio open3d transformations warp-lang einops kornia \
    pyrender scikit-image meshcat webdataset omegaconf pypng roma seaborn openpyxl wandb h5py fast_simplification \
    gdown "tensorrt-cu12==10.7.0.post1" "pybind11==2.10.4"
  pip install -q kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
  pip install -q "$WHEELS"/pytorch3d-*.whl "$WHEELS"/nvdiffrast-*.whl
  pip install -q opencv-contrib-python "numpy<2"   # V2D installs contrib after opencv-python, as here
  local fpdir=$SRC/HOI/foundation_pose/FoundationPose
  if ! ls "$fpdir"/mycpp/build/mycpp*.so >/dev/null 2>&1; then
    mkdir -p "$fpdir/mycpp/build"
    (cd "$fpdir/mycpp/build" && cmake .. -DPYTHON_EXECUTABLE="$(which python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
       && make -j"$MAX_JOBS")
  fi
  (cd "$fpdir/bundlesdf/mycuda" && python -c "import common, gridencoder" 2>/dev/null) || \
    (cd "$fpdir/bundlesdf/mycuda" && pip install -q --no-build-isolation -e .)
  python -c "import kaolin, pytorch3d, nvdiffrast.torch, tensorrt, warp, open3d; print('hoi-fp: kaolin, pytorch3d, nvdiffrast, tensorrt', tensorrt.__version__, 'OK')"
  (cd "$fpdir" && PYTHONPATH="$fpdir:$fpdir/mycpp/build:${PYTHONPATH:-}" python -c "import mycpp; print('hoi-fp: mycpp OK')")
}

trtexec() {  # FoundationPose's default TensorRT backend builds its engines with trtexec (HOI.foundation_pose.trt_engine).
             # pip's tensorrt-cu12 has the libraries but not trtexec, and V2D's image does not add it ("use the
             # FoundationPose TensorRT container"). Take trtexec from NVIDIA's CUDA apt repo, same version as the pip
             # libraries, unpacked (not installed) under $EXTERNAL/tensorrt; a wrapper points it at hoi-fp's libraries.
  local ver=10.7.0.23-1+cuda12.6 dir=$EXTERNAL/tensorrt
  local deb=$DOWNLOADS/libnvinfer-bin_${ver}_amd64.deb
  if [ ! -x "$dir/bin/trtexec.real" ]; then
    wget -q -c -O "$deb" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvinfer-bin_${ver}_amd64.deb"
    rm -rf "$dir/deb" && mkdir -p "$dir/bin" && dpkg-deb -x "$deb" "$dir/deb"
    cp "$dir/deb/usr/src/tensorrt/bin/trtexec" "$dir/bin/trtexec.real"
  fi
  local sp=$HOME/miniforge3/envs/hoi-fp/lib/python3.11/site-packages
  cat > "$dir/bin/trtexec" <<WRAP
#!/usr/bin/env bash
export LD_LIBRARY_PATH=$sp/tensorrt_libs:$sp/nvidia/cuda_runtime/lib:$sp/nvidia/cublas/lib:$sp/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}
exec "$dir/bin/trtexec.real" "\$@"
WRAP
  chmod +x "$dir/bin/trtexec"
  ("$dir/bin/trtexec" --help 2>&1 || true) | grep -m1 "TensorRT.trtexec"
}

# ---------------- weights (into E:\ZY_ego\model\<module>) ----------------
weights_hands() {
  conda activate hoi-hands
  export PYTHONPATH=$SRC
  # WiLoR-mini reads <weights_dir>/pretrained_models/{wilor_final.ckpt,detector.pt,MANO_RIGHT.pkl,mano_mean_params.npz};
  # the pipeline's manotorch layers (mano_assets_root = <weights_dir>/pretrained_models) read models/MANO_RIGHT.pkl
  mkdir -p "$MODEL/wilor/pretrained_models/models"
  cp -n "$MODEL/mano/MANO_RIGHT.pkl" "$MODEL/wilor/pretrained_models/"
  cp -n "$MODEL/mano/MANO_RIGHT.pkl" "$MODEL/wilor/pretrained_models/models/"
  python -m HOI.wilor.download_weights --weights_dir "$MODEL/wilor"
  python -m HOI.hamer.download_weights --weights_dir "$MODEL/hamer"
  # MANO needs registration: use the files already in model/mano. HaMeR reads _DATA/data/mano/; the pipeline's
  # manotorch renders (mano_assets_root = <hamer>/_DATA/data) read _DATA/data/models/
  mkdir -p "$MODEL/hamer/_DATA/data/mano" "$MODEL/hamer/_DATA/data/models"
  cp -n "$MODEL/mano/MANO_RIGHT.pkl" "$MODEL/hamer/_DATA/data/mano/"
  cp -n "$MODEL/mano/MANO_RIGHT.pkl" "$MODEL/hamer/_DATA/data/models/"
}
weights_mediapipe() {
  mkdir -p "$MODEL/mediapipe"
  [ -f "$MODEL/mediapipe/hand_landmarker.task" ] || wget -q -O "$MODEL/mediapipe/hand_landmarker.task" \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
}
weights_hawor() {
  conda activate "${WEIGHTS_ENV:-hoi-hawor}"
  PYTHONPATH=$SRC python -m HOI.hawor.download_weights --weights_dir "$MODEL/hawor"
}
weights_seg() {
  conda activate "${WEIGHTS_ENV:-hoi-seg}"
  export PYTHONPATH=$SRC
  [ -f "$MODEL/sam2/sam2.1_hiera_large.pt" ] || hf download facebook/sam2.1-hiera-large --local-dir "$MODEL/sam2"
  python -m HOI.grounding_dino.download_weights --output_dir "$MODEL/grounding_dino"
  HF_HOME=$MODEL/huggingface python -c "from transformers import AutoTokenizer, BertModel; \
AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased'); print('bert-base-uncased cached')"
}
weights_sam3d() {
  # facebook/sam-3d-objects is gated. The user downloads it into model/sam3d/hf-download/ (so that
  # hf-download/checkpoints/pipeline.yaml exists). The rest of V2D's download_weights.py is public and done here:
  # MoGe v1 (Ruicheng/moge-vitl) into hf_home/, DINOv2 ViT-L/B reg4 into torch_home/hub/checkpoints/.
  conda activate "${WEIGHTS_ENV:-hoi-sam3d}"
  local out=$MODEL/sam3d
  mkdir -p "$out/hf_home" "$out/torch_home/hub/checkpoints"
  if [ ! -f "$out/hf-download/checkpoints/pipeline.yaml" ]; then
    hf_token
    if [ -z "$HF_TOKEN" ]; then
      echo "SAM 3D Objects not found at $out/hf-download/checkpoints/pipeline.yaml (gated repo: download it there," \
           "or put HF_TOKEN in secrets.env after access is granted)" >&2
    else
      hf download facebook/sam-3d-objects --local-dir "$out/hf-download"
    fi
  fi
  HF_HOME=$out/hf_home hf download Ruicheng/moge-vitl
  for name in dinov2_vitl14_reg4_pretrain.pth dinov2_vitb14_reg4_pretrain.pth; do
    local arch=${name%%_reg4*}
    [ -f "$out/torch_home/hub/checkpoints/$name" ] || \
      wget -q -c -O "$out/torch_home/hub/checkpoints/$name" "https://dl.fbaipublicfiles.com/dinov2/$arch/$name"
  done
  ls -la "$out/torch_home/hub/checkpoints"
}
weights_fp() {  # NVlabs PyTorch weights (NVIDIA source code license) and the default TensorRT backend's models.
                # The latter need --accept_nvidia_model_eula: the user accepted the NVIDIA Open Model License on
                # 2026-11-10, so it is on by default here (ACCEPT_NVIDIA_MODEL_EULA=0 skips it)
  conda activate hoi-fp
  export PYTHONPATH=$SRC
  python -m HOI.foundation_pose.download_weights --output_dir "$MODEL/foundation_pose" --backend nvlabs_pytorch
  if [ "${ACCEPT_NVIDIA_MODEL_EULA:-1}" = 1 ]; then
    python -m HOI.foundation_pose.download_weights --output_dir "$MODEL/foundation_pose" --backend nvidia_tensorrt \
      --accept_nvidia_model_eula
  fi
}

check() {
  for e in hoi-hands hoi-mediapipe hoi-hawor hoi-seg hoi-sam3d hoi-fp; do
    if conda env list | grep -q "^$e "; then echo "env $e: present"; else echo "env $e: MISSING"; fi
  done
  du -sh "$MODEL"/{wilor,hamer,mediapipe,hawor,sam2,grounding_dino,huggingface,sam3d,foundation_pose} 2>/dev/null || true
}

case "$STEP" in
  all) system; cuda124; wheels; hands; mediapipe; hawor; seg; sam3d; fp; trtexec
       weights_hands; weights_mediapipe; weights_hawor; weights_seg; weights_sam3d; weights_fp; check ;;
  system|cuda124|wheels|hands|mediapipe|hawor|seg|sam3d|fp|trtexec|weights_hands|weights_mediapipe|weights_hawor|weights_seg|weights_sam3d|weights_fp|check)
       "$STEP" ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
