#!/usr/bin/env bash
# Import the src/HOI model wrappers inside the env each one runs in (GPU visible, weights present).
#   wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/check_hoi.sh
source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$HOME"
SRC=/mnt/e/ZY_ego/src
EXT=$HOME/zyego-external
FP=$SRC/HOI/foundation_pose/FoundationPose
export TRTEXEC_PATH=$EXT/tensorrt/bin/trtexec
status=0
try() {  # try <env> <extra PYTHONPATH> <python statements>
  local env=$1 extra=$2 code=$3 out
  export LIDRA_SKIP_INIT=${LIDRA_SKIP_INIT:-}
  conda activate "$env"
  if out=$(PYTHONPATH="$SRC${extra:+:$extra}" python -c "$code" 2>&1); then
    echo "[ok]   $env: $(echo "$out" | tail -1)"
  else
    echo "[FAIL] $env:"; echo "$out" | tail -6; status=1
  fi
  conda deactivate
}
try hoi-hands "" "import torch, HOI.wilor._wilor, HOI.wilor.video_to_hands; print('wilor')"
try hoi-seg "" "import torch, HOI.sam2.video_to_masks, HOI.grounding_dino.image_to_object_bboxes; print('sam2, grounding_dino')"
LIDRA_SKIP_INIT=1 try hoi-sam3d "" "import torch, sam3d_objects, HOI.sam3d.image_to_mesh; print('sam3d (sam3d_objects importable; runs with LIDRA_SKIP_INIT=1 as V2D)')"
try hoi-fp "$FP" "import torch, HOI.foundation_pose.foundation_pose_tracker, HOI.mesh.mesh, \
HOI.foundation_pose.trt_engine; import Utils; assert Utils.mycpp is not None, 'mycpp'; \
import os; assert os.access(os.environ['TRTEXEC_PATH'], os.X_OK), 'trtexec'; \
print('foundation_pose tracker, TensorRT engine, mycpp, trtexec')"
echo "== weights"
for d in wilor sam2 grounding_dino huggingface sam3d foundation_pose; do
  printf "  %-16s %s\n" "$d" "$(du -sh /mnt/e/ZY_ego/model/$d 2>/dev/null | cut -f1)"
done
[ -f /mnt/e/ZY_ego/model/sam3d/hf-download/checkpoints/pipeline.yaml ] && echo "  SAM 3D Objects: present" \
  || echo "  SAM 3D Objects: missing (user downloads facebook/sam-3d-objects into model/sam3d/hf-download)"
exit $status
