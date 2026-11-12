#!/usr/bin/env bash
# Import the migrated src/HOI modules inside the env each one runs in (GPU visible, weights present).
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
try hoi-hands "" "import torch, HOI.wilor._wilor, HOI.wilor.video_to_hands, HOI.hamer._hamer, HOI.hamer.align_hands, \
HOI.hand_alignment.recover_mano_params, HOI.mesh.mesh; print('wilor, hamer, hand_alignment, mesh')"
try hoi-mediapipe "" "import HOI.mediapipe.image_to_hand_bboxes; print('mediapipe (no torch)')"
try hoi-hawor "$EXT/HaWoR:$EXT/HaWoR/thirdparty/DROID-SLAM:$EXT/HaWoR/thirdparty/DROID-SLAM/droid_slam" \
  "import torch, HOI.hawor.run_hawor; import demo; print('hawor (+ HaWoR demo, DROID-SLAM, Metric3D)')"
try hoi-seg "" "import torch, HOI.sam2.video_to_masks, HOI.grounding_dino.image_to_object_bboxes; print('sam2, grounding_dino')"
LIDRA_SKIP_INIT=1 try hoi-sam3d "" "import torch, sam3d_objects, HOI.sam3d.image_to_mesh; print('sam3d (sam3d_objects importable; runs with LIDRA_SKIP_INIT=1 as V2D)')"
try hoi-fp "$FP" "import torch, HOI.foundation_pose.foundation_pose_tracker, HOI.foundation_pose.run_ekf_smoothing, \
HOI.foundation_pose.run_estimate_mesh_scale, HOI.foundation_pose.trt_engine; import Utils; assert Utils.mycpp is not None, 'mycpp'; \
import os; assert os.access(os.environ['TRTEXEC_PATH'], os.X_OK), 'trtexec'; \
print('foundation_pose tracker, EKF, mesh scale, TensorRT engine, mycpp, trtexec')"
try zyego-cuda "" "import torch, HOI.gsplat_refinement.refine_simple, HOI.gsplat_refinement.refine; print('gsplat_refinement')"
echo "== weights"
for d in mano wilor hamer mediapipe hawor sam2 grounding_dino huggingface sam3d foundation_pose; do
  printf "  %-16s %s\n" "$d" "$(du -sh /mnt/e/ZY_ego/model/$d 2>/dev/null | cut -f1)"
done
[ -f /mnt/e/ZY_ego/model/sam3d/hf-download/checkpoints/pipeline.yaml ] && echo "  SAM 3D Objects: present" \
  || echo "  SAM 3D Objects: missing (user downloads facebook/sam-3d-objects into model/sam3d/hf-download)"
exit $status
