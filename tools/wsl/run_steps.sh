#!/usr/bin/env bash
# Run setup_hoi_env.sh steps in order, one log per step in ~/zyego-external/logs/<step>.log;
# stop at the first failing step and show the end of its log.
#   wsl -d V2D-Ubuntu -- bash /mnt/e/ZY_ego/tools/wsl/run_steps.sh hands mediapipe
here=$(cd "$(dirname "$0")" && pwd)
# wsl.exe starts in the current Windows folder; e.g. from E:\ZY_ego\model, `import sam2` would find the weights
# folder model/sam2 as a namespace package. Run from a neutral directory.
cd "$HOME"
logs=$HOME/zyego-external/logs
mkdir -p "$logs"
for step in "$@"; do
  start=$(date +%s)
  if bash "$here/setup_hoi_env.sh" "$step" > "$logs/$step.log" 2>&1; then
    echo "[ok]   $step ($(( $(date +%s) - start ))s) -- $(tail -1 "$logs/$step.log")"
  else
    code=$?
    echo "[FAIL] $step exit $code ($(( $(date +%s) - start ))s); end of $logs/$step.log:"
    tail -25 "$logs/$step.log"
    exit "$code"
  fi
done
