"""camera_pose: per-frame camera position/orientation in a metric world frame (DROID-SLAM + MoGe scale).

Adapted from NVIDIA video_to_data (V2D), Apache-2.0:
  reconstruction/modules/v2d_droid_slam/lib/_slam.py, video_to_slam.py -> slam.py
DROID-SLAM needs compiled CUDA extensions, so slam.py runs inside WSL (env zyego-cuda, see
tools/wsl/setup_cuda_env.sh); the Windows command in cli.py prepares inputs and calls it, the way
V2D calls its Docker containers. See docs/camera_pose.md.
"""
