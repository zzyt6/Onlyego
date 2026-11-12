"""depth_estimation: video -> per-frame metric depth + camera intrinsics (MoGe-2).

Adapted from NVIDIA video_to_data (V2D), Apache-2.0:
  reconstruction/modules/v2d_moge/lib/video_to_depth.py      -> moge_depth.py
  reconstruction/modules/v2d_depth/lib/stabilize_intrinsics.py -> intrinsics.py
  reconstruction/modules/v2d_depth/lib/align_depth_sequence.py -> align.py
  reconstruction/modules/v2d_common/datatypes.py (DepthImage, CameraIntrinsics, Mask) -> datatypes.py
See docs/depth_estimation.md for how their scheme works and what was changed here.
"""
