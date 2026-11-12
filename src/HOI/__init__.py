"""HOI: the model wrappers our parts call, kept from NVIDIA V2D (video_to_data); V2D's own pipeline is gone.

    wilor.video_to_hands                      hands (object_model)
    grounding_dino.image_to_object_bboxes     object box from the text prompt (object_model)
    sam2.video_to_masks                       object / hand masks over the video (object_model)
    sam3d.image_to_mesh                       object mesh (object_model)
    foundation_pose.foundation_pose_tracker   object pose per frame (object_pose)
    common, mesh                              data types and mesh I/O used by the above

Each runs in its own WSL conda env (tools/wsl/setup_hoi_env.sh), started by object_model/wsl.py.
"""
