"""hand_model: check the MANO files and the left-hand shapedirs fix (as in Wuji MINT)."""

import argparse
import hashlib
import sys

from .mano import create_mano, left_shapedirs_mismatch, load_mano_data, mano_dir, mano_path, mirror_axis_angle


def mirror_error(left, right, *, samples: int, beta_std: float, pose_std: float, seed: int = 0) -> float:
    """Max |left(beta, mirrored pose) - mirror_x(right(beta, pose))| in metres over random hands."""
    import torch

    g = torch.Generator().manual_seed(seed)
    betas = torch.randn(samples, 10, generator=g) * beta_std
    orient = torch.randn(samples, 3, generator=g) * pose_std
    pose = torch.randn(samples, 45, generator=g) * pose_std
    dev = next(right.buffers()).device
    with torch.no_grad():
        vr = right(betas=betas.to(dev), global_orient=orient.to(dev), hand_pose=pose.to(dev)).vertices
        vl = left(betas=betas.to(dev), global_orient=mirror_axis_angle(orient).to(dev),
                  hand_pose=mirror_axis_angle(pose).to(dev)).vertices
    vr = vr * vr.new_tensor([-1.0, 1.0, 1.0])
    return float((vl - vr).abs().max())


def check(args) -> int:
    root = mano_dir(args.mano_dir)
    print(f"MANO folder: {root}")
    for side in ("left", "right"):
        path = mano_path(side, root)
        print(f"  {path.name}  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")
    same, opposite = left_shapedirs_mismatch(load_mano_data("left", root), load_mano_data("right", root))
    state = "v1.2 bug (same sign as right) -> will be fixed" if same < opposite else "already mirrored"
    print(f"left shapedirs x: sum|L-R| = {same:.4f}, sum|L+R| = {opposite:.4f}  [{state}]")

    right = create_mano("right", root=root, device=args.device)
    left = create_mano("left", root=root, device=args.device)
    raw_left = create_mano("left", root=root, device=args.device, fix_left=False)
    print(f"create_mano('left') applied the fix: {left.left_shapedirs_fixed}")
    print(f"left vs mirrored right, {args.samples} random hands (betas std {args.beta_std}, pose std {args.pose_std}):")
    before = mirror_error(raw_left, right, samples=args.samples, beta_std=args.beta_std, pose_std=args.pose_std)
    after = mirror_error(left, right, samples=args.samples, beta_std=args.beta_std, pose_std=args.pose_std)
    print(f"  unfixed file : max error {before * 1000:8.3f} mm")
    print(f"  fixed        : max error {after * 1000:8.3f} mm")
    ok = after < 1e-4
    print("OK" if ok else "FAILED: the fixed left hand is still not the mirror of the right hand")
    return 0 if ok else 1


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="hand_model", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="Verify the MANO files and that the fixed left hand mirrors the right")
    c.add_argument("--mano-dir", help="Default: $ZYEGO_MANO_DIR or model/mano")
    c.add_argument("--device", default="cpu")
    c.add_argument("--samples", type=int, default=64)
    c.add_argument("--beta-std", type=float, default=2.0)
    c.add_argument("--pose-std", type=float, default=0.5)
    args = parser.parse_args(argv)
    sys.exit(check(args))


if __name__ == "__main__":
    main()
