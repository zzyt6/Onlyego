"""Interactive 3D viewer for depth results, in the browser (viser). Chinese interface.

    depth_estimation view demo01                         # then open http://localhost:8080
    depth_estimation view runs\\video_depth\\demo01        # same sources, MegaSaM depth selected first
    depth_estimation view demo01 --poses demo01          # points in world coordinates from a camera_pose run

Depth sources (dropdown "深度来源"): every result of the same video is found from whichever one is opened -- the
depth_estimation run, the video_depth run made from it (its run.json names the depth run) and the HOI runs that
used it (their zyego_sources.json names it):
  MoGe-2 逐帧（V2D 的深度）                  depth_estimation/depth          what V2D uses: MoGe-2 per frame
  V2D 修正后（MoGe-2 逐帧 + gsplat 精修）    HOI run with --depth moge: depth_refined_simple (rendered refined scene)
  MoGe-2 + 逐帧尺度对齐                      depth_estimation/depth_aligned  (run --scale-align / align)
  MoGe-2 + MegaSaM（全分辨率）               video_depth/depth               consistent over the video
  MegaSaM 原始输出（低分辨率）                video_depth/depth_megasam
  MegaSaM + gsplat 精修                      HOI run with MegaSaM depth: depth_refined_simple

What it shows:
  - the point cloud of one frame: every `stride`-th pixel back-projected with that frame's intrinsics (pixel
    centres, as MoGe uses), coloured by the image, by depth, or per frame;
  - "同时显示": a second source drawn in one colour on top, to see where the two disagree;
  - "多帧叠加": several frames at once, each in its own colour. With a still camera they should coincide;
    per-frame scale jitter shows up as layers pushed apart along the viewing rays;
  - the stability table of `video_depth compare` (compare.json), and this frame's static-background depth
    per source;
  - with --poses: points and frustum placed by the camera_pose trajectory (camera->world), plus the path.

Coordinates are OpenCV camera axes (x right, y down, z forward); the viewer's "up" is -y, or +z when the
gravity-aligned poses (poses_gravity/) are used.
"""

import json
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .datatypes import CameraIntrinsics, frame_indices, frame_name, load_depth, load_mask

PROJECT = Path(__file__).resolve().parents[2]
FRAME_COLOURS = np.array([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                          (70, 240, 240), (240, 50, 230), (210, 245, 60)], dtype=np.uint8)
SECOND_COLOUR = np.array((255, 0, 255), dtype=np.uint8)  # "同时显示" source

# (run kind, folder) -> label, in display order
SOURCE_LABELS = {
    ("depth_estimation", "depth"): "MoGe-2 逐帧（V2D 的深度）",
    ("hoi_moge", "depth_refined_simple"): "V2D 修正后（MoGe-2 逐帧 + gsplat 精修）",
    ("depth_estimation", "depth_aligned"): "MoGe-2 + 逐帧尺度对齐",
    ("video_depth", "depth"): "MoGe-2 + MegaSaM（全分辨率）",
    ("video_depth", "depth_megasam"): "MegaSaM 原始输出（低分辨率）",
    ("hoi_megasam", "depth_refined_simple"): "MegaSaM + gsplat 精修",
}
SHORT_LABELS = {  # for the per-frame line
    ("depth_estimation", "depth"): "V2D",
    ("hoi_moge", "depth_refined_simple"): "V2D 精修",
    ("depth_estimation", "depth_aligned"): "尺度对齐",
    ("video_depth", "depth"): "MegaSaM",
    ("video_depth", "depth_megasam"): "MegaSaM 低分辨率",
    ("hoi_megasam", "depth_refined_simple"): "MegaSaM 精修",
}
ENGLISH_LABELS = {  # for video_depth compare (OpenCV text)
    ("depth_estimation", "depth"): "MoGe-2 per frame (V2D)",
    ("hoi_moge", "depth_refined_simple"): "V2D + gsplat refine",
    ("depth_estimation", "depth_aligned"): "MoGe-2 + scale align",
    ("video_depth", "depth"): "MoGe-2 + MegaSaM",
    ("video_depth", "depth_megasam"): "MegaSaM low-res",
    ("hoi_megasam", "depth_refined_simple"): "MegaSaM + gsplat refine",
}
NOT_SHOWN = "不显示"


@dataclass(frozen=True)
class DepthSource:
    label: str
    short: str
    kind: str          # "depth_estimation" | "video_depth"
    run: Path          # run folder (intrinsics, mask)
    folder: Path       # depth PNGs


def _record(run: Path) -> dict:
    path = run / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _hoi_sources(run: Path) -> dict:
    path = run / "zyego_sources.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _kind(run: Path) -> str:
    if (run / "megasam.json").exists():
        return "video_depth"
    hoi = _hoi_sources(run)
    if hoi:
        return "hoi_megasam" if str(hoi.get("depth", "")).startswith("video_depth") else "hoi_moge"
    return "depth_estimation"


def _depth_run_of(run: Path) -> Path | None:
    kind = _kind(run)
    if kind == "video_depth":
        value = _record(run).get("depth_run")
    elif kind.startswith("hoi"):
        value = _hoi_sources(run).get("depth_estimation")
    else:
        return run
    return Path(value) if value and Path(value).is_dir() else None


def linked_runs(run: Path) -> list[Path]:
    """`run` plus every other result of the same video: its depth_estimation run, the video_depth run made from that
    (runs/video_depth/<name>) and the HOI runs that used it (runs/HOI/*/zyego_sources.json)."""
    run = Path(run)
    runs = [run]
    depth_run = _depth_run_of(run)
    if depth_run is None:
        return runs
    candidates = [depth_run, PROJECT / "runs" / "video_depth" / depth_run.name]
    hoi_root = PROJECT / "runs" / "HOI"
    if hoi_root.is_dir():
        candidates += sorted(p for p in hoi_root.iterdir() if (p / "zyego_sources.json").exists())
    for c in candidates:
        if c.is_dir() and c.resolve() not in {r.resolve() for r in runs}:
            other = _depth_run_of(c)
            if other is not None and other.resolve() == depth_run.resolve():
                runs.append(c)
    return runs


def find_sources(run: Path) -> dict:
    """(kind, folder) -> (run folder, depth folder) for every depth source of the video, in display order."""
    found = {}
    for r in linked_runs(run):
        kind = _kind(r)
        for (k, folder) in SOURCE_LABELS:
            if k == kind and (k, folder) not in found and (r / folder).is_dir() and any((r / folder).glob("*.png")):
                found[(k, folder)] = (r, r / folder)
    return dict(sorted(found.items(), key=lambda kv: list(SOURCE_LABELS).index(kv[0])))


class RunData:
    """Lazy access to a run's depth sources (and optionally camera poses), with small caches."""

    def __init__(self, run: Path, poses_dir: Path | None = None):
        self.run = Path(run)
        record = _record(self.run)
        self.images = self.run / record.get("image_dir", "frames")  # image_dir may be absolute (video_depth)
        self.indices = frame_indices(self.run / "depth", ".png")
        if not self.indices:
            raise FileNotFoundError(f"No depth maps in {self.run / 'depth'}")
        self.sources: dict[str, DepthSource] = {
            SOURCE_LABELS[key]: DepthSource(SOURCE_LABELS[key], SHORT_LABELS[key], key[0], r, folder)
            for key, (r, folder) in find_sources(self.run).items()}
        main_kind = _kind(self.run)
        preferred = {"video_depth": [("video_depth", "depth")],
                     "hoi_moge": [("hoi_moge", "depth_refined_simple")],
                     "hoi_megasam": [("hoi_megasam", "depth_refined_simple")]}.get(main_kind, [])
        preferred += [("depth_estimation", "depth_aligned"), ("depth_estimation", "depth")]
        self.default_source = next(SOURCE_LABELS[p] for p in preferred if SOURCE_LABELS[p] in self.sources)
        self.fps = float(record.get("video", {}).get("fps", 30.0)) / int(record.get("settings", {}).get("stride", 1))
        self.stability = self._load_stability()
        self.poses: dict[int, np.ndarray] = {}
        if poses_dir is not None:
            from camera_pose.poses import load_trajectory
            idx, mats = load_trajectory(poses_dir)
            self.poses = dict(zip(idx, mats))
            if not self.poses:
                raise FileNotFoundError(f"No camera poses in {poses_dir}")

    def _load_stability(self) -> dict | None:
        """compare.json of `video_depth compare`, keyed by our source labels (matched on the depth folders)."""
        for r in linked_runs(self.run):
            path = r / "compare.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            by_folder = {s.folder.resolve(): label for label, s in self.sources.items()}
            rows = {}
            for name, folder in summary.get("sources", {}).items():
                label = by_folder.get(Path(folder).resolve())
                if label and name in summary.get("stability", {}):
                    rows[label] = summary["stability"][name]
            if rows:
                return {"rows": rows, "indices": summary.get("indices"), "path": path,
                        "static_fraction": summary.get("static_background_fraction")}
        return None

    def intrinsics(self, i: int, source: str) -> CameraIntrinsics:
        s = self.sources[source]
        if s.folder.name == "depth_megasam" and (s.run / "intrinsics_megasam.json").exists():
            return CameraIntrinsics.load(s.run / "intrinsics_megasam.json")
        path = s.run / "intrinsics" / frame_name(i, ".json")
        return CameraIntrinsics.load(path if path.exists() else s.run / "intrinsics_stable.json")

    @lru_cache(maxsize=16)
    def image(self, i: int) -> np.ndarray:
        bgr = cv2.imread(str(self.images / frame_name(i, ".png")))
        if bgr is None:
            raise FileNotFoundError(f"Missing frame {self.images / frame_name(i, '.png')}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=64)
    def cloud(self, i: int, source: str, stride: int, use_mask: bool):
        """(points in camera axes (N,3) float32, colours (N,3) uint8) for valid pixels, every stride-th pixel."""
        s = self.sources[source]
        depth = load_depth(s.folder / frame_name(i, ".png"))
        k = self.intrinsics(i, source)
        rgb = self.image(i)
        if rgb.shape[:2] != depth.shape:  # e.g. MegaSaM's native depth is at tracking resolution
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        mask_path = s.run / "mask" / frame_name(i, ".png")
        mask = None
        if use_mask and mask_path.exists():
            mask = load_mask(mask_path)
            if mask.shape != depth.shape:
                mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        return backproject(depth, rgb, k, stride, mask=mask)

    def background_depth(self, i: int) -> dict[str, float]:
        """This frame's static-background median depth per source, from compare.json."""
        if not self.stability or not self.stability.get("indices"):
            return {}
        try:
            k = self.stability["indices"].index(i)
        except ValueError:
            return {}
        return {label: row["per_frame_median_m"][k] for label, row in self.stability["rows"].items()}

    def pose(self, i: int) -> np.ndarray | None:
        return self.poses.get(i) if self.poses else None


def backproject(depth: np.ndarray, rgb: np.ndarray, k: CameraIntrinsics, stride: int = 1, mask=None):
    """Pixel (u, v) with depth z -> ((u+0.5-cx) z/fx, (v+0.5-cy) z/fy, z); MoGe's cx = W/2 is the image centre
    in continuous coordinates, i.e. pixel centres sit at +0.5."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask[v, u]
    u, v, z = u[valid] + 0.5, v[valid] + 0.5, z[valid]
    points = np.stack([(u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z], axis=1).astype(np.float32)
    colours = rgb[v.astype(int), u.astype(int)]
    return points, colours


def depth_colours(z: np.ndarray, near: float, far: float) -> np.ndarray:
    """TURBO on inverse depth, fixed range, so colours mean the same distance in every frame."""
    t = ((1.0 / np.maximum(z, 1e-6) - 1.0 / far) / (1.0 / near - 1.0 / far)).clip(0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].copy()


def rotation_to_wxyz(R: np.ndarray) -> np.ndarray:
    from camera_pose.poses import matrix_to_quat_wxyz
    return np.array(matrix_to_quat_wxyz(R))


def stability_markdown(data: RunData) -> str:
    if not data.stability:
        return ("**深度稳定性**：还没有统计。运行 `video_depth compare <名字>` 后重开查看器，"
                "这里会显示各深度来源在静止背景上的抖动。")
    lines = ["**深度稳定性（静止背景，越小越稳）**", "",
             "| 深度来源 | 整帧尺度抖动 | 像素晃动 | 晃动幅度 | 相邻帧跳变 | 统计覆盖 |",
             "|---|---|---|---|---|---|"]
    for label, r in data.stability["rows"].items():
        lines.append(f"| {label} | {r['scale_jitter_pct']:.2f}% | {r['pixel_wobble_pct']:.2f}% | "
                     f"{r['pixel_wobble_mm']:.1f} mm | {r['frame_to_frame_pct']:.3f}% | {r.get('coverage', 1) * 100:.0f}% |")
    frac = data.stability.get("static_fraction")
    lines += ["", f"静止背景 = 整段视频颜色基本不变的像素（占画面 {frac * 100:.1f}%，只适用于固定机位）。"
              if frac is not None else "",
              "整帧尺度抖动：背景深度中位数随帧变化的标准差 / 均值；像素晃动：每个像素深度随时间的标准差 / 均值"
              "（取中位数）；相邻帧跳变：相邻两帧深度的相对变化；统计覆盖：静止背景里这种深度有效（≥80% 帧）的比例，"
              "gsplat 渲染深度有空洞时低于 100%。"]
    return "\n".join(lines)


def serve(run: Path, poses_dir: Path | None = None, host: str = "127.0.0.1", port: int = 8080,
          block: bool = True):
    import viser

    data = RunData(run, poses_dir)
    gravity_world = poses_dir is not None and Path(poses_dir).name == "poses_gravity"
    server = viser.ViserServer(host=host, port=port, label=f"深度：{Path(run).name}")
    server.scene.set_up_direction("+z" if gravity_world else "-y")
    n = len(data.indices)
    lock = threading.Lock()
    handles: list = []
    labels = list(data.sources)

    # depth range for colouring, from the depth run's statistics if present
    near, far = 0.2, 3.0
    for r in linked_runs(data.run):
        stats_path = r / "depth_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            p05 = [s["depth_p05"] for s in stats if s.get("depth_p05")]
            p95 = [s["depth_p95"] for s in stats if s.get("depth_p95")]
            if p05 and p95:
                near, far = max(0.05, float(np.min(p05))), float(np.max(p95))
                break

    with server.gui.add_folder("深度来源"):
        g_source = server.gui.add_dropdown("深度来源", labels, initial_value=data.default_source,
                                           hint="V2D 的深度 = MoGe-2 每帧单独推理；MegaSaM = 以 MoGe-2 为先验、整段视频联合优化的一致深度")
        g_second = server.gui.add_dropdown("同时显示", [NOT_SHOWN, *labels], initial_value=NOT_SHOWN,
                                           hint="再画一种深度（洋红色），和上面选的直接比较哪里不一致")
    g_stats = server.gui.add_markdown(stability_markdown(data))
    with server.gui.add_folder("帧"):
        g_frame = server.gui.add_slider("帧位置", 0, n - 1, 1, 0, hint="在运行结果里的位置；下面的信息显示视频帧号")
        g_play = server.gui.add_checkbox("播放", False)
        g_fps = server.gui.add_slider("播放帧率", 1, 30, 1, min(10, int(round(data.fps))))
    with server.gui.add_folder("点云"):
        g_colour = server.gui.add_dropdown("着色", ["图像颜色", "按深度", "按帧"], initial_value="图像颜色",
                                           hint="按深度：整段视频同一个范围，颜色一样 = 距离一样")
        k0 = data.intrinsics(data.indices[0], data.default_source)
        steps = ["1", "2", "3", "4", "6", "8"]
        target = np.sqrt(k0.width * k0.height / 200_000)  # ~200k points per frame by default
        g_stride = server.gui.add_dropdown("像素步长", steps,
                                           initial_value=min(steps, key=lambda s: abs(int(s) - target)),
                                           hint="每隔 n 个像素取一个点；1 = 全部（1080p 约 200 万点，较慢）")
        g_size = server.gui.add_slider("点大小 (mm)", 0.5, 10.0, 0.5, 2.0)
        g_max = server.gui.add_slider("最大深度 (m)", 0.1, max(1.0, round(far * 2, 1)), 0.05, round(far * 1.2, 2))
        g_mask = server.gui.add_checkbox("只显示 MoGe 有效像素", True)
    with server.gui.add_folder("多帧叠加（看深度一致性）"):
        g_over = server.gui.add_slider("显示帧数", 1, 8, 1, 1, hint="同时画几帧；相机不动时它们应该重合")
        g_gap = server.gui.add_slider("帧间隔", 1, max(1, n // 2), 1, max(1, min(30, n // 8)))
    with server.gui.add_folder("相机"):
        g_frustum = server.gui.add_checkbox("视锥和图像", True)
        g_traj = server.gui.add_checkbox("轨迹", bool(data.poses), visible=bool(data.poses))
    g_info = server.gui.add_markdown("")

    if data.poses:
        idx = [i for i in data.indices if i in data.poses]
        centres = np.array([data.poses[i][:3, 3] for i in idx], dtype=np.float32)
        if len(centres) > 1:
            segs = np.stack([centres[:-1], centres[1:]], axis=1)
            t = np.linspace(0, 1, len(segs))
            cols = (np.stack([t, 0.3 + 0 * t, 1 - t], axis=1)[:, None, :].repeat(2, 1) * 255).astype(np.uint8)
            traj = server.scene.add_line_segments("/trajectory", segs, cols, thickness=2.0, thickness_units="screen")
            g_traj.on_update(lambda _: setattr(traj, "visible", g_traj.value))

    def placement(i: int):
        M = data.pose(i)
        return (rotation_to_wxyz(M[:3, :3]), M[:3, 3]) if M is not None else ((1, 0, 0, 0), (0, 0, 0))

    def render(_=None):
        with lock:
            pos = int(g_frame.value)
            frames = [data.indices[(pos + j * int(g_gap.value)) % n] for j in range(int(g_over.value))]
            stride = int(g_stride.value)
            for h in handles:
                h.remove()
            handles.clear()
            lines = []
            for j, i in enumerate(frames):
                pts, cols = data.cloud(i, g_source.value, stride, g_mask.value)
                keep = pts[:, 2] <= g_max.value
                pts, cols = pts[keep], cols[keep]
                if g_colour.value == "按深度":
                    cols = depth_colours(pts[:, 2], near, far)
                elif g_colour.value == "按帧":
                    cols = np.broadcast_to(FRAME_COLOURS[j % len(FRAME_COLOURS)], pts.shape).copy()
                wxyz, position = placement(i)
                handles.append(server.scene.add_point_cloud(
                    f"/cloud/{j}", pts, cols, point_size=g_size.value / 1000.0, point_shape="rounded",
                    precision="float32", wxyz=wxyz, position=position))
                if g_second.value != NOT_SHOWN and g_second.value != g_source.value:
                    p2, _ = data.cloud(i, g_second.value, stride, g_mask.value)
                    p2 = p2[p2[:, 2] <= g_max.value]
                    handles.append(server.scene.add_point_cloud(
                        f"/second/{j}", p2, np.broadcast_to(SECOND_COLOUR, p2.shape).copy(),
                        point_size=g_size.value / 1000.0, point_shape="rounded", precision="float32",
                        wxyz=wxyz, position=position))
                k = data.intrinsics(i, g_source.value)
                if g_frustum.value:
                    img = data.image(i)
                    small = cv2.resize(img, (320, max(1, round(320 * img.shape[0] / img.shape[1]))))
                    handles.append(server.scene.add_camera_frustum(
                        f"/camera/{j}", fov=2 * np.arctan(k.height / (2 * k.fy)), aspect=k.width / k.height,
                        scale=0.1, image=small, wxyz=wxyz, position=position,
                        color=tuple(int(c) for c in FRAME_COLOURS[j % len(FRAME_COLOURS)]) if len(frames) > 1 else (20, 20, 20)))
                z = pts[:, 2]
                swatch = f"<span style='color:rgb{tuple(int(c) for c in FRAME_COLOURS[j % 8])}'>■</span> " \
                    if len(frames) > 1 and g_colour.value == "按帧" else ""
                lines.append(f"{swatch}第 **{i}** 帧：深度中位数 **{np.median(z):.3f} m**，{len(z):,} 个点"
                             if len(z) else f"第 {i} 帧：没有点")
                if j == 0:
                    lines.insert(0, f"fx {k.fx:.1f} fy {k.fy:.1f}，水平视场角 **{k.fov_x_degrees:.1f}°**，"
                                    f"{k.width}×{k.height}")
                    bg = data.background_depth(i)
                    if bg:
                        parts = [f"{data.sources[label].short} **{v:.3f} m**" if label == g_source.value
                                 else f"{data.sources[label].short} {v:.3f} m" for label, v in bg.items()]
                        lines.insert(1, "本帧静止背景深度：" + "｜".join(parts))
            if g_second.value != NOT_SHOWN and g_second.value != g_source.value:
                lines.append(f"<span style='color:rgb(255,0,255)'>■</span> 洋红色 = {g_second.value}")
            g_info.content = "  \n".join(lines)

    for g in (g_frame, g_source, g_second, g_colour, g_stride, g_size, g_max, g_mask, g_over, g_gap, g_frustum):
        g.on_update(render)
    render()
    print(f"Viewer for {run}: open http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}  "
          f"(Ctrl+C to stop)", flush=True)

    if not block:  # for tests / scripting: the server plus its controls and the current scene handles
        from types import SimpleNamespace
        return server, SimpleNamespace(frame=g_frame, play=g_play, source=g_source, second=g_second,
                                       colour=g_colour, stride=g_stride, size=g_size, max_depth=g_max,
                                       mask=g_mask, overlay=g_over, gap=g_gap, frustum=g_frustum, info=g_info,
                                       stats=g_stats, handles=handles, render=render)
    try:
        while True:
            if g_play.value:
                g_frame.value = (int(g_frame.value) + 1) % n  # triggers render
                time.sleep(1.0 / max(1, g_fps.value))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        server.stop()
