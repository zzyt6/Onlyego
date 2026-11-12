"""Video -> numbered PNG frames (V2D v2d_pipelines/extract_images.py, plus stride and a frame limit).

HDR video (ours): iPhones record HDR by default (HEVC 10-bit, BT.2020, HLG). Decoders that ignore the colour
metadata -- OpenCV here, plain ffmpeg in V2D -- hand over the HLG signal as if it were SDR BT.709, and the frames
come out washed out (demo01: median saturation 66 instead of 139). For HDR input (transfer HLG or PQ) the frames
are decoded by ffmpeg with a tone mapping to SDR BT.709 instead:

    zscale -> linear light (nominal peak `npl` nits) -> BT.709 primaries -> tonemap (hable by default) -> BT.709

There is no single correct HDR->SDR mapping; hable with npl=100 kept demo01's highlights and looked the most
natural of hable 203 / hable 100 / mobius 100 / clip 203. SDR video is read with OpenCV as before.
"""

import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

HDR_TRANSFERS = {"arib-std-b67": "HLG", "smpte2084": "PQ"}
TONEMAPS = ("hable", "mobius", "reinhard", "clip")
DEFAULT_TONEMAP = "hable"
DEFAULT_NPL = 100.0

_STREAM = re.compile(r"Stream #\S+.*?Video: (?P<codec>\w+).*?, (?P<pix>\w+)\((?P<rest>[^)]*)\)")
_SIZE = re.compile(r", (\d{2,5})x(\d{2,5})[ ,\[]")
_ROTATION = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise FileNotFoundError("ffmpeg not found: pip install imageio-ffmpeg (needed for HDR video)")
        return exe


def parse_color_info(ffmpeg_stderr: str) -> dict:
    """Colour description of the first video stream from `ffmpeg -i <video>` output."""
    info = {"pix_fmt": None, "range": None, "matrix": None, "primaries": None, "transfer": None, "hdr": None,
            "width": None, "height": None}
    for line in ffmpeg_stderr.splitlines():
        m = _STREAM.search(line)
        if not m:
            continue
        info["codec"] = m["codec"]
        info["pix_fmt"] = m["pix"]
        parts = [p.strip() for p in m["rest"].split(",")]
        if parts and parts[0] in ("tv", "pc"):
            info["range"] = parts[0]
        for p in parts:
            if p.count("/") == 2:
                info["matrix"], info["primaries"], info["transfer"] = p.split("/")
            elif p in HDR_TRANSFERS or p.startswith(("bt", "smpte", "arib")):
                info["matrix"] = info["primaries"] = info["transfer"] = p  # "bt709" shorthand: all three
        size = _SIZE.search(line)
        if size:
            info["width"], info["height"] = int(size[1]), int(size[2])
        break
    rotation = _ROTATION.search(ffmpeg_stderr)
    if rotation and round(abs(float(rotation[1]))) % 180 == 90 and info["width"]:
        info["width"], info["height"] = info["height"], info["width"]  # ffmpeg auto-rotates
    info["hdr"] = HDR_TRANSFERS.get(info["transfer"])
    return info


def color_info(video: Path) -> dict:
    result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return parse_color_info(result.stderr)


def tonemap_filter(info: dict, tonemap: str = DEFAULT_TONEMAP, npl: float = DEFAULT_NPL,
                   out_range: str = "pc", out_format: str = "bgr24") -> str:
    """ffmpeg -vf chain HDR (HLG/PQ, BT.2020) -> SDR BT.709."""
    if tonemap not in TONEMAPS:
        raise ValueError(f"tonemap must be one of {TONEMAPS}")
    zin = (f"zscale=tin={info['transfer']}:min={info.get('matrix') or 'bt2020nc'}:"
           f"pin={info.get('primaries') or 'bt2020'}:rin={info.get('range') or 'tv'}:t=linear:npl={npl:g}")
    return (f"{zin},format=gbrpf32le,zscale=p=bt709,tonemap=tonemap={tonemap}:desat=0,"
            f"zscale=t=bt709:m=bt709:r={out_range},format={out_format}")


def video_info(video: Path) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    info = {"fps": capture.get(cv2.CAP_PROP_FPS) or 30.0,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # Container metadata only; can be off by a few frames. The extracted count is authoritative.
            "frame_count_reported": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    capture.release()
    try:
        colour = color_info(video)
        info["color"] = {k: colour[k] for k in ("pix_fmt", "range", "matrix", "primaries", "transfer", "hdr")}
    except FileNotFoundError:  # no ffmpeg: fine for SDR video, extract_frames refuses HDR without it
        info["color"] = None
    return info


def _hdr_frames(video: Path, tonemap: str, npl: float):
    """Tone-mapped BGR frames of an HDR video, decoded by ffmpeg."""
    info = color_info(video)
    w, h = info["width"], info["height"]
    if not w or not h:
        raise ValueError(f"Could not read the frame size of {video} from ffmpeg")
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", tonemap_filter(info, tonemap, npl),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = w * h * 3
    finished = False
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                finished = True
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if not finished:  # the caller stopped early (max_frames): stop ffmpeg, that is not a failure
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
        if finished and code != 0:
            raise RuntimeError(f"ffmpeg tone mapping failed: {err[-500:]}")


def _sdr_frames(video: Path):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def extract_frames(video: Path, folder: Path, stride: int = 1, max_frames: int | None = None,
                   tonemap: str | None = DEFAULT_TONEMAP, npl: float = DEFAULT_NPL) -> int:
    """Write every `stride`-th frame as <original index>.png, so indices always refer to the source video.
    HDR video is tone-mapped to SDR (see module docstring) unless tonemap is None/"off"."""
    folder.mkdir(parents=True, exist_ok=True)
    hdr = tonemap not in (None, "off") and color_info(video)["hdr"]
    frames = _hdr_frames(video, tonemap, npl) if hdr else _sdr_frames(video)
    index = written = 0
    for frame in frames:
        if max_frames is not None and written >= max_frames:
            break
        if index % stride == 0:
            if not cv2.imwrite(str(folder / f"{index:06d}.png"), frame):
                raise OSError(f"Failed to write frame {index} to {folder}")
            written += 1
        index += 1
    frames.close()
    if written == 0:
        raise ValueError(f"No frames decoded from {video}")
    return written


def write_sdr_video(video: Path, output: Path, tonemap: str = DEFAULT_TONEMAP, npl: float = DEFAULT_NPL,
                    crf: int = 12) -> Path:
    """The whole HDR video tone-mapped to an SDR BT.709 H.264 file, for steps that read the video itself."""
    info = color_info(video)
    tmp = output.with_suffix(".tmp.mp4")
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(video),
                    "-vf", tonemap_filter(info, tonemap, npl, out_range="tv", out_format="yuv420p"),
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-an", str(tmp)],
                   check=True)
    tmp.replace(output)
    return output
