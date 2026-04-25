"""Image and video compression core.

All actual encoding logic lives here — pure functions that take a path,
options, and a cancel flag, and produce a result dict. The UI layer in
`app.py` calls these from worker threads and reports progress.

This is the same logic that drove every previous Cove Compressor build,
just lifted out of the monolithic GUI file so the redesign can keep things
clean. No behavior changes.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

AVIF_AVAILABLE = Image.registered_extensions().get(".avif") is not None

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".wmv", ".flv"}

DEFAULT_OUTPUT = str(Path.home() / "Downloads" / "cove-compressed")

IMAGE_PRESETS = {
    "Light":      {"jpeg_q": 90, "webp_q": 88, "avif_q": 80, "png_colors": None},
    "Balanced":   {"jpeg_q": 78, "webp_q": 75, "avif_q": 65, "png_colors": None},
    "Aggressive": {"jpeg_q": 62, "webp_q": 55, "avif_q": 45, "png_colors": 256},
}

FORMAT_OPTIONS = ["Keep original", "Force JPEG", "Force PNG", "Force WebP"]
if AVIF_AVAILABLE:
    FORMAT_OPTIONS.append("Force AVIF")

FORMAT_KEY_MAP = {
    "Keep original": "keep",
    "Force JPEG":    "jpeg",
    "Force PNG":     "png",
    "Force WebP":    "webp",
    "Force AVIF":    "avif",
}

VIDEO_MODES = ["Target file size", "Target reduction", "Quality preset"]

# Per-codec CRF values per quality preset.
VIDEO_QUALITY_PRESETS = {
    "Web Small":     {"x265": 30, "x264": 26, "vp9": 37, "speed": "medium"},
    "Balanced":      {"x265": 25, "x264": 22, "vp9": 31, "speed": "medium"},
    "Archive Light": {"x265": 22, "x264": 20, "vp9": 27, "speed": "slow"},
}

# Container/codec presets. libvpx-vp9 is markedly slower than x265 — users
# opt into that tradeoff by picking WebM.
VIDEO_FORMATS = {
    "MP4 (H.265)": {
        "ext": ".mp4", "muxer": "mp4",
        "codec": "libx265", "codec_key": "x265",
        "audio": "aac", "container_flags": ["-movflags", "+faststart"],
        "supports_two_pass": True,
    },
    "MP4 (H.264)": {
        "ext": ".mp4", "muxer": "mp4",
        "codec": "libx264", "codec_key": "x264",
        "audio": "aac", "container_flags": ["-movflags", "+faststart"],
        "supports_two_pass": True,
    },
    "MKV (H.265)": {
        "ext": ".mkv", "muxer": "matroska",
        "codec": "libx265", "codec_key": "x265",
        "audio": "aac", "container_flags": [],
        "supports_two_pass": True,
    },
    "WebM (VP9)": {
        "ext": ".webm", "muxer": "webm",
        "codec": "libvpx-vp9", "codec_key": "vp9",
        "audio": "libopus", "container_flags": [],
        "supports_two_pass": True,
    },
}

RESOLUTION_CAPS = {
    "Original": None,
    "1080p":    1920,
    "720p":     1280,
    "480p":     854,
}

RESIZE_CAPS_IMG = {
    "No cap":  None,
    "4000 px": 4000,
    "2560 px": 2560,
    "1920 px": 1920,
    "1280 px": 1280,
}

AUDIO_BITRATES = ["128", "192", "320"]


# Windowed PyInstaller builds on Windows have no console; without
# CREATE_NO_WINDOW every ffmpeg/ffprobe Popen flashes a black cmd window.
if sys.platform == "win32":
    SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW}
else:
    SUBPROCESS_FLAGS = {}


def _resolve_binary(name: str) -> str:
    """Locate ffmpeg/ffprobe. Prefer a binary shipped next to the app
    (bundled release), then next to the package (dev), then PATH."""
    exe = f"{name}.exe" if sys.platform == "win32" else name
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / exe)
    candidates.append(Path(__file__).resolve().parent.parent.parent / exe)
    for c in candidates:
        if c.is_file():
            return str(c)
    return shutil.which(name) or name


FFMPEG_BIN  = _resolve_binary("ffmpeg")
FFPROBE_BIN = _resolve_binary("ffprobe")


# ── Helpers ──────────────────────────────────────────────────────────────────

def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} TB"


def pct_saved(original: int, new: int) -> float:
    return 0.0 if original <= 0 else (original - new) / original * 100.0


def scan_files(folder: Path, exts: set) -> list:
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


def unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suf, parent = base.stem, base.suffix, base.parent
    i = 1
    while True:
        c = parent / f"{stem}_{i}{suf}"
        if not c.exists():
            return c
        i += 1


def reserve_output(base: Path) -> tuple[Path, Path]:
    """Atomically claim an output path via O_CREAT|O_EXCL. Concurrent callers
    targeting the same name bump to _1, _2, …  Returns (output, tmp)."""
    stem, suf, parent = base.stem, base.suffix, base.parent
    i = 0
    while True:
        candidate = base if i == 0 else parent / f"{stem}_{i}{suf}"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return candidate, candidate.with_suffix(candidate.suffix + ".tmp")
        except FileExistsError:
            i += 1


def ffprobe_duration(path: Path) -> float | None:
    if not shutil.which(FFPROBE_BIN):
        return None
    try:
        r = subprocess.run(
            [FFPROBE_BIN, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
            **SUBPROCESS_FLAGS,
        )
        out = r.stdout.strip()
        return float(out) if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")


def parse_ffmpeg_time(line: str) -> float | None:
    m = _TIME_RE.search(line)
    if m:
        h, mi, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return h * 3600 + mi * 60 + s + cs / 100.0
    return None


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 360000:
        return "calculating…"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def clean_subprocess_env() -> dict:
    """Env for external helpers (xdg-open, nautilus, etc). Inside an AppImage
    we inherit LD_LIBRARY_PATH / QT_PLUGIN_PATH / PYTHONHOME pointing at the
    bundle's libs — those break system helpers. Strip them and restore the
    host's original LD_LIBRARY_PATH if AppRun stashed it."""
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
                "PYTHONHOME", "PYTHONPATH", "GTK_EXE_PREFIX", "GTK_DATA_PREFIX"):
        env.pop(key, None)
    orig = os.environ.get("LD_LIBRARY_PATH_ORIG")
    if orig:
        env["LD_LIBRARY_PATH"] = orig
    return env


def open_in_file_manager(path: Path) -> None:
    """Open a folder in the OS file manager."""
    if not path.exists():
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], env=clean_subprocess_env())
        else:
            subprocess.Popen(["xdg-open", str(path)],
                             env=clean_subprocess_env(),
                             **SUBPROCESS_FLAGS)
    except Exception:  # noqa: BLE001
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


# ── Image compression ────────────────────────────────────────────────────────

def compress_image(
    input_path: Path,
    output_dir: Path,
    preset_name: str,
    force_format: str,
    resize_cap,
) -> dict:
    preset = IMAGE_PRESETS[preset_name]
    original_size = input_path.stat().st_size
    output_path: Path | None = None
    tmp_path: Path | None = None

    try:
        img = Image.open(input_path)
        img.load()
    except Exception as e:
        return {"file": input_path, "status": "error", "msg": f"Could not open: {e}"}

    try:
        img = ImageOps.exif_transpose(img)

        if resize_cap is not None:
            w, h = img.size
            longest = max(w, h)
            if longest > resize_cap:
                scale = resize_cap / longest
                img = img.resize((int(w * scale), int(h * scale)), LANCZOS)

        src_ext = input_path.suffix.lower()
        if force_format == "jpeg":
            out_ext, save_format = ".jpg", "JPEG"
        elif force_format == "png":
            out_ext, save_format = ".png", "PNG"
        elif force_format == "webp":
            out_ext, save_format = ".webp", "WEBP"
        elif force_format == "avif":
            out_ext, save_format = ".avif", "AVIF"
        else:
            if src_ext in (".jpg", ".jpeg"):
                out_ext, save_format = ".jpg", "JPEG"
            elif src_ext == ".png":
                out_ext, save_format = ".png", "PNG"
            elif src_ext == ".avif":
                out_ext, save_format = ".avif", "AVIF"
            elif src_ext in (".bmp", ".tiff", ".tif"):
                out_ext, save_format = ".webp", "WEBP"
            else:
                out_ext, save_format = ".webp", "WEBP"

        if save_format == "JPEG":
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.getchannel("A"))
                img = bg
            elif img.mode == "P":
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.getchannel("A"))
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

        output_path, tmp_path = reserve_output(output_dir / f"{input_path.stem}{out_ext}")

        if save_format == "JPEG":
            img.save(tmp_path, "JPEG", quality=preset["jpeg_q"],
                     optimize=True, progressive=True)
        elif save_format == "WEBP":
            img.save(tmp_path, "WEBP", quality=preset["webp_q"], method=6)
        elif save_format == "AVIF":
            img.save(tmp_path, "AVIF", quality=preset["avif_q"])
        else:
            save_img = img
            if preset["png_colors"] is not None and img.mode == "RGB":
                save_img = img.quantize(colors=preset["png_colors"])
            save_img.save(tmp_path, "PNG", optimize=True)

    except Exception as e:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        return {"file": input_path, "status": "error", "msg": f"Save failed: {e}"}
    finally:
        img.close()

    new_size = tmp_path.stat().st_size
    if new_size >= original_size and force_format == "keep":
        tmp_path.unlink()
        output_path.unlink(missing_ok=True)
        return {"file": input_path, "status": "skipped",
                "original": original_size, "new": original_size,
                "msg": "compression would increase size"}

    tmp_path.replace(output_path)
    return {"file": input_path, "output": output_path, "status": "ok",
            "original": original_size, "new": new_size}


# ── Video compression ────────────────────────────────────────────────────────

def build_scale_filter(long_side: int) -> str:
    return (
        f"scale={long_side}:{long_side}:force_original_aspect_ratio=decrease,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


def calc_video_bitrate_kbps(target_bytes: int, duration: float, audio_kbps: int) -> int:
    usable = target_bytes * 0.97
    total_kbps = (usable * 8) / duration / 1000.0
    return max(int(total_kbps - audio_kbps), 80)


def run_ffmpeg(cmd: list, cancel_flag: threading.Event,
               duration: float | None = None,
               on_progress=None) -> tuple:
    """Run ffmpeg, parse progress from stderr, honor cancel."""
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True,
                                **SUBPROCESS_FLAGS)
    except FileNotFoundError:
        return -1, f"{FFMPEG_BIN} not found on PATH"

    stderr_tail: deque = deque(maxlen=40)
    assert proc.stderr is not None
    while True:
        line = proc.stderr.readline()
        if line:
            stderr_tail.append(line.rstrip())
            if duration and duration > 0 and on_progress:
                t = parse_ffmpeg_time(line)
                if t is not None:
                    on_progress(min(t / duration * 100, 100.0))
        elif proc.poll() is not None:
            break
        if cancel_flag.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return -2, "cancelled"

    tail = list(stderr_tail)[-5:]
    return proc.returncode, "\n".join(tail)


def compress_video(
    input_path: Path,
    output_dir: Path,
    mode: str,
    mode_value,
    video_format_key: str,
    resolution_cap,
    audio_kbps: str,
    cancel_flag: threading.Event,
    progress_cb=None,
) -> dict:
    fmt = VIDEO_FORMATS[video_format_key]
    encoder    = fmt["codec"]
    codec_key  = fmt["codec_key"]
    audio_enc  = fmt["audio"]
    out_ext    = fmt["ext"]
    muxer      = fmt["muxer"]
    container_flags = fmt["container_flags"]
    two_pass_ok     = fmt["supports_two_pass"]

    original_size = input_path.stat().st_size
    output_path = unique_path(output_dir / f"{input_path.stem}{out_ext}")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    duration = ffprobe_duration(input_path)

    use_two_pass = False
    video_kbps = None
    crf = None
    speed_preset = "medium"

    if mode in ("Target file size", "Target reduction"):
        if not duration or duration <= 0:
            return {"file": input_path, "status": "error",
                    "msg": "Could not read duration (ffprobe failed)"}

        if mode == "Target file size":
            target_bytes = int(float(mode_value) * 1024 * 1024)
        else:
            keep_pct = max(5.0, min(95.0, 100.0 - float(mode_value)))
            target_bytes = int(original_size * keep_pct / 100.0)

        if target_bytes >= original_size:
            return {"file": input_path, "status": "skipped",
                    "original": original_size, "new": original_size,
                    "msg": "target size >= original; nothing to do"}

        video_kbps = calc_video_bitrate_kbps(target_bytes, duration, int(audio_kbps))
        use_two_pass = two_pass_ok
    else:
        p = VIDEO_QUALITY_PRESETS[str(mode_value)]
        crf = p[codec_key]
        speed_preset = p["speed"]

    vf = build_scale_filter(resolution_cap) if resolution_cap else None
    ffmpeg_base = [FFMPEG_BIN, "-nostdin", "-hide_banner", "-y"]
    common_in = ["-i", str(input_path)]

    def vargs(pass_num):
        a = ["-c:v", encoder]
        if encoder in ("libx264", "libx265"):
            a += ["-preset", speed_preset]
        elif encoder == "libvpx-vp9":
            # libvpx-vp9 is slow. row-mt + cpu-used 4 keeps quality reasonable
            # without taking forever.
            a += ["-row-mt", "1", "-cpu-used", "4"]
        if vf:
            a += ["-vf", vf]
        if use_two_pass:
            a += ["-b:v", f"{video_kbps}k"]
            if pass_num:
                a += ["-pass", str(pass_num)]
        elif video_kbps is not None:
            a += ["-b:v", f"{video_kbps}k"]
        else:
            a += ["-crf", str(crf)]
            if encoder == "libvpx-vp9":
                a += ["-b:v", "0"]
        if encoder == "libx265":
            a += ["-x265-params", "log-level=error"]
        return a

    def _make_progress(offset: float, scale: float, label: str):
        if not progress_cb:
            return None
        def cb(raw_pct):
            progress_cb(offset + raw_pct * scale / 100.0, label)
        return cb

    with tempfile.TemporaryDirectory(prefix="cove_") as td:
        passlog = os.path.join(td, "ffpass")

        if use_two_pass:
            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + vargs(1) + [
                    "-passlogfile", passlog, "-an", "-f", "null", os.devnull],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(0, 35, "pass 1/2"))
            if rc == -2:
                return {"file": input_path, "status": "error", "msg": "cancelled"}
            if rc != 0:
                return {"file": input_path, "status": "error",
                        "msg": f"pass 1 failed: {err}"}

            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + vargs(2) + [
                    "-passlogfile", passlog,
                    "-c:a", audio_enc, "-b:a", f"{audio_kbps}k",
                ] + container_flags + ["-f", muxer, str(tmp_path)],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(35, 65, "pass 2/2"))
        else:
            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + vargs(None) + [
                    "-c:a", audio_enc, "-b:a", f"{audio_kbps}k",
                ] + container_flags + ["-f", muxer, str(tmp_path)],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(0, 100, "encoding"))

    if rc == -2:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"file": input_path, "status": "error", "msg": "cancelled"}
    if rc != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"file": input_path, "status": "error", "msg": f"ffmpeg failed: {err}"}
    if not tmp_path.exists():
        return {"file": input_path, "status": "error", "msg": "no output file produced"}

    new_size = tmp_path.stat().st_size
    if mode == "Quality preset" and new_size >= original_size:
        tmp_path.unlink()
        return {"file": input_path, "status": "skipped",
                "original": original_size, "new": original_size,
                "msg": "compression would increase size (try Target reduction mode)"}

    tmp_path.rename(output_path)
    return {"file": input_path, "output": output_path, "status": "ok",
            "original": original_size, "new": new_size}
