#!/usr/bin/env python3
"""
Cove Compressor — offline batch image and video compressor.
No cloud. No API keys. No accounts.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit(
        "Pillow required.\n"
        "  pip install Pillow"
    )

try:
    import pillow_avif  # noqa: F401  — registers AVIF plugin with Pillow
except ImportError:
    pass
AVIF_AVAILABLE = Image.registered_extensions().get(".avif") is not None

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget,
        QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
        QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
        QTextEdit, QProgressBar, QFileDialog, QMessageBox,
        QStackedWidget,
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPalette
except ImportError:
    sys.exit(
        "PySide6 required.\n"
        "  pip install PySide6"
    )

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def resource_path(relative: str) -> Path:
    """Resolve path to a bundled resource. Works both in dev and PyInstaller --onefile."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative
    return Path(__file__).resolve().parent / relative


# =============================================================================
# Configuration
# =============================================================================

APP_NAME = "Cove Compressor"
__version__ = "1.1.0"

import updater  # noqa: E402  (must follow APP_NAME / __version__ for context)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}

DEFAULT_OUTPUT = str(Path.home() / "Downloads" / "cove-compressed")

IMAGE_PRESETS = {
    "Light":      {"jpeg_q": 90, "webp_q": 88, "avif_q": 80, "png_colors": None},
    "Balanced":   {"jpeg_q": 78, "webp_q": 75, "avif_q": 65, "png_colors": None},
    "Aggressive": {"jpeg_q": 62, "webp_q": 55, "avif_q": 45, "png_colors": 256},
}

VIDEO_MODES = ["Target file size", "Target reduction", "Quality preset"]

VIDEO_QUALITY_PRESETS = {
    "Web Small":     {"crf_x265": 30, "crf_x264": 26, "speed": "medium"},
    "Balanced":      {"crf_x265": 25, "crf_x264": 22, "speed": "medium"},
    "Archive Light": {"crf_x265": 22, "crf_x264": 20, "speed": "slow"},
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
FORMAT_OPTIONS = ["Keep original", "Force JPEG", "Force WebP"]
if AVIF_AVAILABLE:
    FORMAT_OPTIONS.append("Force AVIF")
CODEC_OPTIONS  = ["H.265 (x265)", "H.264 (x264)"]

# On Windows a --windowed PyInstaller build has no console, and every
# subprocess.Popen for ffmpeg/ffprobe flashes a cmd window unless we pass
# CREATE_NO_WINDOW. No-op on other platforms.
if sys.platform == "win32":
    SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW}
else:
    SUBPROCESS_FLAGS = {}


def _resolve_binary(name: str) -> str:
    """Locate ffmpeg/ffprobe. Prefer a binary shipped next to the app
    (bundled release), then next to the .py source (dev-from-source), then
    anything on PATH. Falls back to the bare name so shutil.which() elsewhere
    can still produce a sensible 'not found' message."""
    exe = f"{name}.exe" if sys.platform == "win32" else name
    candidates = []
    # Frozen: PyInstaller --onefile sets sys._MEIPASS and sys.executable
    # points at the extracted exe. The bundled ffmpeg.exe lives next to the
    # exe, not inside _MEIPASS.
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / exe)
    # Dev from source:
    candidates.append(Path(__file__).resolve().parent / exe)
    for c in candidates:
        if c.is_file():
            return str(c)
    return shutil.which(name) or name


FFMPEG_BIN  = _resolve_binary("ffmpeg")
FFPROBE_BIN = _resolve_binary("ffprobe")


# =============================================================================
# Helpers
# =============================================================================

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
    """Atomically claim an output path by creating a zero-byte placeholder
    via O_CREAT|O_EXCL. Concurrent callers targeting the same name will
    bump to _1, _2, … Returns (output_path, tmp_path).

    The placeholder is later overwritten by renaming tmp → output, or
    must be cleaned up with output_path.unlink(missing_ok=True) on error.
    """
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


# Regex for ffmpeg's progress line: time=HH:MM:SS.cc
_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")


def parse_ffmpeg_time(line: str) -> float | None:
    """Extract the current encode position in seconds from an ffmpeg stderr line."""
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


# =============================================================================
# Image compression
# =============================================================================

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


# =============================================================================
# Video compression
# =============================================================================

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
    """Run ffmpeg, parse progress from stderr, honor cancel.
    on_progress(pct) is called with 0-100 for this encode pass."""
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
    codec: str,
    resolution_cap,
    audio_kbps: str,
    cancel_flag: threading.Event,
    progress_cb=None,
) -> dict:
    """Compress a single video.
    progress_cb(file_pct, label) is called with 0-100 progress for this file
    and a label like 'pass 1/2', 'pass 2/2', or 'encoding'."""
    original_size = input_path.stat().st_size
    encoder = "libx265" if codec == "x265" else "libx264"
    output_path = unique_path(output_dir / f"{input_path.stem}.mp4")
    tmp_path = output_path.with_suffix(".mp4.tmp")

    # Always probe duration — needed for progress tracking in every mode.
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
        use_two_pass = True
    else:
        p = VIDEO_QUALITY_PRESETS[str(mode_value)]
        crf = p["crf_x265"] if codec == "x265" else p["crf_x264"]
        speed_preset = p["speed"]

    vf = build_scale_filter(resolution_cap) if resolution_cap else None
    ffmpeg_base = [FFMPEG_BIN, "-nostdin", "-hide_banner", "-y"]
    common_in = ["-i", str(input_path)]

    def vargs(pass_num):
        a = ["-c:v", encoder, "-preset", speed_preset]
        if vf:
            a += ["-vf", vf]
        if use_two_pass:
            a += ["-b:v", f"{video_kbps}k"]
            if pass_num:
                a += ["-pass", str(pass_num)]
        else:
            a += ["-crf", str(crf)]
        if encoder == "libx265":
            a += ["-x265-params", "log-level=error"]
        return a

    # Build weighted progress callbacks:
    # 2-pass: pass 1 = 0–35%, pass 2 = 35–100%
    # 1-pass: 0–100%
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
                    "-passlogfile", passlog, "-an", "-f", "mp4", os.devnull],
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
                    "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                    "-movflags", "+faststart", "-f", "mp4", str(tmp_path)],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(35, 65, "pass 2/2"))
        else:
            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + vargs(None) + [
                    "-c:a", "aac", "-b:a", f"{audio_kbps}k",
                    "-movflags", "+faststart", "-f", "mp4", str(tmp_path)],
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


# =============================================================================
# GUI — PySide6
# =============================================================================

class CoveCompressor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.setAcceptDrops(True)
        self.resize(900, 800)

        icon_file = resource_path("cove_icon.png")
        if icon_file.exists():
            self.setWindowIcon(QIcon(str(icon_file)))

        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_flag = threading.Event()

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_queue)
        self._timer.start(80)

        threading.Thread(target=self._check_deps, daemon=True).start()

        self._updater = updater.UpdateController(
            parent=self,
            current_version=__version__,
            repo="Sin213/cove-compressor",
            app_display_name=APP_NAME,
            cache_subdir="cove-compressor",
        )
        QTimer.singleShot(4000, self._updater.check)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_image_tab(), "Images")
        self.tabs.addTab(self._build_video_tab(), "Videos")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setAcceptRichText(False)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(9)
        self.log.setFont(mono)
        self.log.setFixedHeight(210)
        self.log.setStyleSheet(
            "QTextEdit { background: #0f0f12; color: #dcdcdc; "
            "border: 1px solid #2a2a2a; }"
        )
        root.addWidget(self.log)

        prog_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        prog_row.addWidget(self.progress, stretch=1)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setMinimumWidth(380)
        prog_row.addWidget(self.status_label)
        root.addLayout(prog_row)

    def _path_row(self, label_text: str, line_edit: QLineEdit,
                  browse_fn) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(112)
        row.addWidget(lbl)
        row.addWidget(line_edit)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(82)
        btn.clicked.connect(browse_fn)
        row.addWidget(btn)
        return row

    def _build_image_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(6)

        self.img_folder = QLineEdit()
        self.img_folder.setPlaceholderText("Folder containing images…")
        self.img_file = QLineEdit()
        self.img_file.setPlaceholderText("Or select a single image file…")
        self.img_output = QLineEdit(DEFAULT_OUTPUT)

        v.addLayout(self._path_row("Input folder:", self.img_folder,
                                   lambda: self._browse_folder(self.img_folder)))
        v.addLayout(self._path_row("Input file:", self.img_file,
                                   lambda: self._browse_file(self.img_file, IMAGE_EXTS)))
        v.addLayout(self._path_row("Output folder:", self.img_output,
                                   lambda: self._browse_folder(self.img_output)))

        hint = QLabel("  ↓  Drag files or folders onto this window to fill the input")
        hint.setStyleSheet("color: #888; font-style: italic;")
        v.addWidget(hint)

        opts = QGroupBox("Options")
        g = QGridLayout(opts)
        g.setSpacing(8)
        g.setContentsMargins(12, 12, 12, 12)

        self.img_preset = QComboBox()
        self.img_preset.addItems(list(IMAGE_PRESETS.keys()))
        self.img_preset.setCurrentText("Balanced")

        self.img_format = QComboBox()
        self.img_format.addItems(FORMAT_OPTIONS)

        self.img_resize = QComboBox()
        self.img_resize.addItems(list(RESIZE_CAPS_IMG.keys()))

        g.addWidget(QLabel("Preset:"), 0, 0)
        g.addWidget(self.img_preset, 0, 1)
        g.addWidget(QLabel("Output format:"), 0, 2)
        g.addWidget(self.img_format, 0, 3)
        g.addWidget(QLabel("Resize cap:"), 1, 0)
        g.addWidget(self.img_resize, 1, 1)
        g.setColumnStretch(3, 1)

        v.addWidget(opts)

        note = QLabel(
            "Metadata (EXIF, GPS, camera info, timestamps) is always stripped.")
        note.setStyleSheet("color: #888;")
        v.addWidget(note)
        v.addStretch()

        btns = QHBoxLayout()
        self.img_start = QPushButton("Start image compression")
        self.img_cancel = QPushButton("Cancel")
        self.img_cancel.setEnabled(False)
        self.img_start.clicked.connect(self._start_images)
        self.img_cancel.clicked.connect(self._cancel)
        btns.addWidget(self.img_start)
        btns.addWidget(self.img_cancel)
        btns.addStretch()
        v.addLayout(btns)

        return w

    def _build_video_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(6)

        self.vid_folder = QLineEdit()
        self.vid_folder.setPlaceholderText("Folder containing videos…")
        self.vid_file = QLineEdit()
        self.vid_file.setPlaceholderText("Or select a single video file…")
        self.vid_output = QLineEdit(DEFAULT_OUTPUT)

        v.addLayout(self._path_row("Input folder:", self.vid_folder,
                                   lambda: self._browse_folder(self.vid_folder)))
        v.addLayout(self._path_row("Input file:", self.vid_file,
                                   lambda: self._browse_file(self.vid_file, VIDEO_EXTS)))
        v.addLayout(self._path_row("Output folder:", self.vid_output,
                                   lambda: self._browse_folder(self.vid_output)))

        hint = QLabel("  ↓  Drag files or folders onto this window to fill the input")
        hint.setStyleSheet("color: #888; font-style: italic;")
        v.addWidget(hint)

        comp = QGroupBox("Compression")
        cv = QVBoxLayout(comp)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Method:"))
        self.vid_mode = QComboBox()
        self.vid_mode.addItems(VIDEO_MODES)
        self.vid_mode.setCurrentText("Quality preset")
        self.vid_mode.currentTextChanged.connect(self._update_vid_mode)
        mode_row.addWidget(self.vid_mode)
        mode_row.addStretch()
        cv.addLayout(mode_row)

        # Stacked widget — one page per mode
        self.vid_stack = QStackedWidget()

        # Page 0 — Target file size
        p0 = QWidget()
        r0 = QHBoxLayout(p0)
        r0.setContentsMargins(0, 0, 0, 0)
        r0.addWidget(QLabel("Target size:"))
        self.vid_size_mb = QSpinBox()
        self.vid_size_mb.setRange(1, 50000)
        self.vid_size_mb.setValue(10)
        self.vid_size_mb.setSuffix(" MB")
        r0.addWidget(self.vid_size_mb)
        lbl0 = QLabel("  2-pass encoded to hit this exact size")
        lbl0.setStyleSheet("color: #888;")
        r0.addWidget(lbl0)
        r0.addStretch()
        self.vid_stack.addWidget(p0)

        # Page 1 — Target reduction (default)
        p1 = QWidget()
        r1 = QHBoxLayout(p1)
        r1.setContentsMargins(0, 0, 0, 0)
        r1.addWidget(QLabel("Reduce by:"))
        self.vid_pct = QSpinBox()
        self.vid_pct.setRange(10, 90)
        self.vid_pct.setSingleStep(5)
        self.vid_pct.setValue(50)
        self.vid_pct.setSuffix(" %")
        r1.addWidget(self.vid_pct)
        lbl1 = QLabel("  50% = output is half the original size")
        lbl1.setStyleSheet("color: #888;")
        r1.addWidget(lbl1)
        r1.addStretch()
        self.vid_stack.addWidget(p1)

        # Page 2 — Quality preset
        p2 = QWidget()
        r2 = QHBoxLayout(p2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(QLabel("Preset:"))
        self.vid_quality = QComboBox()
        self.vid_quality.addItems(list(VIDEO_QUALITY_PRESETS.keys()))
        self.vid_quality.setCurrentText("Balanced")
        r2.addWidget(self.vid_quality)
        lbl2 = QLabel("   CRF-based, skips if output would be larger")
        lbl2.setStyleSheet("color: #888;")
        r2.addWidget(lbl2)
        r2.addStretch()
        self.vid_stack.addWidget(p2)

        self.vid_stack.setCurrentIndex(2)
        cv.addWidget(self.vid_stack)

        g = QGridLayout()
        g.setSpacing(8)

        self.vid_codec = QComboBox()
        self.vid_codec.addItems(CODEC_OPTIONS)

        self.vid_res = QComboBox()
        self.vid_res.addItems(list(RESOLUTION_CAPS.keys()))

        self.vid_audio = QComboBox()
        self.vid_audio.addItems(AUDIO_BITRATES)
        self.vid_audio.setCurrentText("192")

        g.addWidget(QLabel("Codec:"), 0, 0)
        g.addWidget(self.vid_codec, 0, 1)
        g.addWidget(QLabel("Resolution cap:"), 0, 2)
        g.addWidget(self.vid_res, 0, 3)
        g.addWidget(QLabel("Audio kbps:"), 1, 0)
        g.addWidget(self.vid_audio, 1, 1)
        g.setColumnStretch(3, 1)
        cv.addLayout(g)

        v.addWidget(comp)

        note = QLabel("Output is always .mp4 (web-optimized, +faststart).")
        note.setStyleSheet("color: #888;")
        v.addWidget(note)
        v.addStretch()

        btns = QHBoxLayout()
        self.vid_start = QPushButton("Start video compression")
        self.vid_cancel = QPushButton("Cancel")
        self.vid_cancel.setEnabled(False)
        self.vid_start.clicked.connect(self._start_videos)
        self.vid_cancel.clicked.connect(self._cancel)
        btns.addWidget(self.vid_start)
        btns.addWidget(self.vid_cancel)
        btns.addStretch()
        v.addLayout(btns)

        return w

    def _update_vid_mode(self, text: str):
        self.vid_stack.setCurrentIndex(
            {"Target file size": 0, "Target reduction": 1, "Quality preset": 2}.get(text, 1)
        )

    # ------------------------------------------------------------------
    # Drag and drop — native Qt, zero extra dependencies
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
        self._handle_drop(paths, is_video=self.tabs.currentIndex() == 1)

    def _handle_drop(self, paths: list, is_video: bool):
        if not paths:
            return
        exts = VIDEO_EXTS if is_video else IMAGE_EXTS
        folder_edit = self.vid_folder if is_video else self.img_folder
        file_edit   = self.vid_file   if is_video else self.img_file

        if len(paths) == 1 and paths[0].is_file():
            if paths[0].suffix.lower() not in exts:
                self._log(f"[drop] ignored (unsupported type): {paths[0].name}")
                return
            file_edit.setText(str(paths[0]))
            folder_edit.clear()
            self._log(f"[drop] file: {paths[0].name}")
        elif len(paths) == 1 and paths[0].is_dir():
            folder_edit.setText(str(paths[0]))
            file_edit.clear()
            self._log(f"[drop] folder: {paths[0]}")
        else:
            parent = next((p.parent for p in paths if p.exists()), None)
            if parent:
                folder_edit.setText(str(parent))
                file_edit.clear()
                self._log(f"[drop] multiple items — using folder: {parent}")

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_folder(self, edit: QLineEdit):
        start = edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            edit.setText(folder)

    def _browse_file(self, edit: QLineEdit, exts: set):
        start = str(Path(edit.text()).parent) if edit.text() else str(Path.home())
        exts_str = " ".join(f"*{e}" for e in sorted(exts))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", start,
            f"Supported files ({exts_str});;All files (*.*)",
        )
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------
    # Queue / thread-safe UI updates
    # ------------------------------------------------------------------

    def _log(self, line: str):
        self.msg_queue.put(("log", line))

    def _poll_queue(self):
        # Coalesce: logs are all applied in order, but status/progress/finish
        # only need the latest value per tick so the UI doesn't thrash.
        latest_status = None
        latest_progress = None
        finish = False
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log.append(payload)
                elif kind == "status":
                    latest_status = payload
                elif kind == "progress":
                    latest_progress = payload
                elif kind == "finish":
                    finish = True
        except queue.Empty:
            pass

        if latest_status is not None:
            self.status_label.setText(latest_status)
        if latest_progress is not None:
            self.progress.setValue(int(latest_progress))
        if finish:
            self._set_running(False)

    def _set_running(self, running: bool):
        self.img_start.setEnabled(not running)
        self.vid_start.setEnabled(not running)
        self.img_cancel.setEnabled(running)
        self.vid_cancel.setEnabled(running)

    def _cancel(self):
        self.cancel_flag.set()
        self.msg_queue.put(("status", "Cancelling…"))

    def _set_status(self, text: str):
        self.msg_queue.put(("status", text))

    def _set_progress(self, pct: float):
        self.msg_queue.put(("progress", pct))

    def _finish(self):
        self.msg_queue.put(("finish", None))

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    def _check_deps(self):
        for name, bin_ in ((FFMPEG_BIN, FFMPEG_BIN), (FFPROBE_BIN, FFPROBE_BIN)):
            path = shutil.which(bin_)
            if path:
                self._log(f"[ok] {name}: {path}")
            else:
                self._log(f"[ERROR] {name} not found on PATH")
        self._log("")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _collect_inputs(self, folder_edit: QLineEdit, file_edit: QLineEdit,
                        exts: set, kind: str):
        folder = folder_edit.text().strip()
        single = file_edit.text().strip()

        if not folder and not single:
            QMessageBox.warning(self, APP_NAME,
                                "Pick an input folder or file first\n"
                                "(or drag one onto this window).")
            return None
        if single:
            p = Path(single)
            if not p.is_file():
                QMessageBox.warning(self, APP_NAME,
                                    f"Input file does not exist:\n{single}")
                return None
            if p.suffix.lower() not in exts:
                QMessageBox.warning(self, APP_NAME,
                                    f"Not a supported {kind}:\n{single}")
                return None
            return [p]

        p = Path(folder)
        if not p.is_dir():
            QMessageBox.warning(self, APP_NAME,
                                f"Input folder does not exist:\n{folder}")
            return None
        files = scan_files(p, exts)
        if not files:
            QMessageBox.information(self, APP_NAME,
                                    f"No {kind}s found in that folder.")
            return None
        return files

    def _prepare_output(self, edit: QLineEdit, input_paths: list):
        out_str = edit.text().strip()
        if not out_str:
            QMessageBox.warning(self, APP_NAME, "Pick an output folder.")
            return None
        out = Path(out_str)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME,
                                f"Could not create output folder:\n{e}")
            return None
        try:
            if out.resolve() in {p.parent.resolve() for p in input_paths}:
                r = QMessageBox.question(
                    self, APP_NAME,
                    "Output folder is the same as the input folder.\n\n"
                    "Compressed files will be written alongside originals "
                    "with unique names. Continue?",
                )
                if r != QMessageBox.Yes:
                    return None
        except OSError:
            pass
        return out

    # ------------------------------------------------------------------
    # Start handlers
    # ------------------------------------------------------------------

    def _start_images(self):
        files = self._collect_inputs(self.img_folder, self.img_file,
                                     IMAGE_EXTS, "image")
        if not files:
            return
        output_dir = self._prepare_output(self.img_output, files)
        if not output_dir:
            return

        preset = self.img_preset.currentText()
        force_format = {"Keep original": "keep",
                        "Force JPEG": "jpeg",
                        "Force WebP": "webp",
                        "Force AVIF": "avif"}[self.img_format.currentText()]
        resize_cap = RESIZE_CAPS_IMG[self.img_resize.currentText()]

        self.cancel_flag.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self._log(f"=== Images • {preset} • {len(files)} file(s) • "
                  f"{datetime.now().strftime('%H:%M:%S')} ===")

        threading.Thread(
            target=self._run_image_batch,
            args=(files, output_dir, preset, force_format, resize_cap),
            daemon=True,
        ).start()

    def _start_videos(self):
        if not shutil.which(FFMPEG_BIN):
            QMessageBox.warning(
                self, APP_NAME,
                "ffmpeg not found.\n\n"
                "Install it:  winget install ffmpeg\n"
                "  — or drop ffmpeg.exe and ffprobe.exe\n"
                "    next to this app.")
            return

        files = self._collect_inputs(self.vid_folder, self.vid_file,
                                     VIDEO_EXTS, "video")
        if not files:
            return
        output_dir = self._prepare_output(self.vid_output, files)
        if not output_dir:
            return

        mode = self.vid_mode.currentText()
        if mode == "Target file size":
            mode_value = float(self.vid_size_mb.value())
            if not shutil.which(FFPROBE_BIN):
                QMessageBox.warning(self, APP_NAME,
                                    "ffprobe required for target size mode.")
                return
        elif mode == "Target reduction":
            mode_value = float(self.vid_pct.value())
            if not shutil.which(FFPROBE_BIN):
                QMessageBox.warning(self, APP_NAME,
                                    "ffprobe required for target reduction mode.")
                return
        else:
            mode_value = self.vid_quality.currentText()

        codec   = "x265" if self.vid_codec.currentText().startswith("H.265") else "x264"
        res_cap = RESOLUTION_CAPS[self.vid_res.currentText()]
        audio   = self.vid_audio.currentText()

        self.cancel_flag.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self._log(f"=== Videos • {mode} ({mode_value}) • {codec} • "
                  f"{len(files)} file(s) • {datetime.now().strftime('%H:%M:%S')} ===")

        threading.Thread(
            target=self._run_video_batch,
            args=(files, output_dir, mode, mode_value, codec, res_cap, audio),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Batch workers
    # ------------------------------------------------------------------

    def _run_image_batch(self, files, output_dir, preset, force_format, resize_cap):
        total = len(files)
        total_orig = total_new = 0
        ok = skipped = errors = 0
        t0 = time.time()
        done = 0

        # Pillow's JPEG/WebP/PNG encoders release the GIL, so threads scale.
        # Cap at 8 to avoid disk thrash on spinning media.
        max_workers = min(os.cpu_count() or 4, 8)
        self._set_status(f"0/{total} • {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(compress_image, f, output_dir,
                            preset, force_format, resize_cap)
                for f in files
            ]
            try:
                for fut in as_completed(futures):
                    if self.cancel_flag.is_set():
                        for pending in futures:
                            pending.cancel()
                        self._log("[cancelled]")
                        break
                    result = fut.result()
                    done += 1
                    self._log(self._fmt(result))
                    if result["status"] == "ok":
                        ok += 1
                        total_orig += result["original"]
                        total_new += result["new"]
                    elif result["status"] == "skipped":
                        skipped += 1
                        total_orig += result["original"]
                        total_new += result["original"]
                    else:
                        errors += 1
                    pct = done / total * 100
                    self._set_progress(pct)
                    if done < total and pct > 0:
                        elapsed = time.time() - t0
                        eta = format_eta(elapsed / pct * (100 - pct))
                        self._set_status(f"{done}/{total} done • ETA {eta}")
                    else:
                        self._set_status(f"{done}/{total} done")
            finally:
                # If we bailed out, don't block shutdown waiting on queued work.
                if self.cancel_flag.is_set():
                    for pending in futures:
                        pending.cancel()

        self._summary(ok, skipped, errors, total_orig, total_new)
        self._set_status("Done")
        self._finish()

    def _run_video_batch(self, files, output_dir, mode, mode_value, codec, res_cap, audio):
        total = len(files)
        total_orig = total_new = 0
        ok = skipped = errors = 0
        t0 = time.time()

        for i, f in enumerate(files, start=1):
            if self.cancel_flag.is_set():
                self._log("[cancelled]")
                break

            # Each file owns an equal slice of the overall progress bar.
            base_pct = (i - 1) / total * 100
            file_span = 100.0 / total

            def on_progress(file_pct, label,
                            _base=base_pct, _span=file_span,
                            _i=i, _f=f):
                overall = _base + file_pct * _span / 100.0
                self._set_progress(overall)
                elapsed = time.time() - t0
                if overall > 1:
                    eta = format_eta(elapsed / overall * (100.0 - overall))
                else:
                    eta = "calculating…"
                name = _f.name if len(_f.name) <= 28 else _f.name[:25] + "…"
                self._set_status(
                    f"{_i}/{total}  {name} — {label} — "
                    f"{file_pct:.0f}% • ETA {eta}"
                )

            self._set_status(f"{i}/{total}  {f.name}")
            result = compress_video(
                f, output_dir, mode, mode_value, codec, res_cap, audio,
                self.cancel_flag,
                progress_cb=on_progress,
            )
            self._log(self._fmt(result))
            if result["status"] == "ok":
                ok += 1; total_orig += result["original"]; total_new += result["new"]
            elif result["status"] == "skipped":
                skipped += 1; total_orig += result["original"]; total_new += result["original"]
            else:
                errors += 1
            # Snap bar to the end of this file's slice.
            self._set_progress((i / total) * 100)

        self._summary(ok, skipped, errors, total_orig, total_new)
        self._set_status("Done")
        self._finish()

    # ------------------------------------------------------------------
    # Log formatting
    # ------------------------------------------------------------------

    def _fmt(self, r: dict) -> str:
        name = r["file"].name
        if len(name) > 42:
            name = name[:39] + "..."
        if r["status"] == "ok":
            s = pct_saved(r["original"], r["new"])
            return (f"[ok]    {name:<42s}  "
                    f"{human_size(r['original']):>10s} → {human_size(r['new']):>10s}  "
                    f"({s:5.1f}% saved)")
        if r["status"] == "skipped":
            return f"[skip]  {name:<42s}  {r.get('msg', 'skipped')}"
        return     f"[err]   {name:<42s}  {r.get('msg', 'error')}"

    def _summary(self, ok, skipped, errors, total_orig, total_new):
        self._log("-" * 76)
        self._log(f"Done.  ok={ok}   skipped={skipped}   errors={errors}")
        if total_orig > 0:
            self._log(f"Total: {human_size(total_orig)} → {human_size(total_new)}  "
                      f"({pct_saved(total_orig, total_new):.1f}% saved)")
        self._log("")


# =============================================================================
# Entry point
# =============================================================================

def apply_dark_theme(app: QApplication) -> None:
    """Force a dark palette app-wide. Qt's default Windows style paints light
    regardless of Windows dark-mode, so we opt into Fusion (which fully
    respects the palette) and install our own dark colors — making the UI
    look identical across Windows, macOS, and Linux."""
    app.setStyle("Fusion")

    BG        = QColor(30, 30, 34)
    BG_ALT    = QColor(40, 40, 44)
    BG_INPUT  = QColor(20, 20, 24)
    BG_BUTTON = QColor(45, 45, 50)
    FG        = QColor(220, 220, 220)
    FG_DIM    = QColor(120, 120, 120)
    ACCENT    = QColor(80, 140, 220)

    p = QPalette()
    p.setColor(QPalette.Window,          BG)
    p.setColor(QPalette.WindowText,      FG)
    p.setColor(QPalette.Base,            BG_INPUT)
    p.setColor(QPalette.AlternateBase,   BG_ALT)
    p.setColor(QPalette.ToolTipBase,     BG_ALT)
    p.setColor(QPalette.ToolTipText,     FG)
    p.setColor(QPalette.Text,            FG)
    p.setColor(QPalette.Button,          BG_BUTTON)
    p.setColor(QPalette.ButtonText,      FG)
    p.setColor(QPalette.BrightText,      QColor(255, 80, 80))
    p.setColor(QPalette.Link,            ACCENT)
    p.setColor(QPalette.Highlight,       ACCENT)
    p.setColor(QPalette.HighlightedText, BG_INPUT)
    p.setColor(QPalette.PlaceholderText, FG_DIM)
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, FG_DIM)
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_dark_theme(app)

    icon_file = resource_path("cove_icon.png")
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))

    window = CoveCompressor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
