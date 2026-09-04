"""Image and video compression core.

All actual encoding logic lives here — pure functions that take a path,
options, and a cancel flag, and produce a result dict. The UI layer in
`app.py` calls these from worker threads and reports progress.

This is the same logic that drove every previous Cove Compressor build,
just lifted out of the monolithic GUI file so the redesign can keep things
clean. No behavior changes.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import stat as stat_mod
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

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
ENCODE_STALL_TIMEOUT = 120  # Seconds without encoding progress.
# `-movflags +faststart` rewrites the finished MP4 to move the moov atom to the
# front. That phase emits no `time=` progress and scales with file size, so it
# gets its own, longer allowance — entered only once ffmpeg says it has begun.
FINALIZE_STALL_TIMEOUT = 600  # Seconds without progress while finalizing.
# Removing a reserved output stub can lose a race with a transient Windows
# file lock. Bounded so a failed job still returns promptly (~200ms worst case).
RESERVED_CLEANUP_ATTEMPTS = 5
RESERVED_CLEANUP_RETRY_DELAY = 0.05

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

# Per-codec quality values per preset. CRF for the software encoders; the
# `nvenc_*` values are NVENC constant-quality (-cq) targets on the same 0-51
# scale (NVENC is less bit-efficient than x264/x265, so the numbers sit a
# little higher for comparable output). `nvenc_preset` is NVENC's p1 (fastest)
# … p7 (best quality) speed dial — the hardware is fast enough that we can
# afford the slower, higher-quality presets by default.
#
# The `amf_*` values are AMD AMF constant-quality (-qp via -rc cqp) targets on
# the same 0-51 scale. `amf_quality` is AMF's -quality dial
# (speed / balanced / quality), mapped 1:1 to the NVENC preset for each Cove
# preset so users get comparable behavior across vendors.
VIDEO_QUALITY_PRESETS = {
    "Web Small":     {"x265": 30, "x264": 26, "vp9": 37,
                      "nvenc_hevc": 32, "nvenc_h264": 30,
                      "amf_hevc": 33, "amf_h264": 31,
                      "speed": "medium", "nvenc_preset": "p5",
                      "amf_quality": "speed"},
    "Balanced":      {"x265": 25, "x264": 22, "vp9": 31,
                      "nvenc_hevc": 27, "nvenc_h264": 25,
                      "amf_hevc": 28, "amf_h264": 26,
                      "speed": "medium", "nvenc_preset": "p6",
                      "amf_quality": "balanced"},
    "Archive Light": {"x265": 22, "x264": 20, "vp9": 27,
                      "nvenc_hevc": 24, "nvenc_h264": 21,
                      "amf_hevc": 25, "amf_h264": 22,
                      "speed": "slow", "nvenc_preset": "p7",
                      "amf_quality": "quality"},
}

# Container/codec presets. libvpx-vp9 is markedly slower than x265 — users
# opt into that tradeoff by picking WebM.
#
# `nvenc_codec` / `nvenc_key` name the NVIDIA hardware encoder that can stand
# in for the software `codec` (H.264 → h264_nvenc, H.265 → hevc_nvenc). VP9
# has no NVENC equivalent, so WebM stays CPU-only.
#
# `amf_codec` / `amf_key` name the AMD AMF hardware encoder that can stand in
# the same way (H.264 → h264_amf, H.265 → hevc_amf). VP9 likewise has no AMF
# equivalent. AMF is treated as a fallback for the Automatic path when NVENC
# isn't available; users can force it explicitly by picking "AMD GPU (AMF)".
VIDEO_FORMATS = {
    "MP4 (H.265)": {
        "ext": ".mp4", "muxer": "mp4",
        "codec": "libx265", "codec_key": "x265",
        "nvenc_codec": "hevc_nvenc", "nvenc_key": "nvenc_hevc",
        "amf_codec": "hevc_amf", "amf_key": "amf_hevc",
        "audio": "aac", "container_flags": ["-movflags", "+faststart"],
        "supports_two_pass": True,
    },
    "MP4 (H.264)": {
        "ext": ".mp4", "muxer": "mp4",
        "codec": "libx264", "codec_key": "x264",
        "nvenc_codec": "h264_nvenc", "nvenc_key": "nvenc_h264",
        "amf_codec": "h264_amf", "amf_key": "amf_h264",
        "audio": "aac", "container_flags": ["-movflags", "+faststart"],
        "supports_two_pass": True,
    },
    "MKV (H.265)": {
        "ext": ".mkv", "muxer": "matroska",
        "codec": "libx265", "codec_key": "x265",
        "nvenc_codec": "hevc_nvenc", "nvenc_key": "nvenc_hevc",
        "amf_codec": "hevc_amf", "amf_key": "amf_hevc",
        "audio": "aac", "container_flags": [],
        "supports_two_pass": True,
    },
    "WebM (VP9)": {
        "ext": ".webm", "muxer": "webm",
        "codec": "libvpx-vp9", "codec_key": "vp9",
        "nvenc_codec": None, "nvenc_key": None,
        "amf_codec": None, "amf_key": None,
        "audio": "libopus", "container_flags": [],
        "supports_two_pass": True,
    },
}

# Encoder preference exposed in the UI. "auto" prefers an available GPU —
# NVENC first, then AMF — and falls back to CPU; "cpu" always uses the
# software encoder; "nvenc" / "amf" force that vendor's GPU (and still fall
# back to CPU when the chosen format or machine can't drive it, rather than
# failing the job).
ENCODER_OPTIONS = [
    "Automatic (GPU if available)",
    "CPU (x264 / x265)",
    "NVIDIA GPU (NVENC)",
    "AMD GPU (AMF)",
]
ENCODER_KEY_MAP = {
    "Automatic (GPU if available)": "auto",
    "CPU (x264 / x265)":            "cpu",
    "NVIDIA GPU (NVENC)":           "nvenc",
    "AMD GPU (AMF)":                "amf",
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


# ── NVENC (NVIDIA hardware encoding) detection ────────────────────────────────
#
# Whether NVENC works is a runtime property of the machine, not just the
# ffmpeg build: the encoders can be compiled in yet fail to initialize with no
# NVIDIA GPU / driver present. So we do a real 1-frame null encode and cache
# the verdict per encoder. The probe is cheap but not free, so callers should
# rely on the cache (warmed once at startup) rather than re-probing per file.

_NVENC_LOCK = threading.Lock()
_nvenc_cache: dict[str, bool] = {}


def _probe_nvenc(encoder: str) -> bool:
    """Return True only if `encoder` is both present in this ffmpeg build and
    can actually initialize on this machine (working NVIDIA GPU + driver)."""
    try:
        listing = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if encoder not in (listing.stdout or ""):
        return False

    # Compiled in — now confirm the hardware actually accepts it. A tiny
    # synthetic clip encoded to the null muxer exercises NVENC init without
    # touching disk or the user's files.
    try:
        probe = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=0.3",
             "-c:v", encoder, "-f", "null", os.devnull],
            capture_output=True, text=True, timeout=30,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def nvenc_available(encoder: str = "hevc_nvenc") -> bool:
    """Cached: can this machine encode with the given NVENC encoder?"""
    with _NVENC_LOCK:
        if encoder in _nvenc_cache:
            return _nvenc_cache[encoder]
    result = _probe_nvenc(encoder)
    with _NVENC_LOCK:
        _nvenc_cache[encoder] = result
    return result


def any_nvenc_available() -> bool:
    """True if either NVENC encoder Cove can use (H.265 or H.264) works here."""
    return nvenc_available("hevc_nvenc") or nvenc_available("h264_nvenc")


# ── AMF (AMD hardware encoding) detection ────────────────────────────────────
#
# Same reasoning as NVENC: AMF encoders can be compiled into ffmpeg yet fail
# to initialize with no AMD GPU / driver present. Probe once per encoder and
# cache the verdict. AMF is only available on Windows and Linux, and the
# bundled builds (gyan.dev on Windows, johnvansickle.com on Linux) ship with
# AMF support compiled in; the probe is what tells us the rest.

_AMF_LOCK = threading.Lock()
_amf_cache: dict[str, bool] = {}


def _probe_amf(encoder: str) -> bool:
    """Return True only if `encoder` is both present in this ffmpeg build and
    can actually initialize on this machine (working AMD GPU + driver)."""
    try:
        listing = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if encoder not in (listing.stdout or ""):
        return False

    # Compiled in — now confirm the hardware actually accepts it. The probe
    # geometry has to be one AMF will really encode at: `hevc_amf` refuses to
    # initialize on tiny frames (an RX 9070 XT answers 320x240 with
    # "encoder->Init() failed with error 5" while encoding 720p HEVC happily),
    # so a thumbnail-sized probe reported no HEVC on hardware that has it.
    # One 1280x720 frame to the null muxer is representative of real work,
    # costs ~0.2s, and touches neither disk nor the user's files.
    try:
        probe = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=1",
             "-frames:v", "1",
             "-c:v", encoder, "-f", "null", os.devnull],
            capture_output=True, text=True, timeout=30,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def amf_available(encoder: str = "hevc_amf") -> bool:
    """Cached: can this machine encode with the given AMF encoder?"""
    with _AMF_LOCK:
        if encoder in _amf_cache:
            return _amf_cache[encoder]
    result = _probe_amf(encoder)
    with _AMF_LOCK:
        _amf_cache[encoder] = result
    return result


def any_amf_available() -> bool:
    """True if either AMF encoder Cove can use (H.265 or H.264) works here."""
    return amf_available("hevc_amf") or amf_available("h264_amf")


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


def output_dir_for(input_file: Path, shared_dir: Path,
                   same_as_source: bool = False) -> Path:
    """Directory a job writes into: its own source folder when opted in."""
    return input_file.parent if same_as_source else shared_dir


def ffprobe_duration(path: Path) -> float | None:
    if not shutil.which(FFPROBE_BIN):
        return None
    try:
        r = subprocess.run(
            [FFPROBE_BIN, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
        out = r.stdout.strip()
        return float(out) if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")

# ffmpeg announces the faststart rewrite as e.g. "Starting second pass: moving
# the moov atom to the beginning of the file".
_MOOV_RE = re.compile(r"moving the moov atom", re.IGNORECASE)


def is_moov_relocation_line(line: str) -> bool:
    """True when ffmpeg says it has started relocating the moov atom.

    This marks the *start* of finalization, never its completion: it says the
    rewrite is under way, so the watchdog should stop expecting `time=` lines.
    It is deliberately not treated as progress, completion, or success.
    """
    return bool(_MOOV_RE.search(line or ""))


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


# ── Optional English subtitle preservation ───────────────────────────────────
#
# Re-encoding a video throws away its embedded subtitle streams. When the user
# opts in, Cove pulls every stream it can confidently identify as English out
# to a sidecar file *before* the encode starts, so the information survives
# even if the original is later deleted.
#
# The whole feature is fail-closed: anything Cove cannot inspect, extract or
# finalize marks the result with `subtitles_failed`, which the Tab 2a deletion
# gate treats as an absolute veto on removing the original.

SUBTITLE_PROBE_TIMEOUT = 30

# Text formats SRT can carry without losing anything that matters.
SRT_SAFE_SUBTITLE_CODECS = {"subrip", "srt", "mov_text", "text", "webvtt", "vtt"}
# Styled text formats: flattening these to SRT would drop the styling, so they
# keep their own container.
ASS_SUBTITLE_CODECS = {"ass", "ssa"}

_ENGLISH_PRIMARY_TAGS = {"en", "eng"}
_UNKNOWN_LANG_TAGS = {"", "und", "unknown", "none"}


class SubtitleProbeError(RuntimeError):
    """ffprobe could not tell us what streams a file contains.

    Deliberately distinct from 'there are no subtitle streams' or 'there is no
    audio': the first is a preservation failure, the others are normal
    outcomes.
    """


class StreamInventory(NamedTuple):
    """What one ffprobe of a source tells every consumer that needs it.

    `subtitles` is the subtitle stream list Tab 3's mapping policy classifies
    against, in file order, carrying absolute indexes. `audio_count` is how
    many audio streams the explicit `-map 0:a?` policy will therefore encode,
    which is what target-size accounting has to reserve bitrate for. Both come
    out of the same probe: counting audio buys no second ffprobe.
    """
    subtitles: list[dict]
    audio_count: int


def ffprobe_stream_inventory(path: Path) -> StreamInventory:
    """Inventory a source's streams with the one probe a job is allowed.

    Each subtitle dict carries at least the *absolute* file stream index,
    which is what extraction and the mapping policy map against. Raises
    SubtitleProbeError when the answer cannot be trusted - never an empty
    subtitle list and never a zero audio count on failure, since both of those
    are legitimate answers a caller is entitled to act on.
    """
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_tags=language,title:"
        "stream_disposition=default,forced,hearing_impaired",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=SUBTITLE_PROBE_TIMEOUT,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise SubtitleProbeError(f"ffprobe could not run: {e}") from e

    if r.returncode != 0:
        detail = (r.stderr or "").strip().splitlines()
        raise SubtitleProbeError(
            detail[-1][:200] if detail else f"ffprobe exit {r.returncode}")

    try:
        data = json.loads(r.stdout or "")
    except (TypeError, ValueError) as e:
        raise SubtitleProbeError(f"unreadable ffprobe output: {e}") from e
    if not isinstance(data, dict):
        raise SubtitleProbeError("unexpected ffprobe output shape")

    streams = data.get("streams")
    if streams is None:
        return StreamInventory(subtitles=[], audio_count=0)
    if not isinstance(streams, list):
        raise SubtitleProbeError("unexpected ffprobe stream list")

    subtitles: list[dict] = []
    audio_count = 0
    for s in streams:
        if not isinstance(s, dict):
            raise SubtitleProbeError("unexpected ffprobe stream entry")
        idx = s.get("index")
        # bool is an int subclass; an index that is not a plain int means we
        # cannot map the stream safely, so refuse to guess. Applied to every
        # stream, not just subtitles: an inventory that cannot describe one
        # stream is not an inventory anything else should be trusted from.
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            raise SubtitleProbeError("stream without a usable index")
        kind = s.get("codec_type")
        if kind == "subtitle":
            subtitles.append(s)
        elif kind == "audio":
            # Counted by codec type, never by index range: absolute indexes
            # are free to be non-contiguous.
            audio_count += 1
    return StreamInventory(subtitles=subtitles, audio_count=audio_count)


def _stream_tag(stream: dict, name: str) -> str:
    tags = stream.get("tags")
    if not isinstance(tags, dict):
        return ""
    value = tags.get(name)
    return "" if value is None else str(value)


def _primary_language(stream: dict) -> str:
    """Lowercase primary subtag of the stream's language tag ('en-US' → 'en')."""
    raw = _stream_tag(stream, "language").strip().lower()
    return raw.replace("_", "-").split("-")[0].strip()


def _disposition(stream: dict, name: str) -> bool:
    d = stream.get("disposition")
    if not isinstance(d, dict):
        return False
    return bool(d.get(name))


def is_english_subtitle_stream(stream: dict) -> bool:
    """True only when the stream's own metadata says English.

    A recognised language tag is authoritative. The title is consulted only
    when the language tag is missing/empty/undetermined - being the default
    track is never evidence of anything.
    """
    lang = _primary_language(stream)
    if lang in _ENGLISH_PRIMARY_TAGS:
        return True
    if lang not in _UNKNOWN_LANG_TAGS:
        return False
    return "english" in _stream_tag(stream, "title").lower()


def subtitle_sidecar_target(codec_name: str | None) -> tuple[str, list[str], str]:
    """Map a subtitle codec to (sidecar extension, ffmpeg codec args, muxer).

    Preservation first: only codecs SRT can actually represent become .srt,
    styled text keeps .ass, and everything else - bitmap subtitles included -
    is copied bit-for-bit into a subtitle-only Matroska (.mks) rather than
    being discarded.
    """
    name = (codec_name or "").strip().lower()
    if name in SRT_SAFE_SUBTITLE_CODECS:
        return ".srt", ["-c:s", "srt"], "srt"
    if name in ASS_SUBTITLE_CODECS:
        return ".ass", ["-c:s", "ass"], "ass"
    return ".mks", ["-c:s", "copy"], "matroska"


def subtitle_sidecar_name(stem: str, stream: dict, ext: str) -> str:
    """`<stem>.eng[.forced][.sdh]<ext>` - markers come from disposition flags
    and title keywords only. Free-form title text never reaches the filesystem.
    """
    title = _stream_tag(stream, "title").lower()
    parts = [stem, "eng"]
    if _disposition(stream, "forced") or "forced" in title:
        parts.append("forced")
    if _disposition(stream, "hearing_impaired") or "sdh" in title:
        parts.append("sdh")
    return ".".join(parts) + ext


def build_subtitle_extract_cmd(input_path: Path, stream_index: int,
                               codec_name: str | None, out_path: Path) -> list:
    """One ffmpeg invocation for exactly one subtitle stream.

    `-map 0:<absolute index>` is the whole safety story: the index comes
    straight from ffprobe, so subtitle-relative numbering never enters the
    picture. `-vn -an` plus the explicit muxer keep the sidecar free of any
    video/audio and independent of filename inference.
    """
    _ext, codec_args, muxer = subtitle_sidecar_target(codec_name)
    return ([FFMPEG_BIN, "-nostdin", "-hide_banner", "-y",
             "-i", str(input_path),
             "-map", f"0:{int(stream_index)}", "-vn", "-an"]
            + codec_args + ["-f", muxer, str(out_path)])


def probe_stream_inventory_once(input_path: Path
                                ) -> tuple[StreamInventory | None, str | None]:
    """The one stream discovery a video job is allowed to make.

    Every consumer - sidecar extraction, the stream-mapping policy of each
    explicitly mapped container (MP4, Matroska), and the target-size audio
    budget - reads the same result, so enabling one never costs a second
    ffprobe. On failure the inventory is None (never an empty subtitle list,
    which would read as "this file has no subtitles", and never a zero audio
    count) alongside a description of what went wrong.
    """
    try:
        return ffprobe_stream_inventory(input_path), None
    except SubtitleProbeError as e:
        return None, str(e)


def _prepare_english_subtitles(input_path: Path, output_path: Path,
                               output_dir: Path,
                               cancel_flag: threading.Event,
                               streams: list[dict] | None,
                               probe_error: str | None
                               ) -> tuple[list[dict], list[str], Path | None]:
    """Extract every English subtitle stream into throwaway temp files.

    `streams` / `probe_error` come from `probe_stream_inventory_once`; this
    function no longer probes, so the MP4 mapping policy can share the answer.

    Returns (pending, errors, temp_dir). `pending` entries are
    {"temp": Path, "dest": Path} pairs awaiting finalization; `temp_dir` is
    the caller's to remove once the job is over, whatever its outcome.
    """
    pending: list[dict] = []
    errors: list[str] = []

    if streams is None:
        # Explicitly not "there were no subtitles" - fail closed.
        return pending, [f"subtitle probe failed: {probe_error}"], None

    english = [s for s in streams if is_english_subtitle_stream(s)]
    if not english:
        return pending, errors, None

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=".cove_subs_", dir=str(output_dir)))
    except OSError as e:
        return pending, [f"could not stage subtitles: {e}"], None

    for i, stream in enumerate(english):
        if cancel_flag.is_set():
            errors.append("cancelled before subtitle extraction finished")
            break
        ext, _codec_args, _muxer = subtitle_sidecar_target(stream.get("codec_name"))
        temp = temp_dir / f"{i}{ext}"
        dest = output_dir / subtitle_sidecar_name(output_path.stem, stream, ext)
        index = stream["index"]
        rc, err = run_ffmpeg(
            build_subtitle_extract_cmd(
                input_path, index, stream.get("codec_name"), temp),
            cancel_flag)
        if rc == -2:
            errors.append("cancelled during subtitle extraction")
            break
        if rc != 0:
            errors.append(f"stream {index}: extraction failed ({_brief(err)})")
            continue
        # A zero exit code is not preservation. Only a real, non-empty file is.
        try:
            st = os.stat(temp)
        except OSError:
            errors.append(f"stream {index}: no subtitle file produced")
            continue
        if not stat_mod.S_ISREG(st.st_mode) or st.st_size <= 0:
            errors.append(f"stream {index}: empty subtitle output")
            continue
        pending.append({"temp": temp, "dest": dest})

    return pending, errors, temp_dir


# ── MP4 stream selection policy ──────────────────────────────────────────────
#
# MP4 is the one container Cove targets that cannot carry everything a source
# might hold. Left to its own devices ffmpeg picks streams implicitly, which
# turns a PGS/VobSub track or a font attachment in the input into a hard mux
# failure at the very end of a long encode. So MP4 gets explicit positive maps
# instead: first video, first audio if there is one, and only subtitle streams
# that can actually become mov_text. Anything else - bitmap subtitles, unknown
# codecs, attachments, data streams - is simply never named, which excludes it
# without needing negative `-map -0:t` style filters.
#
# Matroska maps explicitly too, but for the opposite reason: it can carry
# more than ffmpeg's automatic selection picks, and attachments are the part
# worth naming (see `build_matroska_stream_map_args`). WebM maps explicitly
# for both reasons at once - see `build_webm_stream_map_args`.

# Text subtitle codecs MP4 can carry once transcoded to mov_text. Deliberately
# the same vocabulary the sidecar path classifies against, so the two features
# can never disagree about what "text subtitle" means.
MP4_TEXT_SUBTITLE_CODECS = SRT_SAFE_SUBTITLE_CODECS | ASS_SUBTITLE_CODECS


def mp4_subtitle_codec_is_compatible(codec_name: str | None) -> bool:
    """True only for codecs Cove knows it can transcode to mov_text."""
    return (codec_name or "").strip().lower() in MP4_TEXT_SUBTITLE_CODECS


def _mappable_subtitle_indexes(subtitle_streams: list[dict],
                               vocabulary: set[str]) -> list[int]:
    """Absolute ffprobe indexes of the streams `vocabulary` admits.

    Absolute file indexes, never subtitle-relative ones: excluding a bitmap
    track must not renumber the text track that follows it. Anything without a
    usable plain-int index is dropped rather than guessed at. The index rule is
    shared by every container; only the vocabulary differs, so a container
    cannot accidentally invent its own idea of what a mappable index is.
    """
    out: list[int] = []
    for s in subtitle_streams:
        if not isinstance(s, dict):
            continue
        idx = s.get("index")
        # bool is an int subclass; only a plain non-negative int can be mapped.
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            continue
        if (s.get("codec_name") or "").strip().lower() in vocabulary:
            out.append(idx)
    return out


def mp4_mappable_subtitle_indexes(subtitle_streams: list[dict]) -> list[int]:
    """Absolute ffprobe indexes of the subtitle streams MP4 may carry."""
    return _mappable_subtitle_indexes(subtitle_streams,
                                      MP4_TEXT_SUBTITLE_CODECS)


def build_mp4_stream_map_args(subtitle_streams: list[dict] | None) -> list:
    """Explicit MP4 stream selection for the muxing pass.

    `subtitle_streams` is the probed stream list, or None when discovery
    failed - in which case subtitles are disabled outright rather than handed
    back to ffmpeg's automatic selection, which is exactly the behaviour this
    policy exists to replace.

    Audio is mapped by type (`0:a?`) rather than by first stream: an MP4 with
    an English, a Japanese and a commentary track keeps all three, in source
    order, and each is encoded with the one audio codec and bitrate the format
    already selected. The trailing `?` is what keeps a silent source
    converting - the map is optional, not mandatory.
    """
    args = ["-map", "0:v:0", "-map", "0:a?"]
    indexes = ([] if subtitle_streams is None
               else mp4_mappable_subtitle_indexes(subtitle_streams))
    for idx in indexes:
        args += ["-map", f"0:{idx}"]
    if indexes:
        args += ["-c:s", "mov_text"]
    else:
        # Nothing safe to carry - say so, so ffmpeg cannot pick something.
        args += ["-sn"]
    return args


def matroska_mappable_subtitle_indexes(subtitle_streams: list[dict]) -> list[int]:
    """Absolute indexes of the subtitle streams Matroska's encoder can reach.

    Matroska's default subtitle codec is text-based, so the usable set is the
    same text vocabulary MP4 classifies against - the difference between the
    two containers is what they do with it, not what counts as text.
    """
    return mp4_mappable_subtitle_indexes(subtitle_streams)


def build_matroska_stream_map_args(
        subtitle_streams: list[dict] | None) -> list:
    """Explicit Matroska stream selection for the muxing pass.

    Matroska can carry attachments - embedded fonts and the like - and losing
    them to a conversion is a fidelity bug. But ffmpeg turns its automatic
    stream selection off entirely as soon as any `-map` appears, so asking for
    attachments means naming everything else too. Each map below therefore has
    to reproduce a decision automatic selection used to make for us.

    The subtitle map is the subtle one. Automatic selection does not simply
    take the first subtitle stream: it takes the first one whose *type* the
    output's default subtitle encoder can produce, and quietly drops the rest.
    Matroska's default encoder is text-based, so a bitmap track (PGS, VobSub,
    DVB, XSUB) used to be skipped and the file converted anyway. Naming
    `0:s:0?` instead would force that bitmap track into a text encoder and fail
    the whole job, so every *text* stream is named by absolute index instead.
    Every one, not just the first: Matroska is a multi-track container, and a
    film with English, Japanese and Spanish subtitles has no reason to arrive
    with two of them missing. Ineligible streams are skipped *around* the
    eligible ones rather than ending the scan, source order is the inventory's
    order, and nothing usable at all still means `-sn`. `subtitle_streams` is
    None when discovery failed, which disables subtitles outright rather than
    handing the choice back to the implicit selection this policy exists to
    replace.

    Audio is mapped by type (`0:a?`), so every source audio stream survives in
    source order - a multi-language film keeps all of its tracks. Audio and
    attachments stay optional: a silent source and a source with no
    attachments must both still convert. That optionality is what makes
    attachments free - `0:t?` needs no classification, so it needs no probe of
    its own. Attachments are stream-copied because there is no such thing as
    re-encoding one. Only `t` (attachment) is named; `d` (data) stays unmapped,
    exactly as it is for MP4.
    """
    args = ["-map", "0:v:0", "-map", "0:a?"]
    indexes = ([] if subtitle_streams is None
               else matroska_mappable_subtitle_indexes(subtitle_streams))
    for idx in indexes:
        args += ["-map", f"0:{idx}"]
    if not indexes:
        args += ["-sn"]
    return args + ["-map", "0:t?", "-c:t", "copy"]


# Text subtitle codecs WebM can carry once transcoded to WebVTT - WebM's only
# subtitle format. Deliberately its own set rather than an alias of MP4's:
# every name here was transcoded into a real WebM by the installed ffmpeg
# (explicit absolute map, `-c:s webvtt`) and came back out as a `webvtt`
# stream with its text intact, and only names that survived that are listed.
# The MP4/Matroska vocabulary additionally carries `srt`, `vtt` and `ssa`,
# which are ffmpeg *encoder* aliases rather than codec names: ffprobe reports
# the streams they produce as `subrip`, `webvtt` and `ass`, so no source can
# present them here and they are left out rather than assumed.
WEBM_TEXT_SUBTITLE_CODECS = {"subrip", "ass", "webvtt", "mov_text", "text"}


def webm_subtitle_codec_is_compatible(codec_name: str | None) -> bool:
    """True only for codecs proven to transcode to WebVTT on this ffmpeg."""
    return (codec_name or "").strip().lower() in WEBM_TEXT_SUBTITLE_CODECS


def webm_mappable_subtitle_indexes(subtitle_streams: list[dict]) -> list[int]:
    """Absolute indexes of the subtitle streams WebM may carry."""
    return _mappable_subtitle_indexes(subtitle_streams,
                                      WEBM_TEXT_SUBTITLE_CODECS)


def build_webm_stream_map_args(subtitle_streams: list[dict] | None) -> list:
    """Explicit WebM stream selection for the muxing pass.

    WebM was the last container Cove produced that still left this to ffmpeg,
    and implicit selection was quietly lossy in two directions at once. It
    takes the first *audio* stream, so a three-language source arrived as a
    one-language file. It takes the first subtitle stream its default text
    encoder can reach and drops the rest, so a film with English, Japanese and
    Spanish subtitles arrived with two of them missing. Both losses were
    silent: the job reported success.

    So WebM names what it carries. `0:a?` keeps every audio stream in source
    order while staying optional, which is what lets a silent source convert.
    Every WebM-safe *text* subtitle stream is named by absolute index - every
    one, not just the first - with ineligible streams skipped around rather
    than ending the scan, so a bitmap track sitting between two text tracks
    costs neither of them. Naming `0:s:0?` instead would force a bitmap track
    into a text encoder and fail the whole job.

    `-c:s webvtt` appears only when something was actually mapped: WebVTT is
    WebM's only subtitle format, and naming an encoder for streams that do not
    exist is how a working encode turns into a mux error. Nothing eligible
    means `-sn`, never a silent hand-back to the implicit selection this
    policy exists to replace - which is also why `subtitle_streams` being None
    (discovery failed) disables subtitles rather than defaulting to anything.

    Attachments and data streams are simply never named. WebM has no
    attachment support to preserve, so unlike Matroska there is nothing here
    worth a `0:t?`; explicit positive mapping excludes both without needing
    negative filters.
    """
    args = ["-map", "0:v:0", "-map", "0:a?"]
    indexes = ([] if subtitle_streams is None
               else webm_mappable_subtitle_indexes(subtitle_streams))
    for idx in indexes:
        args += ["-map", f"0:{idx}"]
    if indexes:
        args += ["-c:s", "webvtt"]
    else:
        args += ["-sn"]
    return args


# The muxers that name their own streams. Membership is what decides three
# things at once - that the job owes itself a stream inventory, that the
# target-size budget must reserve one audio bitrate per mapped stream, and
# that pass 1 must name its video stream - so they cannot drift apart.
EXPLICITLY_MAPPED_MUXERS = ("mp4", "matroska", "webm")


def build_stream_map_args(muxer: str, subtitle_streams: list[dict] | None) -> list:
    """The one place a container's stream-selection policy is decided.

    Both the directly requested encode and the MP4 -> MKV fallback attempt come
    through here, so the fallback inherits Matroska's attachment handling by
    construction rather than by a second copy of the rule. Every container Cove
    targets now maps explicitly; anything else keeps ffmpeg's implicit
    selection, which is the honest answer for a muxer with no policy written
    for it.
    """
    if muxer == "mp4":
        return build_mp4_stream_map_args(subtitle_streams)
    if muxer == "matroska":
        return build_matroska_stream_map_args(subtitle_streams)
    if muxer == "webm":
        return build_webm_stream_map_args(subtitle_streams)
    return []


def build_pass1_stream_map_args() -> list:
    """Two-pass pass 1 analyses video and nothing else.

    It writes to the null muxer, so it is not a mux at all; giving it subtitle
    or attachment maps would only invent failure modes for a pass whose entire
    output is a statistics log.
    """
    return ["-map", "0:v:0"]


def build_pass1_map_args_for(muxer: str) -> list:
    """Pass-1 maps for a container that maps explicitly at all.

    A container that names its streams for the mux must name one for the
    analysis pass too, or ffmpeg would be free to analyse a different video
    stream than the one being encoded.
    """
    if muxer in EXPLICITLY_MAPPED_MUXERS:
        return build_pass1_stream_map_args()
    return []


def _brief(msg: str, limit: int = 120) -> str:
    """Last line of an ffmpeg tail, trimmed - logs stay readable."""
    lines = [ln for ln in (msg or "").strip().splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else "no detail"


def _finalize_subtitle_sidecars(result: dict, pending: list[dict],
                                errors: list[str]) -> dict:
    """Move successfully extracted temps to collision-safe sidecar paths.

    `reserve_output` claims each destination with O_CREAT|O_EXCL, so an
    existing sidecar - the user's or a previous run's - is never touched.
    Only paths that exist and are non-empty afterwards are reported.
    """
    finalized: list[Path] = []
    for item in pending:
        dest = item["dest"]
        claimed: Path | None = None
        try:
            claimed, _tmp = reserve_output(dest)
            os.replace(str(item["temp"]), str(claimed))
            st = os.stat(claimed)
            if not stat_mod.S_ISREG(st.st_mode) or st.st_size <= 0:
                raise OSError("finalized sidecar is empty")
            finalized.append(claimed)
        except OSError as e:
            errors.append(f"{dest.name}: could not finalize sidecar ({e})")
            if claimed is not None:
                try:
                    claimed.unlink()
                except OSError:
                    pass

    result["subtitles_extracted"] = finalized
    if errors:
        result["subtitles_failed"] = True
        result["subtitle_errors"] = errors
    return result


# ── Optional source deletion ─────────────────────────────────────────────────

def _same_filesystem_object(src_st, out_st, src: Path, out: Path) -> bool:
    """True when source and output are provably the same file. Any doubt is
    resolved as 'same' by the callers, which fail closed."""
    if os.path.normcase(os.path.abspath(str(src))) == \
       os.path.normcase(os.path.abspath(str(out))):
        return True
    if os.path.normcase(os.path.realpath(str(src))) == \
       os.path.normcase(os.path.realpath(str(out))):
        return True
    # Hard links and bind-mount aliases: same device + same inode. On Windows
    # st_ino is only meaningful when non-zero, so ignore a zero inode pair.
    if src_st.st_ino and src_st.st_ino == out_st.st_ino \
       and src_st.st_dev == out_st.st_dev:
        return True
    return False


def delete_source_if_eligible(result: dict, enabled: bool = False) -> dict:
    """Permanently delete the original input of a *successful* conversion.

    Opt-in only: the default is off, so any caller that forgets the flag keeps
    the source. Deletion is a plain unlink - no Trash, no Recycle Bin - and is
    only attempted when the finalized output is proven to be a non-empty
    regular file distinct from the source. Any filesystem error fails closed:
    the source is kept and the failure is reported on the result. A deletion
    failure never downgrades a successful conversion.

    Mutates and returns the same result dict.
    """
    if enabled is not True:
        return result
    if result.get("status") != "ok":
        return result
    if result.get("subtitles_failed"):
        return result

    out = result.get("output")
    src = result.get("file")
    if not out or not src:
        return result
    out = Path(out)
    src = Path(src)

    try:
        try:
            out_st = os.stat(out)
            src_st = os.stat(src)
        except FileNotFoundError:
            # Nothing to delete, or no output to justify deleting. Not an
            # error worth reporting - stay quiet.
            return result

        if not stat_mod.S_ISREG(out_st.st_mode) or out_st.st_size <= 0:
            return result
        if not stat_mod.S_ISREG(src_st.st_mode):
            return result
        if _same_filesystem_object(src_st, out_st, src, out):
            return result
    except OSError as e:
        result["source_deleted"] = False
        result["delete_error"] = f"could not verify source: {e}"
        return result

    try:
        os.unlink(src)
    except OSError as e:
        result["source_deleted"] = False
        result["delete_error"] = str(e)
        return result

    result["source_deleted"] = True
    return result


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

        # Strip all embedded metadata (EXIF, XMP, ICC profile, etc.) so the
        # README claim of EXIF removal is actually honoured.  exif_transpose
        # alone only rotates the image and does not remove metadata.
        img = img.copy()
        img.info.clear()

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


# The lowest video bitrate worth asking an encoder for. Reaching it by
# subtraction means the target cannot hold what is being put in it, which is a
# refusal (see `target_fits_audio_budget`), not a value to clamp to.
MIN_VIDEO_KBPS = 80


def usable_total_kbps(target_bytes: int, duration: float) -> float:
    """The whole file's bitrate budget, less the 3% container allowance."""
    usable = target_bytes * 0.97
    return (usable * 8) / duration / 1000.0


def total_audio_kbps(audio_kbps: int, audio_count: int) -> int:
    """The selected bitrate is per track, so N tracks cost N times it."""
    return audio_kbps * audio_count


def target_fits_audio_budget(target_bytes: int, duration: float,
                             audio_kbps: int, audio_count: int) -> bool:
    """Whether any video at all fits alongside the audio being preserved.

    Four 128 kbps tracks in a 500 kbps budget leave nothing for video. The
    predecessor's clamp would have handed the encoder 80 kbps anyway and
    missed the target by the whole difference; preserving every track means
    saying so up front instead.
    """
    return (usable_total_kbps(target_bytes, duration)
            - total_audio_kbps(audio_kbps, audio_count)) >= MIN_VIDEO_KBPS


def calc_video_bitrate_kbps(target_bytes: int, duration: float,
                            audio_kbps: int, audio_count: int = 1) -> int:
    """What is left for video once every mapped audio stream is paid for.

    `audio_count` defaults to 1, which is the single-stream arithmetic this
    has always done, so a one-audio source computes exactly what it used to.
    """
    return max(int(usable_total_kbps(target_bytes, duration)
                   - total_audio_kbps(audio_kbps, audio_count)),
               MIN_VIDEO_KBPS)


def run_ffmpeg(cmd: list, cancel_flag: threading.Event,
               duration: float | None = None,
               on_progress=None, on_start=None) -> tuple:
    """Run ffmpeg while reporting progress and enforcing a stall watchdog."""
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True,
                                env=clean_subprocess_env(),
                                **SUBPROCESS_FLAGS)
    except FileNotFoundError:
        return -1, f"{FFMPEG_BIN} not found on PATH"

    stderr_tail: deque = deque(maxlen=40)
    # The one line that said the disk was full, held outside the rolling
    # window. ffmpeg announces ENOSPC once and then emits as much teardown
    # chatter as it likes, so by the time it exits the window is often all
    # chatter and the decisive line is gone - which is how a full disk used to
    # arrive downstream looking like an ordinary mux failure. This is a single
    # latched line, not a transcript: retention stays O(the existing window).
    no_space_line: str | None = None
    # Which `Metadata:` block, if any, each retained line sat inside. ffmpeg
    # quotes the user's own tags back at it, and a title may legitimately read
    # "No space left on device" - so what a line *is* has to be decided while
    # the block that framed it is still in view. Same window as the tail it
    # annotates: one header reference per retained line, no transcript.
    metadata = _MetadataBlockTracker()
    stderr_provenance: deque = deque(maxlen=40)
    stderr_queue: queue.Queue = queue.Queue()
    assert proc.stderr is not None

    def _read_stderr():
        try:
            for stderr_line in proc.stderr:
                stderr_queue.put(stderr_line)
        finally:
            stderr_queue.put(None)

    def _stop_process():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    threading.Thread(target=_read_stderr, daemon=True).start()
    last_progress = time.monotonic()
    started = False
    eof_received = False
    # Set once ffmpeg reports the moov relocation; swaps the stall allowance
    # for the longer finalization one. Never a completion signal.
    finalizing = False

    while True:
        if cancel_flag.is_set():
            _stop_process()
            return -2, "cancelled"

        allowance = FINALIZE_STALL_TIMEOUT if finalizing else ENCODE_STALL_TIMEOUT
        if time.monotonic() - last_progress > allowance:
            _stop_process()
            phase = "finalizing" if finalizing else "encoding"
            return -3, f"no {phase} progress for {allowance}s (skipped)"

        try:
            line = stderr_queue.get(timeout=1.0)
        except queue.Empty:
            line = ""

        if line is None:
            eof_received = True
        elif line:
            stderr_tail.append(line.rstrip())
            echoed_metadata = metadata.feed(line)
            stderr_provenance.append(metadata.quoted_under)
            # Same predicate the failure classifier uses, asked here while the
            # line is still in hand - and while the block it belongs to is
            # still known, which a single line can never say for itself.
            # Latched once: later lines can only add chatter, never unsay a
            # disk that filled up.
            if (no_space_line is None and not echoed_metadata
                    and _is_no_space_failure(line)):
                no_space_line = line.rstrip()
            progress_time = parse_ffmpeg_time(line)
            if progress_time is not None:
                last_progress = time.monotonic()
                if not started:
                    started = True
                    if on_start:
                        on_start()
                if duration and duration > 0 and on_progress:
                    on_progress(min(progress_time / duration * 100, 100.0))
            elif is_moov_relocation_line(line):
                # The file is being rewritten, not finished. Give the watchdog
                # something recent to measure from and widen its allowance —
                # but report no progress and claim no success. Only ffmpeg's
                # real exit status ends this job.
                last_progress = time.monotonic()
                finalizing = True

        if eof_received and proc.poll() is not None:
            break

    tail = list(stderr_tail)[-5:]
    # The tail is cut by line count, not by structure, so it can open partway
    # through a metadata block - leaving quoted tag values with nothing left
    # in view to say they are quoted tag values. When that would change the
    # answer, the block's own header goes back in front of them: one real
    # ffmpeg line, restoring context the truncation removed rather than
    # inventing any. Read-only otherwise, so an ordinary tail is untouched.
    open_header = (list(stderr_provenance)[-5:] or [None])[0]
    if open_header is not None and _is_no_space_failure("\n".join(tail)):
        tail = [open_header] + tail
    # A failure that ffmpeg blamed on space keeps saying so. If the tail has
    # already scrolled past that line, it goes back in front of it - one extra
    # line, on the one failure that needs it, so the tail itself keeps its
    # size. Cancellation and stall returns are unaffected: they never reach
    # here. Neither does a zero exit, whose message nothing consults, so a
    # warning cannot turn a success into a disk-full report.
    if (proc.returncode != 0 and no_space_line is not None
            and not _is_no_space_failure("\n".join(tail))):
        tail = [no_space_line] + tail
    return proc.returncode, "\n".join(tail)


def build_video_encoder_args(
    encoder: str,
    vf: str | None,
    use_two_pass: bool,
    pass_num: int | None,
    video_kbps: int | None,
    crf: int | None,
    speed_preset: str,
    nvenc_preset: str,
    amf_quality: str = "balanced",
) -> list:
    """Build the video-encoder portion of an ffmpeg command for one pass.

    Kept module-level (rather than closed over `compress_video`'s locals) so
    the codec × rate-control matrix — including the NVENC and AMF hardware
    paths — is unit-testable without spawning ffmpeg.

    Rate control:
      • CPU bitrate target → ABR, optionally as a log-file two-pass.
      • CPU quality        → -crf (plus -b:v 0 for VP9's constant-quality mode).
      • NVENC bitrate      → VBR with a capped maxrate and NVENC's own single-
                             invocation `-multipass fullres` (never the log-file
                             two-pass, which NVENC doesn't use).
      • NVENC quality      → VBR constant-quality via -cq with -b:v 0.
      • AMF bitrate        → peak-constrained VBR (`vbr_peak`, AMF's spelling —
                             it has no bare `vbr`) with the same capped-maxrate
                             heuristic; AMF has no `-multipass`, so size
                             targeting is a single-pass approximation like
                             NVENC's.
      • AMF quality        → constant-quality via -rc cqp -qp on the 0-51 scale.
    """
    is_nvenc = encoder.endswith("_nvenc")
    is_amf = encoder.endswith("_amf")
    a = ["-c:v", encoder]

    if is_nvenc:
        # NVENC presets run p1 (fastest) … p7 (best quality); -tune hq biases
        # the encoder toward quality rather than low-latency streaming.
        a += ["-preset", nvenc_preset, "-tune", "hq"]
    elif is_amf:
        # AMF's -quality dial runs speed → balanced → quality (best). -usage
        # transcoding is the right mode for offline file compression (vs. the
        # low-latency / ultralowlatency modes meant for live streaming).
        a += ["-quality", amf_quality, "-usage", "transcoding"]
    elif encoder in ("libx264", "libx265"):
        a += ["-preset", speed_preset]
    elif encoder == "libvpx-vp9":
        # libvpx-vp9 is slow. row-mt + cpu-used 4 keeps quality reasonable
        # without taking forever.
        a += ["-row-mt", "1", "-cpu-used", "4"]

    if vf:
        a += ["-vf", vf]

    if use_two_pass:
        # CPU ABR two-pass. NVENC and AMF never reach this branch — they do
        # their own single-pass bitrate handling below.
        a += ["-b:v", f"{video_kbps}k"]
        if pass_num:
            a += ["-pass", str(pass_num)]
    elif video_kbps is not None:
        if is_nvenc:
            a += ["-rc", "vbr", "-b:v", f"{video_kbps}k",
                  "-maxrate", f"{int(video_kbps * 1.4)}k",
                  "-bufsize", f"{int(video_kbps * 2)}k",
                  "-multipass", "fullres"]
        elif is_amf:
            # AMF's rate-control vocabulary has no bare `vbr` - the peak-
            # constrained mode is spelled `vbr_peak`, and anything else fails
            # at option parsing before a frame is encoded. NVENC's own `vbr`
            # above is a different encoder's different option; the two must
            # not be unified.
            # AMF has no equivalent of NVENC's -multipass fullres; sizes land
            # close to target but aren't as exact as the software two-pass.
            a += ["-rc", "vbr_peak", "-b:v", f"{video_kbps}k",
                  "-maxrate", f"{int(video_kbps * 1.4)}k",
                  "-bufsize", f"{int(video_kbps * 2)}k"]
        else:
            a += ["-b:v", f"{video_kbps}k"]
    else:
        if is_nvenc:
            a += ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        elif is_amf:
            # AMF constant-quality uses cqp + -qp on the same 0-51 scale.
            a += ["-rc", "cqp", "-qp", str(crf)]
        else:
            a += ["-crf", str(crf)]
            if encoder == "libvpx-vp9":
                a += ["-b:v", "0"]

    if encoder == "libx265":
        a += ["-x265-params", "log-level=error"]
    return a


def _discard_reserved_output(output_path: Path) -> None:
    """Remove a reservation the job never filled.

    Only the empty placeholder `reserve_output` created is ever removed - a
    non-empty file at that path is somebody's real output (or a finished
    encode) and is left strictly alone.

    Windows routinely holds a brief handle on a file that was just created -
    search indexers and AV scanners are the usual culprits - so the first
    unlink can fail on a stub that becomes deletable milliseconds later. The
    emptiness check is therefore re-taken before *every* attempt: during a
    retry delay some other writer could have filled this path in, and deleting
    that would destroy real data. stat-then-unlink is still not atomic, but
    re-checking closes the whole retry window rather than leaving it open.

    Giving up quietly is deliberate: this runs in a `finally` after ffmpeg has
    already exited, and raising would replace the job's real error/timeout/
    cancelled result with an unrelated OSError.
    """
    for attempt in range(RESERVED_CLEANUP_ATTEMPTS):
        try:
            st = os.stat(output_path)
        except OSError:
            return
        if not stat_mod.S_ISREG(st.st_mode) or st.st_size != 0:
            # Gone, replaced, or filled in by somebody else - not our stub.
            return
        try:
            os.unlink(output_path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == RESERVED_CLEANUP_ATTEMPTS - 1:
                return
            time.sleep(RESERVED_CLEANUP_RETRY_DELAY)


def _final_output_is_present(path: Path) -> bool:
    """Whether an apparently successful encode actually left a usable file.

    The counterpart to `_discard_reserved_output`: that one reclaims a
    reservation nothing filled, this one asks whether anything filled it. A
    zero exit code says ffmpeg believed it finished, which is not the same as
    a file existing - and because Cove reserves the destination up front, the
    path exists either way. `exists()` therefore proves nothing on its own;
    only a non-empty regular file distinguishes a finished encode from the
    placeholder Cove created before ffmpeg ever ran.

    Deliberately structural and nothing more. No ffprobe, no codec/duration/
    stream check, no container parsing, and no size floor above zero: a short
    low-bitrate encode is legitimately tiny, and any threshold would throw it
    away. An unreadable path is unverified, and unverified fails closed.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    return stat_mod.S_ISREG(st.st_mode) and st.st_size > 0


# A validation step that can outlive the encode it validates is a worse bug
# than the one it fixes. The same bound the source probes already use: long
# enough that no honest file misses it, short enough to stay a step.
FINAL_READABILITY_TIMEOUT = 30

# The one status that means "ffprobe read this file and refuses it".
#
# ffprobe funnels every internal failure into a single `ret < 0` at the end of
# main, so it has exactly one way of saying no, and it is this. Measured
# against the real binary on garbage bytes, an empty file, ASCII text, 4 KB of
# zeros, a truncated MP4, a missing path and a directory: every one exits 1,
# and only a genuinely readable file exits 0.
#
# Matching that single value rather than a range is what keeps a *crash* from
# being read as a verdict. Process death arrives as a number too, and on
# Windows it can be a small one - the CRT terminates an `abort()`ed process
# with status 3, and ffmpeg's `av_assert0` aborts - so any rule of the form
# "nonzero" or "1..255" would delete an artifact that ffprobe never actually
# judged. Anything that is neither 0 nor 1 is the probe failing, not the file
# failing, and is treated like a launch failure or a timeout: preserve.
_FFPROBE_REFUSAL_STATUS = 1

# What turns the readability probe into a *playable video* readability probe,
# without turning it into a second process.
#
# A stream index proves a video stream exists, and that is not the question. A
# container may carry picture that was never meant to be played: cover art, a
# poster frame, a thumbnail, flagged `attached_pic` precisely so players show it
# instead of decoding it. An audio track with artwork glued to it prints an
# index exactly as a film does, and asking for the index cannot tell them apart.
# `attached_pic` can, ffprobe already knows it, and asking for it costs the same
# single invocation - so the probe asks for it instead.
#
# `-select_streams v` narrows the report to video streams,
# `stream_disposition=attached_pic` asks each of them the one question that
# separates a picture from a thumbnail, and `csv=p=1` prints one `stream,<0|1>`
# line per video stream. The `stream` prefix is what `p=1` buys and it is load-
# bearing: it proves a stream was reported at all, so empty stdout unambiguously
# means "no video streams" rather than "a stream that printed nothing", and a
# stream whose disposition ever went missing would print a bare `stream,` that
# is visibly not the contract. Measured against the real binary on ffmpeg-built
# media: ordinary MP4/MKV/WebM video prints `stream,0`, an MP4 carrying
# `-disposition attached_pic` artwork and an MKV carrying an `-attach`ed cover
# both print `stream,1`, a file holding both prints one of each, and audio-only
# containers print nothing at all.
#
# Deliberately `v` and not `v:0`: artwork routinely sorts ahead of the real
# picture, so the playable stream is not guaranteed to be the first one and
# every video stream has to be asked. It is still a presence rule and not a
# parity rule - the lines are weighed for containing a `0`, never counted.
_FFPROBE_VIDEO_STREAM_ARGS = ("-select_streams", "v",
                              "-show_entries",
                              "stream_disposition=attached_pic",
                              "-of", "csv=p=1")

# The two whole lines the probe above is defined to print. `attached_pic` takes
# 0 - picture the file exists to show in motion - or 1 - picture attached to
# something else; `csv=p=1` puts the section name in front of it, and both
# halves are checked.
#
# The section name is not decoration. Matching only the value after the last
# comma would let any line that happens to end in `,0` vouch for a playable
# stream that was never reported, and - the dangerous direction - let any line
# ending in `,1` be counted as artwork, completing an "every picture here is a
# thumbnail" verdict that *removes the file*. A payload this code cannot parse
# must never do either: it is the validator failing, not the artifact, and it
# is classed inconclusive below. Anything that is not exactly one of these two
# strings is that.
_PLAYABLE_VIDEO_RECORD = "stream,0"
_ATTACHED_PICTURE_RECORD = "stream,1"


class _Readability:
    """The verdict of one final probe of a finished artifact.

    Four outcomes, not two, because "not readable" collapses facts that are not
    the same fact. ffprobe running to completion and exiting nonzero is evidence
    about the *bytes*: they are not media. ffprobe parsing the file and finding
    no video stream in it - or none that is anything but attached artwork - is
    evidence about the *content*: they are media, and they are not a video that
    can be watched. ffprobe failing to launch, timing out, or a
    cancellation landing mid-check is evidence about the *probe*: the artifact
    may be perfectly good video that Cove merely failed to verify. All three
    fail the conversion closed and all three keep the source; only the first two
    are verdicts about the file, and only they may be acted on destructively.

    Falsy unless the file is media *and* carries picture, so every `if not ...`
    caller - and the suites that stub this seam with a plain `True` - keep
    working unchanged. The extra distinctions are read in exactly one place,
    where cleanup and the user's error message are decided.
    """

    __slots__ = ("readable", "rejected", "no_video")

    def __init__(self, readable: bool, rejected: bool, no_video: bool = False):
        self.readable = readable
        self.rejected = rejected      # ffprobe finished and said no
        self.no_video = no_video      # ...and said so by finding no picture

    def __bool__(self) -> bool:
        return self.readable


_READABLE = _Readability(readable=True, rejected=False)
_REJECTED = _Readability(readable=False, rejected=True)
# Media, and not a watchable video: no video stream at all, or none carrying
# anything but attached artwork. A verdict about the artifact exactly as
# `_REJECTED` is, so it carries the same destructive licence - it is only told
# apart to say something true to the user, because "could not be verified as
# readable media" would send them debugging bytes that are perfectly fine.
_NO_VIDEO = _Readability(readable=False, rejected=True, no_video=True)
_INCONCLUSIVE = _Readability(readable=False, rejected=False)


def _playable_video_verdict(stdout: str) -> _Readability:
    """Read one probe's video-stream report as a verdict about playable picture.

    Three answers, and the difference between the last two decides whether a
    file is deleted. A printed `0` is positive proof that a stream exists which
    is not a thumbnail, and that is the whole success condition: one is enough,
    five are equally fine, and where in the list it sits does not matter. Proof
    once found is not retracted by anything printed after it, so noise beside a
    `0` still passes - only the *absence* of proof has to be classified.

    Absent, it can be absent two ways. Every line being `1`, or there being no
    lines at all, is ffprobe answering completely: this file's only pictures are
    attachments, or it has no picture at all. That is a verdict about the
    artifact and carries the licence to remove it. A line that is neither value
    is ffprobe not answering in the vocabulary asked for - a changed output
    format, a missing disposition, a truncated write - and that is the validator
    failing rather than the file, indistinguishable in consequence from a
    timeout. It fails the conversion closed like every other unverified result,
    and it keeps the bytes, because deleting media no probe ever judged is the
    larger of the two mistakes.

    Each non-blank line is matched whole against the two forms the contract can
    take. Nothing is extracted from a line that is not one of them, because a
    line the parser cannot read has to stay outside both counts - a stray `,0`
    must not vouch for picture that was never reported, and a stray `,1` must
    not help complete the verdict that deletes the file.
    """
    playable = False
    unrecognized = False
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line == _PLAYABLE_VIDEO_RECORD:
            playable = True
        elif line != _ATTACHED_PICTURE_RECORD:
            unrecognized = True
    if playable:
        return _READABLE
    return _INCONCLUSIVE if unrecognized else _NO_VIDEO


def _final_output_is_readable(path: Path,
                              cancel_flag: threading.Event | None = None
                              ) -> _Readability:
    """Whether ffprobe can open the finished file as media *containing video*.

    `_final_output_is_present` proves a file arrived; this proves a video is in
    it. They are different questions and the second one cannot be answered from
    metadata: a successful ffmpeg exit that leaves a kilobyte of noise at the
    destination is structurally indistinguishable from a short valid encode,
    and Cove would report it as a conversion, finalize its sidecars, and offer
    to delete the original that was the only readable copy.

    Being media is not enough, because an exit code answers "are these bytes
    parseable?" and for a *video compressor* that question comes apart from the
    one that matters at exactly one place. An audio-only container is media:
    ffprobe opens it, parses it, and exits 0. It is also not the thing the user
    asked for, and every success-only consequence would fire behind that 0 -
    including the offer to delete the only copy of the picture. So the same
    invocation is asked for one more fact, `_FFPROBE_VIDEO_STREAM_ARGS`, and
    both are read off it: the exit code says whether this is media, and stdout
    says whether any of it is video the user can watch.

    Watchable, not merely present, because containers carry picture that was
    never meant to be played. An audio file with cover art holds a real
    codec_type=video stream - flagged `attached_pic`, sized like a thumbnail,
    and shown rather than decoded - so "has a video stream" accepts it and
    hands the user an album track named like a film. `attached_pic` is the one
    field that tells the two apart, and `_playable_video_verdict` weighs the
    report for at least one stream that is not artwork.

    That pair is the whole policy. Duration, video/audio/subtitle counts,
    codecs, resolution, container names and every other disposition flag are
    separate policies, each with its own false rejections - a legitimately odd
    but playable file must not be thrown away here - so the report is weighed
    only for a non-attached picture existing. One is enough and five are
    equally fine, and artwork alongside real picture is an ordinary good file.

    Fails closed on every way the check itself can fail: a missing or
    unlaunchable ffprobe, a timeout, or a cancellation landing either side of
    the run. An unverified output is not a verified one - but it is also not a
    rejected one, which is why those paths return `_INCONCLUSIVE` rather than
    `_REJECTED`. Only a probe that actually ran to completion has said
    anything about the file itself.
    """
    if cancel_flag is not None and cancel_flag.is_set():
        return _INCONCLUSIVE
    try:
        r = subprocess.run(
            [FFPROBE_BIN, "-v", "error", *_FFPROBE_VIDEO_STREAM_ARGS,
             str(path)],
            capture_output=True, text=True,
            timeout=FINAL_READABILITY_TIMEOUT,
            env=clean_subprocess_env(), **SUBPROCESS_FLAGS,
        )
    except (OSError, subprocess.SubprocessError):
        return _INCONCLUSIVE
    if cancel_flag is not None and cancel_flag.is_set():
        return _INCONCLUSIVE
    if r.returncode != 0:
        # ffprobe has exactly one way of refusing a file, and any other status
        # is the probe dying rather than answering - a POSIX signal death, or a
        # Windows crash, which can be a large number (0xC0000005 arrives as
        # 3221225477) or a small one (an aborted CRT process exits 3). A death
        # says exactly as much about the bytes as a launch failure or a timeout
        # does, so it is classed with them and the artifact is kept. Whatever a
        # failing probe managed to print is not evidence of anything and is not
        # read: stdout only means something once the exit code says this is
        # media at all.
        return (_REJECTED if r.returncode == _FFPROBE_REFUSAL_STATUS
                else _INCONCLUSIVE)
    # Media. Now: is any of it picture the user can watch? The per-stream
    # dispositions are the evidence, and the only evidence.
    return _playable_video_verdict(r.stdout)


def _discard_rejected_output(output_path: Path) -> None:
    """Remove a final artifact ffprobe ran on and refused.

    The narrow counterpart to `_discard_reserved_output`. That one reclaims
    only the zero-byte placeholder and leaves anything with bytes in it
    strictly alone, precisely because it cannot prove whose bytes those are.
    Here that proof exists, and only here: `reserve_output` claimed this exact
    path with O_CREAT|O_EXCL so no pre-existing file was ever at it, the
    encode filled that same claimed path from this job's own temp via
    `os.replace`, and ownership has not passed to the user - readability
    failed before the ok result, before any sidecar was finalized, and before
    the source became eligible for deletion. The file is Cove's, it is not
    media, and leaving it behind hands the user a broken video named like a
    good one.

    Deliberately one exact path and one ordinary unlink: no glob over
    stem-alikes, no recursion, no overwrite pass, no retry loop. The
    regular-file check is belt-and-braces - Tab 10 rejects a directory or a
    non-regular path long before a probe runs - and exists so that a future
    bug pointing this at a directory does nothing at all.

    The conversion has already failed on readability, and that is the answer
    the user needs; a cleanup that cannot run must not replace it. So an
    OSError is swallowed exactly as `_discard_reserved_output` swallows its
    own, and the artifact simply stays.
    """
    try:
        st = os.stat(output_path)
    except OSError:
        return
    if not stat_mod.S_ISREG(st.st_mode):
        return
    try:
        os.unlink(output_path)
    except OSError:
        pass


# ── MP4 → MKV container fallback ─────────────────────────────────────────────
#
# MP4 is the one container Cove targets that can refuse a stream combination
# outright, and it does so at the very end of the job. Tab 2a removed the
# predictable causes with explicit stream maps, but a final mux/trailer failure
# is still possible - and for the user it is a dead end: a long encode that
# produced nothing. Matroska carries everything MP4 can, so one retry into MKV
# turns that dead end into a usable file.
#
# The retry is deliberately minimal: one requested format, one failure class,
# one extra attempt, and no state that outlives the `compress_video` call. The
# next file starts as MP4 again because nothing here is remembered anywhere.

MKV_FALLBACK_SOURCE_FORMAT = "MP4 (H.265)"
MKV_FALLBACK_TARGET_FORMAT = "MKV (H.265)"

# Canonical no-space signals, deliberately two exact phrases rather than an
# error taxonomy. A full disk is not a container problem: retrying into another
# container on the same filesystem cannot succeed and only enlarges the mess.
NO_SPACE_SIGNALS = ("no space left on device", "enospc")

# ffmpeg opens its report by announcing every file the job names:
#
#     Input #0, matroska,webm, from 'C:\Videos\Movie.mkv':
#     Output #0, mp4, to 'C:\Output\Movie.mp4':
#
# Those declare what the job *is*; they say nothing about what went wrong, and
# the only part of them the user writes is the path. A file may legitimately be
# called "No space left on device.mp4", and reading that name as a diagnostic
# turned an unrelated mux failure into a full disk - which then cost that
# failure the single MKV retry it had earned. Matched to the end of the line so
# the path itself, quotes and all, is what gets excluded; ordinary prose that
# merely opens with the same word ("Output failed: ENOSPC") is untouched.
_FFMPEG_PATH_ANNOUNCEMENT = re.compile(
    r"(?:Input #\d+, [^']*, from |Output #\d+, [^']*, to )'.*':$")

# The other half of what ffmpeg quotes back before it says anything of its own:
# the container's tags. Captured verbatim from the 8.1.1 build this project
# ships against, over media deliberately tagged with the recognized phrases:
#
#     Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'signal.mp4':
#       Metadata:
#         compatible_brands: isomiso2avc1mp41
#         title           : No space left on device
#         comment         : ENOSPC
#       Duration: 00:00:02.00, start: 0.000000, bitrate: 122 kb/s
#       Stream #0:0[0x1](und): Video: h264 ...
#         Metadata:
#           handler_name    : VideoHandler
#
# A block opens on an indented bare `Metadata:` (2 spaces for a container, 4
# for a stream, 6 for a chapter) and its values sit one level deeper. Nothing
# else in the report is shaped that way, and ffmpeg's own complaints are
# unindented - so the block closes on the first line back at or left of its
# header, which is every real terminator: `Duration:`, `Stream #`, `Chapters:`,
# `Side data:`, the next `Input #`/`Output #`, `Conversion failed!`.
#
# What makes a value line is the padded key column, not the punctuation in the
# key. Two real shapes the obvious pattern misses, both reachable from a file a
# user can hand this program:
#
#     WEIRD:KEY       : ENOSPC          a tag name may contain a colon
#     COMMENT         : first line      a tag value may contain newlines, and
#                     : ENOSPC          its continuations have a blank key
#
# The value pattern is required as well as the indentation. Excluding too
# little only restores the predecessor's behaviour; excluding too much would
# hide a real full disk, which is the failure this whole path exists to report.
_FFMPEG_METADATA_HEADER = re.compile(r"^(\s{2,})Metadata:$")
_FFMPEG_METADATA_VALUE = re.compile(r"^(\s+)\S* *:(?: |$)")

# The one prefix Cove itself adds before a message reaches the classifier.
_FFMPEG_FAILURE_PREFIX = "ffmpeg failed: "


class _MetadataBlockTracker:
    """Line-by-line provenance for ffmpeg's `Metadata:` blocks.

    State is one indent and one header line - bounded, and created per
    subprocess, so nothing survives an attempt, a fallback or a file.
    """

    __slots__ = ("_indent", "_header", "_opened")

    def __init__(self) -> None:
        self._indent: int | None = None
        self._header: str | None = None
        self._opened = False

    @property
    def quoted_under(self) -> str | None:
        """The header framing the line just fed, if it sat inside a block.

        Asked of every retained line rather than only of recognized values, so
        that restoring a truncated tail depends on the block being open and not
        on the shape of whichever line the tail happens to begin with.
        """
        return None if self._opened else self._header

    def feed(self, line: str) -> bool:
        """Consume one stderr line; True if it is an echoed metadata value."""
        stripped = line.rstrip()
        self._opened = False
        header = _FFMPEG_METADATA_HEADER.match(stripped)
        if header is not None:
            self._indent = len(header.group(1))
            self._header = stripped
            self._opened = True
            return False
        if self._indent is None:
            return False
        value = _FFMPEG_METADATA_VALUE.match(stripped)
        if value is not None and len(value.group(1)) > self._indent:
            return True
        if len(stripped) - len(stripped.lstrip()) <= self._indent:
            self._indent = None
            self._header = None
        return False


def _is_no_space_failure(msg: str | None) -> bool:
    """Whether ffmpeg blamed this failure on space, judged line by line.

    Each line is classified with the path it announced removed, rather than
    skipped outright: a line is only ever partly an announcement, so anything
    ffmpeg said around one still counts. Lines ffmpeg was quoting rather than
    complaining - the values inside a metadata block - are not diagnostics at
    all and are passed over, which is provenance, not a rule about colons or
    indentation: the same shapes outside a block still classify. Matching is
    otherwise the predecessor's: lowercase substring, anywhere, over exactly
    the committed vocabulary.
    """
    text = msg or ""
    # Stripped so a re-inserted block header still reads as one after
    # `compress_video` has labelled the message.
    if text.startswith(_FFMPEG_FAILURE_PREFIX):
        text = text[len(_FFMPEG_FAILURE_PREFIX):]
    metadata = _MetadataBlockTracker()
    for line in text.splitlines():
        if metadata.feed(line):
            continue
        lowered = _FFMPEG_PATH_ANNOUNCEMENT.sub("", line.rstrip()).lower()
        if any(signal in lowered for signal in NO_SPACE_SIGNALS):
            return True
    return False


def _mkv_fallback_eligible(video_format_key: str, result: dict,
                           cancel_flag: threading.Event) -> bool:
    """Whether this one failed attempt has earned its single MKV retry.

    Everything else is terminal. `mux_failed` is what narrows this to an
    ordinary positive ffmpeg exit from the final encode/mux invocation, so
    cancellation, stalls, a missing ffmpeg, pass-1 failures, pre-encode
    validation, reservation failures and skips can never reach here. The
    cancel flag is re-read because a terminated ffmpeg often exits positive:
    the user's intent outranks the exit code.
    """
    return (
        video_format_key == MKV_FALLBACK_SOURCE_FORMAT
        and not cancel_flag.is_set()
        and result.get("status") == "error"
        and bool(result.get("mux_failed"))
        and not _is_no_space_failure(result.get("msg"))
    )


def _retarget_subtitle_sidecars(pending: list[dict], old_stem: str,
                                new_stem: str) -> None:
    """Re-aim not-yet-finalized sidecars at the container that actually lands.

    Extraction happened once, before the first attempt, against the MP4
    destination's stem. A fallback usually keeps that stem (`Movie.mp4` →
    `Movie.mkv`) but a collision can bump it, and a sidecar only works when it
    sits beside the video it names. This renames destinations only - no
    re-probe, no re-extraction, no second subtitle lifecycle.
    """
    if old_stem == new_stem:
        return
    for item in pending:
        dest = Path(item["dest"])
        if dest.name.startswith(old_stem):
            item["dest"] = dest.with_name(new_stem + dest.name[len(old_stem):])


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
    encoder_pref: str = "auto",
    on_start=None,
    extract_english_subtitles: bool = False,
) -> dict:
    """Compress one video, optionally rescuing its English subtitle streams.

    `extract_english_subtitles` defaults to False so every existing caller
    keeps the original behavior byte for byte. When enabled, English streams
    are pulled to sidecars *before* the encode begins (the encode is what
    destroys access to them) and only finalized once the video itself lands.
    """
    # ffprobe is a dependency of every video conversion, not of the two modes
    # that happen to read a duration with it. The finished file is validated
    # with ffprobe before any success is reported, so a machine that cannot
    # run it cannot produce a successful conversion in any mode - and the
    # honest place to say so is before the encode, not after it. The
    # application blocks this at its start boundary; this is the same refusal
    # for anything that calls the core directly.
    if not shutil.which(FFPROBE_BIN):
        return {"file": input_path, "status": "error",
                "msg": "ffprobe is required for video compression."}

    fmt = VIDEO_FORMATS[video_format_key]
    encoder    = fmt["codec"]
    codec_key  = fmt["codec_key"]
    audio_enc  = fmt["audio"]
    out_ext    = fmt["ext"]
    muxer      = fmt["muxer"]
    container_flags = fmt["container_flags"]
    two_pass_ok     = fmt["supports_two_pass"]

    # Prefer a hardware encoder when the user allows it, the format has a
    # hardware equivalent, and the encoder actually initializes on this
    # machine. NVENC wins when both vendors are reachable and the user is on
    # Automatic; AMF is the fallback. Forcing a vendor on an unsupported
    # format (e.g. WebM/VP9) or a machine without that GPU quietly falls back
    # to CPU rather than failing the whole job.
    nvenc_codec = fmt.get("nvenc_codec")
    amf_codec = fmt.get("amf_codec")
    use_nvenc = bool(
        nvenc_codec
        and encoder_pref in ("auto", "nvenc")
        and nvenc_available(nvenc_codec)
    )
    use_amf = bool(
        amf_codec
        and encoder_pref in ("auto", "amf")
        and amf_available(amf_codec)
        and not use_nvenc
    )
    if use_nvenc:
        encoder    = nvenc_codec
        quality_key = fmt["nvenc_key"]
    elif use_amf:
        encoder    = amf_codec
        quality_key = fmt["amf_key"]
    else:
        quality_key = codec_key

    original_size = input_path.stat().st_size
    duration = ffprobe_duration(input_path)

    use_two_pass = False
    video_kbps = None
    crf = None
    speed_preset = "medium"
    nvenc_preset = "p6"
    amf_quality = "balanced"

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

        # The video budget itself is settled below, once the inventory says
        # how many audio streams this file will be paying for.
        # NVENC and AMF each do their own single-invocation rate control, so
        # neither uses the log-file two-pass ABR path the software encoders do.
        use_two_pass = two_pass_ok and not use_nvenc and not use_amf
    else:
        p = VIDEO_QUALITY_PRESETS[str(mode_value)]
        crf = p[quality_key]
        speed_preset = p["speed"]
        nvenc_preset = p["nvenc_preset"]
        amf_quality = p["amf_quality"]

    # One subtitle discovery per file, shared by every consumer: the sidecar
    # rescue feature and the explicit stream-mapping policy of any container
    # that has one. No feature pays for another, and enabling them all still
    # costs exactly one ffprobe.
    # Every explicitly mapped container needs it: MP4 to know what it can
    # transcode to mov_text, Matroska to know which subtitle stream its
    # text-based encoder can actually reach, WebM to know which it can
    # transcode to WebVTT. Attachments add nothing here - `-map 0:t?`
    # classifies nothing - so enabling them costs no probe.
    # The same probe also counts the audio streams the explicit `-map 0:a?`
    # policy will encode, which is what the target-size budget has to reserve
    # bitrate for; one inventory answers both questions.
    needs_stream_probe = (extract_english_subtitles
                          or muxer in EXPLICITLY_MAPPED_MUXERS)
    inventory: StreamInventory | None = None
    probe_error: str | None = None
    if needs_stream_probe:
        inventory, probe_error = probe_stream_inventory_once(input_path)
    sub_streams = None if inventory is None else inventory.subtitles

    if mode in ("Target file size", "Target reduction"):
        if muxer in EXPLICITLY_MAPPED_MUXERS:
            # These containers map every audio stream, so the budget depends
            # on how many there are. An untrustworthy count would invalidate
            # the whole target, and no guess is safer than any other: fail
            # closed before anything is reserved or encoded.
            if inventory is None:
                return {"file": input_path, "status": "error",
                        "msg": f"Could not read audio streams "
                               f"(ffprobe failed: {probe_error})"}
            audio_count = inventory.audio_count
        else:
            # A muxer with no explicit policy keeps ffmpeg's implicit
            # selection, and with it the single-stream budget that matches it.
            audio_count = 1

        if not target_fits_audio_budget(target_bytes, duration,
                                        int(audio_kbps), audio_count):
            return {"file": input_path, "status": "error",
                    "msg": f"target too small for {audio_count} audio "
                           f"track{'' if audio_count == 1 else 's'} at "
                           f"{int(audio_kbps)} kbps each"}

        video_kbps = calc_video_bitrate_kbps(
            target_bytes, duration, int(audio_kbps), audio_count)

    # Containers that cannot (MP4) or should not (Matroska, which would
    # otherwise drop attachments; WebM, which would otherwise drop every
    # audio and subtitle track but the first) leave selection to ffmpeg get
    # explicit positive maps.
    stream_map_args = build_stream_map_args(muxer, sub_streams)
    pass1_map_args = build_pass1_map_args_for(muxer)

    # Claim the destination atomically now that the job is certain to encode.
    # `reserve_output` creates the file with O_CREAT|O_EXCL, so a concurrent
    # job - or a pre-existing file - can never end up sharing this name. The
    # placeholder is zero bytes and belongs to Cove until the encode replaces
    # it; every non-success exit below removes it again.
    output_path, tmp_path = reserve_output(
        output_dir / f"{input_path.stem}{out_ext}")

    # Subtitle rescue runs here: after the deterministic pre-encode skips
    # above (no point extracting for a job that never encodes) and before the
    # first real encode invocation below.
    sub_pending: list[dict] = []
    sub_errors: list[str] = []
    sub_temp_dir: Path | None = None
    result: dict | None = None

    def _attempt(out_path: Path, tmp: Path, attempt_muxer: str,
                 attempt_flags: list, attempt_audio: str,
                 attempt_maps: list, attempt_pass1_maps: list) -> dict:
        """One encode of this file into one container. Never more than one."""
        return _encode_video(
            input_path=input_path, output_path=out_path, tmp_path=tmp,
            original_size=original_size, duration=duration, mode=mode,
            encoder=encoder, audio_enc=attempt_audio,
            audio_kbps=audio_kbps,
            muxer=attempt_muxer, container_flags=attempt_flags,
            stream_map_args=attempt_maps,
            pass1_map_args=attempt_pass1_maps,
            resolution_cap=resolution_cap, use_two_pass=use_two_pass,
            video_kbps=video_kbps, crf=crf, speed_preset=speed_preset,
            nvenc_preset=nvenc_preset, amf_quality=amf_quality,
            use_hw=(use_nvenc or use_amf), cancel_flag=cancel_flag,
            progress_cb=progress_cb, on_start=on_start,
            extract_english_subtitles=extract_english_subtitles,
            sub_pending=sub_pending, sub_errors=sub_errors,
        )

    try:
        try:
            if extract_english_subtitles:
                sub_pending, sub_errors, sub_temp_dir = \
                    _prepare_english_subtitles(
                        input_path, output_path, output_dir, cancel_flag,
                        sub_streams, probe_error)
                if cancel_flag.is_set():
                    result = {"file": input_path, "status": "error",
                              "msg": "cancelled"}
            if result is None:
                result = _attempt(output_path, tmp_path, muxer,
                                  container_flags, audio_enc,
                                  stream_map_args, pass1_map_args)

                # The single MP4 → MKV retry. It lives here, inside the
                # subtitle temp lifecycle, so the second attempt reuses the
                # subtitles the first one prepared; `_encode_video` itself
                # stays a one-attempt primitive that knows nothing about it.
                if _mkv_fallback_eligible(video_format_key, result,
                                          cancel_flag):
                    fb = VIDEO_FORMATS[MKV_FALLBACK_TARGET_FORMAT]
                    # This temp belongs to this job alone (it is derived from
                    # an exclusively reserved name), so deleting it outright is
                    # safe - unlike the output placeholder below. `_encode_video`
                    # normally removes it already; this covers the case where it
                    # could not.
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    # Release only the stub this job still owns. The
                    # emptiness-checked cleanup is what keeps this from
                    # deleting a path another writer has since filled in.
                    _discard_reserved_output(output_path)
                    mp4_stem = output_path.stem
                    # A fresh O_CREAT|O_EXCL claim: the MP4 name having been
                    # free says nothing about the MKV name being free.
                    output_path, tmp_path = reserve_output(
                        output_dir / f"{input_path.stem}{fb['ext']}")
                    _retarget_subtitle_sidecars(
                        sub_pending, mp4_stem, output_path.stem)
                    # Same H.265 encoder and rate control; only the container
                    # changes. The retry goes through the same policy seam as
                    # any other job, so it carries what MP4 refused - including
                    # attachments - without a second copy of the rule. No new
                    # probe: the already-discovered `sub_streams` is all the
                    # policy ever consults.
                    result = _attempt(
                        output_path, tmp_path, fb["muxer"],
                        fb["container_flags"], fb["audio"],
                        build_stream_map_args(fb["muxer"], sub_streams),
                        build_pass1_map_args_for(fb["muxer"]))
                    # True means "a fallback attempt happened", not "it
                    # worked" - the status field remains the verdict.
                    result["fallback_used"] = True
        finally:
            # Temps are throwaway either way: finalization moves the keepers
            # out first, and a failed/skipped/cancelled job leaves nothing.
            if sub_temp_dir is not None:
                shutil.rmtree(sub_temp_dir, ignore_errors=True)
    finally:
        # Error, timeout, cancellation, skip, or an exception on the way out:
        # anything short of a finished video leaves no reserved stub behind.
        if result is None or result.get("status") != "ok":
            _discard_reserved_output(output_path)
    # Internal coordination only: callers see the outcome, not the mechanism.
    result.pop("mux_failed", None)
    return result


def _encode_video(*, input_path: Path, output_path: Path, tmp_path: Path,
                  original_size: int, duration, mode: str, encoder: str,
                  audio_enc: str, audio_kbps, muxer: str,
                  container_flags: list, stream_map_args: list,
                  pass1_map_args: list, resolution_cap, use_two_pass: bool,
                  video_kbps, crf, speed_preset: str, nvenc_preset: str,
                  amf_quality: str, use_hw: bool,
                  cancel_flag: threading.Event, progress_cb, on_start,
                  extract_english_subtitles: bool,
                  sub_pending: list, sub_errors: list) -> dict:
    """The encode + finalize half of `compress_video`.

    Split out so subtitle temp cleanup and output-reservation cleanup each get
    one honest `finally` around every exit path in the caller. `stream_map_args`
    is the container's stream-selection policy (empty only where ffmpeg's
    implicit selection is kept) and `pass1_map_args` the video-only variant for
    two-pass analysis; encoder, rate-control and container settings are
    otherwise untouched.
    """
    vf = build_scale_filter(resolution_cap) if resolution_cap else None
    ffmpeg_base = [FFMPEG_BIN, "-nostdin", "-hide_banner", "-y"]
    common_in = ["-i", str(input_path)]

    def vargs(pass_num):
        return build_video_encoder_args(
            encoder=encoder, vf=vf, use_two_pass=use_two_pass,
            pass_num=pass_num, video_kbps=video_kbps, crf=crf,
            speed_preset=speed_preset, nvenc_preset=nvenc_preset,
            amf_quality=amf_quality,
        )

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
                ffmpeg_base + common_in + pass1_map_args + vargs(1) + [
                    "-passlogfile", passlog, "-an", "-f", "null", os.devnull],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(0, 35, "pass 1/2"),
                on_start=on_start)
            if rc == -3:
                if tmp_path.exists():
                    tmp_path.unlink()
                return {"file": input_path, "status": "timeout",
                        "original": original_size, "msg": err}
            if rc == -2:
                return {"file": input_path, "status": "error", "msg": "cancelled"}
            if rc != 0:
                return {"file": input_path, "status": "error",
                        "msg": f"pass 1 failed: {err}"}

            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + stream_map_args + vargs(2) + [
                    "-passlogfile", passlog,
                    "-c:a", audio_enc, "-b:a", f"{audio_kbps}k",
                ] + container_flags + ["-f", muxer, str(tmp_path)],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(35, 65, "pass 2/2"),
                on_start=on_start)
        else:
            rc, err = run_ffmpeg(
                ffmpeg_base + common_in + stream_map_args + vargs(None) + [
                    "-c:a", audio_enc, "-b:a", f"{audio_kbps}k",
                ] + container_flags + ["-f", muxer, str(tmp_path)],
                cancel_flag,
                duration=duration,
                on_progress=_make_progress(
                    0, 100,
                    "encoding · GPU" if use_hw else "encoding"),
                on_start=on_start)

    if rc == -3:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"file": input_path, "status": "timeout",
                "original": original_size, "msg": err}
    if rc == -2:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"file": input_path, "status": "error", "msg": "cancelled"}
    if rc != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        result = {"file": input_path, "status": "error",
                  "msg": f"{_FFMPEG_FAILURE_PREFIX}{err}"}
        if rc > 0:
            # An ordinary nonzero exit from the *final* encode/mux invocation
            # - single-pass, or pass 2 of two-pass. At this layer an encoder
            # failure and a container/trailer failure are indistinguishable,
            # so the marker says only "this was the retryable invocation" and
            # leaves the policy decision to `compress_video`. Negative codes
            # are Cove's own (-1 launch, -2 cancel, -3 stall) and are never
            # marked; neither is pass 1, which returns above.
            result["mux_failed"] = True
        return result
    if not tmp_path.exists():
        return {"file": input_path, "status": "error", "msg": "no output file produced"}

    new_size = tmp_path.stat().st_size
    if mode == "Quality preset" and new_size >= original_size:
        tmp_path.unlink()
        return {"file": input_path, "status": "skipped",
                "original": original_size, "new": original_size,
                "msg": "compression would increase size (try Target reduction mode)"}

    # `os.replace`, not `rename`: the destination is the zero-byte placeholder
    # this job reserved, and on Windows `rename` refuses an existing target.
    os.replace(str(tmp_path), str(output_path))
    # The last thing between "ffmpeg exited 0" and telling the user their
    # video converted. Everything success-only hangs off this line - the ok
    # result, sidecar finalization, and the source deletion those two make
    # eligible - so a destination that is missing, empty, not a regular file
    # or unreadable stops here rather than becoming a result somebody trusts.
    # No `mux_failed` marker: a successful exit that produced nothing is a
    # different anomaly from the ordinary mux failure the MKV retry exists
    # for, and it stays terminal instead of widening that retry.
    if not _final_output_is_present(output_path):
        return {"file": input_path, "status": "error",
                "msg": "FFmpeg reported success but no valid output file "
                       "was created"}
    # ...and a file with bytes in it is not yet a video. Second and last: can
    # ffprobe open what was produced, and is there picture in it? Only a path
    # that got this far is worth probing - a missing, empty, non-regular or
    # unstatable one has already failed above, and a failed encode returned
    # long before either gate. Like the structural check this is deliberately
    # terminal: an apparent success that is not a video is a different anomaly
    # from the mux failure the MP4 -> MKV retry exists for, so it carries no
    # `mux_failed` marker and earns no second encode.
    verdict = _final_output_is_readable(output_path, cancel_flag)
    if not verdict:
        if cancel_flag.is_set():
            # The flag landed between ffmpeg exiting and the gate running.
            # A cancelled job is not a successful conversion whatever the
            # bytes would have probed as - and an interrupted job's output is
            # not Cove's to throw away, so it stays exactly where it is.
            return {"file": input_path, "status": "error", "msg": "cancelled"}
        if verdict.rejected:
            # ffprobe launched, finished, and either refused these bytes or
            # found no video in them. Either way that is a verdict about the
            # artifact rather than about the probe, so the file Cove reserved
            # and filled - and is about to disown - goes with it. A probe that
            # never got to answer (unlaunchable, timed out) reaches here too
            # and deliberately does not: it may have been a good video, and
            # deleting an output Cove merely failed to verify is the larger
            # mistake of the two.
            _discard_rejected_output(output_path)
        return {"file": input_path, "status": "error",
                # "no video stream" was true while the only rejected case was a
                # file with none. An artwork-only output has one, and telling
                # the user it does not would send them debugging a stream they
                # can see; "playable" is true of both cases and of neither
                # good one.
                "msg": ("The output file contains no playable video stream"
                        if verdict.no_video else
                        "The output file could not be verified as readable "
                        "media")}
    result = {"file": input_path, "output": output_path, "status": "ok",
              "original": original_size, "new": new_size, "encoder": encoder}
    # Sidecars are finalized only now, against a video output that exists.
    # Any probe/extraction/finalization failure sets `subtitles_failed`, which
    # `delete_source_if_eligible` treats as a veto on removing the original.
    if extract_english_subtitles:
        _finalize_subtitle_sidecars(result, sub_pending, sub_errors)
    return result
