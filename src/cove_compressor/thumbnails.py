"""Background thumbnail cache.

Generates queue-row previews for images (via Pillow) and videos (via a
single-frame ffmpeg grab). Results are emitted as `QImage` — thread-safe;
callers convert to `QPixmap` on the UI thread. Daemon threads so closing
the app while a video extract is running won't hang shutdown.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from .compressor import FFMPEG_BIN, SUBPROCESS_FLAGS, LANCZOS


class ThumbnailCache(QObject):
    loaded = Signal(object, object)  # Path, QImage

    THUMB_PX = 160  # max source-edge — kept bigger than the on-screen size
                    # so HiDPI downscales stay sharp.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache: dict[Path, QImage] = {}
        self._pending: set[Path] = set()
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(2)  # cap concurrent ffmpeg + PIL workers

    def get(self, path: Path) -> QImage | None:
        return self._cache.get(path)

    def request(self, path: Path, is_video: bool) -> None:
        with self._lock:
            if path in self._cache or path in self._pending:
                return
            self._pending.add(path)
        t = threading.Thread(
            target=self._worker, args=(path, is_video), daemon=True,
        )
        t.start()

    def _worker(self, path: Path, is_video: bool) -> None:
        with self._sem:
            img = None
            try:
                img = self._video_thumb(path) if is_video else self._image_thumb(path)
            except Exception:  # noqa: BLE001
                img = None
            if img is not None and not img.isNull():
                with self._lock:
                    self._cache[path] = img
                    self._pending.discard(path)
                self.loaded.emit(path, img)
            else:
                with self._lock:
                    self._pending.discard(path)

    def _image_thumb(self, path: Path) -> QImage | None:
        try:
            pil = Image.open(path)
            pil = ImageOps.exif_transpose(pil)
            pil.thumbnail((self.THUMB_PX, self.THUMB_PX), LANCZOS)
            if pil.mode == "RGBA":
                data = pil.tobytes("raw", "RGBA")
                img = QImage(data, pil.width, pil.height,
                             pil.width * 4, QImage.Format_RGBA8888)
            else:
                pil = pil.convert("RGB")
                data = pil.tobytes("raw", "RGB")
                img = QImage(data, pil.width, pil.height,
                             pil.width * 3, QImage.Format_RGB888)
            return img.copy()
        except Exception:  # noqa: BLE001
            return None

    def _video_thumb(self, path: Path) -> QImage | None:
        if not shutil.which(FFMPEG_BIN):
            return None
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix="cove_thumb_", delete=False,
        ) as tmp:
            tmp_name = tmp.name
        try:
            subprocess.run(
                [FFMPEG_BIN, "-nostdin", "-hide_banner", "-y",
                 "-ss", "1",
                 "-i", str(path),
                 "-frames:v", "1",
                 "-vf", f"scale={self.THUMB_PX}:{self.THUMB_PX}:"
                        "force_original_aspect_ratio=decrease",
                 tmp_name],
                capture_output=True, timeout=10, **SUBPROCESS_FLAGS,
            )
            img = QImage(tmp_name)
            return img.copy() if not img.isNull() else None
        except Exception:  # noqa: BLE001
            return None
        finally:
            Path(tmp_name).unlink(missing_ok=True)
