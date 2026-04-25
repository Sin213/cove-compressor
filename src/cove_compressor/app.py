"""Cove Compressor 2.0 main window — redesigned UI.

Two-column layout:
  • Main column: header (title + Images/Videos pills with badge counts),
    drop zone / file table, action bar (status + progress + Show log + Start),
    optional log panel.
  • Side column: Options panel (preset / output format / resize cap, plus
    video method/quality/res/audio when on the Videos tab), privacy note,
    Destination panel (save folder + browse + open).

Every compression knob from the previous build is preserved — Light /
Balanced / Aggressive image presets, the four video formats, the three
video methods (Target file size / Target reduction / Quality preset),
resolution caps, audio bitrate, the works.

Window is frameless with our own titlebar matching the rest of the
Cove suite (skull badge + version pill + min/max/close).
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QEvent, QObject, QSettings, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedWidget,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from . import __version__, theme, updater
from .compressor import (
    AUDIO_BITRATES, AVIF_AVAILABLE, DEFAULT_OUTPUT,
    FFMPEG_BIN, FFPROBE_BIN, FORMAT_KEY_MAP, FORMAT_OPTIONS,
    IMAGE_EXTS, IMAGE_PRESETS, RESIZE_CAPS_IMG, RESOLUTION_CAPS,
    VIDEO_EXTS, VIDEO_FORMATS, VIDEO_MODES, VIDEO_QUALITY_PRESETS,
    compress_image, compress_video,
    format_eta, human_size, open_in_file_manager, pct_saved, scan_files,
)
from .thumbnails import ThumbnailCache
from .titlebar import TitleBar, FramelessResizer

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_PATH = ASSETS_DIR / "cove_icon.png"

APP_NAME = "Cove Compressor"
ORG_NAME = "Cove"


# ───────────────────────────────────────────────────────────────────────────
# Small reusable bits
# ───────────────────────────────────────────────────────────────────────────

class CovePanel(QFrame):
    """Card-style panel with optional uppercase title row. Matches the
    `CmpPanel` shape from the redesign reference."""

    def __init__(self, title: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("CovePanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        if title:
            head = QLabel(title.upper())
            head.setObjectName("PanelHead")
            outer.addWidget(head)
        self._inner = QWidget()
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(0, 0, 0, 0)
        self._inner_lay.setSpacing(10)
        outer.addWidget(self._inner)

    def add(self, widget: QWidget) -> None:
        self._inner_lay.addWidget(widget)

    def add_layout(self, lay) -> None:
        self._inner_lay.addLayout(lay)


def _field(label: str, widget: QWidget) -> QWidget:
    """Vertical 'LABEL\\nwidget' field used inside the Options panel."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(3)
    l = QLabel(label.upper())
    l.setObjectName("FieldLabel")
    v.addWidget(l)
    v.addWidget(widget)
    return w


# ───────────────────────────────────────────────────────────────────────────
# Eliding label — used in queue rows so long names don't push the ✕ off
# ───────────────────────────────────────────────────────────────────────────

class _ElidingLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self._apply()

    def setFullText(self, text: str) -> None:
        self._full = text
        self._apply()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, Qt.ElideMiddle, max(self.width() - 2, 10)))


# ───────────────────────────────────────────────────────────────────────────
# File queue — drop zone + table-style list
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class QueueEntry:
    path: Path
    is_dir: bool


class _ItemList(QListWidget):
    """List widget that stretches custom row widgets to viewport width and
    forwards Delete/Backspace as a signal.

    Qt's ``::item:selected`` highlight gets painted *behind* a setItemWidget
    custom row, so an opaque row background hides it. Instead of relying on
    the item-level highlight at all we forward selection to a `selected`
    dynamic property on the row QFrame and let stylesheet rules paint a mint
    border + tint there."""

    deleteRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setObjectName("QueueList")
        self.currentItemChanged.connect(self._on_current_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.viewport().width()
        for i in range(self.count()):
            item = self.item(i)
            h = item.sizeHint().height() or 30
            item.setSizeHint(QSize(w, h))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.currentItem() is not None:
                self.deleteRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_current_changed(self, current, previous):
        for item, sel in ((previous, False), (current, True)):
            if item is None:
                continue
            w = self.itemWidget(item)
            if w is None:
                continue
            w.setProperty("selected", sel)
            st = w.style()
            if st is not None:
                st.unpolish(w)
                st.polish(w)
            w.update()


class FileQueue(QFrame):
    """Drop zone that hosts an empty-state pitch when empty and a table-style
    list when populated. Filters drops/browse by `exts`."""

    itemsChanged = Signal()

    THUMB_W = 96
    THUMB_H = 56

    def __init__(self, exts: set, kind_label: str, is_video: bool,
                 thumb_cache: ThumbnailCache, parent=None):
        super().__init__(parent)
        self._exts = exts
        self._kind = kind_label
        self._is_video = is_video
        self._thumb_cache = thumb_cache
        self._thumb_labels: dict[Path, QLabel] = {}
        self._entries: list[QueueEntry] = []

        self.setObjectName("DropFrame")
        self.setAcceptDrops(True)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack, stretch=1)

        # ── Empty state ──────────────────────────────────────
        empty = QWidget()
        ev = QVBoxLayout(empty)
        ev.setContentsMargins(20, 30, 20, 30)
        ev.setSpacing(12)
        ev.addStretch()

        icon_box = _DropIcon(is_video)
        ev.addWidget(icon_box, 0, Qt.AlignCenter)

        head = QLabel(f"Drop {kind_label} or folders here")
        head.setObjectName("DropHead")
        head.setAlignment(Qt.AlignCenter)
        ev.addWidget(head)

        sub = QLabel("Files stay on your machine. Nothing is uploaded.")
        sub.setObjectName("DropSub")
        sub.setAlignment(Qt.AlignCenter)
        ev.addWidget(sub)

        # Format chips row
        chips_row = QHBoxLayout()
        chips_row.setSpacing(6)
        chips_row.addStretch()
        for e in sorted(exts):
            chip = QLabel(e.lstrip("."))
            chip.setObjectName("FormatChip")
            chips_row.addWidget(chip)
        chips_row.addStretch()
        ev.addLayout(chips_row)
        ev.addStretch()

        self._stack.addWidget(empty)

        # ── Filled state ──────────────────────────────────────
        filled = QWidget()
        fv = QVBoxLayout(filled)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.setSpacing(0)

        # Toolbar (count + Add/Clear)
        bar = QFrame()
        bar.setObjectName("DropToolbar")
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(12, 8, 8, 8)
        bh.setSpacing(8)
        self._count_lbl = QLabel("")
        self._count_lbl.setObjectName("ToolbarCount")
        bh.addWidget(self._count_lbl)
        bh.addStretch()

        self._add_more_btn = QPushButton("Add more…")
        self._add_more_btn.setObjectName("GhostButton")
        self._add_more_btn.clicked.connect(self._browse_files)
        bh.addWidget(self._add_more_btn)

        self._add_folder_btn = QPushButton("Add folder…")
        self._add_folder_btn.setObjectName("GhostButton")
        self._add_folder_btn.clicked.connect(self._browse_folder)
        bh.addWidget(self._add_folder_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("DangerButton")
        self._clear_btn.clicked.connect(self.clear)
        bh.addWidget(self._clear_btn)

        fv.addWidget(bar)

        # Column headers
        head_row = QFrame()
        head_row.setObjectName("QueueHead")
        hh = QHBoxLayout(head_row)
        hh.setContentsMargins(14, 6, 14, 6)
        hh.setSpacing(10)
        for caption, w in (("FILE", -1), ("SIZE", 90), ("STATUS", 90)):
            lbl = QLabel(caption)
            lbl.setObjectName("QueueHeader")
            if w >= 0:
                lbl.setFixedWidth(w)
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                hh.addWidget(lbl)
            else:
                hh.addWidget(lbl, stretch=1)
        # spacer for the ✕ column
        spacer = QLabel("")
        spacer.setFixedWidth(36)
        hh.addWidget(spacer)
        fv.addWidget(head_row)

        # The actual list
        self._list = _ItemList()
        self._list.deleteRequested.connect(self._delete_current)
        fv.addWidget(self._list, stretch=1)

        self._stack.addWidget(filled)
        self._stack.setCurrentIndex(0)

        self._update_style(False)

    # ── drag-and-drop ──────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._update_style(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._update_style(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.toLocalFile()]
        self.add_paths(paths)
        self._update_style(False)
        event.acceptProposedAction()

    def _update_style(self, drag: bool) -> None:
        if drag:
            self.setStyleSheet(
                f"QFrame#DropFrame {{"
                f"  border: 1.5px dashed {theme.ACCENT};"
                f"  border-radius: 12px;"
                f"  background: rgba(94,234,212,0.04);"
                f"}}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#DropFrame {{"
                f"  border: 1.5px dashed {theme.BORDER_HI};"
                f"  border-radius: 12px;"
                f"  background: #0c1417;"
                f"}}"
            )

    # ── public API ─────────────────────────────────────────────────────

    def add_paths(self, paths: list[Path]) -> int:
        added = 0
        existing = {e.path.resolve() for e in self._entries}
        for p in paths:
            if not p.exists():
                continue
            rp = p.resolve()
            if rp in existing:
                continue
            if p.is_dir():
                self._entries.append(QueueEntry(p, True))
                existing.add(rp)
                added += 1
            else:
                if p.suffix.lower() in self._exts:
                    self._entries.append(QueueEntry(p, False))
                    existing.add(rp)
                    added += 1
        if added:
            self._rebuild_list()
            self.itemsChanged.emit()
        return added

    def clear(self):
        if not self._entries:
            return
        self._entries.clear()
        self._rebuild_list()
        self.itemsChanged.emit()

    def is_empty(self) -> bool:
        return not self._entries

    def resolve_files(self) -> list[Path]:
        seen = set()
        out: list[Path] = []
        for entry in self._entries:
            if entry.is_dir:
                for f in scan_files(entry.path, self._exts):
                    r = f.resolve()
                    if r not in seen:
                        seen.add(r)
                        out.append(f)
            else:
                r = entry.path.resolve()
                if r not in seen:
                    seen.add(r)
                    out.append(entry.path)
        return out

    # ── internal ──────────────────────────────────────────────────────

    def _rebuild_list(self):
        self._thumb_labels.clear()
        self._list.clear()
        for entry in self._entries:
            self._append_item(entry)
        self._stack.setCurrentIndex(0 if not self._entries else 1)
        if self._entries:
            total = sum(0 if e.is_dir else (e.path.stat().st_size if e.path.exists() else 0)
                        for e in self._entries)
            n = len(self._entries)
            self._count_lbl.setText(
                f"{n} item{'s' if n != 1 else ''} · {human_size(total)}"
            )

    def _append_item(self, entry: QueueEntry) -> None:
        item = QListWidgetItem()
        row = QFrame()
        row.setObjectName("QueueRow")
        row.setProperty("selected", False)
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 6, 6, 6)
        h.setSpacing(10)

        icon_lbl = QLabel()
        icon_lbl.setFixedSize(self.THUMB_W, self.THUMB_H)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setAttribute(Qt.WA_TranslucentBackground)

        if entry.is_dir:
            icon_lbl.setPixmap(_make_folder_pixmap(self.THUMB_H))
        else:
            cached = self._thumb_cache.get(entry.path)
            if cached is not None:
                self._apply_thumb(icon_lbl, cached)
            else:
                icon_lbl.setPixmap(_make_doc_pixmap(self.THUMB_H, entry.path.suffix))
                self._thumb_cache.request(entry.path, self._is_video)
            self._thumb_labels[entry.path] = icon_lbl
        h.addWidget(icon_lbl)

        name_lbl = _ElidingLabel(entry.path.name or str(entry.path))
        name_lbl.setToolTip(str(entry.path))
        name_lbl.setStyleSheet(f"color: {theme.TEXT};")
        h.addWidget(name_lbl, stretch=1)

        if entry.is_dir:
            size_text = "folder"
        else:
            try:
                size_text = human_size(entry.path.stat().st_size)
            except OSError:
                size_text = "—"
        size_lbl = QLabel(size_text)
        size_lbl.setObjectName("Mono")
        size_lbl.setFixedWidth(90)
        size_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(size_lbl)

        status_lbl = QLabel("queued")
        status_lbl.setObjectName("StatusQueued")
        status_lbl.setFixedWidth(90)
        status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(status_lbl)

        rm_btn = QToolButton()
        rm_btn.setText("✕")
        rm_btn.setToolTip("Remove from queue (or press Delete)")
        rm_btn.setAutoRaise(True)
        rm_btn.setFixedSize(28, 28)
        rm_btn.setObjectName("RowRemoveBtn")
        rm_btn.clicked.connect(lambda _=False, e=entry: self._remove_entry(e))
        h.addWidget(rm_btn)

        # Height only — width is reset by _ItemList.resizeEvent.
        hint_h = max(row.sizeHint().height(), self.THUMB_H + 12)
        item.setSizeHint(QSize(self._list.viewport().width(), hint_h))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

    def _apply_thumb(self, label: QLabel, qimg: QImage) -> None:
        pm = QPixmap.fromImage(qimg).scaled(
            self.THUMB_W, self.THUMB_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        label.setPixmap(pm)

    def _on_thumb_loaded(self, path: Path, qimg: QImage) -> None:
        lbl = self._thumb_labels.get(path)
        if lbl is not None:
            self._apply_thumb(lbl, qimg)

    def _remove_entry(self, entry: QueueEntry):
        self._entries = [e for e in self._entries if e is not entry]
        self._rebuild_list()
        self.itemsChanged.emit()

    def _delete_current(self):
        row = self._list.currentRow()
        if 0 <= row < len(self._entries):
            self._remove_entry(self._entries[row])
            new_count = len(self._entries)
            if new_count > 0:
                self._list.setCurrentRow(min(row, new_count - 1))

    def _browse_files(self):
        exts_str = " ".join(f"*{e}" for e in sorted(self._exts))
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select {self._kind}", str(Path.home()),
            f"Supported files ({exts_str});;All files (*.*)",
        )
        if paths:
            self.add_paths([Path(p) for p in paths])

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder", str(Path.home()))
        if folder:
            self.add_paths([Path(folder)])


# ───────────────────────────────────────────────────────────────────────────
# Painted icons (drop zone, queue placeholders)
# ───────────────────────────────────────────────────────────────────────────

class _DropIcon(QLabel):
    """52×52 mint-tinted square with a stylized image / film glyph."""

    def __init__(self, is_video: bool, parent=None):
        super().__init__(parent)
        self._is_video = is_video
        self.setFixedSize(56, 56)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # rounded square bg
        p.setPen(QPen(QColor(theme.BORDER_HI), 1))
        p.setBrush(QColor("#0e171b"))
        r = self.rect().adjusted(0, 0, -1, -1)
        p.drawRoundedRect(r, 12, 12)
        # glyph
        accent = QColor(theme.ACCENT)
        pen = QPen(accent, 1.8)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        cx, cy = self.width() / 2, self.height() / 2
        if self._is_video:
            # film strip
            p.drawRoundedRect(int(cx - 12), int(cy - 9), 24, 18, 2, 2)
            for x in (cx - 9, cx + 6):
                p.drawLine(int(x), int(cy - 9), int(x), int(cy + 9))
            for y in (cy - 4, cy + 4):
                p.drawLine(int(cx - 12), int(y), int(cx - 9), int(y))
                p.drawLine(int(cx + 6), int(y), int(cx + 12), int(y))
        else:
            # mountain / photo glyph
            p.drawRoundedRect(int(cx - 12), int(cy - 10), 24, 20, 2, 2)
            p.setBrush(accent)
            p.drawEllipse(int(cx - 7), int(cy - 6), 4, 4)
            p.setBrush(Qt.NoBrush)
            p.drawLine(int(cx - 11), int(cy + 8), int(cx - 2), int(cy - 2))
            p.drawLine(int(cx - 4), int(cy + 8), int(cx + 6), int(cy - 4))
            p.drawLine(int(cx + 1), int(cy + 8), int(cx + 11), int(cy + 2))
        p.end()


def _make_folder_pixmap(h: int) -> QPixmap:
    pm = QPixmap(h + 6, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(theme.BORDER_HI), 1.4))
    p.setBrush(QColor("#0e171b"))
    pad = 8
    body = pm.rect().adjusted(pad, pad + 4, -pad, -pad)
    p.drawRoundedRect(body, 4, 4)
    # tab
    p.drawLine(body.left() + 4, body.top(), body.left() + 14, body.top())
    p.end()
    return pm


def _make_doc_pixmap(h: int, ext: str) -> QPixmap:
    pm = QPixmap(h + 6, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(theme.BORDER_HI), 1.2))
    p.setBrush(QColor("#0a1013"))
    pad = 10
    rect = pm.rect().adjusted(pad, pad - 2, -pad, -pad + 2)
    p.drawRoundedRect(rect, 4, 4)
    label = (ext or "").lstrip(".").upper()[:4]
    if label:
        f = p.font()
        f.setPointSize(7); f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_3))
        p.drawText(rect, Qt.AlignCenter, label)
    p.end()
    return pm


# ───────────────────────────────────────────────────────────────────────────
# Tab pill (header)
# ───────────────────────────────────────────────────────────────────────────

class TabPill(QPushButton):
    """A single tab pill — icon glyph + label + count badge."""

    def __init__(self, label: str, kind: str, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._label = label
        self._count = 0
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("TabPill")
        self.setMinimumHeight(32)
        self._refresh()

    def set_count(self, n: int) -> None:
        self._count = n
        self._refresh()

    def _refresh(self) -> None:
        self.setText(f"  {self._label}   {self._count}")


# ───────────────────────────────────────────────────────────────────────────
# Main window
# ───────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.resize(1180, 820)
        self.setMinimumSize(960, 660)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self._resizer = FramelessResizer(self)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.msg_queue: queue.Queue = queue.Queue()
        self.cancel_flag = threading.Event()
        self._last_output_dir: Path | None = None
        self._thumb_cache = ThumbnailCache(self)

        self._build_ui()
        self._restore_settings()

        self._timer = QTimer(self)
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

    # ── frameless helpers ──────────────────────────────────────────────

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.WindowStateChange and hasattr(self, "titlebar"):
            self.titlebar.set_maximized(self.isMaximized())
        super().changeEvent(event)

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralRoot")
        self.setCentralWidget(central)
        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        self.titlebar = TitleBar(self, title=APP_NAME, version=__version__)
        self.titlebar.minimizeRequested.connect(self.showMinimized)
        self.titlebar.maxRestoreRequested.connect(self._toggle_maximize)
        self.titlebar.closeRequested.connect(self.close)
        root_v.addWidget(self.titlebar, 0)

        body = QWidget()
        body.setObjectName("CoveBody")
        root_v.addWidget(body, 1)
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(18, 14, 18, 14)
        body_lay.setSpacing(14)

        # ── Main column ────────────────────────────────────────
        main_col = QVBoxLayout()
        main_col.setSpacing(12)

        # Completion banner (hidden until a batch finishes — savings + open
        # output folder go here so they're not buried in the log).
        self.banner = QFrame()
        self.banner.setObjectName("Banner")
        bh = QHBoxLayout(self.banner)
        bh.setContentsMargins(14, 10, 8, 10)
        bh.setSpacing(10)
        self.banner_icon = QLabel("✓")
        self.banner_icon.setObjectName("BannerIcon")
        self.banner_icon.setFixedWidth(16)
        bh.addWidget(self.banner_icon)
        self.banner_text = QLabel("")
        self.banner_text.setObjectName("BannerText")
        self.banner_text.setWordWrap(True)
        bh.addWidget(self.banner_text, stretch=1)
        self.banner_open_btn = QPushButton("Open output folder")
        self.banner_open_btn.setObjectName("BannerButton")
        self.banner_open_btn.clicked.connect(self._open_last_output_folder)
        bh.addWidget(self.banner_open_btn)
        self.banner_dismiss = QToolButton()
        self.banner_dismiss.setText("✕")
        self.banner_dismiss.setObjectName("BannerDismiss")
        self.banner_dismiss.setAutoRaise(True)
        self.banner_dismiss.setFixedSize(26, 26)
        self.banner_dismiss.clicked.connect(self.banner.hide)
        bh.addWidget(self.banner_dismiss)
        self.banner.hide()
        main_col.addWidget(self.banner)

        # Header (title + tab pills)
        header = QHBoxLayout()
        header.setSpacing(12)
        head_text = QVBoxLayout()
        head_text.setSpacing(2)
        h1 = QLabel("Compress")
        h1.setObjectName("H1")
        head_text.addWidget(h1)
        sub = QLabel("Drop images or videos. Cove strips metadata, transcodes, "
                     "and saves to your folder.")
        sub.setObjectName("Sub")
        sub.setWordWrap(True)
        head_text.addWidget(sub)
        header.addLayout(head_text, stretch=1)

        tabs_box = QFrame()
        tabs_box.setObjectName("TabsBox")
        tb = QHBoxLayout(tabs_box)
        tb.setContentsMargins(4, 4, 4, 4)
        tb.setSpacing(2)
        self.images_tab = TabPill("Images", "images")
        self.images_tab.setChecked(True)
        self.images_tab.clicked.connect(lambda: self._set_tab("images"))
        self.videos_tab = TabPill("Videos", "videos")
        self.videos_tab.clicked.connect(lambda: self._set_tab("videos"))
        tb.addWidget(self.images_tab)
        tb.addWidget(self.videos_tab)
        header.addWidget(tabs_box, 0, Qt.AlignBottom)
        main_col.addLayout(header)

        # Drop zone — both queues exist; only the active tab's is shown.
        self.img_queue = FileQueue(IMAGE_EXTS, "images", False, self._thumb_cache)
        self.vid_queue = FileQueue(VIDEO_EXTS, "videos", True, self._thumb_cache)
        self._thumb_cache.loaded.connect(self.img_queue._on_thumb_loaded, Qt.QueuedConnection)
        self._thumb_cache.loaded.connect(self.vid_queue._on_thumb_loaded, Qt.QueuedConnection)
        self.img_queue.itemsChanged.connect(self._refresh_tab_counts)
        self.vid_queue.itemsChanged.connect(self._refresh_tab_counts)

        self.queue_stack = QStackedWidget()
        self.queue_stack.addWidget(self.img_queue)
        self.queue_stack.addWidget(self.vid_queue)
        main_col.addWidget(self.queue_stack, stretch=1)

        # Action bar (status + progress + show-log + start/cancel)
        action = QFrame()
        action.setObjectName("ActionBar")
        ah = QHBoxLayout(action)
        ah.setContentsMargins(14, 12, 14, 12)
        ah.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(6)
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusText")
        status_row.addWidget(self.status_label, stretch=1)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("Mono")
        status_row.addWidget(self.eta_label)
        left.addLayout(status_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        left.addWidget(self.progress)

        self.log_toggle = QToolButton()
        self.log_toggle.setCheckable(True)
        self.log_toggle.setObjectName("LogToggle")
        self.log_toggle.setText("▸  Show log")
        self.log_toggle.setAutoRaise(True)
        self.log_toggle.clicked.connect(self._on_log_toggled)
        left.addWidget(self.log_toggle, 0, Qt.AlignLeft)

        ah.addLayout(left, stretch=1)

        # Right side — start / cancel
        self.start_btn = QPushButton("Start image compression")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(220)
        self.start_btn.clicked.connect(self._on_start)
        ah.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("DangerButton")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        ah.addWidget(self.cancel_btn)

        main_col.addWidget(action)

        # Log panel (hidden by default)
        self.log_panel = CovePanel("Log")
        log_top = QHBoxLayout()
        log_top.setSpacing(6)
        log_top.addStretch()
        copy_btn = QToolButton()
        copy_btn.setText("Copy")
        copy_btn.setAutoRaise(True)
        copy_btn.setObjectName("LogActionBtn")
        copy_btn.clicked.connect(self._copy_log)
        log_top.addWidget(copy_btn)
        clr_btn = QToolButton()
        clr_btn.setText("Clear")
        clr_btn.setAutoRaise(True)
        clr_btn.setObjectName("LogActionBtn")
        clr_btn.clicked.connect(lambda: self.log.clear())
        log_top.addWidget(clr_btn)
        self.log_panel.add_layout(log_top)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setAcceptRichText(False)
        self.log.setObjectName("LogView")
        self.log.setFixedHeight(200)
        self.log_panel.add(self.log)
        self.log_panel.setVisible(False)
        main_col.addWidget(self.log_panel)

        body_lay.addLayout(main_col, stretch=1)

        # ── Side column ────────────────────────────────────────
        side = QVBoxLayout()
        side.setSpacing(12)
        side.setContentsMargins(0, 0, 0, 0)
        body_lay.addLayout(side, 0)

        # Options panel — hosts a stacked widget so we can swap controls
        # by tab while keeping every original setting available.
        self.options_panel = CovePanel("Options")
        self.options_stack = QStackedWidget()
        self.options_stack.addWidget(self._build_image_options())
        self.options_stack.addWidget(self._build_video_options())
        self.options_panel.add(self.options_stack)
        side.addWidget(self.options_panel)

        # Privacy note
        meta = QFrame()
        meta.setObjectName("MetaNote")
        ml = QHBoxLayout(meta)
        ml.setContentsMargins(12, 10, 12, 10)
        ml.setSpacing(10)
        meta_icon = QLabel("◆")
        meta_icon.setObjectName("MetaIcon")
        meta_icon.setFixedWidth(14)
        ml.addWidget(meta_icon, 0, Qt.AlignTop)
        meta_text = QLabel(
            "Metadata (EXIF, GPS, camera info, timestamps) is "
            "<b>always stripped</b>."
        )
        meta_text.setWordWrap(True)
        meta_text.setObjectName("MetaText")
        ml.addWidget(meta_text, stretch=1)
        side.addWidget(meta)

        # Destination panel
        dest = CovePanel("Destination")
        save_label = QLabel("SAVE TO")
        save_label.setObjectName("FieldLabel")
        dest.add(save_label)
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self.output_edit = QLineEdit(DEFAULT_OUTPUT)
        self.output_edit.setPlaceholderText("Output folder…")
        path_row.addWidget(self.output_edit, stretch=1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_output)
        path_row.addWidget(browse_btn)
        open_btn = QToolButton()
        open_btn.setText("↗")
        open_btn.setToolTip("Open output folder")
        open_btn.setObjectName("OpenFolderBtn")
        open_btn.setFixedSize(34, 34)
        open_btn.clicked.connect(self._open_output_folder_from_edit)
        path_row.addWidget(open_btn)
        dest.add_layout(path_row)
        side.addWidget(dest)

        side.addStretch(1)

        self._apply_extra_qss()

    def _build_image_options(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        self.img_preset = QComboBox()
        self.img_preset.addItems(list(IMAGE_PRESETS.keys()))
        self.img_preset.setCurrentText("Balanced")
        v.addWidget(_field("Preset", self.img_preset))

        self.img_format = QComboBox()
        self.img_format.addItems(FORMAT_OPTIONS)
        v.addWidget(_field("Output format", self.img_format))

        self.img_resize = QComboBox()
        self.img_resize.addItems(list(RESIZE_CAPS_IMG.keys()))
        v.addWidget(_field("Resize cap", self.img_resize))

        # Without this, the stacked-widget's image page is stretched to match
        # the (taller) video page's preferred height, and the empty space is
        # distributed between the fields as fat gaps.
        v.addStretch(1)
        return w

    def _build_video_options(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        self.vid_mode = QComboBox()
        self.vid_mode.addItems(VIDEO_MODES)
        self.vid_mode.setCurrentText("Quality preset")
        self.vid_mode.currentTextChanged.connect(self._update_vid_mode)
        v.addWidget(_field("Method", self.vid_mode))

        # Mode-specific value control (size MB / reduction % / quality preset)
        self._vid_mode_stack = QStackedWidget()
        self._vid_mode_stack.setObjectName("VidModeStack")

        # 0 — Target file size (MB)
        p0 = QWidget(); p0v = QVBoxLayout(p0)
        p0v.setContentsMargins(0, 0, 0, 0); p0v.setSpacing(5)
        p0v_lbl = QLabel("TARGET SIZE")
        p0v_lbl.setObjectName("FieldLabel")
        p0v.addWidget(p0v_lbl)
        self.vid_size_mb = QSpinBox()
        self.vid_size_mb.setRange(1, 50000)
        self.vid_size_mb.setValue(10)
        self.vid_size_mb.setSuffix(" MB")
        p0v.addWidget(self.vid_size_mb)
        self._vid_mode_stack.addWidget(p0)

        # 1 — Target reduction (%)
        p1 = QWidget(); p1v = QVBoxLayout(p1)
        p1v.setContentsMargins(0, 0, 0, 0); p1v.setSpacing(5)
        p1v_lbl = QLabel("REDUCE BY")
        p1v_lbl.setObjectName("FieldLabel")
        p1v.addWidget(p1v_lbl)
        self.vid_pct = QSpinBox()
        self.vid_pct.setRange(10, 90)
        self.vid_pct.setSingleStep(5)
        self.vid_pct.setValue(50)
        self.vid_pct.setSuffix(" %")
        p1v.addWidget(self.vid_pct)
        self._vid_mode_stack.addWidget(p1)

        # 2 — Quality preset
        p2 = QWidget(); p2v = QVBoxLayout(p2)
        p2v.setContentsMargins(0, 0, 0, 0); p2v.setSpacing(5)
        p2v_lbl = QLabel("QUALITY")
        p2v_lbl.setObjectName("FieldLabel")
        p2v.addWidget(p2v_lbl)
        self.vid_quality = QComboBox()
        self.vid_quality.addItems(list(VIDEO_QUALITY_PRESETS.keys()))
        self.vid_quality.setCurrentText("Balanced")
        p2v.addWidget(self.vid_quality)
        self._vid_mode_stack.addWidget(p2)

        self._vid_mode_stack.setCurrentIndex(2)
        v.addWidget(self._vid_mode_stack)

        self.vid_format = QComboBox()
        self.vid_format.addItems(list(VIDEO_FORMATS.keys()))
        self.vid_format.setCurrentText("MP4 (H.265)")
        v.addWidget(_field("Output format", self.vid_format))

        self.vid_res = QComboBox()
        self.vid_res.addItems(list(RESOLUTION_CAPS.keys()))
        v.addWidget(_field("Resolution cap", self.vid_res))

        self.vid_audio = QComboBox()
        self.vid_audio.addItems(AUDIO_BITRATES)
        self.vid_audio.setCurrentText("192")
        v.addWidget(_field("Audio kbps", self.vid_audio))

        v.addStretch(1)
        return w

    def _update_vid_mode(self, text: str) -> None:
        self._vid_mode_stack.setCurrentIndex(
            {"Target file size": 0, "Target reduction": 1,
             "Quality preset": 2}.get(text, 1)
        )

    # ── extra QSS for objects we styled inline ─────────────────────────

    def _apply_extra_qss(self) -> None:
        extra = f"""
        /* theme.py applies QWidget {{ background-color: BG_2 }} which QLabel
           inherits — every label paints a black rectangle against panels and
           queue rows. Override globally; labels that need an explicit bg
           still set it via objectName-targeted rules below. */
        QLabel {{ background: transparent; }}

        #CentralRoot, #CoveBody {{
            background: {theme.BG_2};
        }}
        QLabel#H1 {{
            font-size: 22px; font-weight: 700; color: {theme.TEXT};
            letter-spacing: -0.3px;
        }}
        QLabel#Sub {{
            font-size: 13px; color: {theme.TEXT_3};
        }}
        QLabel#PanelHead, QLabel#FieldLabel {{
            font-size: 10.5px; font-weight: 600; color: {theme.TEXT_3};
            letter-spacing: 0.6px;
        }}
        QLabel#Mono {{
            font-family: "JetBrains Mono", monospace;
            font-size: 11.5px; color: {theme.TEXT_2};
        }}
        QLabel#StatusText {{
            color: {theme.TEXT_2}; font-size: 12.5px;
        }}
        QLabel#StatusQueued {{
            color: {theme.TEXT_3};
            font-family: "JetBrains Mono", monospace; font-size: 11px;
        }}
        QLabel#FormatChip {{
            padding: 2px 8px; border-radius: 999px;
            background: #0c1317;
            border: 1px solid {theme.BORDER};
            color: {theme.TEXT_2};
            font-family: "JetBrains Mono", monospace; font-size: 11px;
        }}
        QLabel#DropHead {{
            font-size: 16px; font-weight: 600; color: {theme.TEXT};
        }}
        QLabel#DropSub {{
            font-size: 12.5px; color: {theme.TEXT_3};
        }}
        QFrame#TabsBox {{
            background: #0a1013;
            border: 1px solid {theme.BORDER};
            border-radius: 10px;
        }}
        QPushButton#TabPill {{
            background: transparent;
            color: {theme.TEXT_3};
            border: 1px solid transparent;
            padding: 5px 14px;
            font-size: 12.5px;
            font-weight: 500;
            border-radius: 7px;
        }}
        QPushButton#TabPill:hover {{
            color: {theme.TEXT_2};
        }}
        QPushButton#TabPill:checked {{
            background: {theme.PANEL_HI};
            color: {theme.TEXT};
            border: 1px solid {theme.BORDER_HI};
        }}
        QFrame#ActionBar {{
            background: {theme.PANEL};
            border: 1px solid {theme.BORDER};
            border-radius: 12px;
        }}
        QProgressBar {{
            background: #0a1013;
            border: 1px solid {theme.BORDER};
            border-radius: 4px;
            min-height: 8px;
            max-height: 8px;
            padding: 0;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {theme.ACCENT_2}, stop:1 {theme.ACCENT});
            border-radius: 3px;
            margin: 0;
        }}
        QToolButton#LogToggle {{
            background: transparent;
            color: {theme.TEXT_3};
            border: none;
            padding: 2px 4px;
            font-size: 11.5px;
        }}
        QToolButton#LogToggle:hover {{
            color: {theme.TEXT};
        }}
        QToolButton#LogActionBtn {{
            color: {theme.TEXT_3};
            background: transparent;
            border: none;
            padding: 2px 6px;
            font-size: 11px;
        }}
        QToolButton#LogActionBtn:hover {{
            color: {theme.TEXT};
        }}
        QTextEdit#LogView {{
            background: #070b0d;
            border: 1px solid {theme.BORDER};
            border-radius: 8px;
            color: {theme.TEXT_2};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
        }}
        QFrame#DropToolbar {{
            background: rgba(255,255,255,0.012);
            border: none;
            border-bottom: 1px solid {theme.BORDER};
        }}
        QLabel#ToolbarCount {{
            color: {theme.TEXT_3};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.6px;
        }}
        QFrame#QueueHead {{
            background: transparent;
            border: none;
            border-bottom: 1px solid {theme.BORDER};
        }}
        QLabel#QueueHeader {{
            color: {theme.TEXT_3};
            font-size: 10.5px;
            font-weight: 600;
            letter-spacing: 0.6px;
        }}
        QListWidget#QueueList {{
            background: transparent;
            border: none;
            outline: 0;
        }}
        QListWidget#QueueList::item {{
            border-bottom: 1px solid {theme.BORDER};
        }}
        /* Suppress Qt's built-in selection painting — it draws *behind*
           setItemWidget rows so we'd never see it anyway. The real highlight
           lives on QueueRow via its `selected` property below. */
        QListWidget#QueueList::item:selected,
        QListWidget#QueueList::item:hover {{
            background: transparent;
        }}
        QFrame#QueueRow {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
        }}
        QFrame#QueueRow:hover {{
            background: rgba(255,255,255,0.025);
        }}
        QFrame#QueueRow[selected="true"] {{
            background: rgba(94,234,212,0.10);
            border: 1px solid rgba(94,234,212,0.45);
        }}
        QToolButton#RowRemoveBtn {{
            color: {theme.TEXT_3};
            background: transparent;
            border: none;
            border-radius: 6px;
            font-size: 14px;
        }}
        QToolButton#RowRemoveBtn:hover {{
            color: {theme.DANGER};
            background: rgba(248,113,113,0.08);
        }}
        QFrame#MetaNote {{
            background: rgba(94,234,212,0.04);
            border: 1px solid rgba(94,234,212,0.18);
            border-radius: 8px;
        }}
        QLabel#MetaIcon {{
            color: {theme.ACCENT};
            font-size: 14px;
        }}
        QLabel#MetaText {{
            color: {theme.TEXT_2};
            font-size: 12px;
        }}
        QToolButton#OpenFolderBtn {{
            background: #141d22;
            color: {theme.TEXT_2};
            border: 1px solid {theme.BORDER};
            border-radius: 6px;
            font-size: 14px;
        }}
        QToolButton#OpenFolderBtn:hover {{
            color: {theme.TEXT};
            background: {theme.PANEL_HI};
            border-color: {theme.BORDER_HI};
        }}
        QFrame#Banner {{
            background: rgba(94,234,212,0.06);
            border: 1px solid rgba(94,234,212,0.28);
            border-radius: 10px;
        }}
        QLabel#BannerIcon {{
            color: {theme.ACCENT};
            font-size: 16px; font-weight: 700;
        }}
        QLabel#BannerText {{
            color: {theme.TEXT};
            font-size: 13px;
        }}
        QPushButton#BannerButton {{
            background: {theme.ACCENT};
            color: {theme.ACCENT_INK};
            border: 1px solid {theme.ACCENT};
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 7px;
        }}
        QPushButton#BannerButton:hover {{
            background: {theme.ACCENT_2};
            border-color: {theme.ACCENT_2};
        }}
        QToolButton#BannerDismiss {{
            color: {theme.TEXT_2};
            background: transparent;
            border: none;
            border-radius: 5px;
            font-size: 13px;
        }}
        QToolButton#BannerDismiss:hover {{
            color: {theme.TEXT};
            background: rgba(255,255,255,0.04);
        }}
        """
        existing = self.styleSheet() or ""
        self.setStyleSheet(existing + extra)

    # ── tab handling ───────────────────────────────────────────────────

    def _set_tab(self, kind: str) -> None:
        is_images = kind == "images"
        self.images_tab.setChecked(is_images)
        self.videos_tab.setChecked(not is_images)
        self.queue_stack.setCurrentIndex(0 if is_images else 1)
        self.options_stack.setCurrentIndex(0 if is_images else 1)
        self.start_btn.setText(
            "Start image compression" if is_images else "Start video compression"
        )

    def _refresh_tab_counts(self) -> None:
        self.images_tab.set_count(len(self.img_queue.resolve_files()) if not self.img_queue.is_empty() else 0)
        self.videos_tab.set_count(len(self.vid_queue.resolve_files()) if not self.vid_queue.is_empty() else 0)

    def _current_kind(self) -> str:
        return "images" if self.images_tab.isChecked() else "videos"

    # ── log toggle / clipboard ─────────────────────────────────────────

    def _on_log_toggled(self) -> None:
        visible = self.log_toggle.isChecked()
        self.log_panel.setVisible(visible)
        self.log_toggle.setText("▾  Hide log" if visible else "▸  Show log")

    def _copy_log(self) -> None:
        QApplication.clipboard().setText(self.log.toPlainText())

    # ── output folder helpers ──────────────────────────────────────────

    def _browse_output(self) -> None:
        start = self.output_edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if folder:
            self.output_edit.setText(folder)

    def _open_output_folder_from_edit(self) -> None:
        p = Path(self.output_edit.text().strip() or DEFAULT_OUTPUT)
        if not p.exists():
            QMessageBox.information(
                self, APP_NAME,
                "Output folder doesn't exist yet — it'll be created on first run.",
            )
            return
        open_in_file_manager(p)

    def _open_last_output_folder(self) -> None:
        target = self._last_output_dir or Path(self.output_edit.text().strip() or DEFAULT_OUTPUT)
        if target and target.exists():
            open_in_file_manager(target)

    # ── settings ───────────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        s = self.settings
        geom = s.value("window/geometry")
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:  # noqa: BLE001
                pass

        def _set_combo(combo: QComboBox, key: str) -> None:
            v = s.value(key)
            if v is not None and combo.findText(str(v)) >= 0:
                combo.setCurrentText(str(v))

        self.output_edit.setText(str(s.value("output/folder", DEFAULT_OUTPUT)))

        _set_combo(self.img_preset,  "img/preset")
        _set_combo(self.img_format,  "img/format")
        _set_combo(self.img_resize,  "img/resize")

        _set_combo(self.vid_format,  "vid/format")
        _set_combo(self.vid_mode,    "vid/mode")
        _set_combo(self.vid_quality, "vid/quality")
        _set_combo(self.vid_res,     "vid/res")
        _set_combo(self.vid_audio,   "vid/audio")
        try:
            self.vid_size_mb.setValue(int(s.value("vid/size_mb", 10)))
            self.vid_pct.setValue(int(s.value("vid/pct", 50)))
        except (TypeError, ValueError):
            pass
        self._update_vid_mode(self.vid_mode.currentText())

        if s.value("log/visible", False, type=bool):
            self.log_toggle.setChecked(True)
            self._on_log_toggled()

        last_tab = s.value("ui/tab", "images")
        if last_tab in ("images", "videos"):
            self._set_tab(str(last_tab))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("output/folder",   self.output_edit.text().strip())
        s.setValue("img/preset",      self.img_preset.currentText())
        s.setValue("img/format",      self.img_format.currentText())
        s.setValue("img/resize",      self.img_resize.currentText())
        s.setValue("vid/format",      self.vid_format.currentText())
        s.setValue("vid/mode",        self.vid_mode.currentText())
        s.setValue("vid/quality",     self.vid_quality.currentText())
        s.setValue("vid/res",         self.vid_res.currentText())
        s.setValue("vid/audio",       self.vid_audio.currentText())
        s.setValue("vid/size_mb",     self.vid_size_mb.value())
        s.setValue("vid/pct",         self.vid_pct.value())
        s.setValue("log/visible",     self.log_toggle.isChecked())
        s.setValue("ui/tab",          self._current_kind())

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    # ── thread-safe UI updates ─────────────────────────────────────────

    def _log(self, line: str) -> None:
        self.msg_queue.put(("log", line))

    def _set_status(self, text: str) -> None:
        self.msg_queue.put(("status", text))

    def _set_progress(self, pct: float) -> None:
        self.msg_queue.put(("progress", pct))

    def _set_eta(self, text: str) -> None:
        self.msg_queue.put(("eta", text))

    def _finish(self) -> None:
        self.msg_queue.put(("finish", None))

    def _poll_queue(self) -> None:
        latest_status = None
        latest_progress = None
        latest_eta = None
        latest_banner = None
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
                elif kind == "eta":
                    latest_eta = payload
                elif kind == "banner":
                    latest_banner = payload
                elif kind == "finish":
                    finish = True
        except queue.Empty:
            pass

        if latest_status is not None:
            self.status_label.setText(latest_status)
        if latest_progress is not None:
            self.progress.setValue(int(latest_progress))
        if latest_eta is not None:
            self.eta_label.setText(latest_eta)
        if latest_banner is not None:
            self.banner_text.setText(latest_banner)
            self.banner.show()
        if finish:
            self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        # Disable tab swapping mid-run to keep the action button context honest.
        self.images_tab.setEnabled(not running)
        self.videos_tab.setEnabled(not running)

    def _cancel(self) -> None:
        self.cancel_flag.set()
        self._set_status("Cancelling…")

    # ── dependency check ──────────────────────────────────────────────

    def _check_deps(self) -> None:
        for label, bin_ in (("ffmpeg", FFMPEG_BIN), ("ffprobe", FFPROBE_BIN)):
            path = shutil.which(bin_)
            if path:
                self._log(f"[ok] {label}: {path}")
            else:
                self._log(f"[ERROR] {label} not found on PATH")

    # ── input validation ──────────────────────────────────────────────

    def _collect_from_queue(self, q: FileQueue, kind: str) -> list[Path] | None:
        if q.is_empty():
            QMessageBox.warning(
                self, APP_NAME,
                f"Drop {kind} files or folders into the queue first\n"
                "(or use Add files… / Add folder…).",
            )
            return None
        files = q.resolve_files()
        if not files:
            QMessageBox.information(
                self, APP_NAME, f"No {kind}s found in the queued items.",
            )
            return None
        return files

    def _prepare_output(self, input_paths: list) -> Path | None:
        out_str = self.output_edit.text().strip()
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
                    "Output folder is the same as an input folder.\n\n"
                    "Compressed files will be written alongside originals "
                    "with unique names. Continue?",
                )
                if r != QMessageBox.Yes:
                    return None
        except OSError:
            pass
        return out

    # ── start handlers ────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._current_kind() == "images":
            self._start_images()
        else:
            self._start_videos()

    def _start_images(self) -> None:
        files = self._collect_from_queue(self.img_queue, "image")
        if not files:
            return
        output_dir = self._prepare_output(files)
        if not output_dir:
            return

        preset       = self.img_preset.currentText()
        force_format = FORMAT_KEY_MAP[self.img_format.currentText()]
        resize_cap   = RESIZE_CAPS_IMG[self.img_resize.currentText()]

        self.banner.hide()
        self._last_output_dir = output_dir
        self.cancel_flag.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self._set_eta("")
        self._log(f"=== Images • {preset} • {len(files)} file(s) • "
                  f"{datetime.now().strftime('%H:%M:%S')} ===")

        threading.Thread(
            target=self._run_image_batch,
            args=(files, output_dir, preset, force_format, resize_cap),
            daemon=True,
        ).start()

    def _start_videos(self) -> None:
        if not shutil.which(FFMPEG_BIN):
            QMessageBox.warning(
                self, APP_NAME,
                "ffmpeg not found.\n\n"
                "Install it or place ffmpeg / ffprobe binaries next to the app.",
            )
            return

        files = self._collect_from_queue(self.vid_queue, "video")
        if not files:
            return
        output_dir = self._prepare_output(files)
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

        vid_format = self.vid_format.currentText()
        res_cap    = RESOLUTION_CAPS[self.vid_res.currentText()]
        audio      = self.vid_audio.currentText()

        self.banner.hide()
        self._last_output_dir = output_dir
        self.cancel_flag.clear()
        self._set_running(True)
        self.progress.setValue(0)
        self._set_eta("")
        self._log(f"=== Videos • {mode} ({mode_value}) • {vid_format} • "
                  f"{len(files)} file(s) • {datetime.now().strftime('%H:%M:%S')} ===")

        threading.Thread(
            target=self._run_video_batch,
            args=(files, output_dir, mode, mode_value, vid_format, res_cap, audio),
            daemon=True,
        ).start()

    # ── batch workers ─────────────────────────────────────────────────

    def _run_image_batch(self, files, output_dir, preset, force_format, resize_cap):
        total = len(files)
        total_orig = total_new = 0
        ok = skipped = errors = 0
        t0 = time.time()
        done = 0

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
                        self._set_status(f"{done}/{total} done")
                        self._set_eta(f"ETA {eta}")
                    else:
                        self._set_status(f"{done}/{total} done")
                        self._set_eta("")
            finally:
                if self.cancel_flag.is_set():
                    for pending in futures:
                        pending.cancel()

        self._summary(ok, skipped, errors, total_orig, total_new, "image")
        self._set_status("Done")
        self._set_eta("")
        self._finish()

    def _run_video_batch(self, files, output_dir, mode, mode_value,
                         vid_format, res_cap, audio):
        total = len(files)
        total_orig = total_new = 0
        ok = skipped = errors = 0
        t0 = time.time()

        for i, f in enumerate(files, start=1):
            if self.cancel_flag.is_set():
                self._log("[cancelled]")
                break

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
                    self._set_eta(f"ETA {eta}")
                name = _f.name if len(_f.name) <= 28 else _f.name[:25] + "…"
                self._set_status(
                    f"{_i}/{total}  {name} — {label} — {file_pct:.0f}%"
                )

            self._set_status(f"{i}/{total}  {f.name}")
            result = compress_video(
                f, output_dir, mode, mode_value, vid_format,
                res_cap, audio,
                self.cancel_flag,
                progress_cb=on_progress,
            )
            self._log(self._fmt(result))
            if result["status"] == "ok":
                ok += 1; total_orig += result["original"]; total_new += result["new"]
            elif result["status"] == "skipped":
                skipped += 1
                total_orig += result["original"]
                total_new += result["original"]
            else:
                errors += 1
            self._set_progress((i / total) * 100)

        self._summary(ok, skipped, errors, total_orig, total_new, "video")
        self._set_status("Done")
        self._set_eta("")
        self._finish()

    # ── log formatting + finish summary ───────────────────────────────

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

    def _summary(self, ok: int, skipped: int, errors: int,
                 total_orig: int, total_new: int, kind: str) -> None:
        self._log("-" * 76)
        self._log(f"Done.  ok={ok}   skipped={skipped}   errors={errors}")
        if total_orig > 0:
            self._log(f"Total: {human_size(total_orig)} → {human_size(total_new)}  "
                      f"({pct_saved(total_orig, total_new):.1f}% saved)")
        self._log("")

        saved = max(total_orig - total_new, 0)
        if ok > 0 and saved > 0:
            banner = (f"Saved {human_size(saved)} "
                      f"({pct_saved(total_orig, total_new):.0f}%) "
                      f"• {ok} {kind}{'s' if ok != 1 else ''} compressed")
            if skipped:
                banner += f" • {skipped} skipped"
            if errors:
                banner += f" • {errors} error{'s' if errors != 1 else ''}"
        elif ok == 0 and skipped > 0 and errors == 0:
            banner = f"All {skipped} {kind}{'s' if skipped != 1 else ''} already at smallest size."
        elif errors and ok == 0:
            banner = f"Finished with {errors} error{'s' if errors != 1 else ''}."
        else:
            banner = "Finished."
        self.msg_queue.put(("banner", banner))
