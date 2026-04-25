"""Custom title bar for the frameless main window.

Matches the cove-nexus / cove-video-downloader look: brand badge (Cove skull)
on the left, centered title + version pill, min/max/close buttons on the right.
Double-clicking the title toggles maximize; dragging moves the window via
`QWindow.startSystemMove()` so the WM handles the actual drag.

`FramelessAssistant` handles window move / resize for a frameless QMainWindow
without re-implementing platform quirks — it just forwards mouse events that
land in the border region to `startSystemResize()`.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolButton,
    QWidget,
)

from . import theme


_ASSETS = Path(__file__).resolve().parent / "assets"
_ICON_PATH = _ASSETS / "cove_icon.png"


# ── Brand badge — scaled Cove icon inside a rounded square ────────────────
class _BrandBadge(QLabel):
    """Square tile with the Cove icon centered inside. The source PNG is
    512x512 — we render to a high-DPR pixmap so the icon stays crisp on
    HiDPI and standard displays alike."""

    def __init__(self, size: int = 26) -> None:
        super().__init__()
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self._src: QPixmap | None = None
        if _ICON_PATH.exists():
            pm = QPixmap(str(_ICON_PATH))
            if not pm.isNull():
                self._src = pm

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # Rounded square background
        p.setPen(QPen(QColor(theme.BORDER_HI), 1))
        p.setBrush(QColor("#0b1114"))
        r = self.rect().adjusted(0, 0, -1, -1)
        p.drawRoundedRect(r, 7, 7)
        if self._src is not None:
            dpr = max(1.0, float(self.devicePixelRatioF()))
            side = self.width() - 6
            # Scale at device pixel ratio — this is the difference between
            # a crisp icon and the blurry one the user saw.
            target = self._src.scaled(
                int(side * dpr), int(side * dpr),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            target.setDevicePixelRatio(dpr)
            x = (self.width() - side) // 2
            y = (self.height() - side) // 2
            p.drawPixmap(x, y, side, side, target)
        p.end()


# ── Control button icons (painted with QPainter; no dependency on SVG) ────
class _IconButton(QToolButton):
    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(32, 28)
        self.setObjectName("CtrlBtn")
        self.setCursor(Qt.ArrowCursor)
        self._kind = kind
        self.setAutoRaise(True)

    def sizeHint(self) -> QSize:
        return QSize(32, 28)

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        super().paintEvent(_event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        color = self.palette().color(self.foregroundRole())
        # Use hover-state-aware color: checked/hover via the CSS; we just
        # pick the current text color from the active palette.
        pen = QPen(color, 1.6)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        cx, cy = self.width() / 2, self.height() / 2
        if self._kind == "min":
            p.drawLine(int(cx - 5), int(cy), int(cx + 5), int(cy))
        elif self._kind == "max":
            p.drawRect(QRect(int(cx - 5), int(cy - 5), 10, 10))
        elif self._kind == "restore":
            # Two overlapping rects to signal "already maximized"
            p.drawRect(QRect(int(cx - 5), int(cy - 3), 8, 8))
            p.drawRect(QRect(int(cx - 3), int(cy - 5), 8, 8))
        elif self._kind == "close":
            p.drawLine(int(cx - 5), int(cy - 5), int(cx + 5), int(cy + 5))
            p.drawLine(int(cx + 5), int(cy - 5), int(cx - 5), int(cy + 5))
        p.end()


class TitleBar(QWidget):
    """Top bar rendered inside the frameless QMainWindow.

    Emits nothing — it reaches into the owning window directly for move /
    min / max / close because it's internal to MainWindow and that keeps
    the plumbing simple.
    """

    minimizeRequested = Signal()
    maxRestoreRequested = Signal()
    closeRequested = Signal()

    def __init__(self, window: QMainWindow, *, title: str, version: str) -> None:
        super().__init__()
        self._window = window
        self.setFixedHeight(44)
        self.setAutoFillBackground(True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 6, 0)
        lay.setSpacing(10)

        self.badge = _BrandBadge()
        lay.addWidget(self.badge, 0, Qt.AlignVCenter)

        lay.addStretch(1)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("BrandTitle")
        lay.addWidget(self.title_lbl, 0, Qt.AlignVCenter)

        self.version_lbl = QLabel(f"v{version}")
        self.version_lbl.setObjectName("VersionPill")
        lay.addWidget(self.version_lbl, 0, Qt.AlignVCenter)

        lay.addStretch(1)

        self.min_btn = _IconButton("min")
        self.min_btn.setToolTip("Minimize")
        self.min_btn.clicked.connect(self.minimizeRequested)
        lay.addWidget(self.min_btn)

        self.max_btn = _IconButton("max")
        self.max_btn.setToolTip("Maximize / Restore")
        self.max_btn.clicked.connect(self.maxRestoreRequested)
        lay.addWidget(self.max_btn)

        self.close_btn = _IconButton("close")
        self.close_btn.setObjectName("CloseBtn")
        self.close_btn.setToolTip("Close")
        self.close_btn.clicked.connect(self.closeRequested)
        lay.addWidget(self.close_btn)

    # ── painting (solid color + bottom border) ───────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0f161a"))
        p.drawRect(self.rect())
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()

    # ── window drag / double-click to maximize ───────────────────────────
    def _clicked_on_button(self, child: QWidget | None) -> bool:
        while child is not None:
            if isinstance(child, QToolButton):
                return True
            child = child.parentWidget()
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and not self._clicked_on_button(self.childAt(event.pos())):
            handle = self._window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and not self._clicked_on_button(self.childAt(event.pos())):
            self.maxRestoreRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_maximized(self, maximized: bool) -> None:
        self.max_btn._kind = "restore" if maximized else "max"
        self.max_btn.update()


# ── Frameless resize assistant ─────────────────────────────────────────────
from PySide6.QtCore import QObject


class FramelessResizer(QObject):
    """Attach to a QMainWindow to support edge-drag resizing via
    `QWindow.startSystemResize()`. Works on X11 / Wayland / Windows without
    any WM-specific glue on our side."""

    BORDER = 6

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._w = window
        window.setMouseTracking(True)
        window.installEventFilter(self)

    def _edge_for(self, pos: QPoint) -> Qt.Edges:
        w = self._w
        if w.isMaximized() or w.isFullScreen():
            return Qt.Edges()
        r = w.rect()
        b = self.BORDER
        edges = Qt.Edges()
        if pos.x() <= b:
            edges |= Qt.LeftEdge
        elif pos.x() >= r.width() - b:
            edges |= Qt.RightEdge
        if pos.y() <= b:
            edges |= Qt.TopEdge
        elif pos.y() >= r.height() - b:
            edges |= Qt.BottomEdge
        return edges

    def _cursor_for(self, edges: Qt.Edges):
        h = bool(edges & (Qt.LeftEdge | Qt.RightEdge))
        v = bool(edges & (Qt.TopEdge | Qt.BottomEdge))
        if h and v:
            tl_br = (
                (edges & (Qt.TopEdge | Qt.LeftEdge)) == (Qt.TopEdge | Qt.LeftEdge)
                or (edges & (Qt.BottomEdge | Qt.RightEdge)) == (Qt.BottomEdge | Qt.RightEdge)
            )
            return Qt.SizeFDiagCursor if tl_br else Qt.SizeBDiagCursor
        if h:
            return Qt.SizeHorCursor
        if v:
            return Qt.SizeVerCursor
        return None

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if obj is not self._w:
            return False
        et = event.type()
        if et == QEvent.MouseMove and not (event.buttons() & Qt.LeftButton):
            edges = self._edge_for(event.position().toPoint())
            shape = self._cursor_for(edges)
            if shape is not None:
                self._w.setCursor(shape)
            else:
                self._w.unsetCursor()
        elif et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            edges = self._edge_for(event.position().toPoint())
            if edges:
                handle = self._w.windowHandle()
                if handle is not None:
                    handle.startSystemResize(edges)
                    return True
        elif et == QEvent.Leave:
            self._w.unsetCursor()
        return False
