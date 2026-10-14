"""Empty queue pickers and keyboard access, without media work or networking."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cove_compressor import app  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    instance = QApplication.instance() or QApplication([])
    previous = instance.styleSheet()
    app.theme.apply_theme(instance)
    yield instance
    instance.setStyleSheet(previous)


@pytest.fixture(params=["images", "videos"])
def window(request, qt, monkeypatch, tmp_path):
    class NoUpdater(QObject):
        def __init__(self, parent, **kwargs):
            super().__init__(parent)

        def check(self):
            pass

    monkeypatch.setattr(app, "is_portable", lambda: True)
    monkeypatch.setattr(app, "portable_data_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(app.updater, "UpdateController", NoUpdater)
    monkeypatch.setattr(app.MainWindow, "_check_deps", lambda self: None)
    monkeypatch.setattr(app.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(app.ThumbnailCache, "request", lambda *args: None)
    monkeypatch.setattr(app.QTimer, "singleShot", lambda *args: None)
    files = Mock(return_value=([], ""))
    folders = Mock(return_value="")
    monkeypatch.setattr(app.QFileDialog, "getOpenFileNames", files)
    monkeypatch.setattr(app.QFileDialog, "getExistingDirectory", folders)
    w = app.MainWindow()
    w.resize(960, 660)
    w._set_tab(request.param)
    w.show()
    w.activateWindow()
    qt.processEvents()
    assert not files.called and not folders.called, "No startup picker"
    w.pickers = files, folders
    yield w
    for timer in w.findChildren(app.QTimer):
        timer.stop()
    w.close()
    w.deleteLater()
    qt.sendPostedEvents(None, QEvent.DeferredDelete)
    qt.processEvents()


def active(w):
    return w.queue_stack.currentWidget()


def button(q, text):
    matches = [b for b in q.findChildren(QPushButton)
               if b.text() == text and b.isVisible()]
    assert len(matches) == 1, f"Expected one visible {text} button"
    return matches[0]


def media(q, tmp_path, name="sample"):
    p = tmp_path / (name + sorted(q._exts)[0])
    p.write_bytes(b"disposable picker test")
    return p


def visible_focus():
    QApplication.processEvents()
    focused = QApplication.focusWidget()
    assert focused is not None and focused.isVisible() and focused.isEnabled()
    return focused


def test_empty_actions_geometry_and_names(window):
    q = active(window)
    assert q.is_empty() and q.acceptDrops()
    for text, accessible in (("Add files…", "Add files"),
                             ("Add folder…", "Add folder")):
        b = button(q, text)
        assert accessible.lower() in b.accessibleName().lower()
        assert b.isEnabled() and b.focusPolicy() & Qt.TabFocus
        for target in (q, window):
            rect = b.rect().translated(b.mapTo(target, QPoint(0, 0)))
            assert target.rect().contains(rect)
    assert window.size().width() == 960 and window.size().height() == 660


@pytest.mark.parametrize("folder", [False, True])
def test_empty_picker_wiring_and_cancel(window, tmp_path, folder):
    q = active(window)
    b = button(q, "Add folder…" if folder else "Add files…")
    files, folders = window.pickers
    b.setFocus()
    b.click()
    spy = folders if folder else files
    assert spy.call_count == 1 and spy.call_args.args[0] is q
    assert q.is_empty() and b.isVisible() and b.isEnabled()
    assert visible_focus() is b
    path = tmp_path if folder else media(q, tmp_path)
    spy.return_value = str(path) if folder else ([str(path)], "")
    b.click()
    assert spy.call_count == 2
    assert len(q._entries) == 1 and q._entries[0].path == path
    assert q._entries[0].is_dir == folder and q._stack.currentIndex() == 1
    assert not b.isVisible()
    visible_focus()
    if not folder:
        expected = " ".join(f"*{ext}" for ext in sorted(q._exts))
        assert files.call_args.args[3] == f"Supported files ({expected});;All files (*.*)"


def test_populated_toolbar_clear_and_last_removal(window, tmp_path):
    q = active(window)
    path = media(q, tmp_path)
    q.add_paths([path])
    files, folders = window.pickers
    files.return_value = ([str(media(q, tmp_path, "second"))], "")
    button(q, "Add more…").click()
    folders.return_value = str(tmp_path)
    button(q, "Add folder…").click()
    assert files.call_count == folders.call_count == 1
    assert len(q._entries) == 3
    clear = button(q, "Clear")
    clear.setFocus()
    clear.click()
    assert q.is_empty() and button(q, "Add files…").isEnabled()
    assert button(q, "Add folder…").isEnabled()
    visible_focus()
    q.add_paths([path])
    q._list.setCurrentRow(0)
    q._list.setFocus()
    QTest.keyClick(q._list, Qt.Key_Delete)
    assert q.is_empty() and button(q, "Add files…").isVisible()
    visible_focus()
    q.add_paths([path])
    row = q._list.itemWidget(q._list.item(0))
    remove = row.findChild(app.QToolButton, "RowRemoveBtn")
    remove.setFocus()
    remove.click()
    assert q.is_empty() and button(q, "Add folder…").isVisible()
    visible_focus()
    window.output_edit.setFocus()
    q.add_paths([path])
    q.clear()
    assert visible_focus() is window.output_edit


@pytest.mark.parametrize("folder", [False, True])
def test_shortcuts_current_queue_and_tab_switch(window, folder):
    files, folders = window.pickers
    spy = folders if folder else files
    mods = Qt.ControlModifier | (Qt.ShiftModifier if folder else Qt.NoModifier)
    for _ in range(2):
        q = active(window)
        window.output_edit.setFocus()
        QTest.keyClick(window, Qt.Key_O, mods)
        QApplication.processEvents()
        assert spy.call_count == 1 and spy.call_args.args[0] is q
        assert (files if folder else folders).call_count == 0
        assert q.is_empty()
        spy.reset_mock()
        button(q, "Add files…").setFocus()
        window._set_tab("videos" if q is window.img_queue else "images")
        assert q is not active(window)
        assert not q.isAncestorOf(visible_focus())
    shortcuts = [s for s in window.findChildren(QShortcut)
                 if s.key().toString() == ("Ctrl+Shift+O" if folder else "Ctrl+O")]
    assert len(shortcuts) == 1 and shortcuts[0].parent() is window
    assert shortcuts[0].context() == Qt.WindowShortcut


@pytest.mark.parametrize("folder", [False, True])
def test_shortcuts_respect_buttons_and_text_input(window, tmp_path, folder):
    q = active(window)
    files, folders = window.pickers
    spy = folders if folder else files
    mods = Qt.ControlModifier | (Qt.ShiftModifier if folder else Qt.NoModifier)
    for populated in (False, True):
        if populated:
            q.add_paths([media(q, tmp_path)])
        text = "Add folder…" if folder else ("Add more…" if populated else "Add files…")
        b = button(q, text)
        window.output_edit.setFocus()
        for change, value in ((b.setEnabled, False), (b.setVisible, False)):
            change(value)
            QTest.keyClick(window, Qt.Key_O, mods)
            assert not spy.called
            change(True)
        QTest.keyClick(window, Qt.Key_O, mods)
        assert spy.call_count == 1 and spy.call_args.args[0] is q
        spy.reset_mock()
    window.output_edit.setEnabled(True)
    window.output_edit.setFocus()
    window.output_edit.clear()
    QTest.keyClicks(window.output_edit, "ordinary input")
    assert window.output_edit.text() == "ordinary input" and not spy.called


def test_tab_space_enter_and_visible_focus(window):
    q = active(window)
    files, folders = window.pickers
    first, second = button(q, "Add files…"), button(q, "Add folder…")
    window.images_tab.setFocus()
    reached = []
    for _ in range(60):
        QTest.keyClick(visible_focus(), Qt.Key_Tab)
        if QApplication.focusWidget() in (first, second):
            reached.append(QApplication.focusWidget())
        if len(reached) == 2:
            break
    assert reached == [first, second]
    for b, spy in ((first, files), (second, folders)):
        window.output_edit.setFocus()
        QApplication.processEvents()
        blurred = b.grab().toImage()
        b.setFocus()
        QApplication.processEvents()
        assert b.grab().toImage() != blurred, "Focus must visibly change the button"
        for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            before = spy.call_count
            QTest.keyClick(b, key)
            assert spy.call_count == before + 1
            assert visible_focus() is b
    window.output_edit.setFocus()
    window._set_tab("videos" if q is window.img_queue else "images")
    assert visible_focus() is window.output_edit, "Unrelated focus should stay put"


@pytest.mark.parametrize("folder", [False, True])
def test_populated_picker_focus_is_visible(window, tmp_path, folder):
    q = active(window)
    files, folders = window.pickers
    path = tmp_path if folder else media(q, tmp_path)
    spy = folders if folder else files
    spy.return_value = str(path) if folder else ([str(path)], "")
    empty_button = button(q, "Add folder…" if folder else "Add files…")
    empty_button.setFocus()
    empty_button.click()
    assert spy.call_count == 1 and not q.is_empty()
    assert visible_focus() is button(q, "Add more…")
    for text in ("Add more…", "Add folder…"):
        toolbar_button = button(q, text)
        toolbar_button.setFocus()
        assert visible_focus() is toolbar_button
        focused = toolbar_button.grab().toImage()
        window.output_edit.setFocus()
        assert visible_focus() is window.output_edit
        unfocused = toolbar_button.grab().toImage()
        focused.save(str(tmp_path / f"{text}-focused.png"))
        unfocused.save(str(tmp_path / f"{text}-unfocused.png"))
        assert focused != unfocused, f"{text} focus must visibly change the button"


def test_nonlast_row_removal_restores_list_focus(window, tmp_path):
    q = active(window)
    first, second = media(q, tmp_path), media(q, tmp_path, "second")
    q.add_paths([first, second])
    row = q._list.itemWidget(q._list.item(0))
    remove = row.findChild(app.QToolButton, "RowRemoveBtn")
    remove.setFocus()
    assert visible_focus() is remove
    remove.click()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert q.resolve_files() == [second]
    focused = visible_focus()
    assert focused is q._list and q.isAncestorOf(focused)


def test_duplicates_and_extensions_unchanged(window, tmp_path):
    q = active(window)
    supported = media(q, tmp_path)
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_text("test")
    assert q.add_paths([supported, supported, unsupported]) == 1
    assert q.add_paths([supported, unsupported, tmp_path / "missing"]) == 0
    assert q.resolve_files() == [supported]
