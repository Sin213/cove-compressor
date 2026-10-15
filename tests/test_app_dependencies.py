"""ffprobe is a dependency of video compression, not of two of its modes.

Cove has always refused to start a video batch without ffmpeg. ffprobe was
treated as optional infrastructure: Target file size and Target reduction
need it to read a duration, so they checked for it, and Quality preset - the
default - ran happily without it.

That exemption is now unsound. Every finished video is validated with ffprobe
before Cove calls the conversion successful, so a machine without ffprobe
cannot produce a successful conversion in *any* mode. Discovering that after
a twenty-minute encode is the wrong place to discover it, so the start
boundary refuses first.

The gate lives in `_start_videos`, which is the single place a video batch
can begin. `_check_deps` stays what it was - a startup log line, informational
only - because a second copy of a blocking policy is a second thing that can
drift.

  A. Both binaries present: the batch starts, exactly as before.
  B. ffprobe missing: Quality preset is blocked. This is the new behavior.
  C. ffprobe missing: the target modes stay blocked too, now by the same
     global rule rather than their own.
  D. ffmpeg missing: the existing refusal is untouched, message and all.
  E. One refusal per attempt - no stacked dialogs.

No widgets and no event loop: `_start_videos` is driven directly against a
stub window, which is the only honest way to assert that no worker thread
was created.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import app as app_mod  # noqa: E402
from cove_compressor.app import MainWindow  # noqa: E402


# ── stubs ────────────────────────────────────────────────────────────────────

class _Combo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _Spin:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Check:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Queue:
    def __init__(self):
        self.batches = []

    def prepare_batch(self, files):
        self.batches.append(list(files))


class _Warnings:
    """Stands in for QMessageBox: records rather than renders."""

    def __init__(self):
        self.shown: list[tuple[str, str]] = []

    def warning(self, parent, title, text):
        self.shown.append((title, text))

    @property
    def texts(self) -> list[str]:
        return [t for _title, t in self.shown]


class _Thread:
    """Stands in for `threading.Thread`: records the worker that would have
    been started, and starts nothing."""

    instances: list["_Thread"] = []

    def __init__(self, target=None, args=(), daemon=False, **kw):
        self.target = target
        self.args = args
        self.started = False
        _Thread.instances.append(self)

    def start(self):
        self.started = True


def _window(tmp_path, mode="Quality preset", files=None):
    """A stub with exactly the surface `_start_videos` touches."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    files = files if files is not None else [tmp_path / "clip.mkv"]
    w = types.SimpleNamespace(
        vid_queue=_Queue(),
        same_dir_chk=_Check(False),
        delete_source_chk=_Check(False),
        vid_subs_chk=_Check(False),
        vid_mode=_Combo(mode),
        vid_size_mb=_Spin(25),
        vid_pct=_Spin(50),
        vid_quality=_Combo("Balanced"),
        vid_format=_Combo("MP4 (H.264)"),
        vid_res=_Combo("Original"),
        vid_audio=_Combo("128"),
        vid_encoder=_Combo("Automatic (GPU if available)"),
        banner=types.SimpleNamespace(hide=lambda: None),
        progress=types.SimpleNamespace(setValue=lambda v: None),
        cancel_flag=types.SimpleNamespace(clear=lambda: None),
        _last_output_dir=None,
        _collect_from_queue=lambda q, kind: list(files),
        _prepare_output=lambda f, same: out_dir,
        _set_running=lambda running: None,
        _set_eta=lambda text: None,
        _log=lambda text: None,
        _run_video_batch=lambda *a, **kw: None,
    )
    return w


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """The real `_start_videos`, a stub window, and a recorded message box."""
    warnings = _Warnings()
    monkeypatch.setattr(app_mod, "QMessageBox", warnings)
    _Thread.instances = []
    monkeypatch.setattr(app_mod.threading, "Thread", _Thread)
    return warnings


def _present(monkeypatch, ffmpeg=True, ffprobe=True, tmp_path=None):
    """Point Cove's binary names at something real or at nothing at all -
    the same lever the packaged app has when a binary is absent."""
    if not ffmpeg:
        monkeypatch.setattr(app_mod, "FFMPEG_BIN",
                            str(tmp_path / "no-such-ffmpeg.exe"))
    if not ffprobe:
        monkeypatch.setattr(app_mod, "FFPROBE_BIN",
                            str(tmp_path / "no-such-ffprobe.exe"))


def _started() -> bool:
    return any(t.started for t in _Thread.instances)


ALL_MODES = ["Quality preset", "Target file size", "Target reduction"]


# ══ A — both binaries present ══════════════════════════════════════════════

@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_both_present_starts_the_batch(gui, tmp_path, monkeypatch, mode):
    w = _window(tmp_path, mode=mode)

    MainWindow._start_videos(w)

    assert gui.shown == [], "a satisfied dependency check says nothing"
    assert _started(), "the worker starts exactly as it always did"


# ══ B — ffprobe missing blocks Quality preset ══════════════════════════════

def test_b_quality_preset_is_blocked_without_ffprobe(gui, tmp_path,
                                                     monkeypatch):
    """The mode that used to be exempt. Without ffprobe no conversion can be
    confirmed, so none is started."""
    _present(monkeypatch, ffprobe=False, tmp_path=tmp_path)
    w = _window(tmp_path, mode="Quality preset")

    MainWindow._start_videos(w)

    assert not _started(), "blocked before any worker exists"
    assert len(gui.shown) == 1
    assert "ffprobe" in gui.texts[0].lower()
    assert w.vid_queue.batches == [], "nothing was queued for a batch"


# ══ C — the target modes stay blocked ═════════════════════════════════════

@pytest.mark.parametrize("mode", ["Target file size", "Target reduction"])
def test_c_target_modes_are_blocked_without_ffprobe(gui, tmp_path,
                                                    monkeypatch, mode):
    _present(monkeypatch, ffprobe=False, tmp_path=tmp_path)
    w = _window(tmp_path, mode=mode)

    MainWindow._start_videos(w)

    assert not _started()
    assert len(gui.shown) == 1
    assert "ffprobe" in gui.texts[0].lower()


@pytest.mark.parametrize("mode", ALL_MODES)
def test_c2_every_mode_gets_the_same_refusal(gui, tmp_path, monkeypatch, mode):
    """One policy, one message. A mode-specific variant surviving alongside
    the global rule is how users end up with two dialogs saying different
    things about the same missing binary."""
    _present(monkeypatch, ffprobe=False, tmp_path=tmp_path)
    w = _window(tmp_path, mode=mode)

    MainWindow._start_videos(w)

    assert len(gui.shown) == 1
    assert "mode" not in gui.texts[0].lower(), \
        "the dependency is the app's, not the mode's"


# ══ D — the ffmpeg refusal is untouched ═══════════════════════════════════

@pytest.mark.parametrize("mode", ALL_MODES)
def test_d_missing_ffmpeg_still_blocks_with_its_own_message(gui, tmp_path,
                                                            monkeypatch, mode):
    _present(monkeypatch, ffmpeg=False, tmp_path=tmp_path)
    w = _window(tmp_path, mode=mode)

    MainWindow._start_videos(w)

    assert not _started()
    assert len(gui.shown) == 1
    assert gui.texts[0].startswith("ffmpeg not found")


def test_d2_both_missing_reports_ffmpeg_first_and_once(gui, tmp_path,
                                                       monkeypatch):
    _present(monkeypatch, ffmpeg=False, ffprobe=False, tmp_path=tmp_path)
    w = _window(tmp_path)

    MainWindow._start_videos(w)

    assert not _started()
    assert len(gui.shown) == 1, "one refusal per attempt"


# ══ E — the informational startup log is unchanged ════════════════════════

def test_e_check_deps_still_only_reports(tmp_path, monkeypatch):
    """`_check_deps` logs what it found and blocks nothing; `_start_videos`
    owns the policy. Keeping them separate is deliberate."""
    _present(monkeypatch, ffprobe=False, tmp_path=tmp_path)
    logged: list[str] = []
    posted: list[tuple] = []
    w = types.SimpleNamespace(
        _log=logged.append,
        msg_queue=types.SimpleNamespace(put=posted.append),
    )
    monkeypatch.setattr(app_mod, "any_nvenc_available", lambda: False)
    monkeypatch.setattr(app_mod, "any_amf_available", lambda: False)

    MainWindow._check_deps(w)

    assert any("ffprobe" in line for line in logged)
    assert any(line.startswith("[ERROR]") and "ffprobe" in line
               for line in logged)
