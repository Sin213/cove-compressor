"""No-space classification survives the whole stderr stream, not just its tail.

`run_ffmpeg` streams ffmpeg's stderr, keeps a bounded window of it, and returns
only the last few lines as the failure message. Everything downstream that
cares whether a job died because the disk filled up - most consequentially
`_mkv_fallback_eligible`, which must refuse to retry a full disk into another
container - reads that message. So a "No space left on device" line that ffmpeg
emits early, followed by enough ordinary cleanup chatter, used to vanish before
anything could classify it: the job was reported as an ordinary mux failure and
MP4 (H.265) went on to burn a second encode against the same full filesystem.

Locked down here:

  A. The tail horizon: in-tail, one line beyond it, far beyond it, and beyond
     it by hundreds of lines all classify as no-space.
  B. Long ordinary stderr with no recognized signal stays an ordinary failure.
  C. The diagnostic tail stays bounded; classification does not depend on
     hoarding stderr.
  D. Only the committed vocabulary counts - buried recognized phrases classify,
     buried plausible-but-unrecognized ones do not.
  E. Every container reports a buried no-space failure as one.
  F. Buried no-space in MP4 (H.265) still refuses the MKV fallback.
  G. An ordinary eligible mux failure still gets its one MKV retry.
  H. No-space state never leaks between attempts.
  I. Pass-1 no-space stops before pass 2 runs.
  J. Pass-2 no-space stays specific and keeps the source.
  K. Cancellation and stall/timeout precedence is unchanged, and a nonzero
     no-space exit is never rewritten into Tab 10's missing-output error.
  L. Success stays success; stderr warnings do not classify anything.
  M. No-space state never leaks between files.
  N. Retained stderr stays bounded no matter how much ffmpeg emits.
  O. Command construction is untouched.

Unlike the fakes in `tests/test_mkv_fallback.py`, which hand `run_ffmpeg` a
pre-built `(rc, message)` pair, everything here drives the *real* `run_ffmpeg`
against a real subprocess that writes real scripted stderr. The stderr reader,
the bounded window and the message the classifier actually sees are all
exercised; only the encoder itself is stood in for.
"""
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import compress_video  # noqa: E402

# Captured before any monkeypatching so the scripted stand-in below can call
# the genuine implementation rather than recurse into itself.
_REAL_RUN_FFMPEG = compressor.run_ffmpeg


# ── the predecessor's bounds, restated as constants ──────────────────────────
#
# `run_ffmpeg` keeps a 40-line window and returns the last 5 of it. Tab 11 does
# not move either number; the tests assert against them so a future change to
# the display tail cannot silently pass for a classification fix.
RETAINED_TAIL_LINES = 5
STDERR_WINDOW_LINES = 40

NO_SPACE_LINE = "av_interleaved_write_frame(): No space left on device"
ENOSPC_LINE = "Error writing trailer: ENOSPC"

# Plausible phrasings the committed classifier does *not* recognize. Tab 11 is
# a visibility fix, not a vocabulary expansion, so these must stay ordinary.
UNRECOGNIZED_LINES = [
    "Error writing trailer: disk full",
    "muxer error: quota exceeded",
    "write failed: insufficient storage",
    "output device full",
]


def _chatter(count: int, start: int = 0) -> list[str]:
    return [f"[hevc @ 0000021f] generic encoder chatter line {i}"
            for i in range(start, start + count)]


# The emitter reads its payload from a file rather than carrying it inline:
# some of these scripts are thousands of lines long and Windows caps a command
# line at 32 KB.
_EMITTER = (
    "import sys, time\n"
    "payload, code, nap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])\n"
    "with open(payload, encoding='utf-8') as fh:\n"
    "    for line in fh:\n"
    "        sys.stderr.write(line)\n"
    "sys.stderr.flush()\n"
    "time.sleep(nap)\n"
    "sys.exit(code)\n"
)


@contextmanager
def _scripted_process(lines, code: int, sleep: float = 0.0):
    """A real process that writes `lines` to stderr and exits with `code`."""
    fd, payload = tempfile.mkstemp(prefix="cove_enospc_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in lines)
        yield [sys.executable, "-c", _EMITTER, payload, str(code), str(sleep)]
    finally:
        os.unlink(payload)


def _run_scripted(lines, code: int, cancel_flag=None, sleep: float = 0.0):
    """Drive the production stderr path over a scripted real subprocess."""
    with _scripted_process(lines, code, sleep) as cmd:
        return _REAL_RUN_FFMPEG(
            cmd, cancel_flag if cancel_flag is not None else threading.Event())


def _classified_no_space(message: str) -> bool:
    """The one committed predicate, asked the one committed way."""
    return compressor._is_no_space_failure(message)


# ── compress_video-level harness ─────────────────────────────────────────────

SRC_BYTES = 4 * 1024 * 1024


def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


class ScriptedStderrFfmpeg:
    """Stands in for the encoder, not for the stderr reader.

    Each scripted invocation is `(exit_code, stderr_lines)`. The command Cove
    built is recorded and then replaced with a real subprocess that emits those
    lines, which the genuine `run_ffmpeg` reads, windows and summarizes exactly
    as it would for ffmpeg. Queues fall back to a silent success once drained.
    """

    def __init__(self, finals=None, pass1=None, subtitle=None,
                 encode_bytes=b"v" * 10):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.subtitle = list(subtitle or [])
        self.encode_bytes = encode_bytes
        self.subtitle_cmds: list[list] = []
        self.pass1_cmds: list[list] = []
        self.final_cmds: list[list] = []
        self.messages: list[str] = []

    @staticmethod
    def _next(q):
        return q.pop(0) if q else (0, [])

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        cmd = list(cmd)
        out = Path(cmd[-1])
        if "-vn" in cmd and "-an" in cmd and "-map" in cmd:
            self.subtitle_cmds.append(cmd)
            kind, spec = "subtitle", self._next(self.subtitle)
        elif _muxer_of(cmd) == "null":
            self.pass1_cmds.append(cmd)
            kind, spec = "pass1", self._next(self.pass1)
        else:
            self.final_cmds.append(cmd)
            kind, spec = "final", self._next(self.finals)

        code, lines = spec
        with _scripted_process(lines, code) as scripted:
            rc, message = _REAL_RUN_FFMPEG(scripted, cancel_flag)
        self.messages.append(message)
        if rc == 0:
            if kind == "subtitle":
                out.write_bytes(b"1\nhello\n")
            elif kind == "final":
                out.write_bytes(self.encode_bytes)
        return rc, message

    @property
    def final_muxers(self) -> list[str]:
        return [_muxer_of(c) for c in self.final_cmds]

    @property
    def subprocess_count(self) -> int:
        return len(self.subtitle_cmds) + len(self.pass1_cmds) \
            + len(self.final_cmds)


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = ScriptedStderrFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(subtitles=[], audio_count=1))
    monkeypatch.setattr(compressor, "nvenc_available",
                        lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available",
                        lambda e="hevc_amf": False)
    return src, out_dir, fake


def _run(src, out_dir, fmt="MP4 (H.265)", mode="Quality preset",
         mode_value="Balanced", cancel_flag=None, **kw):
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


def _two_pass(src, out_dir, **kw):
    """Target-size mode on a CPU encoder, which is the two-pass path."""
    return _run(src, out_dir, mode="Target file size", mode_value="1", **kw)


# A buried signal: the decisive line first, then far more ordinary lines than
# the returned tail can hold.
def _buried(signal: str = NO_SPACE_LINE, trailing: int = 20) -> list[str]:
    return [signal] + _chatter(trailing)


# ══ GROUP A — the tail horizon ═══════════════════════════════════════════════

def test_a1_no_space_inside_the_retained_tail_is_classified():
    """The case that already worked, kept as the control."""
    lines = _chatter(3) + [NO_SPACE_LINE] + _chatter(3, start=3)
    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_a2_no_space_one_line_beyond_the_retained_tail_is_classified():
    """Off by exactly one line is the whole defect in its smallest form."""
    lines = [NO_SPACE_LINE] + _chatter(RETAINED_TAIL_LINES)
    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_a3_no_space_far_beyond_the_retained_tail_is_classified():
    lines = _buried(trailing=30)
    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_a4_no_space_beyond_hundreds_of_trailing_lines_is_classified():
    """Well past the 40-line window, so no larger buffer could rescue it."""
    lines = [NO_SPACE_LINE] + _chatter(600)
    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert _classified_no_space(message)


# ══ GROUP B — ordinary failures stay ordinary ════════════════════════════════

def test_b1_long_ordinary_stderr_is_not_classified_as_no_space():
    rc, message = _run_scripted(_chatter(600), 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_b2_ordinary_failure_message_is_the_ordinary_tail():
    lines = _chatter(20)
    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert message.splitlines() == lines[-RETAINED_TAIL_LINES:]


# ══ GROUP C — the diagnostic tail stays bounded ══════════════════════════════

def test_c1_generic_tail_is_still_the_predecessor_window():
    """Classification is not paid for by showing the user more chatter."""
    lines = _buried(trailing=200)
    rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)
    generic = [ln for ln in message.splitlines()
               if not _classified_no_space(ln)]
    assert generic == lines[-RETAINED_TAIL_LINES:]


def test_c2_buried_signal_does_not_inflate_the_message():
    lines = _buried(trailing=200)
    _rc, message = _run_scripted(lines, 1)

    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 1


def test_c3_in_tail_signal_is_not_duplicated():
    lines = _chatter(3) + [NO_SPACE_LINE] + _chatter(3, start=3)
    _rc, message = _run_scripted(lines, 1)

    assert message.lower().count("no space left on device") == 1
    assert len(message.splitlines()) == RETAINED_TAIL_LINES


# ══ GROUP D — the committed vocabulary, and only that ════════════════════════

@pytest.mark.parametrize("signal", [NO_SPACE_LINE, ENOSPC_LINE])
def test_d1_every_committed_signal_survives_burial(signal):
    rc, message = _run_scripted(_buried(signal), 1)

    assert rc == 1
    assert _classified_no_space(message)


@pytest.mark.parametrize("signal", UNRECOGNIZED_LINES)
def test_d2_unrecognized_phrasings_stay_ordinary_failures(signal):
    """Tab 11 widens the horizon, never the vocabulary."""
    rc, message = _run_scripted(_buried(signal), 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_d3_vocabulary_is_still_exactly_two_signals():
    assert compressor.NO_SPACE_SIGNALS == (
        "no space left on device", "enospc")


# ══ GROUP E — every container ════════════════════════════════════════════════

@pytest.mark.parametrize("fmt,muxer", [
    ("MP4 (H.264)", "mp4"),
    ("MP4 (H.265)", "mp4"),
    ("MKV (H.265)", "matroska"),
    ("WebM (VP9)", "webm"),
])
def test_e1_buried_no_space_fails_specifically_in_every_container(
        env, fmt, muxer):
    src, out_dir, fake = env
    fake.finals = [(1, _buried())]

    result = _run(src, out_dir, fmt=fmt)

    assert fake.final_muxers == [muxer]
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert src.exists(), "a failed encode never removes the source"
    assert list(out_dir.iterdir()) == [], "the reservation is released"


# ══ GROUP F — buried no-space must not license the MKV fallback ══════════════

def test_f1_buried_no_space_mp4_h265_does_not_fall_back(env):
    """The load-bearing regression: the only thing separating this from
    `test_g1` is a no-space line ffmpeg emitted before the tail begins."""
    src, out_dir, fake = env
    fake.finals = [(1, _buried())]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"], "a full disk cannot be re-muxed away"
    assert len(fake.final_cmds) == 1
    assert not result.get("fallback_used")
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert src.exists()


def test_f2_buried_no_space_adds_no_subprocess(env):
    src, out_dir, fake = env
    fake.finals = [(1, _buried(trailing=200))]

    _run(src, out_dir)

    assert fake.subprocess_count == 1


# ══ GROUP G — the legitimate retry is untouched ══════════════════════════════

def test_g1_ordinary_buried_mux_failure_still_retries_once_as_mkv(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20))]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert Path(result["output"]).suffix == ".mkv"


def test_g2_fallback_is_still_capped_at_two_attempts(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)), (1, _chatter(20, start=20))]

    result = _run(src, out_dir)

    assert len(fake.final_cmds) == 2
    assert result["status"] == "error"


# ══ GROUP H — per-attempt isolation ══════════════════════════════════════════

def test_h1_no_space_in_the_second_attempt_is_classified(env):
    """Attempt 1 is an ordinary failure, so the retry is earned; attempt 2
    then buries a no-space line and must still be reported as one."""
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)), (1, _buried())]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert src.exists()


def test_h2_no_space_in_the_first_attempt_never_reaches_a_second(env):
    src, out_dir, fake = env
    fake.finals = [(1, _buried()), (0, [])]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"


def test_h3_each_attempt_is_classified_from_its_own_stderr(env):
    """Attempt 2's buried signal must not be read back onto attempt 1, nor
    attempt 1's clean stderr forward onto attempt 2."""
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)), (1, _buried())]

    _run(src, out_dir)

    first, second = fake.messages
    assert not _classified_no_space(first)
    assert _classified_no_space(second)


def test_h4_no_space_in_subtitle_extraction_does_not_taint_the_encode(
        env, monkeypatch):
    """The other invocation ordering the same job can produce.

    A sidecar extraction that hit a full disk runs *before* the encode. If its
    verdict carried forward, the encode's ordinary mux failure would read as
    no-space and quietly lose the MKV retry it had earned - so this is the
    attempt-to-attempt leak in the one shape a single job can actually show.
    """
    src, out_dir, fake = env
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(
            subtitles=[{"index": 2, "codec_name": "subrip",
                        "tags": {"language": "eng"}}], audio_count=1))
    fake.subtitle = [(1, _buried())]
    fake.finals = [(1, _chatter(20))]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert len(fake.subtitle_cmds) == 1
    assert _classified_no_space(fake.messages[0]), "extraction really did"
    assert not _classified_no_space(fake.messages[1]), "the encode did not"
    assert fake.final_muxers == ["mp4", "matroska"], "the retry is still earned"
    assert result["status"] == "ok"


# ══ GROUP I — two-pass, pass 1 ═══════════════════════════════════════════════

def test_i1_buried_no_space_in_pass_1_stops_before_pass_2(env):
    src, out_dir, fake = env
    fake.pass1 = [(1, _buried())]

    result = _two_pass(src, out_dir)

    assert len(fake.pass1_cmds) == 1
    assert fake.final_cmds == [], "pass 2 must not run against a full disk"
    assert fake.subprocess_count == 1
    assert result["status"] == "error"
    assert "pass 1 failed" in result["msg"]
    assert _classified_no_space(result["msg"])
    assert src.exists()


# ══ GROUP J — two-pass, pass 2 ═══════════════════════════════════════════════

def test_j1_buried_no_space_in_pass_2_stays_specific(env):
    src, out_dir, fake = env
    fake.finals = [(1, _buried())]

    result = _two_pass(src, out_dir)

    assert len(fake.pass1_cmds) == 1
    assert len(fake.final_cmds) == 1
    assert fake.subprocess_count == 2
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert src.exists()


def test_j2_pass_2_no_space_is_not_rewritten_as_a_missing_output(env):
    """Tab 10's artifact gate answers a *successful* exit that produced
    nothing. A nonzero no-space exit keeps its own, more specific verdict."""
    src, out_dir, fake = env
    fake.finals = [(1, _buried())]

    result = _two_pass(src, out_dir)

    assert "no valid output file" not in result["msg"]
    assert "no output file produced" not in result["msg"]
    assert _classified_no_space(result["msg"])


# ══ GROUP K — precedence is unchanged ════════════════════════════════════════

def test_k1_cancellation_still_outranks_a_no_space_line():
    cancel = threading.Event()
    cancel.set()

    rc, message = _run_scripted(_buried(), 1, cancel_flag=cancel, sleep=2)

    assert (rc, message) == (-2, "cancelled")


def test_k1b_cancelled_job_reports_cancellation_not_disk_space(env):
    src, out_dir, fake = env
    cancel = threading.Event()
    cancel.set()
    fake.finals = [(1, _buried())]

    result = _run(src, out_dir, cancel_flag=cancel)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"


def test_k2_stall_timeout_still_outranks_a_no_space_line():
    with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 0.05):
        rc, message = _run_scripted(_buried(), 1, sleep=10)

    assert rc == -3
    assert "no encoding progress" in message
    assert not _classified_no_space(message)


def test_k3_nonzero_exit_with_a_buried_signal_is_the_no_space_verdict(env):
    src, out_dir, fake = env
    fake.finals = [(1, _buried())]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "error"
    assert result["msg"].startswith("ffmpeg failed: ")
    assert _classified_no_space(result["msg"])


def test_k4_successful_exit_with_no_output_keeps_its_own_error(env):
    """rc == 0 never reaches the no-space verdict, buried line or not."""
    src, out_dir, fake = env

    def no_output(cmd, cancel_flag, duration=None, on_progress=None,
                  on_start=None):
        fake.final_cmds.append(list(cmd))
        with _scripted_process(_buried(), 0) as scripted:
            return _REAL_RUN_FFMPEG(scripted, cancel_flag)

    with patch.object(compressor, "run_ffmpeg", no_output):
        result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "error"
    assert result["msg"] == "no output file produced"


# ══ GROUP L — success is untouched ═══════════════════════════════════════════

def test_l1_warning_chatter_on_a_successful_encode_still_succeeds(env):
    src, out_dir, fake = env
    fake.finals = [(0, _chatter(200))]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).exists()
    assert Path(result["output"]).stat().st_size > 0
    assert fake.subprocess_count == 1


def test_l2_a_successful_run_returns_the_ordinary_tail():
    lines = _chatter(20)
    rc, message = _run_scripted(lines, 0)

    assert rc == 0
    assert message.splitlines() == lines[-RETAINED_TAIL_LINES:]


def test_l4_a_zero_exit_is_never_rewritten_into_a_disk_full_report():
    """Classification answers a failure. A warning ffmpeg shrugged off and
    exited 0 on must not be promoted into one."""
    lines = _buried()
    rc, message = _run_scripted(lines, 0)

    assert rc == 0
    assert message.splitlines() == lines[-RETAINED_TAIL_LINES:]
    assert not _classified_no_space(message)


def test_l3_no_space_wording_in_the_command_never_classifies(env, tmp_path):
    """Classification reads ffmpeg's diagnostics, not the job's arguments.

    The output path here carries the phrase verbatim and the encode fails for
    an unrelated reason; the command Cove built is not evidence about disks.
    """
    src, out_dir, fake = env
    named = tmp_path / "No space left on device.mov"
    named.write_bytes(b"s" * SRC_BYTES)
    fake.finals = [(1, _chatter(20)), (1, _chatter(20, start=20))]

    result = _run(named, out_dir)

    assert any("No space left on device" in str(a)
               for a in fake.final_cmds[0]), "the phrase really is in the cmd"
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert fake.final_muxers == ["mp4", "matroska"], "an ordinary retry"


# ══ GROUP M — per-file isolation ═════════════════════════════════════════════

def test_m1_no_space_state_never_leaks_between_files(env, tmp_path):
    src, out_dir, fake = env
    sources = []
    for name in ("A", "B", "C", "D"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)

    fake.finals = [
        (1, _buried()),          # A - buried no-space
        (1, _chatter(20)),       # B - ordinary MP4 failure ...
        (0, _chatter(20)),       # ... and its earned MKV retry
        (0, _chatter(20)),       # C - clean success
        (1, _buried()),          # D - buried no-space again
    ]
    results = [_run(p, out_dir, fmt="MP4 (H.265)") for p in sources]

    a, b, c, d = results
    assert a["status"] == "error" and _classified_no_space(a["msg"])
    assert b["status"] == "ok" and b["fallback_used"] is True
    assert c["status"] == "ok" and not c.get("fallback_used")
    assert d["status"] == "error" and _classified_no_space(d["msg"])
    assert fake.final_muxers == ["mp4", "mp4", "matroska", "mp4", "mp4"]


# ══ GROUP N — retention stays bounded ════════════════════════════════════════

def test_n1_retained_window_never_exceeds_the_predecessor_bound():
    """The stderr window is a fixed-size deque, not a transcript."""
    windows: list[int] = []
    real_deque = compressor.deque

    def recording(*args, **kwargs):
        d = real_deque(*args, **kwargs)
        if kwargs.get("maxlen") is not None or (
                len(args) > 1 and args[1] is not None):
            windows.append(d.maxlen)
        return d

    with patch.object(compressor, "deque", recording):
        _rc, message = _run_scripted([NO_SPACE_LINE] + _chatter(2000), 1)

    assert windows and all(w == STDERR_WINDOW_LINES for w in windows)
    assert _classified_no_space(message)
    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 1


def test_n2_thousands_of_lines_do_not_grow_the_message():
    _rc, small = _run_scripted([NO_SPACE_LINE] + _chatter(20), 1)
    _rc, large = _run_scripted([NO_SPACE_LINE] + _chatter(2000), 1)

    assert len(small.splitlines()) == len(large.splitlines())
    assert _classified_no_space(small) and _classified_no_space(large)


# ══ GROUP O — command construction is untouched ══════════════════════════════

@pytest.mark.parametrize("fmt,muxer,maps", [
    ("MP4 (H.265)", "mp4", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
    ("MKV (H.265)", "matroska",
     ["-map", "0:v:0", "-map", "0:a?", "-sn", "-map", "0:t?", "-c:t", "copy"]),
    ("WebM (VP9)", "webm", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
])
def test_o1_stream_mapping_is_unchanged(env, fmt, muxer, maps):
    src, out_dir, fake = env
    fake.finals = [(0, _chatter(20))]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    cmd = fake.final_cmds[0]
    assert _muxer_of(cmd) == muxer
    joined = " ".join(cmd)
    assert " ".join(maps) in joined


def test_o2_two_pass_analysis_command_is_unchanged(env):
    src, out_dir, fake = env

    result = _two_pass(src, out_dir)

    assert result["status"] == "ok"
    cmd = fake.pass1_cmds[0]
    assert " ".join(["-map", "0:v:0"]) in " ".join(cmd)
    assert _muxer_of(cmd) == "null"
    assert "-an" in cmd and os.devnull in cmd
