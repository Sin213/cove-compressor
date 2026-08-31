"""A filename is not a diagnostic: ffmpeg's path echoes must not read as ENOSPC.

Tab 11 made a "No space left on device" line count no matter where in ffmpeg's
stderr it appeared, which closed a real hole and opened a smaller one. ffmpeg
announces every file it touches by echoing the user's own path:

    Input #0, matroska,webm, from 'C:\\Videos\\No space left on device.mkv':
    Output #0, mp4, to 'C:\\Output\\ENOSPC.mp4':

Those are declarations, not complaints. A user is entitled to name a file
anything, and when one of those names happens to contain a phrase the disk-full
classifier recognizes, an unrelated failure was being reported as a full disk -
and, worse, the legitimate MP4 (H.265) -> MKV retry that failure had earned was
being refused on the strength of a filename.

Locked down here:

  A. An Output path echo carrying either signal, still inside the returned
     tail, stays an ordinary failure.
  B. The same echo pushed out of the tail - so Tab 11's latch is what sees it -
     also stays an ordinary failure.
  C. The same, for Input path echoes.
  D. Genuine diagnostics still classify: in the tail, buried far past it, and
     when the diagnostic line happens to mention a path of its own.
  E. A false path echo never suppresses a genuine diagnostic that follows it,
     and never unsays one that preceded it.
  F. Only the announcement *shape* is excluded. Ordinary lines that merely
     begin with "Input"/"Output" still classify.
  G. Announcement indexes are not assumed to be #0.
  H. A false path echo no longer costs MP4 (H.265) its one MKV retry.
  I. A genuine diagnostic still refuses that retry, signal-bearing filename or
     not.
  J. Attempt state stays isolated across the fallback pair.
  K. Every direct container behaves the same way.
  L. Two-pass classification is unchanged on either pass.
  M. Signal-bearing filenames still convert successfully. This is a
     classification fix, not a filename policy.
  N. Tab 11's buried-signal guarantee is intact.
  O. Retention stays bounded; nothing here is fixed by keeping more stderr.
  P. No state leaks between files.
  Q. Command construction is untouched.

The header lines used throughout are byte-for-byte the shapes emitted by the
ffmpeg 8.1.1 build this project ships against, captured from real runs over
disposable files deliberately named with the recognized phrases.

Like `tests/test_enospc_classification.py`, and unlike the fakes in
`tests/test_mkv_fallback.py`, this drives the *real* `run_ffmpeg` over a real
subprocess emitting real scripted stderr. Only the encoder is stood in for.
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

_REAL_RUN_FFMPEG = compressor.run_ffmpeg

# The predecessor's bounds. Tab 12 moves neither.
RETAINED_TAIL_LINES = 5
STDERR_WINDOW_LINES = 40

# ── real ffmpeg 8.1.1 announcement lines ─────────────────────────────────────
#
# Captured, not remembered: each of these is a verbatim line from an ffmpeg run
# over a disposable file named with the phrase it carries.
OUT_HDR_NO_SPACE = (
    r"Output #0, mp4, to 'C:\Output\No space left on device.mp4':")
OUT_HDR_ENOSPC = r"Output #0, mp4, to 'C:\Output\ENOSPC.mp4':"
IN_HDR_NO_SPACE = (
    r"Input #0, matroska,webm, from 'C:\Videos\No space left on device.mkv':")
IN_HDR_ENOSPC = r"Input #0, matroska,webm, from 'C:\Videos\ENOSPC.mkv':"
# An announcement with nothing to say about disks - the control that proves the
# exclusion is about provenance, not about headers.
OUT_HDR_CLEAN = r"Output #0, mp4, to 'C:\Output\Movie.mp4':"

PATH_ECHOES = [OUT_HDR_NO_SPACE, OUT_HDR_ENOSPC,
               IN_HDR_NO_SPACE, IN_HDR_ENOSPC]

# Nonzero indexes, also captured from a real multi-output run.
OUT_HDR_INDEX_2 = r"Output #2, mp4, to 'C:\Output\No space left on device.mp4':"
IN_HDR_INDEX_1 = r"Input #1, matroska,webm, from 'C:\Videos\ENOSPC.mkv':"

# ── genuine diagnostics ──────────────────────────────────────────────────────
NO_SPACE_LINE = "av_interleaved_write_frame(): No space left on device"
ENOSPC_LINE = "Error writing trailer: ENOSPC"
NO_SPACE_WITH_PATH = (
    r"Error writing trailer of C:\Output\video.mp4: No space left on device")

# Lines that merely start with the announcement keywords. These are ordinary
# prose and must keep classifying.
LOOKALIKES = [
    "Output failed: ENOSPC",
    "Input write failed: No space left on device",
    "Output #0 aborted: No space left on device",
    "Outputting frame failed: ENOSPC",
]

GENERIC_MUX_FAILURE = [
    "  Metadata:",
    "    encoder         : Lavf62.6.100",
    "Could not write header for output file #0: Invalid argument",
    "Conversion failed!",
]


def _chatter(count: int, start: int = 0) -> list[str]:
    return [f"[hevc @ 0000021f] generic encoder chatter line {i}"
            for i in range(start, start + count)]


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
    fd, payload = tempfile.mkstemp(prefix="cove_pathecho_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(line + "\n" for line in lines)
        yield [sys.executable, "-c", _EMITTER, payload, str(code), str(sleep)]
    finally:
        os.unlink(payload)


def _run_scripted(lines, code: int, cancel_flag=None, sleep: float = 0.0):
    with _scripted_process(lines, code, sleep) as cmd:
        return _REAL_RUN_FFMPEG(
            cmd, cancel_flag if cancel_flag is not None else threading.Event())


def _classified_no_space(message: str) -> bool:
    """The one committed predicate, asked the one committed way."""
    return compressor._is_no_space_failure(message)


def _in_tail(header: str) -> list[str]:
    """A failure short enough that the echo is still in the returned tail."""
    return [header] + GENERIC_MUX_FAILURE


def _out_of_tail(header: str) -> list[str]:
    """A failure long enough that only Tab 11's latch can still see the echo."""
    return [header] + _chatter(30) + GENERIC_MUX_FAILURE


# ── compress_video-level harness ─────────────────────────────────────────────

SRC_BYTES = 4 * 1024 * 1024


def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


class ScriptedStderrFfmpeg:
    """Stands in for the encoder, not for the stderr reader."""

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
    return _run(src, out_dir, mode="Target file size", mode_value="1", **kw)


# ══ GROUP A — Output path echo inside the returned tail ══════════════════════

def test_a1_output_path_echo_of_the_phrase_in_the_tail_is_not_no_space():
    """The original caveat, at its shortest: the echo is the only signal."""
    _rc, message = _run_scripted(_in_tail(OUT_HDR_NO_SPACE), 1)

    assert OUT_HDR_NO_SPACE in message, "the echo really is in the tail"
    assert not _classified_no_space(message)


def test_a2_output_path_echo_of_enospc_in_the_tail_is_not_no_space():
    _rc, message = _run_scripted(_in_tail(OUT_HDR_ENOSPC), 1)

    assert OUT_HDR_ENOSPC in message
    assert not _classified_no_space(message)


def test_a3_an_ordinary_output_announcement_is_still_an_ordinary_failure():
    """The control: excluding announcements changes nothing for clean ones."""
    _rc, message = _run_scripted(_in_tail(OUT_HDR_CLEAN), 1)

    assert not _classified_no_space(message)


# ══ GROUP B — Output path echo beyond the tail (Tab 11's latch) ══════════════

@pytest.mark.parametrize("header", [OUT_HDR_NO_SPACE, OUT_HDR_ENOSPC])
def test_b1_output_path_echo_beyond_the_tail_is_not_latched(header):
    """Only the latch can still see this line, and it must not take it."""
    _rc, message = _run_scripted(_out_of_tail(header), 1)

    assert header not in message, "the echo scrolled out, as intended"
    assert not _classified_no_space(message)


@pytest.mark.parametrize("header", [OUT_HDR_NO_SPACE, OUT_HDR_ENOSPC])
def test_b2_a_latched_path_echo_never_reaches_the_returned_message(header):
    """A false echo must not be resurrected in front of the tail either."""
    _rc, message = _run_scripted(_out_of_tail(header), 1)

    assert len(message.splitlines()) == RETAINED_TAIL_LINES


# ══ GROUP C — Input path echoes ══════════════════════════════════════════════

@pytest.mark.parametrize("header", [IN_HDR_NO_SPACE, IN_HDR_ENOSPC])
def test_c1_input_path_echo_in_the_tail_is_not_no_space(header):
    """A source the user named unfortunately cannot poison its own encode."""
    _rc, message = _run_scripted(_in_tail(header), 1)

    assert header in message
    assert not _classified_no_space(message)


@pytest.mark.parametrize("header", [IN_HDR_NO_SPACE, IN_HDR_ENOSPC])
def test_c2_input_path_echo_beyond_the_tail_is_not_latched(header):
    _rc, message = _run_scripted(_out_of_tail(header), 1)

    assert not _classified_no_space(message)


# ══ GROUP D — genuine diagnostics still classify ═════════════════════════════

def test_d1_genuine_no_space_in_the_tail_is_still_classified():
    _rc, message = _run_scripted(_chatter(3) + [NO_SPACE_LINE], 1)

    assert _classified_no_space(message)


def test_d2_genuine_no_space_beyond_the_tail_is_still_latched():
    """Tab 11's whole point, restated so Tab 12 cannot quietly undo it."""
    _rc, message = _run_scripted([NO_SPACE_LINE] + _chatter(200), 1)

    assert _classified_no_space(message)


def test_d3_the_enospc_spelling_still_classifies_as_a_substring():
    _rc, message = _run_scripted(["Error while encoding: ENOSPC"]
                                 + _chatter(200), 1)

    assert _classified_no_space(message)


def test_d4_a_genuine_diagnostic_that_mentions_a_path_still_classifies():
    """The exclusion is for announcements, not for any line with a path in it."""
    _rc, message = _run_scripted([NO_SPACE_WITH_PATH] + _chatter(200), 1)

    assert _classified_no_space(message)


# ══ GROUP E — a false echo never overrides a genuine diagnostic ══════════════

def test_e1_genuine_no_space_after_a_false_echo_still_classifies():
    lines = ([OUT_HDR_NO_SPACE] + _chatter(5)
             + [NO_SPACE_LINE] + _chatter(30, start=5))
    _rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)


def test_e2_a_later_false_echo_never_clears_an_earlier_genuine_latch():
    lines = ([NO_SPACE_LINE] + _chatter(5)
             + [OUT_HDR_ENOSPC] + _chatter(30, start=5))
    _rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)


# ══ GROUP F — only the announcement shape is excluded ════════════════════════

@pytest.mark.parametrize("line", LOOKALIKES)
def test_f1_lines_that_merely_begin_with_the_keywords_still_classify(line):
    """"Output"/"Input" are not the signal; the announcement grammar is."""
    _rc, message = _run_scripted([line] + _chatter(200), 1)

    assert _classified_no_space(message), line


# ══ GROUP G — announcement indexes are not assumed to be #0 ══════════════════

@pytest.mark.parametrize("header", [OUT_HDR_INDEX_2, IN_HDR_INDEX_1])
def test_g1_nonzero_announcement_indexes_are_excluded_too(header):
    _rc, message = _run_scripted(_in_tail(header), 1)

    assert not _classified_no_space(message)


def test_g2_several_announcements_at_once_are_all_excluded():
    """A real multi-input, multi-output run announces every file it touches."""
    lines = ([IN_HDR_NO_SPACE, IN_HDR_INDEX_1,
              OUT_HDR_ENOSPC, OUT_HDR_INDEX_2]
             + GENERIC_MUX_FAILURE)
    _rc, message = _run_scripted(lines, 1)

    assert not _classified_no_space(message)


# ══ GROUP H — the retry a false echo used to cost ════════════════════════════

def test_h1_a_false_output_echo_no_longer_refuses_the_mkv_retry(env):
    """The consequence the whole slice exists for."""
    src, out_dir, fake = env
    fake.finals = [(1, _in_tail(OUT_HDR_NO_SPACE)),
                   (0, _chatter(5))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"], "the retry was earned"
    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert fake.subprocess_count == 2


def test_h2_a_false_input_echo_no_longer_refuses_the_mkv_retry(env):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(IN_HDR_ENOSPC)),
                   (0, _chatter(5))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "ok"
    assert result["fallback_used"] is True


# ══ GROUP I — a genuine diagnostic still refuses it ══════════════════════════

def test_i1_genuine_no_space_still_refuses_the_retry_despite_the_echo(env):
    """Same unfortunate filename, but ffmpeg really did run out of room."""
    src, out_dir, fake = env
    fake.finals = [(1, [OUT_HDR_NO_SPACE] + _chatter(5)
                    + [NO_SPACE_LINE] + _chatter(30, start=5))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4"], "no second encode onto a full disk"
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert fake.subprocess_count == 1
    assert src.exists()


# ══ GROUP J — the fallback pair stays isolated ═══════════════════════════════

def test_j1_a_false_echo_in_the_mkv_attempt_stays_an_ordinary_failure(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)),
                   (1, _in_tail(OUT_HDR_ENOSPC))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert fake.subprocess_count == 2, "no third attempt"


def test_j2_genuine_no_space_in_the_mkv_attempt_is_classified(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)),
                   (1, [OUT_HDR_ENOSPC] + [NO_SPACE_LINE] + _chatter(30))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert fake.subprocess_count == 2


def test_j3_a_tainted_extraction_never_reaches_an_echoing_encode(
        env, monkeypatch):
    """The one intra-job ordering where a verdict could actually carry forward.

    The sidecar extraction really did hit a full disk, and it runs before the
    encode. The encode then fails for an ordinary reason while announcing an
    output path that merely reads like one. Neither the genuine earlier verdict
    nor the later echo may reach the encode's own classification, or the retry
    it earned disappears.
    """
    src, out_dir, fake = env
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(
            subtitles=[{"index": 2, "codec_name": "subrip",
                        "tags": {"language": "eng"}}], audio_count=1))
    fake.subtitle = [(1, [NO_SPACE_LINE] + _chatter(30))]
    fake.finals = [(1, _in_tail(OUT_HDR_NO_SPACE)), (0, _chatter(5))]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert _classified_no_space(fake.messages[0]), "extraction really did"
    assert not _classified_no_space(fake.messages[1]), "the encode did not"
    assert fake.final_muxers == ["mp4", "matroska"], "the retry is still earned"
    assert result["status"] == "ok"


# ══ GROUP K — every direct container ═════════════════════════════════════════

@pytest.mark.parametrize("fmt,muxer", [
    ("MP4 (H.264)", "mp4"),
    ("MKV (H.265)", "matroska"),
    ("WebM (VP9)", "webm"),
])
@pytest.mark.parametrize("header", [OUT_HDR_NO_SPACE, IN_HDR_ENOSPC])
def test_k1_a_false_echo_is_an_ordinary_failure_in_every_container(
        env, fmt, muxer, header):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(header))]

    result = _run(src, out_dir, fmt=fmt)

    assert fake.final_muxers == [muxer]
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert src.exists(), "a failed encode never removes the source"
    assert list(out_dir.iterdir()) == [], "the reservation is released"


# ══ GROUP L — two-pass ═══════════════════════════════════════════════════════

def test_l1_a_false_echo_in_pass_1_is_an_ordinary_failure(env):
    src, out_dir, fake = env
    fake.pass1 = [(1, _in_tail(OUT_HDR_NO_SPACE))]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert fake.subprocess_count == 1, "pass 2 never started"


def test_l2_genuine_no_space_in_pass_1_is_still_classified(env):
    src, out_dir, fake = env
    fake.pass1 = [(1, [NO_SPACE_LINE] + _chatter(30))]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert fake.subprocess_count == 1


def test_l3_a_false_echo_in_pass_2_is_an_ordinary_failure(env):
    """MKV, so the pass count is the only thing the assertion is reading: an
    ordinary MP4 (H.265) pass-2 failure would legitimately earn a retry, which
    group H already covers."""
    src, out_dir, fake = env
    fake.pass1 = [(0, _chatter(5))]
    fake.finals = [(1, _out_of_tail(OUT_HDR_ENOSPC))]

    result = _two_pass(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert fake.subprocess_count == 2


def test_l4_genuine_no_space_in_pass_2_is_still_classified(env):
    src, out_dir, fake = env
    fake.pass1 = [(0, _chatter(5))]
    fake.finals = [(1, [OUT_HDR_ENOSPC, NO_SPACE_LINE] + _chatter(30))]

    result = _two_pass(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert fake.subprocess_count == 2


# ══ GROUP M — signal-bearing filenames are legal ═════════════════════════════

def test_m1_a_source_named_with_the_phrase_still_converts(env, tmp_path):
    src, out_dir, fake = env
    named = tmp_path / "No space left on device.mov"
    named.write_bytes(b"s" * SRC_BYTES)
    fake.finals = [(0, _in_tail(IN_HDR_NO_SPACE))]

    result = _run(named, out_dir)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.exists() and out.stat().st_size > 0
    assert out.stem == "No space left on device", "no renaming, no rejection"


def test_m2_an_output_named_with_enospc_still_converts(env, tmp_path):
    src, out_dir, fake = env
    named = tmp_path / "ENOSPC.mov"
    named.write_bytes(b"s" * SRC_BYTES)
    fake.finals = [(0, _in_tail(OUT_HDR_ENOSPC))]

    result = _run(named, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).exists()
    assert "ENOSPC" in Path(result["output"]).name


# ══ GROUP N — Tab 11 non-regression ══════════════════════════════════════════

def test_n1_a_buried_genuine_signal_is_still_classified(env):
    """Tab 11's core scenario, re-driven through the public path."""
    src, out_dir, fake = env
    fake.finals = [(1, [NO_SPACE_LINE] + _chatter(600))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert len(result["msg"].splitlines()) <= RETAINED_TAIL_LINES + 1


def test_n2_a_buried_enospc_spelling_is_still_classified():
    _rc, message = _run_scripted([ENOSPC_LINE] + _chatter(600), 1)

    assert _classified_no_space(message)


# ══ GROUP O — retention stays bounded ════════════════════════════════════════

@pytest.mark.parametrize("lines", [
    [OUT_HDR_NO_SPACE] + _chatter(2000),
    [NO_SPACE_LINE] + _chatter(2000),
    _chatter(2000),
])
def test_o1_the_retained_window_is_still_the_predecessor_bound(lines):
    windows: list[int] = []
    real_deque = compressor.deque

    def recording(*args, **kwargs):
        d = real_deque(*args, **kwargs)
        if kwargs.get("maxlen") is not None or (
                len(args) > 1 and args[1] is not None):
            windows.append(d.maxlen)
        return d

    with patch.object(compressor, "deque", recording):
        _rc, message = _run_scripted(lines, 1)

    assert windows and all(w == STDERR_WINDOW_LINES for w in windows)
    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 1


def test_o2_a_false_echo_returns_exactly_the_predecessor_tail():
    """Nothing is prepended for an echo, so the tail keeps its own size."""
    _rc, message = _run_scripted(_out_of_tail(OUT_HDR_NO_SPACE), 1)

    assert len(message.splitlines()) == RETAINED_TAIL_LINES


# ══ GROUP P — per-file isolation ═════════════════════════════════════════════

def test_p1_classification_never_leaks_between_files(env, tmp_path):
    src, out_dir, fake = env
    sources = []
    for name in ("A", "B", "C", "D"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)

    fake.finals = [
        (1, _out_of_tail(OUT_HDR_NO_SPACE)),  # A - echo only, MP4 ...
        (1, _chatter(20)),                    # ... and its earned MKV retry
        (1, [NO_SPACE_LINE] + _chatter(30)),  # B - genuine, buried
        (0, _chatter(20)),                    # C - clean success
        (1, _in_tail(IN_HDR_ENOSPC)),         # D - echo only, MP4 ...
        (0, _chatter(20)),                    # ... and its earned MKV retry
    ]
    a, b, c, d = [_run(p, out_dir, fmt="MP4 (H.265)") for p in sources]

    assert a["status"] == "error" and not _classified_no_space(a["msg"])
    assert b["status"] == "error" and _classified_no_space(b["msg"])
    assert c["status"] == "ok" and not c.get("fallback_used")
    assert d["status"] == "ok" and d["fallback_used"] is True
    assert fake.final_muxers == ["mp4", "matroska", "mp4", "mp4",
                                "mp4", "matroska"]


# ══ GROUP Q — command construction is untouched ══════════════════════════════

@pytest.mark.parametrize("fmt,muxer,maps", [
    ("MP4 (H.265)", "mp4", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
    ("MKV (H.265)", "matroska",
     ["-map", "0:v:0", "-map", "0:a?", "-sn", "-map", "0:t?", "-c:t", "copy"]),
    ("WebM (VP9)", "webm", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
])
def test_q1_stream_mapping_is_unchanged_under_a_false_echo(
        env, fmt, muxer, maps):
    src, out_dir, fake = env
    fake.finals = [(0, _in_tail(OUT_HDR_NO_SPACE))]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    cmd = fake.final_cmds[0]
    assert _muxer_of(cmd) == muxer
    joined = " ".join(str(a) for a in cmd)
    assert " ".join(maps) in joined


def test_q2_the_vocabulary_is_still_exactly_two_signals():
    assert compressor.NO_SPACE_SIGNALS == (
        "no space left on device", "enospc")
