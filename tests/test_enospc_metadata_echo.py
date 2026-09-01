"""A tag is not a diagnostic: ffmpeg's metadata echoes must not read as ENOSPC.

Tab 12 stopped ffmpeg's *path* announcements from being read as disk-full
diagnostics, and while proving that fix over real media it surfaced a second,
independent source of the same false positive. ffmpeg opens its report by
echoing the container's metadata back to the user:

    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'signal.mp4':
      Metadata:
        title           : No space left on device
        comment         : ENOSPC

Those are the user's own tags, quoted. They describe the movie, not the
filesystem, and a user is as entitled to a title as to a filename. Read as
diagnostics they turned an unrelated mux failure into a full disk - and with it
cost that failure the single MP4 (H.265) -> MKV retry it had earned.

Locked down here:

  A. A container metadata value carrying either signal, still inside the
     returned tail, stays an ordinary failure - including when the `Metadata:`
     header that gave it its provenance has itself scrolled out of that tail.
  B. The same value pushed clear of the tail - so Tab 11's latch is what sees
     it - also stays an ordinary failure.
  C. Provenance is the block, not the key: an arbitrary user-defined tag is
     excluded exactly as `title` and `comment` are.
  D. Genuine diagnostics still classify: in the tail, buried far past it,
     either spelling, and when the diagnostic itself mentions metadata.
  E. A block that has ended stops excluding: a real diagnostic immediately
     after one still classifies.
  F. A false echo never unsays a genuine diagnostic that preceded it.
  G. Input and Output blocks in one run are both excluded.
  H. Every real terminator ffmpeg emits closes the block.
  I. Only metadata provenance excludes. Indented, colon-shaped diagnostics
     outside a block still classify.
  J. A false echo no longer costs MP4 (H.265) its one MKV retry.
  K. A genuine diagnostic still refuses that retry, signal-bearing tags or not.
  L. Attempt state stays isolated across the fallback pair.
  M. Every direct container behaves the same way.
  N. Two-pass classification is unchanged on either pass.
  O. Tab 12's path-echo exclusion is intact.
  P. Tab 11's buried-signal guarantee is intact.
  Q. Signal-bearing tags still convert successfully. This is a classification
     fix, not a metadata policy.
  R. No state leaks between attempts or between files.
  S. Retention stays bounded; nothing here is fixed by keeping more stderr.
  T. Command construction is untouched.

Every metadata line used below is byte-for-byte a line emitted by the ffmpeg
8.1.1 build this project ships against, captured from real runs over disposable
media deliberately tagged with the recognized phrases:

    ffmpeg -i signal.mp4 -c:v libx265 -c:a aac -t 1 out.mkv

    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'signal.mp4':
      Metadata:
        major_brand     : isom
        minor_version   : 512
        compatible_brands: isomiso2avc1mp41
        title           : No space left on device
        encoder         : Lavf62.12.101
        comment         : ENOSPC
        description     : Testing No space left on device behavior
      Duration: 00:00:02.00, start: 0.000000, bitrate: 122 kb/s
      Stream #0:0[0x1](und): Video: h264 ...
        Metadata:
          handler_name    : VideoHandler

Like `tests/test_enospc_path_echo.py`, this drives the *real* `run_ffmpeg` over
a real subprocess emitting real scripted stderr. Only the encoder is stood in
for.
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

# The predecessor's bounds. Tab 13 moves neither.
RETAINED_TAIL_LINES = 5
STDERR_WINDOW_LINES = 40

# ── real ffmpeg 8.1.1 structural lines ───────────────────────────────────────
IN_HDR = r"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'signal.mp4':"
OUT_HDR = r"Output #0, matroska, to 'out.mkv':"
CONTAINER_META = "  Metadata:"
STREAM_META = "    Metadata:"
CHAPTER_META = "      Metadata:"

DURATION = "  Duration: 00:00:02.00, start: 0.000000, bitrate: 122 kb/s"
STREAM_LINE = ("  Stream #0:0[0x1](und): Video: h264 (High 4:4:4 Predictive) "
               "(avc1 / 0x31637661), yuv444p(progressive), 320x240 "
               "[SAR 1:1 DAR 4:3], 42 kb/s, 10 fps, 10 tbr, 10240 tbn "
               "(default)")
SIDE_DATA = "    Side data:"
CHAPTERS = "  Chapters:"
CHAPTER_LINE = "    Chapter #0:0: start 0.000000, end 1.000000"
STREAM_MAPPING = "Stream mapping:"

# ── real ffmpeg 8.1.1 metadata value lines ───────────────────────────────────
#
# Captured, not remembered. `compatible_brands` is kept because it is the one
# real key long enough to collapse the padding and sit flush against its colon.
M_MAJOR = "    major_brand     : isom"
M_MINOR = "    minor_version   : 512"
M_BRANDS = "    compatible_brands: isomiso2avc1mp41"
M_TITLE_NO_SPACE = "    title           : No space left on device"
M_ENCODER = "    encoder         : Lavf62.12.101"
M_COMMENT_ENOSPC = "    comment         : ENOSPC"
M_DESCRIPTION = "    description     : Testing No space left on device behavior"
# From an MKV re-read, where ffmpeg preserves arbitrary user-defined tags.
M_ARBITRARY_ENOSPC = "    COVE_TEST_TAG   : ENOSPC"
# Stream- and chapter-level blocks, also carrying user-controlled values.
M_STREAM_TITLE_ENOSPC = "      title           : ENOSPC"
M_STREAM_HANDLER = "      handler_name    : VideoHandler"
M_CHAPTER_TITLE_NO_SPACE = "        title           : No space left on device"

CONTAINER_VALUES = [M_TITLE_NO_SPACE, M_COMMENT_ENOSPC, M_ARBITRARY_ENOSPC,
                    M_DESCRIPTION]

# A key may contain a colon. Matroska normalizes whitespace in a tag name to
# an underscore (`display name` is echoed as `DISPLAY_NAME`) but leaves a colon
# alone, so this is a real shape a user can put in a real file:
#
#     ffmpeg -i in.mp4 -i meta.txt -map_metadata 1 -c copy out.mkv
#     ;FFMETADATA1
#     weird:key=ENOSPC
M_COLON_KEY_ENOSPC = "    WEIRD:KEY       : ENOSPC"

# A tag value containing newlines is echoed across continuation lines, whose
# key column is blank. Captured from a real run over a file tagged with
# `-metadata "comment=first line<LF>No space left on device<LF>third line"`.
M_MULTILINE_HEAD = "    COMMENT         : first line"
M_MULTILINE_CONT = "                    : No space left on device"
M_MULTILINE_END = "                    : third line"
MULTILINE_VALUE = [M_MULTILINE_HEAD, M_MULTILINE_CONT, M_MULTILINE_END]

# ── genuine diagnostics ──────────────────────────────────────────────────────
NO_SPACE_LINE = "av_interleaved_write_frame(): No space left on device"
ENOSPC_LINE = "Error writing trailer: ENOSPC"
NO_SPACE_TRAILER = "Error writing trailer: No space left on device"
# A real complaint that happens to be *about* metadata. The word must not be
# what excludes anything.
NO_SPACE_ABOUT_METADATA = "Failed to write metadata: No space left on device"

# Colon-shaped, sometimes indented, but not metadata. These are ordinary
# diagnostics and must keep classifying.
LOOKALIKES = [
    "Error: ENOSPC",
    "write_packet: No space left on device",
    "av_interleaved_write_frame(): No space left on device",
    "  Error           : ENOSPC",
    "    write_packet    : No space left on device",
]

GENERIC_MUX_FAILURE = [
    "[out#0/matroska @ 0000020520420b40] Could not write header for output "
    "file #0: Invalid argument",
    "Conversion failed!",
]

# Tab 12's shapes, kept here so a shared-provenance refactor cannot quietly
# reopen the hole it closed.
OUT_PATH_ECHO = r"Output #0, mp4, to 'C:\Output\No space left on device.mp4':"
IN_PATH_ECHO = r"Input #0, matroska,webm, from 'C:\Videos\ENOSPC.mkv':"


def _chatter(count: int, start: int = 0) -> list[str]:
    return [f"[hevc @ 0000021f] generic encoder chatter line {i}"
            for i in range(start, start + count)]


def _input_block(*values) -> list[str]:
    """A real Input announcement and its container metadata block."""
    return [IN_HDR, CONTAINER_META, M_MAJOR, M_MINOR, M_BRANDS, *values,
            M_ENCODER, DURATION]


def _output_block(*values) -> list[str]:
    return [OUT_HDR, CONTAINER_META, M_MAJOR, M_MINOR, M_BRANDS, *values,
            M_ENCODER]


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
    fd, payload = tempfile.mkstemp(prefix="cove_metaecho_", suffix=".txt")
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


def _as_result_msg(message: str) -> str:
    """What `compress_video` hands the classifier for a nonzero exit."""
    return f"ffmpeg failed: {message}"


def _in_tail(*values) -> list[str]:
    """A failure short enough that the block is intact in the returned tail."""
    return [IN_HDR, CONTAINER_META, *values] + GENERIC_MUX_FAILURE


def _tail_truncated(*values) -> list[str]:
    """A failure whose metadata values survive into the tail but whose
    `Metadata:` header does not.

    This is the case the returned five-line tail cannot classify on its own:
    by the time the classifier sees these lines, the only thing that ever said
    they were metadata has scrolled away.
    """
    return _input_block(*values)


def _out_of_tail(*values) -> list[str]:
    """A failure long enough that only Tab 11's latch can still see the echo."""
    return _input_block(*values) + _chatter(30) + GENERIC_MUX_FAILURE


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


# ══ GROUP A — container metadata inside the returned tail ════════════════════

@pytest.mark.parametrize("value", CONTAINER_VALUES)
def test_a1_a_metadata_value_in_the_tail_is_not_a_disk_full_report(value):
    """The block is intact in the tail: provenance is right there to read."""
    rc, message = _run_scripted(_in_tail(value), 1)

    assert rc == 1
    assert value in message
    assert not _classified_no_space(message)
    assert not _classified_no_space(_as_result_msg(message))


@pytest.mark.parametrize("value", CONTAINER_VALUES)
def test_a2_a_metadata_value_outlives_its_header_in_the_tail(value):
    """The header has scrolled out of the five-line tail and the value has not.

    Nothing left in the raw tail says these lines are metadata, so provenance
    has to survive the truncation rather than be re-derived from it.
    """
    rc, message = _run_scripted(_tail_truncated(value), 1)

    lines = message.splitlines()
    assert rc == 1
    assert any(value in line for line in lines)
    assert not _classified_no_space(message)
    assert not _classified_no_space(_as_result_msg(message))


def test_a3_an_output_block_in_the_tail_is_excluded_too():
    rc, message = _run_scripted(
        [OUT_HDR, CONTAINER_META, M_TITLE_NO_SPACE] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert not _classified_no_space(message)


@pytest.mark.parametrize("header,value", [
    (STREAM_META, M_STREAM_TITLE_ENOSPC),
    (CHAPTER_META, M_CHAPTER_TITLE_NO_SPACE),
])
def test_a4_stream_and_chapter_blocks_are_excluded(header, value):
    """Both are real, both carry user-controlled values, both were captured."""
    rc, message = _run_scripted(
        [STREAM_LINE, header, value] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert not _classified_no_space(message)


# ══ GROUP B — metadata clear of the tail, seen only by Tab 11's latch ════════

@pytest.mark.parametrize("value", CONTAINER_VALUES)
def test_b1_a_metadata_value_past_the_tail_never_latches(value):
    rc, message = _run_scripted(_out_of_tail(value), 1)

    assert rc == 1
    assert not _classified_no_space(message)
    assert not _classified_no_space(_as_result_msg(message))
    # Nothing was prepended, so the tail keeps exactly its own size.
    assert len(message.splitlines()) == RETAINED_TAIL_LINES


def test_b2_both_signals_in_one_block_past_the_tail_never_latch():
    rc, message = _run_scripted(
        _out_of_tail(M_TITLE_NO_SPACE, M_COMMENT_ENOSPC), 1)

    assert rc == 1
    assert not _classified_no_space(message)


# ══ GROUP C — provenance is the block, not the key ═══════════════════════════

def test_c1_an_arbitrary_user_tag_is_excluded_like_title():
    """`COVE_TEST_TAG` is not a key ffmpeg knows; it is a key a user wrote.

    Captured from a real MKV re-read, where ffmpeg preserves arbitrary tags.
    """
    rc, message = _run_scripted(_in_tail(M_ARBITRARY_ENOSPC), 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_c2_an_arbitrary_user_tag_past_the_tail_never_latches():
    rc, message = _run_scripted(_out_of_tail(M_ARBITRARY_ENOSPC), 1)

    assert not _classified_no_space(message)


def test_c3_a_signal_buried_in_a_longer_tag_value_is_excluded():
    """Substring matching does not make an echoed value a diagnostic."""
    rc, message = _run_scripted(_in_tail(M_DESCRIPTION), 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_c4_a_key_containing_a_colon_is_still_a_metadata_value():
    """Matroska keeps a colon in a tag name. The padded key column is what
    makes this a value line, not the absence of punctuation in the key."""
    rc, message = _run_scripted(_in_tail(M_COLON_KEY_ENOSPC), 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_c5_a_colon_key_past_the_tail_never_latches():
    rc, message = _run_scripted(_out_of_tail(M_COLON_KEY_ENOSPC), 1)

    assert not _classified_no_space(message)


def test_c6_a_multi_line_value_continuation_is_excluded():
    """A tag value with newlines in it is echoed over several lines, and only
    the first carries the key. The rest are still the user's own text."""
    rc, message = _run_scripted([IN_HDR, CONTAINER_META] + MULTILINE_VALUE
                                + [DURATION] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_c7_a_multi_line_value_continuation_never_latches():
    rc, message = _run_scripted(
        [IN_HDR, CONTAINER_META] + MULTILINE_VALUE + [DURATION]
        + _chatter(30) + GENERIC_MUX_FAILURE, 1)

    assert not _classified_no_space(message)


def test_c8_a_continuation_that_opens_the_tail_keeps_its_provenance():
    """The tail begins on a continuation line: neither the key that named it
    nor the header that framed it is left in view."""
    rc, message = _run_scripted(
        [IN_HDR, CONTAINER_META, M_MAJOR, M_MULTILINE_HEAD, M_MULTILINE_CONT,
         M_MULTILINE_END, M_TITLE_NO_SPACE, M_ENCODER, M_DESCRIPTION], 1)

    assert rc == 1
    assert not _classified_no_space(message)
    assert not _classified_no_space(_as_result_msg(message))


def test_c9_a_deeper_non_value_line_does_not_lose_the_block(env):
    """Provenance follows the open block, not the shape of the line that
    happens to start the tail."""
    rc, message = _run_scripted(
        [IN_HDR, CONTAINER_META, M_MAJOR, "      unlabelled continuation",
         M_TITLE_NO_SPACE, M_ENCODER, M_COMMENT_ENOSPC, M_DESCRIPTION], 1)

    assert not _classified_no_space(message)
    assert not _classified_no_space(_as_result_msg(message))


# ══ GROUP D — genuine diagnostics still classify ═════════════════════════════

@pytest.mark.parametrize("line", [
    NO_SPACE_LINE, ENOSPC_LINE, NO_SPACE_TRAILER, NO_SPACE_ABOUT_METADATA])
def test_d1_a_genuine_diagnostic_in_the_tail_still_classifies(line):
    rc, message = _run_scripted([line] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert _classified_no_space(message)
    assert _classified_no_space(_as_result_msg(message))


@pytest.mark.parametrize("line", [
    NO_SPACE_LINE, ENOSPC_LINE, NO_SPACE_TRAILER, NO_SPACE_ABOUT_METADATA])
def test_d2_a_genuine_diagnostic_buried_past_the_tail_still_classifies(line):
    rc, message = _run_scripted([line] + _chatter(600), 1)

    assert rc == 1
    assert _classified_no_space(message)
    assert _classified_no_space(_as_result_msg(message))


def test_d3_a_diagnostic_that_mentions_metadata_is_not_excluded():
    """The word "metadata" is not provenance."""
    assert _classified_no_space(NO_SPACE_ABOUT_METADATA)


# ══ GROUP E — the block ends where ffmpeg ends it ════════════════════════════

def test_e1_a_diagnostic_right_after_a_block_still_classifies():
    """`Duration:` closed the block; the next line is ffmpeg's own complaint."""
    rc, message = _run_scripted(
        _input_block(M_TITLE_NO_SPACE) + [NO_SPACE_TRAILER], 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_e2_the_block_closing_does_not_depend_on_the_tail():
    rc, message = _run_scripted(
        _input_block(M_TITLE_NO_SPACE) + [NO_SPACE_TRAILER] + _chatter(600), 1)

    assert _classified_no_space(message)


# ══ GROUP F — a false echo never unsays a genuine diagnostic ═════════════════

def test_f1_metadata_after_a_genuine_diagnostic_keeps_the_latch():
    rc, message = _run_scripted(
        [NO_SPACE_LINE] + _out_of_tail(M_COMMENT_ENOSPC), 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_f2_a_genuine_diagnostic_survives_hundreds_of_lines_and_a_block():
    rc, message = _run_scripted(
        [ENOSPC_LINE] + _chatter(300) + _out_of_tail(M_TITLE_NO_SPACE)
        + _chatter(300), 1)

    assert _classified_no_space(message)


# ══ GROUP G — several blocks in one run ══════════════════════════════════════

def test_g1_input_and_output_blocks_are_both_excluded():
    lines = (_input_block(M_TITLE_NO_SPACE)
             + [STREAM_LINE, STREAM_META, M_STREAM_HANDLER, STREAM_MAPPING]
             + _output_block(M_COMMENT_ENOSPC)
             + GENERIC_MUX_FAILURE)

    rc, message = _run_scripted(lines, 1)

    assert rc == 1
    assert not _classified_no_space(message)


def test_g2_a_genuine_diagnostic_after_several_blocks_still_classifies():
    lines = (_input_block(M_TITLE_NO_SPACE)
             + _output_block(M_COMMENT_ENOSPC)
             + [NO_SPACE_TRAILER, "Conversion failed!"])

    rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)


# ══ GROUP H — every real terminator closes the block ═════════════════════════

@pytest.mark.parametrize("terminator", [
    DURATION, STREAM_LINE, SIDE_DATA, CHAPTERS, CHAPTER_LINE,
    STREAM_MAPPING, OUT_HDR, IN_HDR, "Conversion failed!",
])
def test_h1_a_real_terminator_lets_the_next_diagnostic_classify(terminator):
    """Captured terminators, each proven to end a block in real ffmpeg output."""
    rc, message = _run_scripted(
        [IN_HDR, CONTAINER_META, M_TITLE_NO_SPACE, terminator,
         NO_SPACE_TRAILER], 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_h2_the_block_does_not_end_one_line_early():
    """Every value in a real block is excluded, not merely the first."""
    rc, message = _run_scripted(
        [IN_HDR, CONTAINER_META, M_MAJOR, M_MINOR, M_BRANDS,
         M_TITLE_NO_SPACE, M_ENCODER, M_COMMENT_ENOSPC, M_DESCRIPTION,
         DURATION] + GENERIC_MUX_FAILURE, 1)

    assert not _classified_no_space(message)


def test_h4_a_block_that_ended_stops_excluding_indented_lines():
    """`Duration:` closed the block. What follows is indented and
    colon-shaped, and is still ffmpeg's own complaint."""
    rc, message = _run_scripted(
        _input_block(M_TITLE_NO_SPACE)
        + ["    write_packet    : No space left on device"], 1)

    assert rc == 1
    assert _classified_no_space(message)


def test_h3_a_deeper_block_does_not_swallow_its_parent_terminator():
    """The stream block closes on `Side data:`, which is shallower than it."""
    rc, message = _run_scripted(
        [STREAM_LINE, STREAM_META, M_STREAM_TITLE_ENOSPC, SIDE_DATA,
         "      CPB properties: bitrate max/min/avg: 0/0/0",
         NO_SPACE_TRAILER], 1)

    assert _classified_no_space(message)


# ══ GROUP I — colon-shaped diagnostics outside a block still classify ════════

@pytest.mark.parametrize("line", LOOKALIKES)
def test_i1_a_lookalike_outside_a_block_is_still_a_diagnostic(line):
    rc, message = _run_scripted([line] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert _classified_no_space(message)


@pytest.mark.parametrize("line", LOOKALIKES)
def test_i2_a_lookalike_buried_past_the_tail_is_still_a_diagnostic(line):
    rc, message = _run_scripted([line] + _chatter(600), 1)

    assert _classified_no_space(message)


def test_i3_a_lookalike_after_a_closed_block_is_still_a_diagnostic():
    rc, message = _run_scripted(
        _input_block(M_TITLE_NO_SPACE)
        + ["    write_packet    : No space left on device"], 1)

    assert _classified_no_space(message)


def test_i4_a_bare_metadata_word_does_not_open_a_block():
    """An unindented line is structure ffmpeg never emits for a block."""
    rc, message = _run_scripted(
        ["Metadata:", "    title           : No space left on device"], 1)

    assert _classified_no_space(message)


# ══ GROUP J — the MKV retry is earned again ══════════════════════════════════

@pytest.mark.parametrize("value", CONTAINER_VALUES)
def test_j1_a_false_echo_no_longer_costs_mp4_h265_its_retry(env, value):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(value)), (0, _chatter(20))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "ok"
    assert result["fallback_used"] is True


def test_j2_the_retry_is_earned_when_the_echo_is_still_in_the_tail(env):
    src, out_dir, fake = env
    fake.finals = [(1, _in_tail(M_COMMENT_ENOSPC)), (0, _chatter(20))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["fallback_used"] is True


def test_j3_the_retry_is_earned_when_the_header_scrolled_out_of_the_tail(env):
    src, out_dir, fake = env
    fake.finals = [(1, _tail_truncated(M_TITLE_NO_SPACE)), (0, _chatter(20))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["fallback_used"] is True


def test_j4_a_failing_retry_is_still_a_terminal_generic_failure(env):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(M_TITLE_NO_SPACE)),
                   (1, _out_of_tail(M_COMMENT_ENOSPC))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])


# ══ GROUP K — a genuine diagnostic still refuses the retry ═══════════════════

def test_k1_a_genuine_diagnostic_refuses_the_retry_despite_the_tags(env):
    src, out_dir, fake = env
    fake.finals = [(1, _input_block(M_TITLE_NO_SPACE, M_COMMENT_ENOSPC)
                    + [NO_SPACE_TRAILER] + _chatter(30))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"
    assert _classified_no_space(result["msg"])
    assert src.exists()


def test_k2_a_genuine_diagnostic_before_the_block_also_refuses_the_retry(env):
    src, out_dir, fake = env
    fake.finals = [(1, [NO_SPACE_LINE] + _out_of_tail(M_COMMENT_ENOSPC))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4"]
    assert _classified_no_space(result["msg"])


# ══ GROUP L — the fallback pair stays isolated ═══════════════════════════════

def test_l1_attempt_two_classifies_on_its_own_stderr(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)),
                   (1, _out_of_tail(M_TITLE_NO_SPACE))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert not _classified_no_space(result["msg"])


def test_l2_a_genuine_diagnostic_on_attempt_two_is_still_specific(env):
    src, out_dir, fake = env
    fake.finals = [(1, _chatter(20)),
                   (1, _out_of_tail(M_TITLE_NO_SPACE) + [NO_SPACE_LINE])]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert _classified_no_space(result["msg"])


def test_l3_there_is_never_a_third_attempt(env):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(M_TITLE_NO_SPACE)),
                   (1, _out_of_tail(M_COMMENT_ENOSPC)),
                   (0, _chatter(5))]

    _run(src, out_dir, fmt="MP4 (H.265)")

    assert len(fake.final_cmds) == 2


# ══ GROUP M — every direct container behaves the same ════════════════════════

@pytest.mark.parametrize("fmt,muxer", [
    ("MP4 (H.264)", "mp4"),
    ("MKV (H.265)", "matroska"),
    ("WebM (VP9)", "webm"),
])
def test_m1_a_false_echo_is_an_ordinary_failure_for_direct_formats(
        env, fmt, muxer):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(M_TITLE_NO_SPACE))]

    result = _run(src, out_dir, fmt=fmt)

    assert fake.final_muxers == [muxer]
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])
    assert src.exists()


@pytest.mark.parametrize("fmt", ["MP4 (H.264)", "MKV (H.265)", "WebM (VP9)"])
def test_m2_a_genuine_diagnostic_is_still_specific_for_direct_formats(
        env, fmt):
    src, out_dir, fake = env
    fake.finals = [(1, _out_of_tail(M_TITLE_NO_SPACE) + [NO_SPACE_LINE])]

    result = _run(src, out_dir, fmt=fmt)

    assert _classified_no_space(result["msg"])


# ══ GROUP N — two-pass ═══════════════════════════════════════════════════════

def test_n1_a_false_echo_on_pass_one_stops_before_pass_two(env):
    src, out_dir, fake = env
    fake.pass1 = [(1, _out_of_tail(M_TITLE_NO_SPACE))]

    result = _run_two = _two_pass(src, out_dir, fmt="MP4 (H.265)")

    assert fake.subprocess_count == 1
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])


def test_n2_a_genuine_diagnostic_on_pass_one_is_still_specific(env):
    src, out_dir, fake = env
    fake.pass1 = [(1, _out_of_tail(M_TITLE_NO_SPACE) + [NO_SPACE_LINE])]

    result = _two_pass(src, out_dir, fmt="MP4 (H.265)")

    assert fake.subprocess_count == 1
    assert _classified_no_space(result["msg"])


def test_n3_a_false_echo_on_pass_two_is_an_ordinary_failure(env):
    src, out_dir, fake = env
    fake.pass1 = [(0, _chatter(5))]
    fake.finals = [(1, _out_of_tail(M_COMMENT_ENOSPC))]

    result = _two_pass(src, out_dir, fmt="MKV (H.265)")

    # MKV direct, so the two subprocesses are pass 1 and pass 2 and nothing
    # else: no fallback is eligible to blur the count.
    assert fake.subprocess_count == 2
    assert result["status"] == "error"
    assert not _classified_no_space(result["msg"])


def test_n4_a_genuine_diagnostic_on_pass_two_is_still_specific(env):
    src, out_dir, fake = env
    fake.pass1 = [(0, _chatter(5))]
    fake.finals = [(1, _out_of_tail(M_COMMENT_ENOSPC) + [NO_SPACE_LINE])]

    result = _two_pass(src, out_dir, fmt="MP4 (H.265)")

    assert len(fake.pass1_cmds) == 1 and len(fake.final_cmds) == 1
    assert _classified_no_space(result["msg"])


# ══ GROUP O — Tab 12's path exclusion is intact ══════════════════════════════

@pytest.mark.parametrize("echo", [OUT_PATH_ECHO, IN_PATH_ECHO])
def test_o1_a_path_echo_is_still_excluded(echo):
    rc, message = _run_scripted([echo] + GENERIC_MUX_FAILURE, 1)

    assert rc == 1
    assert not _classified_no_space(message)


@pytest.mark.parametrize("echo", [OUT_PATH_ECHO, IN_PATH_ECHO])
def test_o2_a_path_echo_past_the_tail_is_still_excluded(echo):
    rc, message = _run_scripted([echo] + _chatter(30) + GENERIC_MUX_FAILURE, 1)

    assert not _classified_no_space(message)


def test_o3_a_path_echo_and_a_metadata_echo_together_stay_excluded():
    lines = ([OUT_PATH_ECHO, CONTAINER_META, M_TITLE_NO_SPACE, DURATION]
             + _chatter(30) + GENERIC_MUX_FAILURE)

    rc, message = _run_scripted(lines, 1)

    assert not _classified_no_space(message)


def test_o4_a_genuine_diagnostic_survives_both_exclusions():
    lines = ([OUT_PATH_ECHO, CONTAINER_META, M_TITLE_NO_SPACE, DURATION,
              NO_SPACE_TRAILER] + _chatter(30))

    rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)


# ══ GROUP P — Tab 11's buried signal is intact ═══════════════════════════════

def test_p1_a_buried_genuine_signal_is_still_classified(env):
    src, out_dir, fake = env
    fake.finals = [(1, [NO_SPACE_LINE] + _chatter(600))]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4"]
    assert _classified_no_space(result["msg"])
    assert len(result["msg"].splitlines()) <= RETAINED_TAIL_LINES + 1


def test_p2_a_buried_enospc_spelling_is_still_classified():
    _rc, message = _run_scripted([ENOSPC_LINE] + _chatter(600), 1)

    assert _classified_no_space(message)


# ══ GROUP Q — signal-bearing tags still convert ══════════════════════════════

@pytest.mark.parametrize("fmt", ["MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"])
def test_q1_a_successful_encode_with_signal_bearing_tags_succeeds(env, fmt):
    src, out_dir, fake = env
    fake.finals = [(0, _input_block(M_TITLE_NO_SPACE, M_COMMENT_ENOSPC)
                    + _chatter(5))]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    assert src.exists()


def test_q2_no_metadata_stripping_argument_is_added(env):
    src, out_dir, fake = env
    fake.finals = [(0, _input_block(M_TITLE_NO_SPACE))]

    _run(src, out_dir, fmt="MKV (H.265)")

    joined = " ".join(str(a) for a in fake.final_cmds[0])
    assert "-map_metadata" not in joined


# ══ GROUP R — no state leaks ═════════════════════════════════════════════════

def test_r1_classification_never_leaks_between_files(env, tmp_path):
    src, out_dir, fake = env
    sources = []
    for name in ("A", "B", "C", "D"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)

    fake.finals = [
        (1, _out_of_tail(M_TITLE_NO_SPACE)),   # A - tags only, MP4 ...
        (1, _chatter(20)),                     # ... and its earned MKV retry
        (1, [NO_SPACE_LINE] + _chatter(30)),   # B - genuine, buried
        (0, _chatter(20)),                     # C - clean success
        (1, _in_tail(M_COMMENT_ENOSPC)),       # D - tags only, MP4 ...
        (0, _chatter(20)),                     # ... and its earned MKV retry
    ]
    a, b, c, d = [_run(p, out_dir, fmt="MP4 (H.265)") for p in sources]

    assert a["status"] == "error" and not _classified_no_space(a["msg"])
    assert b["status"] == "error" and _classified_no_space(b["msg"])
    assert c["status"] == "ok" and not c.get("fallback_used")
    assert d["status"] == "ok" and d["fallback_used"] is True
    assert fake.final_muxers == ["mp4", "matroska", "mp4", "mp4",
                                "mp4", "matroska"]


def test_r2_an_unclosed_block_does_not_carry_into_the_next_process():
    """The first run ends mid-block; the second opens with a real diagnostic."""
    _rc, first = _run_scripted([IN_HDR, CONTAINER_META, M_TITLE_NO_SPACE], 1)
    _rc, second = _run_scripted(
        ["    write_packet    : No space left on device"], 1)

    assert not _classified_no_space(first)
    assert _classified_no_space(second)


# ══ GROUP S — retention stays bounded ════════════════════════════════════════

@pytest.mark.parametrize("lines", [
    _out_of_tail(M_TITLE_NO_SPACE) + _chatter(2000),
    [NO_SPACE_LINE] + _chatter(2000),
    (_input_block(M_TITLE_NO_SPACE) + _output_block(M_COMMENT_ENOSPC)) * 40
    + _chatter(2000),
])
def test_s1_the_retained_window_is_still_the_predecessor_bound(lines):
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
    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 2


def test_s2_thousands_of_metadata_lines_do_not_grow_the_message():
    lines = ((_input_block(M_TITLE_NO_SPACE, M_COMMENT_ENOSPC)) * 200
             + _chatter(1000) + [NO_SPACE_LINE] + _chatter(1000))

    _rc, message = _run_scripted(lines, 1)

    assert _classified_no_space(message)
    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 2


def test_s3_a_false_echo_returns_no_more_than_one_extra_line():
    _rc, message = _run_scripted(_tail_truncated(M_TITLE_NO_SPACE), 1)

    assert len(message.splitlines()) <= RETAINED_TAIL_LINES + 1


# ══ GROUP T — vocabulary and command construction are untouched ══════════════

def test_t1_the_vocabulary_is_still_exactly_two_signals():
    assert compressor.NO_SPACE_SIGNALS == (
        "no space left on device", "enospc")


@pytest.mark.parametrize("phrase", [
    "disk full", "quota exceeded", "insufficient space",
    "out of disk space", "device full",
])
def test_t2_no_new_vocabulary_was_admitted(phrase):
    assert not _classified_no_space(f"Error writing trailer: {phrase}")


@pytest.mark.parametrize("fmt,muxer,maps", [
    ("MP4 (H.265)", "mp4", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
    ("MKV (H.265)", "matroska",
     ["-map", "0:v:0", "-map", "0:a?", "-sn", "-map", "0:t?", "-c:t", "copy"]),
    ("WebM (VP9)", "webm", ["-map", "0:v:0", "-map", "0:a?", "-sn"]),
])
def test_t3_stream_mapping_is_unchanged_under_a_false_echo(
        env, fmt, muxer, maps):
    src, out_dir, fake = env
    fake.finals = [(0, _in_tail(M_TITLE_NO_SPACE))]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    cmd = fake.final_cmds[0]
    assert _muxer_of(cmd) == muxer
    joined = " ".join(str(a) for a in cmd)
    assert " ".join(maps) in joined


def test_t4_a_successful_exit_is_never_reclassified(env):
    """Tab 10's gate decides success; stderr chatter cannot override it."""
    src, out_dir, fake = env
    fake.finals = [(0, _input_block(M_TITLE_NO_SPACE) + [NO_SPACE_LINE])]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert len(fake.final_cmds) == 1
