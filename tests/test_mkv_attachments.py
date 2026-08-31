"""Matroska attachment preservation.

Every final Matroska encode Cove produces - the directly requested
`MKV (H.265)` conversion and the MP4 -> MKV fallback attempt alike - maps the
source's attachment streams and stream-copies them into the output. Embedded
fonts and similar attachments survive a conversion that Matroska is perfectly
capable of carrying them through.

Locked down here:

  A. Container policy: Matroska maps `0:t?` once and copies with `-c:t copy`;
     MP4 and WebM do not, and no container ever maps data streams.
  B. The single-pass final encode carries the policy on the ordinary public
     `compress_video` path, with zero or many attachments alike.
  C. Two-pass analysis (pass 1) stays video-only; only pass 2 muxes
     attachments, and preservation adds no third invocation.
  D. The Tab 2b MP4 -> MKV fallback inherits the policy: attempt 1 (MP4) has
     no attachment map, attempt 2 (Matroska) does, and there is never a third.
  E. The neighbouring stream contract survives: one video map, one optional
     audio map (`0:a?`), and the subtitle map is not dropped.
  F. Attachment preservation costs no new ffprobe. The optional `0:t?`
     selector replaces classification entirely.
  G. A Matroska encode that fails is terminal - no "safe retry without
     attachments" - and leaves no reserved stub or temp behind.

The optionality of `0:t?` is the whole trick: a source with no attachments
must still convert, so the selector is optional rather than mandatory and no
attachment probe is ever needed to find that out.

No ffmpeg and no real media: `run_ffmpeg` and the probes are faked, but the
fakes reproduce the filesystem effects production actually checks for (an
encode writes its temp output; a subtitle extraction writes its sidecar temp).
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    build_matroska_stream_map_args,
    build_mp4_stream_map_args,
    build_pass1_stream_map_args,
    build_stream_map_args,
    compress_video,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _pair_count(cmd, flag, value) -> int:
    return sum(1 for i in range(len(cmd) - 1)
               if cmd[i] == flag and cmd[i + 1] == value)


def _sub(index: int, codec: str, **tags) -> dict:
    s: dict = {"index": index, "codec_name": codec, "tags": {"language": "eng"}}
    if tags:
        s["tags"].update(tags)
    return s


TEXT_SUB = [_sub(3, "subrip")]

# Bitmap subtitles have no text encoder to reach matroska's text-based default
# subtitle codec. ffmpeg's implicit selection quietly skipped them; an explicit
# map does not, which is exactly the trap this policy has to avoid.
BITMAP_CODECS = ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"]

# Big enough that the 1 MB targets below are a real reduction *and* leave a
# usable video budget once the audio bitrate is reserved. A target that cannot
# hold its audio is refused before encoding (see `tests/test_multi_audio.py`),
# which is not the subject of any test in this file.
SRC_BYTES = 4 * 1024 * 1024


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, classifying and recording every invocation.

    Three kinds exist and each gets its own scripted result queue, because the
    contracts here are about *which* invocation carried what: subtitle
    extraction (`-vn` and `-an`, the structural discriminator the sibling
    suites use), two-pass analysis (the null muxer) and the final encode/mux.
    Queues fall back to success once exhausted.

    `attachments` models how many attachment streams the source holds. It is
    deliberately never *probed* - it only decides whether a mandatory `0:t`
    map could be satisfied, which is how the optionality of `0:t?` becomes
    observable rather than merely asserted about a string.
    """

    def __init__(self, finals=None, pass1=None, subtitle=None,
                 attachments=1, encode_bytes=b"v" * 10):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.subtitle = list(subtitle or [])
        self.attachments = attachments
        self.encode_bytes = encode_bytes
        self.subtitle_cmds: list[list] = []
        self.pass1_cmds: list[list] = []
        self.final_cmds: list[list] = []

    @staticmethod
    def _next(q):
        return q.pop(0) if q else (0, "")

    def _unsatisfied_map(self, cmd) -> str | None:
        """Real ffmpeg fails a mandatory map that matches no stream."""
        if self.attachments == 0 and "0:t" in _maps(cmd):
            return "Stream map '0:t' matches no streams."
        return None

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        cmd = list(cmd)
        out = Path(cmd[-1])
        if "-vn" in cmd and "-an" in cmd and "-map" in cmd:
            self.subtitle_cmds.append(cmd)
            rc, err = self._next(self.subtitle)
            if rc == 0:
                out.write_bytes(b"1\nhello\n")
            return rc, err
        if _muxer_of(cmd) == "null":
            self.pass1_cmds.append(cmd)
            bad = self._unsatisfied_map(cmd)
            return (1, bad) if bad else self._next(self.pass1)
        self.final_cmds.append(cmd)
        bad = self._unsatisfied_map(cmd)
        if bad:
            return 1, bad
        rc, err = self._next(self.finals)
        if rc == 0:
            out.write_bytes(self.encode_bytes)
        return rc, err

    @property
    def final_muxers(self) -> list[str]:
        return [_muxer_of(c) for c in self.final_cmds]

    @property
    def mux_cmd(self) -> list:
        assert self.final_cmds, "no final muxing invocation was recorded"
        return self.final_cmds[-1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder/probe stack."""
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(subtitles=[], audio_count=1))
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": False)
    return src, out_dir, fake


def _probe(monkeypatch, streams, calls=None, audio_count=1):
    """Fake the shared stream inventory; `streams` may be an exception."""
    def fake(path):
        if calls is not None:
            calls.append(Path(path))
        if isinstance(streams, Exception):
            raise streams
        return compressor.StreamInventory(
            subtitles=[dict(s) for s in streams], audio_count=audio_count)
    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", fake)


def _run(src, out_dir, fmt="MKV (H.265)", mode="Quality preset",
         mode_value="Balanced", cancel_flag=None, **kw):
    """The ordinary public `compress_video` entry point - no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


def _names(out_dir, pattern="*") -> list[str]:
    return sorted(p.name for p in out_dir.glob(pattern))


# ══ GROUP A — container stream policy ════════════════════════════════════════

def test_a1_matroska_maps_optional_attachments_exactly_once():
    """`0:t?` - optional, so a source without attachments still converts."""
    args = build_matroska_stream_map_args(TEXT_SUB)
    assert _maps(args).count("0:t?") == 1
    assert "0:t" not in _maps(args)


def test_a2_matroska_stream_copies_attachments():
    """Copied, never re-encoded - there is no attachment 'encoder'."""
    args = build_matroska_stream_map_args(TEXT_SUB)
    assert _pair_count(args, "-c:t", "copy") == 1


@pytest.mark.parametrize("streams", [None, [], TEXT_SUB,
                                     [_sub(2, "hdmv_pgs_subtitle")]])
def test_a2b_matroska_policy_is_reached_through_the_shared_seam(streams):
    """Direct MKV and the fallback MKV must share one policy, not two."""
    assert build_stream_map_args("matroska", streams) == \
        build_matroska_stream_map_args(streams)


@pytest.mark.parametrize("streams", [None, [], [_sub(3, "subrip")],
                                     [_sub(3, "hdmv_pgs_subtitle")]])
def test_a3_mp4_never_maps_or_copies_attachments(streams):
    args = build_stream_map_args("mp4", streams)
    assert args == build_mp4_stream_map_args(streams)
    assert not [m for m in _maps(args) if m.startswith("0:t")]
    assert "-c:t" not in args


def test_a4_webm_never_maps_or_copies_attachments():
    """WebM maps explicitly now, and still names no attachment.

    Attachments are Matroska's to preserve; WebM has no support for them, so
    the explicit policy leaves them out by simply never naming them.
    """
    args = build_stream_map_args("webm", [_sub(3, "subrip")])
    assert args == ["-map", "0:v:0", "-map", "0:a?", "-map", "0:3",
                    "-c:s", "webvtt"]
    assert not [m for m in _maps(args) if m.startswith("0:t")]
    assert "-c:t" not in args


@pytest.mark.parametrize("muxer", ["mp4", "matroska", "webm"])
def test_a5_no_container_policy_maps_data_streams(muxer):
    """`t` is an attachment; `d` is a data stream. Never confuse the two."""
    args = build_stream_map_args(muxer, [_sub(3, "subrip")])
    assert not [m for m in _maps(args) if m.startswith("0:d")]
    assert "-c:d" not in args
    assert "-map" not in build_pass1_stream_map_args()[2:]


# ══ GROUP B — the single-pass final encode ═══════════════════════════════════

def test_b1_direct_mkv_single_pass_maps_and_copies_attachments(env, monkeypatch):
    """Drives the ordinary public path, not just the helper."""
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip")])

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert len(fake.final_cmds) == 1
    cmd = fake.mux_cmd
    assert _muxer_of(cmd) == "matroska"
    assert _maps(cmd) == ["0:v:0", "0:a?", "0:3", "0:t?"]
    assert _pair_count(cmd, "-c:t", "copy") == 1


def test_b2_source_without_attachments_still_succeeds(env):
    """The optional selector is load-bearing: zero attachments is normal."""
    src, out_dir, fake = env
    fake.attachments = 0

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert "0:t?" in _maps(fake.mux_cmd)
    assert Path(result["output"]).suffix == ".mkv"


def test_b3_many_attachments_do_not_multiply_the_command(env):
    """One optional type selector covers N attachments - no per-index maps."""
    src, out_dir, fake = env
    fake.attachments = 7

    assert _run(src, out_dir)["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert [m for m in maps if m.startswith("0:t")] == ["0:t?"]
    assert _pair_count(fake.mux_cmd, "-c:t", "copy") == 1


# ══ GROUP C — two-pass safety ════════════════════════════════════════════════

def _two_pass(src, out_dir, **kw):
    return _run(src, out_dir, mode="Target file size", mode_value=1, **kw)


def test_c1_pass_one_never_mentions_attachments(env):
    src, out_dir, fake = env

    assert _two_pass(src, out_dir)["status"] == "ok"

    assert len(fake.pass1_cmds) == 1
    cmd = fake.pass1_cmds[0]
    assert _maps(cmd) == ["0:v:0"]
    assert "-an" in cmd
    assert "0:t?" not in _maps(cmd)
    assert "-c:t" not in cmd


def test_c2_pass_two_preserves_attachments(env):
    src, out_dir, fake = env

    assert _two_pass(src, out_dir)["status"] == "ok"

    cmd = fake.mux_cmd
    assert _muxer_of(cmd) == "matroska"
    assert _maps(cmd).count("0:t?") == 1
    assert _pair_count(cmd, "-c:t", "copy") == 1


def test_c3_preservation_adds_no_third_invocation(env):
    src, out_dir, fake = env

    assert _two_pass(src, out_dir)["status"] == "ok"

    assert len(fake.pass1_cmds) == 1
    assert len(fake.final_cmds) == 1


# ══ GROUP D — Tab 2b MP4 -> MKV fallback integration ═════════════════════════

def _fallback(src, out_dir, monkeypatch, fake, finals):
    """A qualifying final-MP4 failure, then whatever `finals` scripts next."""
    fake.finals = list(finals)
    return _run(src, out_dir, fmt="MP4 (H.265)")


def test_d1_first_mp4_attempt_excludes_attachments(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error")]

    _run(src, out_dir, fmt="MP4 (H.265)")

    first = fake.final_cmds[0]
    assert _muxer_of(first) == "mp4"
    assert not [m for m in _maps(first) if m.startswith("0:t")]
    assert "-c:t" not in first


def test_d2_mkv_fallback_attempt_includes_attachments(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert fake.final_muxers == ["mp4", "matroska"]
    second = fake.final_cmds[1]
    assert _maps(second).count("0:t?") == 1
    assert _pair_count(second, "-c:t", "copy") == 1


def test_d3_failing_attachment_bearing_fallback_is_still_only_two_attempts(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error"), (1, "could not copy attachment")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] != "ok"
    assert len(fake.final_cmds) == 2
    assert fake.final_muxers == ["mp4", "matroska"]


def test_d4_fallback_success_result_shape_is_unchanged(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).suffix == ".mkv"
    assert result["fallback_used"] is True
    for key in ("attachments_preserved", "attachment_count",
                "attachment_names"):
        assert key not in result


# ══ GROUP E — the neighbouring stream contract survives ══════════════════════

def test_e1_matroska_maps_exactly_one_video_stream():
    maps = _maps(build_matroska_stream_map_args(TEXT_SUB))
    assert [m for m in maps if m.startswith("0:v")] == ["0:v:0"]


def test_e2_matroska_maps_audio_once_and_optionally():
    """`0:t?` sits beside the audio map, never in place of or duplicating it.

    The audio map itself is the all-streams `0:a?` (see
    `tests/test_multi_audio.py`); what matters here is that attachment
    preservation neither widens nor suppresses it.
    """
    maps = _maps(build_matroska_stream_map_args(TEXT_SUB))
    assert [m for m in maps if m.startswith("0:a")] == ["0:a?"]
    assert "0:a" not in maps, "a mandatory map breaks silent sources"


def test_e3_matroska_maps_every_text_subtitle_by_absolute_index():
    """Absolute indexes, never subtitle-relative ones - and all of them.

    Implicit selection took one text stream and dropped the rest; Matroska is
    a multi-track container and has no reason to. See
    `tests/test_mkv_multi_subtitles.py` for the full cardinality contract.
    """
    args = build_matroska_stream_map_args([_sub(3, "subrip"),
                                           _sub(5, "ass")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:3", "0:5", "0:t?"]
    assert "-sn" not in args


def test_e5_matroska_leaves_the_subtitle_codec_unset():
    """Explicit maps must not bring an explicit subtitle codec with them.

    Matroska's default subtitle encoder is text-based, and letting ffmpeg use
    it is what makes `subrip`/`ass`/`mov_text` sources all work. `-c:s copy`
    looks like the safer choice and is not: matroska cannot carry `mov_text`,
    so copying turns every MP4-with-subtitles -> MKV conversion into a hard
    "Subtitle codec mov_text is not supported" mux failure. Verified against
    real ffmpeg 8.1.1; leave this alone.
    """
    args = build_matroska_stream_map_args(TEXT_SUB)
    assert "-c:s" not in args
    assert "copy" not in args[:args.index("-c:t")]
    assert "-sn" not in args


@pytest.mark.parametrize("codec", BITMAP_CODECS)
def test_e6_matroska_never_maps_a_bitmap_subtitle(codec):
    """The regression Codex caught, locked down.

    Matroska's default subtitle encoder is text-based, so a bitmap stream
    cannot reach it. ffmpeg's implicit selection used to skip such a stream
    and convert happily; naming it explicitly instead turns the whole job into
    a hard "Automatic encoder selection failed" mux failure. Verified against
    real ffmpeg 8.1.1 - see the handoff.
    """
    args = build_matroska_stream_map_args([_sub(2, codec)])
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args


def test_e7_matroska_skips_a_bitmap_stream_to_reach_a_text_one():
    """Bitmap first, text second: the text one is what implicit would pick."""
    args = build_matroska_stream_map_args([_sub(2, "hdmv_pgs_subtitle"),
                                           _sub(4, "subrip")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:4", "0:t?"]
    assert "-sn" not in args


@pytest.mark.parametrize("streams", [None, [], [_sub(2, "unknown_codec")]])
def test_e8_matroska_disables_subtitles_when_none_are_mappable(streams):
    """No usable subtitle - including a failed probe (None) - means `-sn`.

    Fail closed, exactly as MP4 does: never hand the choice back to ffmpeg's
    implicit selection, which is the behaviour this policy replaces.
    """
    args = build_matroska_stream_map_args(streams)
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args
    assert _pair_count(args, "-c:t", "copy") == 1


def test_e9_bitmap_subtitle_source_still_converts_end_to_end(env, monkeypatch):
    """Drives the public path: a PGS-only source must not fail the job."""
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "hdmv_pgs_subtitle")])

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert not [m for m in _maps(fake.mux_cmd) if m.startswith("0:2")]
    assert "-sn" in fake.mux_cmd
    assert _maps(fake.mux_cmd).count("0:t?") == 1


def test_e4_mp4_bitmap_and_unknown_exclusion_is_untouched():
    args = build_stream_map_args(
        "mp4", [_sub(2, "hdmv_pgs_subtitle"), _sub(3, "subrip"),
                _sub(4, "nonsense_codec")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:3"]


# ══ GROUP F — no new probes, no extra subtitle work ══════════════════════════

def test_f1_attachments_add_no_probe_of_their_own(env, monkeypatch):
    """MKV probes once - for subtitles. Attachments never add a second one.

    `0:t?` needs no classification at all, so the attachment half of this
    policy is free: one probe per file covers both consumers, exactly as it
    already did for MP4.
    """
    src, out_dir, fake = env
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip")], calls)

    assert _run(src, out_dir)["status"] == "ok"

    assert len(calls) == 1
    assert _maps(fake.mux_cmd).count("0:t?") == 1


def test_f2_subtitle_probe_count_does_not_increase(env, monkeypatch):
    """MKV with extraction on: still exactly the one established probe."""
    src, out_dir, _fake = env
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip")], calls)

    assert _run(src, out_dir, extract_english_subtitles=True)["status"] == "ok"

    assert len(calls) == 1


def test_f3_fallback_still_prepares_subtitles_once(env, monkeypatch):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error")]
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip")], calls)

    result = _run(src, out_dir, fmt="MP4 (H.265)",
                  extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert len(fake.subtitle_cmds) == 1


# ══ GROUP G — failure and cleanup ════════════════════════════════════════════

def test_g1_direct_mkv_failure_is_terminal(env):
    src, out_dir, fake = env
    fake.finals = [(1, "could not copy attachment")]

    result = _run(src, out_dir)

    assert result["status"] != "ok"
    assert len(fake.final_cmds) == 1
    assert src.exists()


def test_g2_fallback_mkv_failure_is_terminal_after_attempt_two(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error"), (1, "could not copy attachment")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] != "ok"
    assert len(fake.final_cmds) == 2
    assert src.exists()


def test_g3_failed_direct_mkv_leaves_no_stub_or_temp(env):
    src, out_dir, fake = env
    fake.finals = [(1, "could not copy attachment")]

    assert _run(src, out_dir)["status"] != "ok"

    assert _names(out_dir) == []


def test_g4_failed_attachment_bearing_fallback_leaves_no_stub_or_temp(env):
    src, out_dir, fake = env
    fake.finals = [(1, "muxer error"), (1, "could not copy attachment")]

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] != "ok"

    assert _names(out_dir) == []
    assert src.exists()
