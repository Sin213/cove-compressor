"""Matroska multi-subtitle preservation.

A final Matroska encode preserves *every* embedded subtitle stream whose codec
is already in Cove's reviewed text-safe vocabulary - not just the first one.
A source with an English SRT, a Japanese ASS and a Spanish SRT track comes out
of an MKV conversion with all three, in source order, named by absolute
ffprobe index.

This is a cardinality change and nothing else. What counts as an eligible text
subtitle is unchanged: the same allowlist, the same bitmap exclusion, the same
fail-closed treatment of unknown codecs, MicroDVD and EIA-608.

Locked down here:

  A. Zero / one / many: no eligible stream still means `-sn`; exactly one is
     byte-for-byte what the predecessor emitted; N eligible means N maps.
  B. Mixed eligibility: bitmap and unknown streams are skipped *around* the
     text ones rather than terminating the scan at the first hit.
  C. Absolute indexes: sparse source indexes are reproduced exactly, and no
     subtitle-relative selector is ever emitted.
  D. Source order: inventory order is authoritative and load-bearing, not a
     set.
  E. Vocabulary freeze: the eligible set is unchanged in both directions.
  F. Tab 3 safety: no `0:s:0?`, no generic `-c:s copy`, attachments intact.
  G. Tab 4 contract: one video map, one optional audio map, untouched by N.
  H. Two-pass: pass 1 stays subtitle-free; only pass 2 carries the N maps.
  I. The public `compress_video` MKV path, end to end.
  J. MP4 non-regression: MP4's own policy is unchanged by the shared helper.
  K. Tab 2b fallback inherits the Matroska seam, still in two attempts.
  L. A failing multi-subtitle encode is terminal - no silent retry that drops
     tracks.
  M. Preservation costs no new probe.
  N. Per-file isolation across sequential jobs.

No ffmpeg and no real media: `run_ffmpeg` and the probes are faked, but the
fakes reproduce the filesystem effects production actually checks for.
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
    build_stream_map_args,
    compress_video,
    matroska_mappable_subtitle_indexes,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _sub_maps(cmd) -> list[str]:
    """Only the explicitly indexed maps - video/audio/attachment excluded."""
    fixed = {"0:v:0", "0:a?", "0:t?"}
    return [m for m in _maps(cmd) if m not in fixed]


def _pair_count(cmd, flag, value) -> int:
    return sum(1 for i in range(len(cmd) - 1)
               if cmd[i] == flag and cmd[i + 1] == value)


def _sub(index: int, codec: str, **tags) -> dict:
    s: dict = {"index": index, "codec_name": codec, "tags": {"language": "eng"}}
    if tags:
        s["tags"].update(tags)
    return s


# The realistic shape this slice exists for: an English, a Japanese and a
# Spanish text track, sparsely indexed around audio and a bitmap stream.
THREE_TEXT = [
    _sub(3, "subrip", language="eng", title="English"),
    _sub(6, "ass", language="jpn", title="Japanese"),
    _sub(9, "subrip", language="spa", title="Spanish"),
]

BITMAP_CODECS = ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"]

# Codecs that are deliberately *not* in the vocabulary. MicroDVD is closed by
# Tab 7 evidence (unreachable as an embedded stream); EIA-608 has no genuine
# fixture and stays fail-closed. Neither becomes eligible by widening N.
INELIGIBLE_TEXTISH = ["microdvd", "eia_608", "mpl2", "jacosub", "subviewer",
                      "pjs", "realtext", "sami", "totally_unknown_codec"]

SRC_BYTES = 4 * 1024 * 1024


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, classifying and recording every invocation.

    Three kinds exist and each gets its own scripted result queue: subtitle
    extraction (`-vn` and `-an`), two-pass analysis (the null muxer) and the
    final encode/mux. Queues fall back to success once exhausted.
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

    @staticmethod
    def _next(q):
        return q.pop(0) if q else (0, "")

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
            return self._next(self.pass1)
        self.final_cmds.append(cmd)
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


# ══ GROUP A — zero / one / many ══════════════════════════════════════════════

def test_a1_no_subtitle_streams_still_disables_subtitles():
    args = build_matroska_stream_map_args([])
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args


@pytest.mark.parametrize("codec", BITMAP_CODECS + INELIGIBLE_TEXTISH)
def test_a2_only_ineligible_subtitles_still_disables_subtitles(codec):
    args = build_matroska_stream_map_args([_sub(2, codec), _sub(5, codec)])
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args


def test_a3_exactly_one_eligible_is_predecessor_equivalent():
    """One eligible stream must produce exactly the predecessor command."""
    args = build_matroska_stream_map_args([_sub(7, "subrip")])
    assert args == ["-map", "0:v:0", "-map", "0:a?", "-map", "0:7",
                    "-map", "0:t?", "-c:t", "copy"]


def test_a4_two_eligible_subtitles_are_both_mapped():
    args = build_matroska_stream_map_args([_sub(3, "subrip"), _sub(5, "ass")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:3", "0:5", "0:t?"]
    assert "-sn" not in args


def test_a5_three_eligible_subtitles_are_all_mapped():
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert _maps(args) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9", "0:t?"]


def test_a6_mapped_subtitles_never_come_with_sn():
    for streams in ([_sub(1, "subrip")],
                    [_sub(1, "subrip"), _sub(2, "ass")],
                    THREE_TEXT):
        assert "-sn" not in build_matroska_stream_map_args(streams)


def test_a7_each_eligible_stream_is_mapped_exactly_once():
    maps = _sub_maps(build_matroska_stream_map_args(THREE_TEXT))
    assert maps == ["0:3", "0:6", "0:9"]
    assert len(maps) == len(set(maps))


def test_a8_identical_codecs_and_tags_are_not_deduplicated():
    """Three input streams means three output streams, however alike."""
    streams = [_sub(4, "subrip", language="eng", title="English"),
               _sub(5, "subrip", language="eng", title="English"),
               _sub(6, "subrip", language="eng", title="English")]
    assert _sub_maps(build_matroska_stream_map_args(streams)) == \
        ["0:4", "0:5", "0:6"]


# ══ GROUP B — mixed eligibility ══════════════════════════════════════════════

def test_b1_bitmap_and_unknown_streams_are_skipped_around_text_ones():
    args = build_matroska_stream_map_args([
        _sub(2, "dvd_subtitle"),
        _sub(4, "subrip"),
        _sub(7, "totally_unknown_codec"),
        _sub(9, "ass"),
        _sub(12, "hdmv_pgs_subtitle"),
        _sub(14, "srt"),
    ])
    maps = _maps(args)
    assert _sub_maps(args) == ["0:4", "0:9", "0:14"]
    for excluded in ("0:2", "0:7", "0:12"):
        assert excluded not in maps
    assert "-sn" not in args


def test_b2_a_leading_bitmap_stream_does_not_end_the_scan():
    """The predecessor stopped at the first text hit; that is the bug."""
    args = build_matroska_stream_map_args([_sub(2, "hdmv_pgs_subtitle"),
                                           _sub(4, "subrip"),
                                           _sub(6, "ass")])
    assert _sub_maps(args) == ["0:4", "0:6"]


def test_b3_a_trailing_bitmap_stream_is_still_excluded():
    args = build_matroska_stream_map_args([_sub(1, "subrip"),
                                           _sub(2, "hdmv_pgs_subtitle"),
                                           _sub(3, "webvtt")])
    assert _sub_maps(args) == ["0:1", "0:3"]


# ══ GROUP C — absolute indexes ═══════════════════════════════════════════════

def test_c1_sparse_absolute_indexes_are_reproduced_exactly():
    args = build_matroska_stream_map_args([_sub(5, "subrip"),
                                           _sub(13, "ass"),
                                           _sub(27, "webvtt")])
    assert _sub_maps(args) == ["0:5", "0:13", "0:27"]


@pytest.mark.parametrize("selector",
                         ["0:s:0", "0:s:1", "0:s:2", "0:s?", "0:s:0?"])
def test_c2_no_subtitle_relative_selector_is_ever_emitted(selector):
    """Tab 3's regression: a relative selector bypasses ffmpeg's own
    text/image compatibility filtering and forces a bitmap track into a text
    encoder. Widening N must not delegate selection back to ffmpeg."""
    for streams in ([], [_sub(2, "hdmv_pgs_subtitle")], [_sub(3, "subrip")],
                    THREE_TEXT):
        assert selector not in _maps(build_matroska_stream_map_args(streams))


def test_c3_helper_returns_every_eligible_absolute_index():
    assert matroska_mappable_subtitle_indexes(THREE_TEXT) == [3, 6, 9]


def test_c4_unindexable_streams_are_dropped_not_guessed_at():
    args = build_matroska_stream_map_args([
        {"codec_name": "subrip"},
        _sub(4, "subrip"),
        {"index": "5", "codec_name": "ass"},
        {"index": True, "codec_name": "ass"},
        {"index": -1, "codec_name": "ass"},
        _sub(8, "ass"),
    ])
    assert _sub_maps(args) == ["0:4", "0:8"]


# ══ GROUP D — source order ═══════════════════════════════════════════════════

def test_d1_maps_follow_inventory_order():
    """`ffprobe_stream_inventory` yields streams in file order, and that
    ordering is what reaches the output. Order is asserted, not membership."""
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert _sub_maps(args) == ["0:3", "0:6", "0:9"]


def test_d2_inventory_order_wins_over_numeric_index_order():
    """Nothing sorts the list: if the inventory ever hands back an order that
    is not ascending, the output follows the inventory, not the integers."""
    args = build_matroska_stream_map_args([_sub(9, "subrip", language="spa"),
                                           _sub(4, "ass", language="jpn"),
                                           _sub(12, "srt", language="eng")])
    assert _sub_maps(args) == ["0:9", "0:4", "0:12"]


def test_d3_order_is_not_language_or_codec_driven():
    """English last in the source stays last in the output."""
    args = build_matroska_stream_map_args([
        _sub(3, "ass", language="jpn", title="Japanese"),
        _sub(5, "webvtt", language="spa", title="Spanish"),
        _sub(8, "subrip", language="eng", title="English"),
    ])
    assert _sub_maps(args) == ["0:3", "0:5", "0:8"]


def test_d4_source_order_survives_the_public_mkv_path(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(11, "subrip", language="eng"),
                         _sub(2, "ass", language="jpn"),
                         _sub(7, "webvtt", language="spa")])

    assert _run(src, out_dir)["status"] == "ok"
    assert _sub_maps(fake.mux_cmd) == ["0:11", "0:2", "0:7"]


# ══ GROUP E — vocabulary freeze ══════════════════════════════════════════════

@pytest.mark.parametrize("codec", ["subrip", "srt", "mov_text", "text",
                                   "webvtt", "vtt", "ass", "ssa"])
def test_e1_every_committed_safe_codec_stays_eligible(codec):
    args = build_matroska_stream_map_args([_sub(4, codec)])
    assert _sub_maps(args) == ["0:4"]


def test_e2_microdvd_stays_ineligible():
    """Closed by Tab 7: unreachable as an embedded stream on ffmpeg 8.1.1."""
    args = build_matroska_stream_map_args([_sub(4, "microdvd")])
    assert _sub_maps(args) == []
    assert "-sn" in args
    assert "microdvd" not in compressor.MP4_TEXT_SUBTITLE_CODECS


def test_e3_eia_608_stays_ineligible():
    args = build_matroska_stream_map_args([_sub(4, "eia_608")])
    assert _sub_maps(args) == []
    assert "-sn" in args
    assert "eia_608" not in compressor.MP4_TEXT_SUBTITLE_CODECS


@pytest.mark.parametrize("codec", BITMAP_CODECS)
def test_e4_bitmap_stays_ineligible(codec):
    assert _sub_maps(build_matroska_stream_map_args([_sub(4, codec)])) == []


@pytest.mark.parametrize("codec", INELIGIBLE_TEXTISH)
def test_e5_arbitrary_unknown_codecs_stay_ineligible(codec):
    """The classifier is an allowlist. "Everything that is not bitmap is
    text" must not pass this."""
    assert _sub_maps(build_matroska_stream_map_args([_sub(4, codec)])) == []


def test_e6_the_vocabulary_itself_is_unchanged():
    assert compressor.SRT_SAFE_SUBTITLE_CODECS == {
        "subrip", "srt", "mov_text", "text", "webvtt", "vtt"}
    assert compressor.ASS_SUBTITLE_CODECS == {"ass", "ssa"}
    assert compressor.MP4_TEXT_SUBTITLE_CODECS == {
        "subrip", "srt", "mov_text", "text", "webvtt", "vtt", "ass", "ssa"}


# ══ GROUP F — Tab 3 safety locks ═════════════════════════════════════════════

def test_f1_no_generic_subtitle_copy_codec():
    """`-c:s copy` fails outright for mov_text -> matroska; the default text
    encoder is what makes every eligible codec work."""
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert "-c:s" not in args
    assert "copy" not in args[:args.index("-c:t")]


def test_f2_no_per_stream_subtitle_codec_either():
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert not [t for t in args if t.startswith("-c:s")]


def test_f3_attachments_are_still_mapped_and_copied_exactly_once():
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert _maps(args).count("0:t?") == 1
    assert _pair_count(args, "-c:t", "copy") == 1


def test_f4_attachment_map_stays_last():
    args = build_matroska_stream_map_args(THREE_TEXT)
    assert _maps(args)[-1] == "0:t?"


@pytest.mark.parametrize("streams", [None, [], THREE_TEXT])
def test_f5_data_streams_stay_unmapped(streams):
    args = build_matroska_stream_map_args(streams)
    assert not [m for m in _maps(args) if m.startswith("0:d")]


def test_f6_failed_discovery_still_fails_closed():
    args = build_matroska_stream_map_args(None)
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args


@pytest.mark.parametrize("streams", [None, [], THREE_TEXT])
def test_f7_the_shared_seam_is_the_only_matroska_policy(streams):
    assert build_stream_map_args("matroska", streams) == \
        build_matroska_stream_map_args(streams)


# ══ GROUP G — audio / video contract ═════════════════════════════════════════

def test_g1_exactly_one_video_map():
    maps = _maps(build_matroska_stream_map_args(THREE_TEXT))
    assert [m for m in maps if m.startswith("0:v")] == ["0:v:0"]


def test_g2_exactly_one_optional_audio_map():
    maps = _maps(build_matroska_stream_map_args(THREE_TEXT))
    assert [m for m in maps if m.startswith("0:a")] == ["0:a?"]


def test_g3_three_subtitle_maps_do_not_duplicate_video_or_audio():
    maps = _maps(build_matroska_stream_map_args(THREE_TEXT))
    assert maps == ["0:v:0", "0:a?", "0:3", "0:6", "0:9", "0:t?"]


def test_g4_audio_codec_and_bitrate_args_are_untouched(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [], audio_count=3)
    assert _run(src, out_dir)["status"] == "ok"
    one = [t for t in fake.mux_cmd if t in ("-c:a", "-b:a")]
    baseline = [(fake.mux_cmd[i], fake.mux_cmd[i + 1])
                for i, t in enumerate(fake.mux_cmd) if t in ("-c:a", "-b:a")]

    fake.final_cmds.clear()
    _probe(monkeypatch, THREE_TEXT, audio_count=3)
    assert _run(src, out_dir)["status"] == "ok"
    assert [t for t in fake.mux_cmd if t in ("-c:a", "-b:a")] == one
    assert [(fake.mux_cmd[i], fake.mux_cmd[i + 1])
            for i, t in enumerate(fake.mux_cmd)
            if t in ("-c:a", "-b:a")] == baseline


# ══ GROUP H — two-pass ═══════════════════════════════════════════════════════

def _two_pass(src, out_dir, **kw):
    return _run(src, out_dir, mode="Target file size", mode_value=1, **kw)


def test_h1_pass_one_carries_no_subtitle_maps(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _two_pass(src, out_dir)["status"] == "ok"

    assert len(fake.pass1_cmds) == 1
    assert _maps(fake.pass1_cmds[0]) == ["0:v:0"]


def test_h2_pass_one_keeps_an(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _two_pass(src, out_dir)["status"] == "ok"
    assert "-an" in fake.pass1_cmds[0]


def test_h3_pass_two_maps_every_eligible_subtitle(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _two_pass(src, out_dir)["status"] == "ok"
    assert _sub_maps(fake.mux_cmd) == ["0:3", "0:6", "0:9"]


def test_h4_exactly_two_ffmpeg_encode_subprocesses(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _two_pass(src, out_dir)["status"] == "ok"
    assert len(fake.pass1_cmds) == 1
    assert len(fake.final_cmds) == 1


# ══ GROUP I — public direct MKV path ═════════════════════════════════════════

def test_i1_public_mkv_path_preserves_all_three_subtitles(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).suffix == ".mkv"
    cmd = fake.mux_cmd
    assert _maps(cmd) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9", "0:t?"]
    assert _pair_count(cmd, "-c:t", "copy") == 1
    assert "-sn" not in cmd
    assert "0:s:0?" not in _maps(cmd)
    assert "-c:s" not in cmd
    assert "mov_text" not in cmd


def test_i2_public_mkv_path_with_no_eligible_subtitles_says_sn(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "hdmv_pgs_subtitle"), _sub(5, "microdvd")])

    assert _run(src, out_dir)["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in fake.mux_cmd


def test_i3_public_result_shape_is_unchanged(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    for absent in ("subtitle_count", "subtitles_embedded",
                   "subtitle_tracks_preserved"):
        assert absent not in result


# ══ GROUP J — MP4 non-regression ═════════════════════════════════════════════

def test_j1_mp4_helper_policy_is_unchanged():
    """MP4 already mapped every compatible stream and transcoded to mov_text.
    The Matroska fix must not touch either half of that."""
    args = build_mp4_stream_map_args(THREE_TEXT)
    assert args == ["-map", "0:v:0", "-map", "0:a?", "-map", "0:3",
                    "-map", "0:6", "-map", "0:9", "-c:s", "mov_text"]


def test_j2_mp4_never_gains_attachment_or_subtitle_disable():
    args = build_mp4_stream_map_args(THREE_TEXT)
    assert "0:t?" not in _maps(args)
    assert "-c:t" not in args
    assert "-sn" not in args


def test_j3_mp4_with_no_eligible_subtitles_is_unchanged():
    args = build_mp4_stream_map_args([_sub(2, "hdmv_pgs_subtitle")])
    assert args == ["-map", "0:v:0", "-map", "0:a?", "-sn"]


@pytest.mark.parametrize("fmt", ["MP4 (H.264)", "MP4 (H.265)"])
def test_j4_public_mp4_path_is_unchanged(env, monkeypatch, fmt):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _run(src, out_dir, fmt=fmt)["status"] == "ok"
    cmd = fake.final_cmds[0]
    assert _muxer_of(cmd) == "mp4"
    assert _maps(cmd) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9"]
    assert cmd[cmd.index("-c:s") + 1] == "mov_text"
    assert "0:t?" not in _maps(cmd)


def test_j5_webm_maps_every_eligible_subtitle_as_webvtt(env, monkeypatch):
    """WebM maps explicitly too now - into its own subtitle format.

    Same absolute indexes and same source order as Matroska; only the output
    codec differs, because WebVTT is the only subtitle format WebM carries.
    """
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _run(src, out_dir, fmt="WebM (VP9)")["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9"]
    assert fake.mux_cmd[fake.mux_cmd.index("-c:s") + 1] == "webvtt"
    assert build_stream_map_args("webm", THREE_TEXT) == [
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:3", "-map", "0:6",
        "-map", "0:9", "-c:s", "webvtt"]


# ══ GROUP K — Tab 2b MP4 -> MKV fallback ═════════════════════════════════════

def test_k1_fallback_mkv_attempt_maps_every_eligible_subtitle(env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    _probe(monkeypatch, THREE_TEXT, calls=calls)
    fake.finals = [(1, "muxer error")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert result.get("fallback_used") is True
    assert fake.final_muxers == ["mp4", "matroska"]
    assert len(fake.final_cmds) == 2
    assert len(calls) == 1

    first, second = fake.final_cmds
    assert _maps(first) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9"]
    assert first[first.index("-c:s") + 1] == "mov_text"

    assert _maps(second) == ["0:v:0", "0:a?", "0:3", "0:6", "0:9", "0:t?"]
    assert _pair_count(second, "-c:t", "copy") == 1
    assert "-c:s" not in second
    assert "0:s:0?" not in _maps(second)


def test_k2_fallback_with_no_eligible_subtitles_still_says_sn(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "dvd_subtitle")])
    fake.finals = [(1, "muxer error")]

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"
    assert "-sn" in fake.final_cmds[1]


# ══ GROUP L — failure does not silently drop tracks ══════════════════════════

def test_l1_a_failing_multi_subtitle_mkv_is_terminal(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)
    fake.finals = [(1, "Error initializing output stream 0:4")]

    result = _run(src, out_dir)

    assert result["status"] != "ok"
    assert len(fake.final_cmds) == 1
    assert _sub_maps(fake.final_cmds[0]) == ["0:3", "0:6", "0:9"]


def test_l2_no_retry_ever_drops_a_subtitle(env, monkeypatch):
    """A second attempt that quietly kept fewer tracks would be worse than a
    visible failure. There is no such attempt."""
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)
    fake.finals = [(1, "boom"), (1, "boom"), (1, "boom")]

    assert _run(src, out_dir)["status"] != "ok"
    assert len(fake.final_cmds) == 1


def test_l3_fallback_double_failure_stops_at_two_attempts(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)
    fake.finals = [(1, "muxer error"), (1, "subtitle encoder failed")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] != "ok"
    assert len(fake.final_cmds) == 2
    assert fake.final_muxers == ["mp4", "matroska"]


# ══ GROUP M — probe counts ═══════════════════════════════════════════════════

def test_m1_direct_mkv_probes_the_inventory_once(env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    _probe(monkeypatch, THREE_TEXT, calls=calls)

    assert _run(src, out_dir)["status"] == "ok"
    assert len(calls) == 1


def test_m2_two_pass_mkv_probes_the_inventory_once(env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    _probe(monkeypatch, THREE_TEXT, calls=calls)

    assert _two_pass(src, out_dir)["status"] == "ok"
    assert len(calls) == 1


def test_m3_fallback_probes_the_inventory_once_for_the_whole_lifecycle(
        env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    _probe(monkeypatch, THREE_TEXT, calls=calls)
    fake.finals = [(1, "muxer error")]

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"
    assert len(calls) == 1


def test_m4_preservation_runs_no_subtitle_extraction_of_its_own(
        env, monkeypatch):
    """Embedded preservation and the opt-in sidecar feature are separate: with
    extraction off, three mapped tracks still cost zero extraction calls."""
    src, out_dir, fake = env
    _probe(monkeypatch, THREE_TEXT)

    assert _run(src, out_dir)["status"] == "ok"
    assert fake.subtitle_cmds == []


# ══ GROUP N — per-file isolation ═════════════════════════════════════════════

def test_n1_subtitle_indexes_do_not_leak_between_sequential_jobs(
        env, monkeypatch, tmp_path):
    src, out_dir, fake = env

    _probe(monkeypatch, THREE_TEXT)
    assert _run(src, out_dir)["status"] == "ok"
    a_maps = _sub_maps(fake.mux_cmd)

    src_b = tmp_path / "B.mov"
    src_b.write_bytes(b"s" * SRC_BYTES)
    _probe(monkeypatch, [_sub(2, "hdmv_pgs_subtitle")])
    assert _run(src_b, out_dir)["status"] == "ok"
    b_cmd = fake.mux_cmd

    src_c = tmp_path / "C.mov"
    src_c.write_bytes(b"s" * SRC_BYTES)
    _probe(monkeypatch, [_sub(8, "ass")])
    assert _run(src_c, out_dir)["status"] == "ok"
    c_maps = _sub_maps(fake.mux_cmd)

    assert a_maps == ["0:3", "0:6", "0:9"]
    assert _sub_maps(b_cmd) == []
    assert "-sn" in b_cmd
    assert c_maps == ["0:8"]


def test_n2_the_helper_does_not_mutate_its_input(env):
    streams = [dict(s) for s in THREE_TEXT]
    before = [dict(s) for s in streams]
    build_matroska_stream_map_args(streams)
    assert streams == before
