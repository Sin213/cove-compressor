"""WebM explicit stream policy and multi-audio target accounting.

WebM was the last video container Cove produced whose stream selection was
still ffmpeg's implicit one. Implicit selection takes the first video, the
first *audio*, and the first subtitle stream its default text encoder can
reach - so a three-language source arrived as a one-language file, and a
source with three subtitle tracks arrived with two of them missing. Nothing
said so; the job reported success.

WebM now names what it carries, exactly like MP4 and Matroska do:

    -map 0:v:0                first video, once
    -map 0:a?                 every audio stream, source order, optional
    -map 0:<absolute index>   every WebM-safe text subtitle stream
    -c:s webvtt               only when at least one is mapped
    -sn                       when none is

Attachments and data streams are simply never named, which excludes them
without negative filters - WebM has no attachment support to preserve.

Mapping every audio stream also means paying for every audio stream. The
selected audio bitrate stays a *per track* value, so the target-size budget
reserves `count x audio_kbps`, and a target too small to hold that plus a
floor of video is refused before the encoder is ever started rather than
clamped to 80 kbps and silently overshot. WebM therefore joins MP4 and
Matroska in needing the shared stream inventory, and joins them in failing
closed when that inventory cannot be trusted.

The subtitle vocabulary here is deliberately its own set rather than MP4's:
every codec in it was transcoded into a real WebM by the installed ffmpeg
before it was written down (subrip, ass, webvtt, mov_text and raw text all
produced a `webvtt` output stream whose text survived). Codecs that could not
be produced at all on this build are not in it.

Locked down here:

  A. Explicit video/audio maps, and MP4/MKV untouched by them.
  B. Zero, one and many audio streams in the budget arithmetic.
  C. `Target file size` reserves `count x audio_kbps`.
  D. `Target reduction` does the same.
  E. An impossible target is refused before any encode.
  F. Subtitle eligibility follows the runtime-proven vocabulary only.
  G. Every eligible subtitle is mapped, in source order, once.
  H. Nothing eligible means `-sn` and no `-c:s`.
  I. Absolute ffprobe indexes, never relative subtitle selectors.
  J. Attachments and data streams are never mapped into WebM.
  K. Quality mode keeps its CRF behaviour and still maps everything.
  L. A failed inventory fails target modes closed and disables WebM
     subtitles, without ever disabling audio.
  M. One stream inventory probe per file, at most.
  N. Two-pass pass 1 stays video-only; both passes share one bitrate.
  O. Nothing leaks between files.
  P. MP4 and Matroska behaviour is unchanged by any of it.

No ffmpeg and no real media here: `run_ffmpeg` and the inventory probe are
faked, but the fakes reproduce what production observes - a *mandatory* map
matching no stream fails the invocation, an encode writes its temp output,
and pass 1 is distinguishable from the final mux by its null muxer. The
codec vocabulary itself is not taken on faith from these fakes; it was
established against the real encoder first.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    MIN_VIDEO_KBPS,
    StreamInventory,
    SubtitleProbeError,
    build_matroska_stream_map_args,
    build_mp4_stream_map_args,
    build_pass1_map_args_for,
    build_stream_map_args,
    build_webm_stream_map_args,
    calc_video_bitrate_kbps,
    compress_video,
    total_audio_kbps,
    webm_mappable_subtitle_indexes,
    webm_subtitle_codec_is_compatible,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _value_after(cmd, flag) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _video_kbps_of(cmd) -> int:
    v = _value_after(cmd, "-b:v")
    assert v is not None, "invocation carried no -b:v"
    return int(str(v).rstrip("k"))


def _sub(index: int, codec: str, language: str = "eng",
         title: str | None = None) -> dict:
    s = {"index": index, "codec_name": codec, "codec_type": "subtitle",
         "tags": {"language": language}}
    if title is not None:
        s["tags"]["title"] = title
    return s


# Every codec below was transcoded into a real WebM by the installed ffmpeg
# and came back out as a `webvtt` stream with its text intact. Nothing is in
# this list on the strength of a fake probe dictionary.
PROVEN_WEBM_TEXT_CODECS = ["subrip", "ass", "webvtt", "mov_text", "text"]

# Bitmap subtitle codecs: no text encoder can reach them, so naming one would
# fail the whole job rather than lose one track.
BITMAP_CODECS = ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"]

# Deliberately excluded even though they are text: MicroDVD is structurally
# unreachable in Cove's single-input architecture (Tab 7), and EIA-608 has
# never been exercised against a genuine fixture.
DEFERRED_TEXT_CODECS = ["microdvd", "eia_608"]

THREE_TEXT = [_sub(4, "subrip", "eng"), _sub(10, "ass", "fre"),
              _sub(17, "webvtt", "ger")]


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, recording and classifying each invocation.

    `audio_streams` / `attachments` model what the source actually holds. They
    are not consulted as a probe: they only decide whether a *mandatory* map
    could be satisfied, which is what makes the optionality of `0:a?`
    observable as behaviour rather than merely asserted about a string.
    """

    def __init__(self, finals=None, pass1=None, subtitle=None,
                 audio_streams=1, attachments=0, encode_bytes=b"v" * 10):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.subtitle = list(subtitle or [])
        self.audio_streams = audio_streams
        self.attachments = attachments
        self.encode_bytes = encode_bytes
        self.subtitle_cmds: list[list] = []
        self.pass1_cmds: list[list] = []
        self.final_cmds: list[list] = []

    @staticmethod
    def _next(q):
        return q.pop(0) if q else (0, "")

    def _unsatisfied_map(self, cmd) -> str | None:
        maps = _maps(cmd)
        if self.audio_streams == 0 and ("0:a" in maps or "0:a:0" in maps):
            return "Stream map '0:a' matches no streams."
        if self.attachments == 0 and "0:t" in maps:
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


def _inventory(monkeypatch, audio_count=1, subtitles=(), calls=None,
               error=None, per_file=None):
    """Fake the one stream inventory probe production is allowed to make.

    Patching the low-level probe rather than its wrapper keeps the real
    failure-to-`None` conversion in play, so "probe failed" and "no audio"
    stay as distinguishable here as they are in production.
    """
    def fake(path):
        p = Path(path)
        if calls is not None:
            calls.append(p)
        if per_file is not None:
            spec = per_file[p.name]
            if isinstance(spec, Exception):
                raise spec
            return StreamInventory(
                subtitles=[dict(s) for s in spec[1]], audio_count=spec[0])
        if error is not None:
            raise error
        return StreamInventory(
            subtitles=[dict(s) for s in subtitles], audio_count=audio_count)

    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", fake)
    return fake


# Sized so every target below is genuinely smaller than the original - a
# target that is not is a legitimate pre-encode skip, and would hide the
# accounting these tests are about.
SRC_BYTES = 8 * 1024 * 1024
DURATION = 10.0


def _expected_video_kbps(target_bytes: int, audio_kbps: int,
                         audio_count: int) -> int:
    """The contract's arithmetic, written out rather than imported.

    Deliberately not a call to `calc_video_bitrate_kbps`: a test that asks
    production to confirm its own arithmetic proves nothing about it.
    """
    usable = (target_bytes * 0.97 * 8) / DURATION / 1000.0
    return max(int(usable - audio_kbps * audio_count), MIN_VIDEO_KBPS)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder/probe stack."""
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    # The encoder is faked, so nothing it "writes" is real media. The final
    # readability gate is a separate contract with its own suite.
    monkeypatch.setattr(compressor, "_final_output_is_readable",
                        lambda path, cancel_flag=None: True)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: DURATION)
    monkeypatch.setattr(compressor, "nvenc_available",
                        lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available",
                        lambda e="hevc_amf": False)
    _inventory(monkeypatch, audio_count=1)
    return src, out_dir, fake


def _run(src, out_dir, fmt="WebM (VP9)", mode="Quality preset",
         mode_value="Balanced", audio_kbps="128", cancel_flag=None, **kw):
    """The ordinary public `compress_video` entry point - no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, audio_kbps,
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


# ══ GROUP A — explicit video and audio maps ══════════════════════════════════

def test_a1_webm_maps_the_first_video_stream_exactly_once(env, monkeypatch):
    """Alternate video streams and cover art are not video Cove re-encodes."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    assert _run(src, out_dir)["status"] == "ok"
    assert _maps(fake.mux_cmd).count("0:v:0") == 1
    assert [m for m in _maps(fake.mux_cmd) if m.startswith("0:v")] == ["0:v:0"]


def test_a2_webm_maps_every_audio_stream_exactly_once(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir)["status"] == "ok"
    assert _maps(fake.mux_cmd).count("0:a?") == 1


def test_a3_webm_never_selects_only_the_first_audio_stream(env, monkeypatch):
    """`0:a:0?` would silently keep exactly one track - the old behaviour."""
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir)
    maps = _maps(fake.mux_cmd)
    assert "0:a:0?" not in maps
    assert "0:a:0" not in maps
    assert not [m for m in maps if m.startswith("0:a") and m != "0:a?"]


def test_a4_webm_still_converts_a_source_with_no_audio(env, monkeypatch):
    """The trailing `?` is the whole point: the map is optional, not absent."""
    src, out_dir, fake = env
    fake.audio_streams = 0
    _inventory(monkeypatch, audio_count=0)

    result = _run(src, out_dir)
    assert result["status"] == "ok"
    assert "0:a?" in _maps(fake.mux_cmd)


def test_a5_webm_video_map_precedes_its_audio_map(env, monkeypatch):
    """Output stream order follows map order; video first is not incidental."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=2, subtitles=THREE_TEXT)

    _run(src, out_dir)
    maps = _maps(fake.mux_cmd)
    assert maps[0] == "0:v:0"
    assert maps[1] == "0:a?"


def test_a6_builder_shape_for_webm_is_video_audio_then_subtitles():
    assert build_webm_stream_map_args(None)[:4] == [
        "-map", "0:v:0", "-map", "0:a?"]
    assert build_stream_map_args("webm", THREE_TEXT)[:4] == [
        "-map", "0:v:0", "-map", "0:a?"]


def test_a7_mp4_and_matroska_maps_are_untouched_by_the_webm_policy():
    assert build_mp4_stream_map_args([_sub(3, "subrip")]) == [
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:3", "-c:s", "mov_text"]
    assert build_matroska_stream_map_args([_sub(3, "subrip")]) == [
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:3",
        "-map", "0:t?", "-c:t", "copy"]


# ══ GROUP B — zero / one / many audio accounting ═════════════════════════════

TARGET_MB = 4
TARGET_BYTES = TARGET_MB * 1024 * 1024


def test_b1_zero_audio_reserves_nothing():
    assert total_audio_kbps(128, 0) == 0
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 0) == \
        _expected_video_kbps(TARGET_BYTES, 128, 0)


def test_b2_one_audio_is_integer_identical_to_the_predecessor():
    """The single-stream arithmetic WebM has always been given."""
    predecessor = int((TARGET_BYTES * 0.97 * 8) / DURATION / 1000.0) - 128
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 1) == predecessor


def test_b3_two_audio_streams_reserve_twice_the_selected_bitrate():
    assert total_audio_kbps(128, 2) == 256
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 2) == \
        _expected_video_kbps(TARGET_BYTES, 128, 2)


def test_b4_three_audio_streams_reserve_three_times_it():
    assert total_audio_kbps(128, 3) == 384
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 3) == \
        _expected_video_kbps(TARGET_BYTES, 128, 3)


def test_b5_the_selected_bitrate_is_never_divided_between_tracks():
    """Three tracks at 128 cost 384, they do not share 128."""
    one = calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 1)
    three = calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 3)
    assert one - three == 256


def test_b6_non_audio_streams_never_enter_the_audio_count(env, monkeypatch):
    """Subtitles, attachments and the video itself are not audio."""
    src, out_dir, fake = env
    fake.audio_streams = 1
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert _video_kbps_of(fake.mux_cmd) == \
        _expected_video_kbps(TARGET_BYTES, 128, 1)


# ══ GROUP C — Target file size through the public path ═══════════════════════

@pytest.mark.parametrize("count", [0, 1, 3])
def test_c1_target_file_size_reserves_one_budget_per_audio_stream(
        env, monkeypatch, count):
    src, out_dir, fake = env
    fake.audio_streams = count
    _inventory(monkeypatch, audio_count=count)

    assert _run(src, out_dir, mode="Target file size",
                mode_value=TARGET_MB)["status"] == "ok"
    assert _video_kbps_of(fake.mux_cmd) == \
        _expected_video_kbps(TARGET_BYTES, 128, count)


def test_c2_a_different_selected_audio_bitrate_scales_per_track(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB,
         audio_kbps="192")
    assert _video_kbps_of(fake.mux_cmd) == \
        _expected_video_kbps(TARGET_BYTES, 192, 3)


def test_c3_three_audio_costs_exactly_two_tracks_more_than_one(
        env, monkeypatch):
    src, out_dir, fake = env

    fake.audio_streams = 1
    _inventory(monkeypatch, audio_count=1)
    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    one = _video_kbps_of(fake.mux_cmd)

    fake.final_cmds.clear()
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)
    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    three = _video_kbps_of(fake.mux_cmd)

    assert one - three == 256


# ══ GROUP D — Target reduction ═══════════════════════════════════════════════

REDUCE_PCT = 50
REDUCED_BYTES = int(SRC_BYTES * 50 / 100.0)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_d1_target_reduction_reserves_one_budget_per_audio_stream(
        env, monkeypatch, count):
    src, out_dir, fake = env
    fake.audio_streams = count
    _inventory(monkeypatch, audio_count=count)

    assert _run(src, out_dir, mode="Target reduction",
                mode_value=REDUCE_PCT)["status"] == "ok"
    assert _video_kbps_of(fake.mux_cmd) == \
        _expected_video_kbps(REDUCED_BYTES, 128, count)


def test_d2_target_byte_derivation_is_unchanged_by_the_audio_count(
        env, monkeypatch):
    """Only the reservation changes; the percentage still means what it did."""
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2)

    _run(src, out_dir, mode="Target reduction", mode_value=REDUCE_PCT)
    assert _video_kbps_of(fake.mux_cmd) + 256 == \
        int((REDUCED_BYTES * 0.97 * 8) / DURATION / 1000.0)


# ══ GROUP E — impossible target ══════════════════════════════════════════════

# 0.5 MB over 10 s leaves ~406 kbps usable: enough for one 128 kbps track plus
# a floor of video, nowhere near enough for three.
TINY_MB = 0.5


def test_e1_a_target_too_small_for_every_track_is_refused(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    result = _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert result["status"] == "error"
    assert "target too small" in result["msg"]
    assert "3 audio tracks" in result["msg"]


def test_e2_the_refusal_happens_before_any_encode(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert fake.final_cmds == []
    assert fake.pass1_cmds == []


def test_e3_the_refusal_leaves_no_output_behind(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert list(out_dir.iterdir()) == []
    assert src.exists()


def test_e4_the_same_target_is_accepted_when_one_track_fits(env, monkeypatch):
    """The refusal is about the budget, not about the number 0.5."""
    src, out_dir, fake = env
    fake.audio_streams = 1
    _inventory(monkeypatch, audio_count=1)

    result = _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert result["status"] == "ok"
    assert _video_kbps_of(fake.mux_cmd) > MIN_VIDEO_KBPS


def test_e5_an_impossible_target_is_never_clamped_to_the_floor(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert not [c for c in fake.final_cmds if "-b:v" in c]


def test_e6_target_reduction_refuses_the_same_way(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    result = _run(src, out_dir, mode="Target reduction", mode_value=94)
    assert result["status"] == "error"
    assert "target too small" in result["msg"]
    assert fake.final_cmds == []


# ══ GROUP F — subtitle eligibility ═══════════════════════════════════════════

@pytest.mark.parametrize("codec", PROVEN_WEBM_TEXT_CODECS)
def test_f1_every_runtime_proven_codec_is_eligible(codec):
    assert webm_subtitle_codec_is_compatible(codec)
    assert webm_mappable_subtitle_indexes([_sub(5, codec)]) == [5]


@pytest.mark.parametrize("codec", PROVEN_WEBM_TEXT_CODECS)
def test_f2_every_proven_codec_maps_by_absolute_index(env, monkeypatch, codec):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[_sub(7, codec)])

    assert _run(src, out_dir)["status"] == "ok"
    assert "0:7" in _maps(fake.mux_cmd)
    assert _value_after(fake.mux_cmd, "-c:s") == "webvtt"


@pytest.mark.parametrize("codec", BITMAP_CODECS)
def test_f3_bitmap_subtitles_are_excluded(env, monkeypatch, codec):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[_sub(5, codec)])

    assert not webm_subtitle_codec_is_compatible(codec)
    assert _run(src, out_dir)["status"] == "ok"
    assert "0:5" not in _maps(fake.mux_cmd)
    assert "-sn" in fake.mux_cmd


@pytest.mark.parametrize("codec", DEFERRED_TEXT_CODECS)
def test_f4_deferred_text_codecs_stay_excluded(env, monkeypatch, codec):
    """MicroDVD (Tab 7: structurally unreachable) and EIA-608 (no fixture)."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[_sub(5, codec)])

    assert not webm_subtitle_codec_is_compatible(codec)
    _run(src, out_dir)
    assert "0:5" not in _maps(fake.mux_cmd)


@pytest.mark.parametrize("codec", ["", None, "not_a_codec", "  ", "arib_caption"])
def test_f5_unknown_codecs_are_excluded(codec):
    assert not webm_subtitle_codec_is_compatible(codec)
    assert webm_mappable_subtitle_indexes([_sub(5, codec)]) == []


def test_f6_codec_matching_ignores_case_and_padding():
    assert webm_subtitle_codec_is_compatible("  SubRip ")
    assert webm_subtitle_codec_is_compatible("WEBVTT")


def test_f7_a_stream_without_a_usable_index_is_never_guessed_at():
    assert webm_mappable_subtitle_indexes([
        {"index": True, "codec_name": "subrip"},
        {"index": -1, "codec_name": "subrip"},
        {"index": "4", "codec_name": "subrip"},
        {"codec_name": "subrip"},
        "not a stream",
    ]) == []


# ══ GROUP G — every eligible subtitle, in source order ═══════════════════════

def test_g1_all_three_eligible_subtitles_are_mapped(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    assert _run(src, out_dir)["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:4", "0:10", "0:17"]


def test_g2_the_subtitle_encoder_is_named_exactly_once(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    assert fake.mux_cmd.count("-c:s") == 1
    assert _value_after(fake.mux_cmd, "-c:s") == "webvtt"
    assert "-sn" not in fake.mux_cmd


def test_g3_source_order_is_preserved_not_sorted(env, monkeypatch):
    """Not by language, codec, title or disposition - by file order."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[
        _sub(9, "webvtt", "zul", title="zzz"),
        _sub(2, "subrip", "aar", title="aaa"),
        _sub(6, "ass", "eng", title="mmm"),
    ])

    _run(src, out_dir)
    assert _maps(fake.mux_cmd)[2:] == ["0:9", "0:2", "0:6"]


def test_g4_ineligible_streams_are_skipped_around_not_scanned_up_to(
        env, monkeypatch):
    """A bitmap track between two text tracks must not truncate the list."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[
        _sub(3, "subrip"), _sub(5, "hdmv_pgs_subtitle"), _sub(8, "ass"),
    ])

    _run(src, out_dir)
    assert _maps(fake.mux_cmd)[2:] == ["0:3", "0:8"]


def test_g5_each_eligible_stream_is_mapped_once(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    maps = _maps(fake.mux_cmd)
    assert len(maps) == len(set(maps))


# ══ GROUP H — nothing eligible ═══════════════════════════════════════════════

def test_h1_no_subtitle_streams_at_all_disables_subtitles(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[])

    assert _run(src, out_dir)["status"] == "ok"
    assert "-sn" in fake.mux_cmd
    assert "-c:s" not in fake.mux_cmd
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?"]


def test_h2_only_ineligible_subtitle_streams_disables_subtitles(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[
        _sub(3, "hdmv_pgs_subtitle"), _sub(4, "microdvd"),
        _sub(5, "who_knows"),
    ])

    assert _run(src, out_dir)["status"] == "ok"
    assert "-sn" in fake.mux_cmd
    assert "webvtt" not in fake.mux_cmd


def test_h3_subtitle_selection_is_never_left_implicit(env, monkeypatch):
    """Either eligible streams are named, or `-sn` says there are none."""
    src, out_dir, fake = env
    for subs in ([], [_sub(3, "subrip")], [_sub(3, "dvd_subtitle")]):
        fake.final_cmds.clear()
        _inventory(monkeypatch, audio_count=1, subtitles=subs)
        _run(src, out_dir)
        cmd = fake.mux_cmd
        named = [m for m in _maps(cmd) if m not in ("0:v:0", "0:a?")]
        assert bool(named) ^ ("-sn" in cmd)


# ══ GROUP I — absolute index safety ══════════════════════════════════════════

def test_i1_sparse_absolute_indexes_are_used_verbatim(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    for token in ("0:4", "0:10", "0:17"):
        assert token in _maps(fake.mux_cmd)


@pytest.mark.parametrize("selector", ["0:s:0", "0:s:1", "0:s?", "0:s:0?",
                                      "0:s:2", "0:s"])
def test_i2_relative_subtitle_selectors_are_never_emitted(
        env, monkeypatch, selector):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    assert selector not in _maps(fake.mux_cmd)


def test_i3_excluding_a_bitmap_track_does_not_renumber_the_next_one(
        env, monkeypatch):
    """The whole reason indexes are absolute: index 8 stays index 8."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=[
        _sub(3, "dvd_subtitle"), _sub(8, "subrip"),
    ])

    _run(src, out_dir)
    assert _maps(fake.mux_cmd)[2:] == ["0:8"]


def test_i4_the_builder_emits_no_relative_selector_of_any_kind():
    args = build_webm_stream_map_args(THREE_TEXT)
    assert not [m for m in _maps(args) if m.startswith("0:s")]


# ══ GROUP J — attachments and data streams ═══════════════════════════════════

def test_j1_webm_never_maps_attachments(env, monkeypatch):
    src, out_dir, fake = env
    fake.attachments = 2
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    assert not [m for m in _maps(fake.mux_cmd) if m.startswith("0:t")]
    assert "-c:t" not in fake.mux_cmd


def test_j2_webm_never_maps_data_streams(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir)
    assert not [m for m in _maps(fake.mux_cmd) if m.startswith("0:d")]
    assert "-c:d" not in fake.mux_cmd


def test_j3_an_attachment_or_data_record_never_becomes_a_subtitle_map():
    """Only subtitle records reach the vocabulary; nothing else is named."""
    args = build_webm_stream_map_args([
        {"index": 2, "codec_type": "attachment", "codec_name": "ttf"},
        {"index": 6, "codec_type": "data", "codec_name": "bin"},
        _sub(9, "subrip"),
    ])
    assert _maps(args) == ["0:v:0", "0:a?", "0:9"]


def test_j4_matroska_still_preserves_its_attachments():
    """The neighbour contract WebM's exclusion must not have disturbed."""
    args = build_stream_map_args("matroska", [_sub(3, "subrip")])
    assert args[-4:] == ["-map", "0:t?", "-c:t", "copy"]


# ══ GROUP K — quality mode ═══════════════════════════════════════════════════

def test_k1_quality_mode_maps_all_audio_and_every_safe_subtitle(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3, subtitles=[
        _sub(4, "subrip"), _sub(9, "ass")])

    assert _run(src, out_dir, mode="Quality preset",
                mode_value="Balanced")["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:4", "0:9"]
    assert _value_after(fake.mux_cmd, "-c:s") == "webvtt"


def test_k2_quality_mode_keeps_vp9_constant_quality_rate_control(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Quality preset", mode_value="Balanced")
    assert _value_after(fake.mux_cmd, "-crf") == "31"
    assert _value_after(fake.mux_cmd, "-b:v") == "0"


@pytest.mark.parametrize("count", [0, 1, 5])
def test_k3_the_audio_count_never_reaches_the_quality_value(
        env, monkeypatch, count):
    src, out_dir, fake = env
    fake.audio_streams = count
    _inventory(monkeypatch, audio_count=count)

    _run(src, out_dir, mode="Quality preset", mode_value="Web Small")
    assert _value_after(fake.mux_cmd, "-crf") == "37"


def test_k4_quality_mode_has_no_impossible_target_gate(env, monkeypatch):
    """There is no target to be impossible against."""
    src, out_dir, fake = env
    fake.audio_streams = 8
    _inventory(monkeypatch, audio_count=8)

    assert _run(src, out_dir, mode="Quality preset",
                mode_value="Balanced")["status"] == "ok"


def test_k5_quality_mode_is_a_single_encode(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1)

    _run(src, out_dir, mode="Quality preset", mode_value="Balanced")
    assert len(fake.final_cmds) == 1
    assert fake.pass1_cmds == []


# ══ GROUP L — inventory failure ══════════════════════════════════════════════

def test_l1_target_file_size_fails_closed_on_a_failed_inventory(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("ffprobe exit 1"))

    result = _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert result["status"] == "error"
    assert "audio streams" in result["msg"]
    assert fake.final_cmds == []
    assert fake.pass1_cmds == []
    assert list(out_dir.iterdir()) == []


def test_l2_target_reduction_fails_closed_on_a_failed_inventory(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("unreadable output"))

    result = _run(src, out_dir, mode="Target reduction", mode_value=REDUCE_PCT)
    assert result["status"] == "error"
    assert fake.final_cmds == []


def test_l3_a_failed_inventory_is_never_read_as_zero_audio(env, monkeypatch):
    """Zero audio is a legitimate answer; "we could not tell" is not."""
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("boom"))

    result = _run(src, out_dir, mode="Target file size", mode_value=TINY_MB)
    assert result["status"] == "error"
    assert "target too small" not in result["msg"]
    assert fake.final_cmds == []


def test_l4_a_failed_inventory_is_never_read_as_one_audio(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("boom"))

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert not [c for c in fake.final_cmds if "-b:v" in c]


def test_l5_quality_mode_keeps_all_audio_when_the_inventory_fails(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, error=SubtitleProbeError("boom"))

    assert _run(src, out_dir)["status"] == "ok"
    assert "0:a?" in _maps(fake.mux_cmd)
    assert "0:v:0" in _maps(fake.mux_cmd)


def test_l6_quality_mode_fails_subtitles_closed_when_the_inventory_fails(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("boom"))

    _run(src, out_dir)
    assert "-sn" in fake.mux_cmd
    assert "-c:s" not in fake.mux_cmd
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?"]


def test_l7_a_failed_inventory_never_returns_to_implicit_selection(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("boom"))

    _run(src, out_dir)
    assert _maps(fake.mux_cmd) != []


def test_l8_the_builder_treats_none_as_disable_not_as_empty():
    assert build_webm_stream_map_args(None) == [
        "-map", "0:v:0", "-map", "0:a?", "-sn"]


# ══ GROUP M — one inventory probe per file ═══════════════════════════════════

def test_m1_webm_target_mode_probes_the_inventory_exactly_once(
        env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=2, subtitles=THREE_TEXT, calls=calls)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert len(calls) == 1


def test_m2_webm_quality_mode_probes_the_inventory_at_most_once(
        env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT, calls=calls)

    _run(src, out_dir, mode="Quality preset", mode_value="Balanced")
    assert len(calls) <= 1
    assert len(calls) == 1


def test_m3_the_two_pass_path_probes_the_inventory_exactly_once(
        env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT, calls=calls)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert len(fake.pass1_cmds) == 1 and len(fake.final_cmds) == 1
    assert len(calls) == 1


def test_m4_sidecar_extraction_adds_no_second_probe(env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=1,
               subtitles=[_sub(3, "subrip", "eng")], calls=calls)

    _run(src, out_dir, extract_english_subtitles=True)
    assert len(calls) == 1


def test_m5_sidecar_extraction_in_target_mode_adds_no_second_probe(
        env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=2,
               subtitles=[_sub(3, "subrip", "eng")], calls=calls)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB,
         extract_english_subtitles=True)
    assert len(calls) == 1


def test_m6_sidecar_extraction_still_writes_its_sidecar(env, monkeypatch):
    """Reusing the inventory must not have cost the feature its output."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1,
               subtitles=[_sub(3, "subrip", "eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)
    assert result["status"] == "ok"
    assert len(fake.subtitle_cmds) == 1
    assert "0:3" in _maps(fake.subtitle_cmds[0])
    assert (out_dir / "Movie.eng.srt").exists()


def test_m7_no_audio_only_or_subtitle_only_probe_is_introduced(
        env, monkeypatch):
    """The one inventory is the only ffprobe the stream policy is allowed."""
    src, out_dir, fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=3, subtitles=THREE_TEXT, calls=calls)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert calls == [src]


# ══ GROUP N — two-pass ═══════════════════════════════════════════════════════

def test_n1_pass_one_maps_only_the_video_stream(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3, subtitles=THREE_TEXT)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert _maps(fake.pass1_cmds[0]) == ["0:v:0"]
    assert build_pass1_map_args_for("webm") == ["-map", "0:v:0"]


def test_n2_pass_one_disables_audio(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert "-an" in fake.pass1_cmds[0]
    assert "0:a?" not in _maps(fake.pass1_cmds[0])


def test_n3_pass_one_carries_no_subtitle_maps(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    cmd = fake.pass1_cmds[0]
    assert not [m for m in _maps(cmd) if m not in ("0:v:0",)]


def test_n4_pass_one_names_no_subtitle_encoder(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1, subtitles=THREE_TEXT)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    cmd = fake.pass1_cmds[0]
    assert "-c:s" not in cmd
    assert "webvtt" not in cmd
    assert "-c:a" not in cmd
    assert "-b:a" not in cmd


def test_n5_pass_two_carries_the_full_stream_policy(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3, subtitles=THREE_TEXT)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:4", "0:10", "0:17"]
    assert _value_after(fake.mux_cmd, "-c:s") == "webvtt"


def test_n6_both_passes_are_handed_the_same_video_bitrate(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    expected = _expected_video_kbps(TARGET_BYTES, 128, 3)
    assert _video_kbps_of(fake.pass1_cmds[0]) == expected
    assert _video_kbps_of(fake.mux_cmd) == expected


def test_n7_one_compression_attempt_is_exactly_two_encode_subprocesses(
        env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1)

    _run(src, out_dir, mode="Target file size", mode_value=TARGET_MB)
    assert len(fake.pass1_cmds) == 1
    assert len(fake.final_cmds) == 1
    assert fake.final_muxers == ["webm"]


# ══ GROUP O — per-file isolation ═════════════════════════════════════════════

def test_o1_audio_counts_and_subtitle_indexes_do_not_leak_between_files(
        tmp_path, monkeypatch):
    a = tmp_path / "A.mov"
    b = tmp_path / "B.mov"
    c = tmp_path / "C.mov"
    for f in (a, b, c):
        f.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    # The encoder is faked, so nothing it "writes" is real media. The final
    # readability gate is a separate contract with its own suite.
    monkeypatch.setattr(compressor, "_final_output_is_readable",
                        lambda path, cancel_flag=None: True)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: DURATION)
    monkeypatch.setattr(compressor, "nvenc_available", lambda e=None: False)
    monkeypatch.setattr(compressor, "amf_available", lambda e=None: False)
    _inventory(monkeypatch, per_file={
        "A.mov": (3, THREE_TEXT),
        "B.mov": (0, []),
        "C.mov": (1, [_sub(2, "webvtt")]),
    })

    seen = []
    for src, audio in ((a, 3), (b, 0), (c, 1)):
        fake.final_cmds.clear()
        fake.audio_streams = audio
        assert _run(src, out_dir, mode="Target file size",
                    mode_value=TARGET_MB)["status"] == "ok"
        seen.append((_maps(fake.mux_cmd), _video_kbps_of(fake.mux_cmd)))

    assert seen[0] == (["0:v:0", "0:a?", "0:4", "0:10", "0:17"],
                       _expected_video_kbps(TARGET_BYTES, 128, 3))
    assert seen[1] == (["0:v:0", "0:a?"],
                       _expected_video_kbps(TARGET_BYTES, 128, 0))
    assert seen[2] == (["0:v:0", "0:a?", "0:2"],
                       _expected_video_kbps(TARGET_BYTES, 128, 1))


def test_o2_a_refused_file_does_not_poison_the_next_one(env, monkeypatch,
                                                        tmp_path):
    src, out_dir, fake = env
    other = tmp_path / "Other.mov"
    other.write_bytes(b"s" * SRC_BYTES)
    _inventory(monkeypatch, per_file={
        "Movie.mov": (4, []),
        "Other.mov": (1, [_sub(3, "subrip")]),
    })

    fake.audio_streams = 4
    assert _run(src, out_dir, mode="Target file size",
                mode_value=TINY_MB)["status"] == "error"

    fake.audio_streams = 1
    assert _run(other, out_dir, mode="Target file size",
                mode_value=TARGET_MB)["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3"]


# ══ GROUP P — MP4 and Matroska are unchanged ═════════════════════════════════

def test_p1_mp4_keeps_its_exact_stream_policy(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3, subtitles=[
        _sub(3, "subrip"), _sub(5, "hdmv_pgs_subtitle"), _sub(8, "ass")])

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3", "0:8"]
    assert _value_after(fake.mux_cmd, "-c:s") == "mov_text"
    assert "webvtt" not in fake.mux_cmd
    assert not [m for m in _maps(fake.mux_cmd) if m.startswith("0:t")]


def test_p2_matroska_keeps_its_exact_stream_policy(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    fake.attachments = 1
    _inventory(monkeypatch, audio_count=3, subtitles=[
        _sub(3, "subrip"), _sub(5, "hdmv_pgs_subtitle"), _sub(8, "ass")])

    assert _run(src, out_dir, fmt="MKV (H.265)")["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3", "0:8", "0:t?"]
    assert "-c:s" not in fake.mux_cmd
    assert _value_after(fake.mux_cmd, "-c:t") == "copy"


def test_p3_webvtt_is_never_named_outside_webm():
    for muxer in ("mp4", "matroska"):
        assert "webvtt" not in build_stream_map_args(muxer, THREE_TEXT)


def test_p4_mp4_target_accounting_is_unchanged(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value=TARGET_MB)
    assert _video_kbps_of(fake.mux_cmd) == \
        _expected_video_kbps(TARGET_BYTES, 128, 3)


def test_p5_the_mp4_vocabulary_did_not_gain_or_lose_a_codec():
    from cove_compressor.compressor import MP4_TEXT_SUBTITLE_CODECS
    assert MP4_TEXT_SUBTITLE_CODECS == {
        "subrip", "srt", "mov_text", "text", "webvtt", "vtt", "ass", "ssa"}


def test_p6_pass1_maps_are_shared_verbatim_across_containers():
    for muxer in ("mp4", "matroska", "webm"):
        assert build_pass1_map_args_for(muxer) == ["-map", "0:v:0"]


def test_p7_an_unknown_muxer_still_keeps_implicit_selection():
    """The policy table gained WebM, not a default for everything."""
    assert build_stream_map_args("avi", THREE_TEXT) == []
    assert build_pass1_map_args_for("avi") == []
