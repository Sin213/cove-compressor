"""Multi-audio preservation and target-size accounting.

Every explicitly mapped container Cove produces - `MP4 (H.265)`,
`MP4 (H.264)` and `MKV (H.265)`, plus the MP4 -> MKV fallback attempt - maps
*every* source audio stream, in source order, with a single optional
`-map 0:a?`. A film with an English, a Japanese and a commentary track keeps
all three; a silent source still converts.

Preserving three audio streams also costs three audio streams' worth of
bitrate, so the target-size arithmetic has to reserve the user's selected
audio bitrate once *per mapped stream*. The setting stays a per-track value:
three tracks at 192 kbps reserve 576 kbps, they do not share 192.

Locked down here:

  A. Container policy: every explicitly mapped container maps `0:a?` exactly
     once. WebM was widened into the same policy later (see
     `tests/test_webm_stream_policy.py`), so it is checked here too.
  B. Zero, one and many audio streams all behave.
  C. `Target file size` reserves `count x audio_kbps`, and one track computes
     exactly what the single-track predecessor computed.
  D. `Target reduction` does the same, without disturbing the reduction math.
  E. A target too small to hold the audio plus a floor of video is rejected
     *before* any encode - never clamped to 80 kbps and silently overshot.
  F. The audio count comes from the one stream inventory the explicit
     mapping policy already probes for; a failed probe is never read as
     "no audio", and target modes fail closed on one.
  G. Two-pass analysis stays video-only; only pass 2 carries audio.
  H. The MP4 -> MKV fallback inherits all of it and re-probes nothing.
  I. The Tab 3 attachment and subtitle contracts survive intact.
  J. Quality-preset mode maps every track without letting the count touch
     CRF/CQ.
  K. Counts are per file: nothing caches one file's audio count into the next.

No ffmpeg and no real media: `run_ffmpeg` and the inventory probe are faked,
but the fakes reproduce what production actually observes - a mandatory map
that matches no stream fails the invocation, an encode writes its temp output,
and pass 1 is distinguishable from the final mux by its null muxer.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    StreamInventory,
    SubtitleProbeError,
    build_matroska_stream_map_args,
    build_mp4_stream_map_args,
    build_pass1_stream_map_args,
    build_stream_map_args,
    calc_video_bitrate_kbps,
    compress_video,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _value_after(cmd, flag) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _video_kbps_of(cmd) -> int:
    """The `-b:v 1424k` the encoder was actually handed."""
    v = _value_after(cmd, "-b:v")
    assert v is not None, "invocation carried no -b:v"
    return int(str(v).rstrip("k"))


def _sub(index: int, codec: str) -> dict:
    return {"index": index, "codec_name": codec, "codec_type": "subtitle",
            "tags": {"language": "eng"}}


TEXT_SUB = [_sub(3, "subrip")]


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, recording and classifying each invocation.

    `audio_streams` models how many audio streams the source holds. It is not
    consulted as a probe - it only decides whether a *mandatory* audio map
    could be satisfied, which is what makes the optionality of `0:a?`
    observable in behaviour rather than merely asserted about a string.
    """

    def __init__(self, finals=None, pass1=None, subtitle=None,
                 audio_streams=1, attachments=1, encode_bytes=b"v" * 10):
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

    Patching the low-level probe (not the wrapper) keeps the real
    failure-to-`None` conversion in play, so "probe failed" and "no audio"
    stay as distinguishable in the test as they are in production.
    """
    def fake(path):
        p = Path(path)
        if calls is not None:
            calls.append(p)
        if per_file is not None:
            spec = per_file[p.name]
            if isinstance(spec, Exception):
                raise spec
            return StreamInventory(subtitles=[], audio_count=spec)
        if error is not None:
            raise error
        return StreamInventory(
            subtitles=[dict(s) for s in subtitles], audio_count=audio_count)

    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", fake)
    return fake


# Sources are sized so every target below is genuinely smaller than the
# original - a target that is not is a legitimate pre-encode skip, and would
# hide the accounting these tests are about.
SRC_BYTES = 8 * 1024 * 1024


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
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": False)
    _inventory(monkeypatch, audio_count=1)
    return src, out_dir, fake


def _run(src, out_dir, fmt="MKV (H.265)", mode="Quality preset",
         mode_value="Balanced", audio_kbps="128", cancel_flag=None, **kw):
    """The ordinary public `compress_video` entry point - no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, audio_kbps,
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


# ══ GROUP A — basic stream mapping ═══════════════════════════════════════════

def test_a1_mp4_h265_maps_every_audio_stream_optionally(env, monkeypatch):
    """The public MP4 path, not just the builder."""
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert "0:a?" in maps
    assert "0:a:0?" not in maps


def test_a2_mp4_h264_maps_every_audio_stream_optionally(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2)

    assert _run(src, out_dir, fmt="MP4 (H.264)")["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert "0:a?" in maps
    assert "0:a:0?" not in maps


def test_a3_matroska_maps_every_audio_stream_optionally(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir, fmt="MKV (H.265)")["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert "0:a?" in maps
    assert "0:a:0?" not in maps


def test_a4_webm_maps_every_audio_stream_optionally(env, monkeypatch):
    """WebM joined the all-audio policy; the selector is the same one."""
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir, fmt="WebM (VP9)")["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert "0:a?" in maps
    assert "0:a:0?" not in maps


@pytest.mark.parametrize("builder", [build_mp4_stream_map_args,
                                     build_matroska_stream_map_args])
def test_a5_audio_map_appears_exactly_once(builder):
    maps = _maps(builder(TEXT_SUB))
    assert maps.count("0:a?") == 1
    assert "0:a" not in maps, "a mandatory map breaks silent sources"


def test_a6_audio_codec_and_bitrate_stay_generic(env, monkeypatch):
    """One `-c:a`/`-b:a` pair covers every mapped stream - no per-track args."""
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, fmt="MP4 (H.265)", audio_kbps="192")
    cmd = fake.mux_cmd
    assert _value_after(cmd, "-c:a") == "aac"
    assert _value_after(cmd, "-b:a") == "192k"
    assert not [t for t in cmd if t.startswith("-c:a:") or t.startswith("-b:a:")]


# ══ GROUP B — zero / one / many audio ════════════════════════════════════════

def test_b1_zero_audio_streams_still_converts(env, monkeypatch):
    """The optional map is what keeps a silent source working."""
    src, out_dir, fake = env
    fake.audio_streams = 0
    _inventory(monkeypatch, audio_count=0)

    result = _run(src, out_dir, fmt="MKV (H.265)")
    assert result["status"] == "ok"
    assert "0:a?" in _maps(fake.mux_cmd)


def test_b2_one_audio_stream_matches_predecessor_target_math(env, monkeypatch):
    """One track reserves exactly one track's bitrate, as it always did."""
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="1", audio_kbps="192")

    target_bytes = 1024 * 1024
    legacy = max(int((target_bytes * 0.97 * 8) / 10.0 / 1000.0 - 192), 80)
    assert _video_kbps_of(fake.mux_cmd) == legacy


def test_b3_two_audio_streams_reserve_two_tracks(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="1", audio_kbps="192")

    total = (1024 * 1024) * 0.97 * 8 / 10.0 / 1000.0
    assert _video_kbps_of(fake.mux_cmd) == int(total - 2 * 192)


def test_b4_three_audio_streams_reserve_three_tracks(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="1", audio_kbps="192")

    total = (1024 * 1024) * 0.97 * 8 / 10.0 / 1000.0
    assert _video_kbps_of(fake.mux_cmd) == int(total - 3 * 192)


def test_b5_only_audio_streams_are_counted(monkeypatch, tmp_path):
    """A mixed inventory counts audio by codec type, not by stream total."""
    payload = {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        {"index": 2, "codec_type": "audio", "codec_name": "ac3"},
        {"index": 3, "codec_type": "audio", "codec_name": "dts"},
        {"index": 4, "codec_type": "subtitle", "codec_name": "subrip"},
        {"index": 5, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"index": 6, "codec_type": "attachment", "codec_name": "ttf"},
        {"index": 7, "codec_type": "data", "codec_name": "bin_data"},
    ]}
    _fake_ffprobe_json(monkeypatch, payload)

    inv = compressor.ffprobe_stream_inventory(tmp_path / "x.mkv")
    assert inv.audio_count == 3
    assert [s["index"] for s in inv.subtitles] == [4, 5]


# ══ GROUP C — target file size accounting ════════════════════════════════════

TARGET_BYTES = 12 * 1024 * 1024
DURATION = 60.0


def _usable_total(target_bytes=TARGET_BYTES, duration=DURATION) -> float:
    return (target_bytes * 0.97 * 8) / duration / 1000.0


def test_c1_one_track_result_is_the_predecessor_result():
    legacy = max(int(_usable_total() - 128), 80)
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 1) == legacy


def test_c2_zero_tracks_reserve_no_audio():
    zero = calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 0)
    one = calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 1)
    assert zero == int(_usable_total())
    assert zero - one == int(_usable_total()) - int(_usable_total() - 128)


def test_c3_two_tracks_reserve_twice():
    assert (calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 2)
            == int(_usable_total() - 256))


def test_c4_three_tracks_reserve_three_times():
    assert (calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 3)
            == int(_usable_total() - 384))


def test_c5_selected_bitrate_is_per_track_not_shared():
    """3 x 192 reserves 576 - the setting is not divided among the tracks."""
    got = calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 192, 3)
    assert got == int(_usable_total() - 576)
    assert got != int(_usable_total() - 192)
    assert got != int(_usable_total() - 64)


def test_c6_three_percent_container_allowance_survives():
    """The usable share is 97% of the target, audio count notwithstanding."""
    no_allowance = int((TARGET_BYTES * 8) / DURATION / 1000.0 - 384)
    assert calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 3) != no_allowance
    assert (calc_video_bitrate_kbps(TARGET_BYTES, DURATION, 128, 3)
            == int(_usable_total() - 384))


# ══ GROUP D — target reduction accounting ════════════════════════════════════

def _reduction_video_kbps(src_size, duration, reduce_pct, audio_kbps, count):
    keep = max(5.0, min(95.0, 100.0 - reduce_pct))
    target_bytes = int(src_size * keep / 100.0)
    return max(int((target_bytes * 0.97 * 8) / duration / 1000.0
                   - audio_kbps * count), 80)


def test_d1_one_track_reduction_matches_legacy(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target reduction",
         mode_value="50", audio_kbps="128")

    assert _video_kbps_of(fake.mux_cmd) == _reduction_video_kbps(
        SRC_BYTES, 10.0, 50.0, 128, 1)


def test_d2_three_track_reduction_reserves_three(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target reduction",
         mode_value="50", audio_kbps="128")

    assert _video_kbps_of(fake.mux_cmd) == _reduction_video_kbps(
        SRC_BYTES, 10.0, 50.0, 128, 3)


def test_d3_zero_track_reduction_reserves_none(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 0
    _inventory(monkeypatch, audio_count=0)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target reduction",
         mode_value="50", audio_kbps="128")

    assert _video_kbps_of(fake.mux_cmd) == _reduction_video_kbps(
        SRC_BYTES, 10.0, 50.0, 128, 0)


def test_d4_reduction_percentage_math_is_unchanged(env, monkeypatch):
    """Audio accounting must not move the target-bytes derivation itself."""
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target reduction",
         mode_value="70", audio_kbps="128")

    target_bytes = int(SRC_BYTES * 30.0 / 100.0)
    expected = max(int((target_bytes * 0.97 * 8) / 10.0 / 1000.0 - 256), 80)
    assert _video_kbps_of(fake.mux_cmd) == expected


# ══ GROUP E — impossible targets ═════════════════════════════════════════════

def test_e1_audio_budget_leaving_too_little_video_is_rejected(env, monkeypatch):
    """4 x 128 kbps against a target with barely any room: no encode at all."""
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    result = _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
                  mode_value="0.7", audio_kbps="128")

    assert result["status"] == "error"
    assert fake.final_cmds == []


def test_e2_audio_alone_exceeding_the_budget_is_rejected(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 6
    _inventory(monkeypatch, audio_count=6)

    result = _run(src, out_dir, fmt="MKV (H.265)", mode="Target file size",
                  mode_value="0.5", audio_kbps="320")

    assert result["status"] == "error"
    assert fake.final_cmds == []


def test_e3_error_names_the_audio_pressure(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    msg = _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
               mode_value="0.7", audio_kbps="128")["msg"].lower()
    assert "audio" in msg
    assert "4" in msg and "128" in msg


def test_e4_source_survives_an_impossible_target(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="0.7", audio_kbps="128")
    assert src.exists()


def test_e5_impossible_target_leaves_no_reserved_output(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="0.7", audio_kbps="128")
    assert list(out_dir.iterdir()) == []


def test_e6_impossible_target_never_reaches_pass_1(env, monkeypatch):
    """The CPU two-pass path must be rejected before it spends an analysis."""
    src, out_dir, fake = env
    fake.audio_streams = 4
    _inventory(monkeypatch, audio_count=4)

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="0.7", audio_kbps="128")
    assert fake.pass1_cmds == []
    assert fake.final_cmds == []


# ══ GROUP F — the shared stream inventory ════════════════════════════════════

def _fake_ffprobe_json(monkeypatch, payload, returncode=0):
    import json
    import subprocess

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)


def test_f1_inventory_counts_zero_audio(monkeypatch, tmp_path):
    _fake_ffprobe_json(monkeypatch, {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
    ]})
    inv = compressor.ffprobe_stream_inventory(tmp_path / "x.mkv")
    assert inv.audio_count == 0
    assert inv.subtitles == []


def test_f2_inventory_counts_many_audio(monkeypatch, tmp_path):
    _fake_ffprobe_json(monkeypatch, {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        {"index": 2, "codec_type": "audio", "codec_name": "aac"},
    ]})
    assert compressor.ffprobe_stream_inventory(tmp_path / "x.mkv").audio_count == 2


def test_f3_subtitle_indexes_and_codecs_survive_the_widening(monkeypatch,
                                                             tmp_path):
    """Tab 3's policy still gets absolute indexes and codec names."""
    _fake_ffprobe_json(monkeypatch, {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        {"index": 4, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"index": 5, "codec_type": "subtitle", "codec_name": "subrip",
         "tags": {"language": "eng"}},
    ]})
    inv = compressor.ffprobe_stream_inventory(tmp_path / "x.mkv")

    assert [(s["index"], s["codec_name"]) for s in inv.subtitles] == [
        (4, "hdmv_pgs_subtitle"), (5, "subrip")]
    assert _maps(build_matroska_stream_map_args(inv.subtitles)) == [
        "0:v:0", "0:a?", "0:5", "0:t?"]


def test_f4_probe_failure_is_not_zero_audio(monkeypatch, tmp_path):
    """A failed probe raises; it never degrades into a count of 0."""
    def boom(cmd, **kw):
        raise OSError("ffprobe vanished")

    monkeypatch.setattr(compressor.subprocess, "run", boom)
    with pytest.raises(SubtitleProbeError):
        compressor.ffprobe_stream_inventory(tmp_path / "x.mkv")

    inv, err = compressor.probe_stream_inventory_once(tmp_path / "x.mkv")
    assert inv is None
    assert err


def test_f5_target_size_fails_closed_on_probe_failure(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("ffprobe exit 1"))

    result = _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
                  mode_value="1", audio_kbps="128")

    assert result["status"] == "error"
    assert fake.final_cmds == []
    assert fake.pass1_cmds == []
    assert list(out_dir.iterdir()) == []


def test_f6_target_reduction_fails_closed_on_probe_failure(env, monkeypatch):
    src, out_dir, fake = env
    _inventory(monkeypatch, error=SubtitleProbeError("ffprobe exit 1"))

    result = _run(src, out_dir, fmt="MKV (H.265)", mode="Target reduction",
                  mode_value="50", audio_kbps="128")

    assert result["status"] == "error"
    assert fake.final_cmds == []


def test_f7_exactly_one_inventory_probe_per_file(env, monkeypatch):
    """Counting audio buys no second ffprobe on top of subtitle discovery."""
    src, out_dir, fake = env
    calls: list[Path] = []
    _inventory(monkeypatch, audio_count=3, subtitles=TEXT_SUB, calls=calls)
    fake.audio_streams = 3

    _run(src, out_dir, fmt="MKV (H.265)", mode="Target file size",
         mode_value="1", audio_kbps="128", extract_english_subtitles=True)

    assert calls == [src]


def test_f8_quality_mode_maps_all_audio_without_a_count(env, monkeypatch):
    """`0:a?` is self-describing: an unknown count cannot revert it."""
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, error=SubtitleProbeError("ffprobe exit 1"))

    result = _run(src, out_dir, fmt="MKV (H.265)", mode="Quality preset",
                  mode_value="Balanced")

    assert result["status"] == "ok"
    maps = _maps(fake.mux_cmd)
    assert "0:a?" in maps
    assert "0:a:0?" not in maps
    # Tab 3's fail-closed subtitle behaviour is untouched by that.
    assert "-sn" in fake.mux_cmd


# ══ GROUP G — two-pass safety ════════════════════════════════════════════════

def _two_pass(env, monkeypatch, count=3):
    src, out_dir, fake = env
    fake.audio_streams = count
    _inventory(monkeypatch, audio_count=count)
    result = _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
                  mode_value="4", audio_kbps="128")
    assert result["status"] == "ok"
    return fake


def test_g1_pass_1_carries_no_audio_map(env, monkeypatch):
    fake = _two_pass(env, monkeypatch)
    assert fake.pass1_cmds, "no analysis pass was recorded"
    assert _maps(fake.pass1_cmds[0]) == ["0:v:0"]
    assert "0:a?" not in _maps(fake.pass1_cmds[0])


def test_g2_pass_1_stays_audio_free(env, monkeypatch):
    fake = _two_pass(env, monkeypatch)
    cmd = fake.pass1_cmds[0]
    assert "-an" in cmd
    assert "-c:a" not in cmd and "-b:a" not in cmd
    assert build_pass1_stream_map_args() == ["-map", "0:v:0"]


def test_g3_pass_2_maps_every_audio_stream(env, monkeypatch):
    fake = _two_pass(env, monkeypatch)
    assert _maps(fake.mux_cmd).count("0:a?") == 1


def test_g4_video_bitrate_is_computed_once_before_pass_1(env, monkeypatch):
    fake = _two_pass(env, monkeypatch)
    total = (4 * 1024 * 1024) * 0.97 * 8 / 10.0 / 1000.0
    expected = int(total - 3 * 128)
    assert _video_kbps_of(fake.pass1_cmds[0]) == expected
    assert _video_kbps_of(fake.mux_cmd) == expected


def test_g5_two_pass_stays_two_invocations(env, monkeypatch):
    fake = _two_pass(env, monkeypatch)
    assert len(fake.pass1_cmds) == 1
    assert len(fake.final_cmds) == 1


# ══ GROUP H — MP4 -> MKV fallback ════════════════════════════════════════════

def _forced_fallback(env, monkeypatch, count=3, **run_kw):
    """Attempt 1 (MP4) fails the way Tab 2b treats as retryable."""
    src, out_dir, fake = env
    fake.audio_streams = count
    fake.finals = [(1, "Could not write header")]
    _inventory(monkeypatch, audio_count=count, subtitles=TEXT_SUB)
    result = _run(src, out_dir, fmt="MP4 (H.265)", **run_kw)
    return fake, result


def test_h1_first_mp4_attempt_maps_all_audio(env, monkeypatch):
    fake, _ = _forced_fallback(env, monkeypatch)
    assert "0:a?" in _maps(fake.final_cmds[0])
    assert _muxer_of(fake.final_cmds[0]) == "mp4"


def test_h2_mkv_fallback_maps_all_audio_and_attachments(env, monkeypatch):
    fake, _ = _forced_fallback(env, monkeypatch)
    maps = _maps(fake.final_cmds[1])
    assert _muxer_of(fake.final_cmds[1]) == "matroska"
    assert "0:a?" in maps
    assert "0:t?" in maps


def test_h3_fallback_reuses_the_one_inventory(env, monkeypatch):
    src, out_dir, fake = env
    calls: list[Path] = []
    fake.audio_streams = 3
    fake.finals = [(1, "Could not write header")]
    _inventory(monkeypatch, audio_count=3, subtitles=TEXT_SUB, calls=calls)

    _run(src, out_dir, fmt="MP4 (H.265)")
    assert calls == [src], "the fallback attempt must not re-probe"


def test_h4_fallback_reuses_the_same_video_bitrate(env, monkeypatch):
    fake, _ = _forced_fallback(env, monkeypatch, mode="Target file size",
                               mode_value="4", audio_kbps="128")
    assert (_video_kbps_of(fake.final_cmds[0])
            == _video_kbps_of(fake.final_cmds[1]))


def test_h5_attempts_stay_capped_at_two(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    fake.finals = [(1, "Could not write header"), (1, "Could not write header")]
    _inventory(monkeypatch, audio_count=3)

    result = _run(src, out_dir, fmt="MP4 (H.265)")
    assert len(fake.final_cmds) == 2
    assert result["status"] == "error"


def test_h6_fallback_success_result_is_unchanged(env, monkeypatch):
    fake, result = _forced_fallback(env, monkeypatch)
    assert result["status"] == "ok"
    assert Path(result["output"]).suffix == ".mkv"
    assert result["fallback_used"] is True
    assert "audio_count" not in result and "audio_tracks" not in result


# ══ GROUP I — Tab 3 fidelity regression ══════════════════════════════════════

def test_i1_matroska_still_maps_and_copies_attachments():
    args = build_matroska_stream_map_args(TEXT_SUB)
    assert _maps(args).count("0:t?") == 1
    assert args[args.index("-c:t") + 1] == "copy"


def test_i2_matroska_still_names_its_text_subtitles_by_absolute_index():
    """The bitmap stream is skipped; both text streams are kept."""
    args = build_matroska_stream_map_args(
        [_sub(2, "hdmv_pgs_subtitle"), _sub(4, "subrip"), _sub(6, "ass")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:4", "0:6", "0:t?"]


def test_i3_matroska_without_usable_subtitles_still_says_sn():
    args = build_matroska_stream_map_args([_sub(2, "hdmv_pgs_subtitle")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:t?"]
    assert "-sn" in args
    assert "0:s:0?" not in _maps(args)


def test_i4_mp4_still_maps_safe_text_subtitles():
    args = build_mp4_stream_map_args([_sub(2, "subrip"), _sub(5, "mov_text")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:2", "0:5"]
    assert args[args.index("-c:s") + 1] == "mov_text"


def test_i5_mp4_still_excludes_attachments():
    assert "0:t?" not in _maps(build_mp4_stream_map_args(TEXT_SUB))


@pytest.mark.parametrize("builder", [build_mp4_stream_map_args,
                                     build_matroska_stream_map_args])
def test_i6_data_streams_stay_unmapped(builder):
    assert not [m for m in _maps(builder(TEXT_SUB)) if m.startswith("0:d")]


# ══ GROUP J — quality preset mode ════════════════════════════════════════════

def _quality_video_args(fake) -> list:
    cmd = fake.mux_cmd
    return [t for t in cmd if t in ("-crf", "-cq") or t.startswith("-preset")] + [
        cmd[cmd.index("-crf") + 1] if "-crf" in cmd else ""]


def test_j1_quality_preset_mp4_maps_all_audio(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"
    assert "0:a?" in _maps(fake.mux_cmd)


def test_j2_quality_preset_mkv_maps_all_audio(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    assert _run(src, out_dir, fmt="MKV (H.265)")["status"] == "ok"
    assert "0:a?" in _maps(fake.mux_cmd)


def test_j3_audio_count_does_not_touch_video_quality(env, monkeypatch,
                                                     tmp_path):
    src, out_dir, fake = env
    _inventory(monkeypatch, audio_count=1)
    _run(src, out_dir, fmt="MKV (H.265)")
    one = _quality_video_args(fake)

    fake.final_cmds.clear()
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)
    src2 = tmp_path / "Other.mov"
    src2.write_bytes(b"s" * SRC_BYTES)
    _run(src2, out_dir, fmt="MKV (H.265)")

    assert _quality_video_args(fake) == one
    assert "-b:v" not in fake.mux_cmd


def test_j4_each_track_gets_the_selected_bitrate(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    _run(src, out_dir, fmt="MKV (H.265)", audio_kbps="192")
    assert _value_after(fake.mux_cmd, "-b:a") == "192k"


# ══ GROUP K — per-file isolation ═════════════════════════════════════════════

def test_k1_each_file_uses_its_own_audio_count(env, monkeypatch, tmp_path):
    """Three sources, three counts, three budgets - and one probe each."""
    _src, out_dir, fake = env
    calls: list[Path] = []
    counts = {"A.mov": 3, "B.mov": 1, "C.mov": 0}
    for name in counts:
        (tmp_path / name).write_bytes(b"s" * SRC_BYTES)
    _inventory(monkeypatch, calls=calls, per_file=counts)

    seen = {}
    for name, count in counts.items():
        fake.audio_streams = count
        fake.final_cmds.clear()
        result = _run(tmp_path / name, out_dir, fmt="MP4 (H.265)",
                      mode="Target file size", mode_value="1",
                      audio_kbps="192")
        assert result["status"] == "ok", name
        seen[name] = _video_kbps_of(fake.mux_cmd)

    total = (1024 * 1024) * 0.97 * 8 / 10.0 / 1000.0
    assert seen == {
        "A.mov": int(total - 3 * 192),
        "B.mov": int(total - 1 * 192),
        "C.mov": int(total - 0 * 192),
    }
    assert [p.name for p in calls] == ["A.mov", "B.mov", "C.mov"]
