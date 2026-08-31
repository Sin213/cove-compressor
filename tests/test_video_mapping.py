"""Deterministic MP4 stream mapping, faststart parity, and atomic output
reservation.

Three things are locked down here, all of them about *which* streams and
*which* destination a video job commits to before ffmpeg ever runs:

  A. MP4 gets explicit positive stream maps instead of ffmpeg's implicit
     selection — first video, optional first audio, and only subtitle streams
     Cove can transcode to mov_text. Bitmap/unknown subtitles, attachments and
     data streams are excluded by construction.
  B. Two-pass pass 1 stays a video-only analysis pass.
  C. `+faststart` stays on both MP4 profiles and off everything else.
  E. The final video destination is claimed atomically with `reserve_output`
     rather than merely name-checked, and no zero-byte placeholder outlives a
     job that did not succeed.

No ffmpeg, no real media: `run_ffmpeg` and the probes are faked, but the fakes
model the filesystem effects production actually checks for.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    SubtitleProbeError,
    build_mp4_stream_map_args,
    build_pass1_stream_map_args,
    compress_video,
    mp4_subtitle_codec_is_compatible,
)


# ── ffprobe-shaped fixtures ──────────────────────────────────────────────────
#
# Shapes match what `ffprobe_subtitle_streams` actually returns: the absolute
# file index, a codec name, and optional tags/disposition dicts.

def _sub(index: int, codec: str, **tags) -> dict:
    s: dict = {"index": index, "codec_name": codec}
    if tags:
        s["tags"] = dict(tags)
    return s


TEXT_CODECS = ["subrip", "srt", "mov_text", "text", "webvtt", "vtt", "ass", "ssa"]
BITMAP_CODECS = ["hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"]


# ── Fake encoder stack ───────────────────────────────────────────────────────

class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, recording every invocation.

    Subtitle extraction is identified structurally (`-vn` *and* `-an`), the
    same discriminator `tests/test_subtitles.py` uses — explicit `-map` is no
    longer unique to extraction. A successful encode writes its output file,
    because production stats that file before calling the job a success.
    """

    def __init__(self, encode_rc=0, encode_bytes=b"v" * 10):
        self.encode_rc = encode_rc
        self.encode_bytes = encode_bytes
        self.encode_cmds: list[list] = []
        self.subtitle_cmds: list[list] = []

    @staticmethod
    def _is_subtitle_extraction(cmd) -> bool:
        return "-vn" in cmd and "-an" in cmd and "-map" in cmd

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        cmd = list(cmd)
        if self._is_subtitle_extraction(cmd):
            self.subtitle_cmds.append(cmd)
            Path(cmd[-1]).write_bytes(b"1\nhello\n")
            return 0, ""
        self.encode_cmds.append(cmd)
        if self.encode_rc != 0:
            return self.encode_rc, "encode failed"
        out = Path(cmd[-1])
        if out.name != "nul" and not str(out).endswith(("null", "NUL")):
            try:
                out.write_bytes(self.encode_bytes)
            except OSError:
                pass
        return 0, ""

    @property
    def mux_cmd(self) -> list:
        """The final muxing invocation (pass 1 writes to the null muxer)."""
        muxing = [c for c in self.encode_cmds if "null" not in _muxer_of(c)]
        assert muxing, "no muxing encode invocation was recorded"
        return muxing[-1]

    @property
    def pass1_cmd(self) -> list:
        for c in self.encode_cmds:
            if "null" in _muxer_of(c):
                return c
        raise AssertionError("no pass-1 (null muxer) invocation was recorded")


def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _pair(cmd, flag, value) -> bool:
    return any(cmd[i] == flag and cmd[i + 1] == value
               for i in range(len(cmd) - 1))


# Big enough that the 1 MB targets below are a real reduction *and* leave a
# usable video budget once the audio bitrate is reserved. A target that cannot
# hold its audio is refused before encoding (see `tests/test_multi_audio.py`),
# which is not the subject of any test in this file.
SRC_BYTES = 4 * 1024 * 1024


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder stack."""
    src = tmp_path / "Movie.mkv"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
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


def _run(src, out_dir, fmt="MP4 (H.265)", mode="Quality preset",
         mode_value="Balanced", **kw):
    """The ordinary `compress_video` entry point — no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        threading.Event(), **kw)


# ══ GROUP A — MP4 stream mapping policy ══════════════════════════════════════

def test_a1_video_stream_mapped_exactly_once():
    args = build_mp4_stream_map_args([])
    assert _maps(args).count("0:v:0") == 1


def test_a2_audio_is_mapped_optionally():
    """`0:a?` — the trailing `?` is what keeps silent video working.

    Which audio streams that covers is `tests/test_multi_audio.py`'s subject;
    what this asserts is that neither form is mandatory.
    """
    maps = _maps(build_mp4_stream_map_args([]))
    assert "0:a?" in maps
    assert "0:a:0" not in maps, "mandatory audio breaks sources with no audio"
    assert "0:a" not in maps, "mandatory audio breaks sources with no audio"


@pytest.mark.parametrize("codec", TEXT_CODECS)
def test_a3_safe_text_subtitles_map_by_absolute_index(codec):
    args = build_mp4_stream_map_args([_sub(3, codec)])
    assert "0:3" in _maps(args)
    assert _pair(args, "-c:s", "mov_text")
    assert "-sn" not in args


def test_a3_multiple_text_subtitles_keep_absolute_indexes():
    args = build_mp4_stream_map_args(
        [_sub(2, "subrip"), _sub(5, "ass"), _sub(9, "mov_text")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:2", "0:5", "0:9"]


@pytest.mark.parametrize("codec", BITMAP_CODECS)
def test_a4_bitmap_subtitles_are_never_mapped(codec):
    args = build_mp4_stream_map_args([_sub(2, codec)])
    assert _maps(args) == ["0:v:0", "0:a?"]
    assert "-sn" in args
    assert not _pair(args, "-c:s", "mov_text")


def test_a4_bitmap_stream_does_not_shift_a_sibling_text_index():
    args = build_mp4_stream_map_args(
        [_sub(2, "hdmv_pgs_subtitle"), _sub(4, "subrip")])
    assert _maps(args) == ["0:v:0", "0:a?", "0:4"]


@pytest.mark.parametrize("codec", ["", None, "eia_608", "unknown_future_codec"])
def test_a5_unknown_subtitle_codecs_fail_closed(codec):
    args = build_mp4_stream_map_args([_sub(1, codec)])
    assert _maps(args) == ["0:v:0", "0:a?"]
    assert "-sn" in args


def test_a5_unusable_stream_index_is_refused():
    """A missing/odd index cannot be mapped safely, so it is dropped."""
    for bad in ({"codec_name": "subrip"},
                {"index": "2", "codec_name": "subrip"},
                {"index": True, "codec_name": "subrip"},
                {"index": -1, "codec_name": "subrip"}):
        assert _maps(build_mp4_stream_map_args([bad])) == ["0:v:0", "0:a?"]


def test_a6_attachments_and_data_are_never_positively_mapped():
    args = build_mp4_stream_map_args([_sub(3, "subrip")])
    maps = _maps(args)
    assert not any(m.startswith(("0:t", "0:d")) for m in maps)
    # Explicit positive mapping already excludes them; a negative map would be
    # redundant (and would silently mask a mapping mistake).
    assert "-0:t" not in maps and "-0:d" not in maps


def test_a7_probe_failure_keeps_av_and_disables_subtitles():
    args = build_mp4_stream_map_args(None)
    assert _maps(args) == ["0:v:0", "0:a?"]
    assert "-sn" in args, "must not fall back to automatic subtitle selection"
    assert not _pair(args, "-c:s", "mov_text")


def test_a8_no_subtitle_streams_is_not_an_error():
    args = build_mp4_stream_map_args([])
    assert _maps(args) == ["0:v:0", "0:a?"]
    assert "-sn" in args


@pytest.mark.parametrize("codec", TEXT_CODECS)
def test_codec_classifier_accepts_text(codec):
    assert mp4_subtitle_codec_is_compatible(codec)
    assert mp4_subtitle_codec_is_compatible(codec.upper())


@pytest.mark.parametrize("codec", BITMAP_CODECS + [None, "", "  "])
def test_codec_classifier_rejects_everything_else(codec):
    assert not mp4_subtitle_codec_is_compatible(codec)


# ── A: the policy reaches the real command through the default call path ─────

def test_default_mp4_path_applies_the_mapping_policy(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "hdmv_pgs_subtitle"), _sub(3, "subrip")])

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3"]
    assert _pair(fake.mux_cmd, "-c:s", "mov_text")


def test_default_mp4_path_disables_subtitles_when_probe_fails(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, SubtitleProbeError("ffprobe exit 1"))

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?"]
    assert "-sn" in fake.mux_cmd


def test_mkv_maps_explicitly_without_the_mp4_policy(env, monkeypatch):
    """MP4's *policy* must not leak into MKV, even though MKV now maps too.

    Both containers map explicitly and share the one subtitle probe, but they
    do different things with it. Both name every compatible stream by absolute
    index, but Matroska lets ffmpeg's default text encoder handle them and
    also maps attachments, where MP4 transcodes to `mov_text` and maps none.
    The `mov_text` codec argument is the part that must never appear here.
    """
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip")])

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"
    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3", "0:t?"]
    assert "-sn" not in fake.mux_cmd
    assert "mov_text" not in fake.mux_cmd


def test_webm_keeps_implicit_selection(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip")])
    assert _run(src, out_dir, fmt="WebM (VP9)")["status"] == "ok"
    assert _maps(fake.mux_cmd) == []


# ── A: one subtitle probe per file, never two ────────────────────────────────

def test_probe_runs_once_for_mp4_with_extraction_on(env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip", language="eng")], calls)

    _run(src, out_dir, extract_english_subtitles=True)

    assert len(calls) <= 1
    assert len(calls) == 1


def test_probe_runs_once_for_mp4_with_extraction_off(env, monkeypatch):
    src, out_dir, _fake = env
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip")], calls)

    _run(src, out_dir)

    assert len(calls) == 1


def test_implicitly_selected_container_adds_no_probe(env, monkeypatch):
    """WebM still keeps ffmpeg's implicit selection, so it classifies nothing.

    MKV no longer qualifies: it maps explicitly now and so must classify its
    subtitle streams first (see `tests/test_mkv_attachments.py`).
    """
    src, out_dir, _fake = env
    calls: list[Path] = []
    _probe(monkeypatch, [_sub(3, "subrip")], calls)

    _run(src, out_dir, fmt="WebM (VP9)")

    assert calls == []


def test_successful_mp4_runs_exactly_one_encode_lifecycle(env, monkeypatch):
    """This slice contains zero fallback attempts and zero retries."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])

    assert _run(src, out_dir)["status"] == "ok"

    assert len(fake.encode_cmds) == 1
    assert fake.subtitle_cmds == []


def test_mp4_extraction_still_produces_sidecars(env, monkeypatch):
    """Sharing the probe must not regress the sidecar lifecycle."""
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip", language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert not result.get("subtitles_failed")
    assert [p.name for p in result["subtitles_extracted"]] == ["Movie.eng.srt"]
    assert len(fake.subtitle_cmds) == 1


def test_mp4_probe_failure_still_marks_subtitles_failed(env, monkeypatch):
    src, out_dir, _fake = env
    _probe(monkeypatch, SubtitleProbeError("boom"))

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True


# ══ GROUP B — pass 1 stays video-only analysis ═══════════════════════════════

def test_b_pass1_helper_maps_video_only():
    assert build_pass1_stream_map_args() == ["-map", "0:v:0"]


def test_b_two_pass_pass1_is_video_only(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip")])

    result = _run(src, out_dir, mode="Target file size", mode_value=1)

    assert result["status"] == "ok"
    p1 = fake.pass1_cmd
    assert _maps(p1) == ["0:v:0"], "pass 1 must analyse video only"
    assert "-an" in p1
    assert not _pair(p1, "-c:s", "mov_text")
    assert "-movflags" not in p1
    assert not any(m.startswith(("0:t", "0:d", "0:s")) for m in _maps(p1))


def test_b_two_pass_final_pass_still_gets_the_full_policy(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(3, "subrip")])

    _run(src, out_dir, mode="Target file size", mode_value=1)

    assert _maps(fake.mux_cmd) == ["0:v:0", "0:a?", "0:3"]
    assert _pair(fake.mux_cmd, "-c:s", "mov_text")


def test_b_mkv_pass1_stays_video_only(env, monkeypatch):
    """MKV maps explicitly now, so pass 1 names the video stream it analyses -
    and still nothing else."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    _run(src, out_dir, fmt="MKV (H.265)", mode="Target file size",
         mode_value=1)
    assert _maps(fake.pass1_cmd) == ["0:v:0"]
    assert "-an" in fake.pass1_cmd
    assert "-c:t" not in fake.pass1_cmd


# ══ GROUP C — faststart parity ═══════════════════════════════════════════════

@pytest.mark.parametrize("fmt", ["MP4 (H.265)", "MP4 (H.264)"])
def test_c_both_mp4_profiles_keep_faststart(env, monkeypatch, fmt):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    assert _run(src, out_dir, fmt=fmt)["status"] == "ok"
    assert _pair(fake.mux_cmd, "-movflags", "+faststart")


@pytest.mark.parametrize("fmt", ["MKV (H.265)", "WebM (VP9)"])
def test_c_non_mp4_never_gains_faststart(env, monkeypatch, fmt):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    assert _run(src, out_dir, fmt=fmt)["status"] == "ok"
    assert "-movflags" not in fake.mux_cmd


@pytest.mark.parametrize("fmt", ["MP4 (H.265)", "MP4 (H.264)"])
def test_c_faststart_survives_the_two_pass_path(env, monkeypatch, fmt):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    _run(src, out_dir, fmt=fmt, mode="Target file size", mode_value=1)
    assert _pair(fake.mux_cmd, "-movflags", "+faststart")


def test_c_encoder_settings_are_untouched_by_the_mapping_patch(env, monkeypatch):
    """Guards the patch against drifting into encoder/quality territory."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    _run(src, out_dir)
    cmd = fake.mux_cmd
    assert _pair(cmd, "-c:v", "libx265")
    assert _pair(cmd, "-crf", "25")
    assert _pair(cmd, "-preset", "medium")
    assert _pair(cmd, "-c:a", "aac")
    assert _pair(cmd, "-b:a", "128k")
    assert _pair(cmd, "-f", "mp4")


# ══ GROUP E — atomic video output reservation ════════════════════════════════

def _placeholders(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.iterdir()
                  if p.is_file() and p.stat().st_size == 0)


def test_e1_collision_bumps_the_name_without_touching_existing_bytes(
        env, monkeypatch):
    src, out_dir, _fake = env
    _probe(monkeypatch, [])
    existing = out_dir / "Movie.mp4"
    existing.write_bytes(b"earlier-result")

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).name == "Movie_1.mp4"
    assert existing.read_bytes() == b"earlier-result"


def test_e2_final_path_is_claimed_before_encoding_begins(env, monkeypatch):
    """Not `exists() → pick a name`: the destination must already be on disk
    when ffmpeg is invoked, so a concurrent job cannot pick the same one."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    seen: list[bool] = []

    real_call = fake.__call__

    def spy(cmd, *a, **kw):
        seen.append((out_dir / "Movie.mp4").exists())
        return real_call(cmd, *a, **kw)

    monkeypatch.setattr(compressor, "run_ffmpeg", spy)

    assert _run(src, out_dir)["status"] == "ok"
    assert seen and all(seen), "reservation must predate the encode"


def test_e2_concurrent_reservations_never_collide(env, monkeypatch):
    src, out_dir, _fake = env
    _probe(monkeypatch, [])
    claimed: list[Path] = []

    real_reserve = compressor.reserve_output

    def reserve_then_race(base):
        out, tmp = real_reserve(base)
        claimed.append(out)
        return out, tmp

    monkeypatch.setattr(compressor, "reserve_output", reserve_then_race)

    first = _run(src, out_dir)
    second = _run(src, out_dir)

    assert {Path(first["output"]).name, Path(second["output"]).name} == \
        {"Movie.mp4", "Movie_1.mp4"}
    assert len(claimed) == len(set(claimed))


def test_e3_successful_conversion_replaces_the_placeholder(env, monkeypatch):
    src, out_dir, _fake = env
    _probe(monkeypatch, [])

    result = _run(src, out_dir)

    out = Path(result["output"])
    assert result["status"] == "ok"
    assert out.exists() and out.stat().st_size > 0
    assert _placeholders(out_dir) == []
    assert not (out_dir / "Movie.mp4.tmp").exists()


def test_e4_encode_error_leaves_no_placeholder(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = 1

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert _placeholders(out_dir) == []
    assert list(out_dir.iterdir()) == []


def test_e5_cancellation_leaves_no_placeholder(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = -2

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"
    assert list(out_dir.iterdir()) == []


def test_e5_cancellation_during_subtitle_stage_leaves_no_placeholder(
        env, monkeypatch):
    src, out_dir, _fake = env
    _probe(monkeypatch, [_sub(3, "subrip", language="eng")])
    cancel = threading.Event()

    def cancel_then_extract(cmd, cancel_flag, **kw):
        cancel.set()
        Path(cmd[-1]).write_bytes(b"1\nhi\n")
        return 0, ""

    monkeypatch.setattr(compressor, "run_ffmpeg", cancel_then_extract)

    result = compress_video(src, out_dir, "Quality preset", "Balanced",
                            "MP4 (H.265)", None, "128", cancel,
                            extract_english_subtitles=True)

    assert result["status"] == "error"
    assert list(out_dir.iterdir()) == []


def test_e6_timeout_leaves_no_placeholder(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = -3

    result = _run(src, out_dir)

    assert result["status"] == "timeout"
    assert list(out_dir.iterdir()) == []


def test_e6_two_pass_timeout_in_pass1_leaves_no_placeholder(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = -3

    result = _run(src, out_dir, mode="Target file size", mode_value=1)

    assert result["status"] == "timeout"
    assert list(out_dir.iterdir()) == []


def test_e7_skipped_result_leaves_no_placeholder(env, monkeypatch):
    """Quality preset that would grow the file skips — and cleans up."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_bytes = b"x" * (SRC_BYTES + 1)  # larger than the source

    result = _run(src, out_dir)

    assert result["status"] == "skipped"
    assert list(out_dir.iterdir()) == []


def test_e7_pre_encode_skip_reserves_nothing(env, monkeypatch):
    """Target >= original short-circuits before any destination is claimed."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])

    result = _run(src, out_dir, mode="Target file size", mode_value=100)

    assert result["status"] == "skipped"
    assert list(out_dir.iterdir()) == []
    assert fake.encode_cmds == []


def test_e4_placeholder_cleanup_survives_a_transient_file_lock(
        env, monkeypatch):
    """A Windows AV/indexer handle makes the first unlink fail; the stub must
    still be gone once the lock clears."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = 1
    target = out_dir / "Movie.mp4"
    attempts: list[int] = []
    real_unlink = compressor.os.unlink

    def flaky(path, *a, **kw):
        if Path(path) == target:
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError(32, "file in use")
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(compressor.os, "unlink", flaky)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert len(attempts) == 3, "must retry rather than give up on first failure"
    assert list(out_dir.iterdir()) == []


def test_e4_placeholder_cleanup_rechecks_emptiness_between_retries(
        env, monkeypatch):
    """If another writer fills the path during a retry delay, that real data
    must not be deleted by the next attempt."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    # -1 (ffmpeg unavailable), not a positive exit: this test asserts that the
    # *only* unlink in the job is the MP4 placeholder's, and a positive exit
    # now earns an MKV fallback attempt that legitimately reserves - and then
    # discards - a second stub. The cleanup semantics under test are identical
    # either way; the failure class is chosen to keep the job single-container.
    fake.encode_rc = -1
    target = out_dir / "Movie.mp4"
    attempts: list[int] = []

    def locked_then_filled(path, *a, **kw):
        if Path(path) == target:
            attempts.append(1)
            # Somebody else claims the path while we back off.
            target.write_bytes(b"someone-elses-real-output")
            raise PermissionError(32, "file in use")
        raise AssertionError(f"unexpected unlink of {path}")

    monkeypatch.setattr(compressor.os, "unlink", locked_then_filled)
    monkeypatch.setattr(compressor, "RESERVED_CLEANUP_RETRY_DELAY", 0)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert len(attempts) == 1, "must re-check emptiness before retrying"
    assert target.read_bytes() == b"someone-elses-real-output"


def test_e4_placeholder_cleanup_never_raises_over_the_real_result(
        env, monkeypatch):
    """A permanently locked stub is left behind quietly - it must never
    replace the job's real error result with an OSError."""
    src, out_dir, fake = env
    _probe(monkeypatch, [])
    fake.encode_rc = 1
    target = out_dir / "Movie.mp4"

    def always_locked(path, *a, **kw):
        raise PermissionError(32, "file in use")

    monkeypatch.setattr(compressor.os, "unlink", always_locked)
    monkeypatch.setattr(compressor, "RESERVED_CLEANUP_RETRY_DELAY", 0)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert result["msg"].startswith("ffmpeg failed")
    assert target.exists()


def test_e8_source_is_never_claimed_as_the_output(env, monkeypatch):
    """Same-directory MP4→MP4: the source must survive untouched."""
    src, out_dir, _fake = env
    _probe(monkeypatch, [])
    same_dir_src = out_dir / "Clip.mp4"
    same_dir_src.write_bytes(b"s" * 4096)

    result = _run(same_dir_src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]) != same_dir_src
    assert Path(result["output"]).name == "Clip_1.mp4"
    assert same_dir_src.read_bytes() == b"s" * 4096
