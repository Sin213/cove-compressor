"""Per-file MP4 (H.265) -> MKV (H.265) fallback.

One narrowly classified retry: when a job the user asked to encode as
MP4 (H.265) fails its *final* encode/mux invocation with an ordinary nonzero
ffmpeg exit, that one file is retried exactly once as MKV (H.265). Nothing
about that retry survives the call - the next file starts as MP4 again.

Locked down here:

  A. Basic control flow: one retry, only when eligible, never a third attempt.
  B. Failure classification: cancellation, stall/timeout, missing ffmpeg,
     pass-1 failure, pre-encode validation, reservation failure, skips,
     subtitle-only failures, ENOSPC and every non-MP4-H.265 format are all
     terminal.
  C. The MKV destination is reserved independently and collision-safely, and
     the failed MP4 reservation never outlives the job.
  D. Subtitle discovery/extraction runs once per file, not once per attempt.
  E. Source deletion stays governed by `delete_source_if_eligible`.
  F. No fallback state leaks between files.
  G. Explicit call counts, so a hidden retry cannot creep in.

No ffmpeg and no real media: `run_ffmpeg` and the probes are faked, but the
fakes reproduce the filesystem effects production actually checks for (an
encode writes its temp output; a subtitle extraction writes its sidecar temp).
"""
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    SubtitleProbeError,
    compress_video,
    delete_source_if_eligible,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _sub(index: int, codec: str, **tags) -> dict:
    s: dict = {"index": index, "codec_name": codec, "tags": {"language": "eng"}}
    if tags:
        s["tags"].update(tags)
    return s


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, classifying and recording every invocation.

    Three kinds of invocation exist and each has its own scripted result queue,
    because the fallback contract is specifically about *which* invocation
    failed: subtitle extraction (`-vn` and `-an`, the same structural
    discriminator the other suites use), two-pass analysis (the null muxer),
    and the final encode/mux. Queues fall back to success once exhausted.
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
    def final_encoders(self) -> list[str]:
        return [c[c.index("-c:v") + 1] if "-c:v" in c else ""
                for c in self.final_cmds]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder/probe stack."""
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * 4096)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(compressor, "ffprobe_subtitle_streams", lambda p: [])
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": False)
    return src, out_dir, fake


def _probe(monkeypatch, streams, calls=None):
    """Fake `ffprobe_subtitle_streams`; `streams` may be an exception."""
    def fake(path):
        if calls is not None:
            calls.append(Path(path))
        if isinstance(streams, Exception):
            raise streams
        return [dict(s) for s in streams]
    monkeypatch.setattr(compressor, "ffprobe_subtitle_streams", fake)


def _run(src, out_dir, fmt="MP4 (H.265)", mode="Quality preset",
         mode_value="Balanced", cancel_flag=None, **kw):
    """The ordinary public `compress_video` entry point - no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


def _names(out_dir, pattern="*") -> list[str]:
    return sorted(p.name for p in out_dir.glob(pattern))


# ══ GROUP A — basic fallback control flow ════════════════════════════════════

def test_a1_eligible_mp4_mux_failure_retries_once_as_mkv(env):
    src, out_dir, fake = env
    fake.finals = [(1, "Could not write header: Invalid argument")]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4", "matroska"]
    assert fake.final_encoders == ["libx265", "libx265"]
    assert result["status"] == "ok"
    assert Path(result["output"]).suffix == ".mkv"
    assert result["fallback_used"] is True
    assert Path(result["output"]).exists()


def test_a2_successful_mp4_never_invokes_mkv(env):
    src, out_dir, fake = env

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "ok"
    assert Path(result["output"]).suffix == ".mp4"
    assert not result.get("fallback_used")


def test_a3_mkv_fallback_failure_does_not_retry_again(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]

    result = _run(src, out_dir)

    assert len(fake.final_cmds) == 2, "a third attempt is never allowed"
    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] != "ok"


def test_a4_internal_mux_marker_does_not_leak_to_callers(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]

    result = _run(src, out_dir)

    assert "mux_failed" not in result


# ══ GROUP B — failure classification (no fallback) ═══════════════════════════

def test_b1_cancellation_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(-2, "cancelled")]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"
    assert result["msg"] == "cancelled"


def test_b1b_cancel_flag_set_during_encode_does_not_fall_back(env, monkeypatch):
    """A nonzero rc cannot license a retry once the user has cancelled.

    ffmpeg terminated mid-write reports an ordinary positive exit code often
    enough that the flag, not the code, has to be authoritative.
    """
    src, out_dir, fake = env
    cancel = threading.Event()
    calls: list[list] = []

    def cancelling(cmd, cancel_flag, duration=None, on_progress=None,
                   on_start=None):
        calls.append(list(cmd))
        cancel_flag.set()
        return 1, "killed mid-write"
    monkeypatch.setattr(compressor, "run_ffmpeg", cancelling)

    result = _run(src, out_dir, cancel_flag=cancel)

    assert len(calls) == 1
    assert result["status"] != "ok"


def test_b2_timeout_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(-3, "no encoding progress for 300s (skipped)")]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "timeout"


def test_b3_missing_ffmpeg_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(-1, "ffmpeg not found on PATH")]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"


def test_b4_pass1_failure_does_not_fall_back(env):
    """Two-pass analysis is not a mux attempt; its failure is terminal."""
    src, out_dir, fake = env
    fake.pass1 = [(1, "pass 1 boom")]

    result = _run(src, out_dir, mode="Target file size", mode_value="0.001")

    assert len(fake.pass1_cmds) == 1
    assert fake.final_cmds == [], "pass-1 failure must never reach a mux retry"
    assert result["status"] == "error"
    assert "pass 1 failed" in result["msg"]


def test_b5_pre_encode_validation_failure_does_not_fall_back(env, monkeypatch):
    src, out_dir, fake = env
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: None)

    result = _run(src, out_dir, mode="Target file size", mode_value="1")

    assert fake.final_cmds == []
    assert result["status"] == "error"


def test_b6_output_reservation_failure_does_not_fall_back(env, monkeypatch):
    src, out_dir, fake = env

    def boom(base):
        raise OSError("no reservation")
    monkeypatch.setattr(compressor, "reserve_output", boom)

    with pytest.raises(OSError):
        _run(src, out_dir)

    assert fake.final_cmds == []


def test_b7_skipped_result_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.encode_bytes = b"v" * 8192  # bigger than the 4096-byte source

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "skipped"


def test_b8_subtitle_probe_failure_alone_does_not_fall_back(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, SubtitleProbeError("ffprobe exploded"))

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True


def test_b8b_subtitle_extraction_failure_alone_does_not_fall_back(
        env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.subtitle = [(1, "extraction boom")]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True


@pytest.mark.parametrize("stderr", [
    "av_interleaved_write_frame(): No space left on device",
    "Error writing trailer: ENOSPC",
])
def test_b9_disk_full_does_not_fall_back(env, stderr):
    src, out_dir, fake = env
    fake.finals = [(1, stderr)]

    result = _run(src, out_dir)

    assert fake.final_muxers == ["mp4"], "another container cannot make room"
    assert result["status"] == "error"


def test_b10_mp4_h264_failure_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(1, "boom")]

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"


def test_b11_native_mkv_failure_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(1, "boom")]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert fake.final_muxers == ["matroska"]
    assert result["status"] == "error"


def test_b12_webm_failure_does_not_fall_back(env):
    src, out_dir, fake = env
    fake.finals = [(1, "boom")]

    result = _run(src, out_dir, fmt="WebM (VP9)")

    assert fake.final_muxers == ["webm"]
    assert result["status"] == "error"


# ══ GROUP C — reservation, collision and cleanup ═════════════════════════════

def test_c1_failed_mp4_reservation_is_released(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir)

    assert _names(out_dir, "*.mp4") == [], "stale zero-byte MP4 left behind"
    assert _names(out_dir, "*.mkv") == ["Movie.mkv"]
    assert Path(result["output"]).name == "Movie.mkv"


def test_c2_existing_mkv_is_never_overwritten(env):
    src, out_dir, fake = env
    squatter = out_dir / "Movie.mkv"
    squatter.write_bytes(b"PRECIOUS")
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir)

    assert Path(result["output"]).name == "Movie_1.mkv"
    assert squatter.read_bytes() == b"PRECIOUS"


def test_c3_mkv_destination_is_reserved_independently(env, monkeypatch):
    """A suffix swap is not a reservation - the MKV name is claimed atomically."""
    src, out_dir, fake = env
    real = compressor.reserve_output
    reserved: list[Path] = []

    def spy(base):
        out, tmp = real(base)
        reserved.append(Path(out))
        return out, tmp
    monkeypatch.setattr(compressor, "reserve_output", spy)
    fake.finals = [(1, "mux boom")]

    _run(src, out_dir)

    assert [p.name for p in reserved] == ["Movie.mp4", "Movie.mkv"]


def test_c4_both_attempts_fail_leaves_nothing_behind(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]

    result = _run(src, out_dir)

    assert _names(out_dir) == [], f"stale artifacts: {_names(out_dir)}"
    assert src.exists()
    assert result["status"] != "ok"


# ══ GROUP D — subtitles run once per file, not once per attempt ══════════════

def test_d1_subtitle_preparation_happens_once_across_both_attempts(
        env, monkeypatch):
    src, out_dir, fake = env
    probe_calls: list[Path] = []
    _probe(monkeypatch, [_sub(2, "subrip")], calls=probe_calls)
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert len(probe_calls) == 1, "fallback must not re-probe"
    assert len(fake.subtitle_cmds) == 1, "fallback must not re-extract"
    assert result["status"] == "ok"


def test_d2_prepared_subtitles_feed_the_fallback_output(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["fallback_used"] is True
    assert len(result["subtitles_extracted"]) == 1
    assert not result.get("subtitles_failed")


def test_d3_fallback_success_produces_no_duplicate_sidecars(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.finals = [(1, "mux boom")]

    _run(src, out_dir, extract_english_subtitles=True)

    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]
    assert _names(out_dir, "*.mkv") == ["Movie.mkv"]
    assert _names(out_dir, "*.mp4") == []


def test_d4_subtitle_failure_still_vetoes_source_deletion(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.subtitle = [(1, "extraction boom")]
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir, extract_english_subtitles=True)
    delete_source_if_eligible(result, enabled=True)

    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert result["subtitles_failed"] is True
    assert src.exists(), "a lost subtitle must still keep the original"


# ══ GROUP E — source deletion stays governed by the existing gate ════════════

def test_e1_clean_fallback_success_allows_ordinary_deletion(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mux boom")]

    result = _run(src, out_dir)
    delete_source_if_eligible(result, enabled=True)

    assert result["source_deleted"] is True
    assert not src.exists()


def test_e1b_fallback_code_never_deletes_the_source_itself(env):
    """Deletion is the caller's gate; `compress_video` never unlinks."""
    src, out_dir, fake = env
    fake.finals = [(1, "mux boom")]

    _run(src, out_dir)

    assert src.exists()


def test_e2_both_attempts_fail_keeps_the_source(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]

    result = _run(src, out_dir)
    delete_source_if_eligible(result, enabled=True)

    assert src.exists()
    assert not result.get("source_deleted")


def test_e4_cancellation_keeps_the_source(env):
    src, out_dir, fake = env
    fake.finals = [(-2, "cancelled")]

    result = _run(src, out_dir)
    delete_source_if_eligible(result, enabled=True)

    assert src.exists()
    assert not result.get("source_deleted")


# ══ GROUP F — per-file isolation: no sticky fallback ═════════════════════════

def test_f1_next_file_starts_as_mp4_after_a_successful_fallback(env):
    src, out_dir, fake = env
    second = src.with_name("Second.mov")
    second.write_bytes(b"s" * 4096)
    fake.finals = [(1, "mux boom")]

    first = _run(src, out_dir)
    fake.final_cmds.clear()
    second_result = _run(second, out_dir)

    assert first["fallback_used"] is True
    assert fake.final_muxers == ["mp4"], "fallback leaked into the next file"
    assert Path(second_result["output"]).suffix == ".mp4"
    assert not second_result.get("fallback_used")


def test_f2_next_file_starts_as_mp4_after_a_failed_fallback(env):
    src, out_dir, fake = env
    second = src.with_name("Second.mov")
    second.write_bytes(b"s" * 4096)
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]

    _run(src, out_dir)
    fake.final_cmds.clear()
    second_result = _run(second, out_dir)

    assert fake.final_muxers == ["mp4"]
    assert Path(second_result["output"]).suffix == ".mp4"


# ══ GROUP G — call counts ════════════════════════════════════════════════════

def test_g1_ordinary_success_is_one_encode_attempt(env):
    src, out_dir, fake = env
    _run(src, out_dir)
    assert len(fake.final_cmds) == 1


def test_g2_eligible_failure_plus_mkv_success_is_two_attempts(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mux boom")]
    _run(src, out_dir)
    assert len(fake.final_cmds) == 2


def test_g3_eligible_failure_plus_mkv_failure_is_two_attempts(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mp4 boom"), (1, "mkv boom")]
    _run(src, out_dir)
    assert len(fake.final_cmds) == 2


@pytest.mark.parametrize("rc", [-1, -2, -3])
def test_g4_non_eligible_rc_never_reaches_a_second_attempt(env, rc):
    src, out_dir, fake = env
    fake.finals = [(rc, "terminal")]
    _run(src, out_dir)
    assert len(fake.final_cmds) == 1


def test_g5_subtitle_preparation_lifecycle_is_one_per_file(env, monkeypatch):
    src, out_dir, fake = env
    probe_calls: list[Path] = []
    _probe(monkeypatch, [_sub(2, "subrip"), _sub(3, "ass")], calls=probe_calls)
    fake.finals = [(1, "mux boom")]

    _run(src, out_dir, extract_english_subtitles=True)

    assert len(probe_calls) == 1
    assert len(fake.subtitle_cmds) == 2, "two streams, one pass - not four"


def test_g6_no_stale_temp_output_after_a_successful_fallback(env):
    src, out_dir, fake = env
    fake.finals = [(1, "mux boom")]

    _run(src, out_dir)

    assert _names(out_dir, "*.tmp") == []
    assert not any(n.endswith(".tmp") for n in os.listdir(out_dir))
