"""Final output artifact verification: a successful exit is not a conversion.

An encode that exits 0 has told Cove only that ffmpeg believed it finished.
Whether a *file* came out of it is a separate question, and until it is asked
Cove can report a conversion as successful while the destination holds
nothing - then offer to delete the original that is now the only copy.

The gate locked down here is structural and nothing more: the expected final
output exists, is a regular file, and is > 0 bytes. It is not a media
validator - no ffprobe, no codec/duration/stream check, no container parsing,
and no size floor above zero, because a genuinely tiny valid encode is still
a valid encode.

  A/B. Missing and zero-byte outputs fail, in every container Cove targets,
       and the empty artifact Cove owns is cleaned up rather than left
       looking like a result.
  C.   One byte is enough - the day this fails is the day a size heuristic
       or a media check crept in.
  D.   Source deletion happens only behind the gate.
  E/F/G. A phantom success is not a mux failure and earns no MP4 -> MKV
       retry; the legitimate Tab 2b fallback still works; a fallback's own
       output faces the same gate, with no third attempt.
  H.   Two-pass verifies pass 2 only - pass 1 targets the null muxer.
  I.   Non-regular and unreadable paths fail closed instead of raising.
  J.   Only Cove's own reservation is ever cleaned up.
  K.   Sidecars are not finalized for a video that failed the gate.
  L/M. Cancellation, stalls and ENOSPC keep their own classifications.
  N.   Stream mapping and target accounting are untouched.
  O.   Nothing is remembered between files.

Plus explicit call counts: verification is filesystem metadata only, so it
must add exactly zero subprocesses.

No ffmpeg and no real media: `run_ffmpeg` and the probes are faked, but the
fakes reproduce the filesystem effect production actually checks for - a
successful encode leaves, or fails to leave, bytes at its temp output.
"""
import os
import stat as stat_mod
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    compress_video,
    delete_source_if_eligible,
)


# ── helpers ──────────────────────────────────────────────────────────────────

# Big enough that the target-size cases below are a real reduction and still
# leave a usable video budget once audio is reserved.
SRC_BYTES = 4 * 1024 * 1024

# The three ways a final encode can report success, as what it leaves behind.
MISSING = None          # exits 0, writes nothing at all
EMPTY = b""             # exits 0, leaves a zero-byte file
VALID = b"v"            # exits 0, leaves one honest byte

PHANTOM = [MISSING, EMPTY]
ALL_FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"]
EXT_OF = {"MP4 (H.264)": ".mp4", "MP4 (H.265)": ".mp4",
          "MKV (H.265)": ".mkv", "WebM (VP9)": ".webm"}


def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _sub(index: int, codec: str) -> dict:
    return {"index": index, "codec_name": codec, "tags": {"language": "eng"}}


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, scripting each invocation's exit *and* its
    filesystem effect.

    Invocation kinds are told apart the same structural way the neighboring
    suites do it: subtitle extraction carries `-vn -an -map`, two-pass
    analysis targets the null muxer, anything else is a final encode/mux.
    `payloads` is per final attempt; once exhausted, later attempts succeed
    and write VALID.
    """

    def __init__(self, finals=None, pass1=None, subtitle=None, payloads=None):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.subtitle = list(subtitle or [])
        self.payloads = list(payloads or [])
        self.subtitle_cmds: list[list] = []
        self.pass1_cmds: list[list] = []
        self.final_cmds: list[list] = []
        self.all_cmds: list[list] = []

    @staticmethod
    def _next(q, default):
        return q.pop(0) if q else default

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        cmd = list(cmd)
        self.all_cmds.append(cmd)
        out = Path(cmd[-1])
        if "-vn" in cmd and "-an" in cmd and "-map" in cmd:
            self.subtitle_cmds.append(cmd)
            rc, err = self._next(self.subtitle, (0, ""))
            if rc == 0:
                out.write_bytes(b"1\nhello\n")
            return rc, err
        if _muxer_of(cmd) == "null":
            self.pass1_cmds.append(cmd)
            return self._next(self.pass1, (0, ""))
        self.final_cmds.append(cmd)
        rc, err = self._next(self.finals, (0, ""))
        payload = self._next(self.payloads, VALID)
        if rc == 0 and payload is not MISSING:
            out.write_bytes(payload)
        return rc, err

    @property
    def final_muxers(self) -> list[str]:
        return [_muxer_of(c) for c in self.final_cmds]

    @property
    def attempts(self) -> int:
        return len(self.final_cmds)


def _fake_stack(monkeypatch, fake):
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(subtitles=[], audio_count=1))
    monkeypatch.setattr(compressor, "nvenc_available",
                        lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder/probe stack."""
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fake = FakeFfmpeg()
    _fake_stack(monkeypatch, fake)
    return src, out_dir, fake


def _probe(monkeypatch, streams, audio_count=1):
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda path: compressor.StreamInventory(
            subtitles=[dict(s) for s in streams], audio_count=audio_count))


def _run(src, out_dir, fmt="MP4 (H.264)", mode="Quality preset",
         mode_value="Balanced", cancel_flag=None, **kw):
    """The ordinary public `compress_video` entry point - no new arguments."""
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


def _two_pass(src, out_dir, fmt="MP4 (H.264)"):
    return _run(src, out_dir, fmt=fmt, mode="Target file size", mode_value=1)


def _names(out_dir, pattern="*") -> list[str]:
    return sorted(p.name for p in out_dir.glob(pattern))


def _replace_output_with_directory(monkeypatch):
    """Let the encode 'succeed', then leave a directory where the finished
    video was supposed to be."""
    def fake(src, dst, *a, **kw):
        os.unlink(src)
        if os.path.exists(dst):
            os.unlink(dst)
        os.mkdir(dst)
    monkeypatch.setattr(compressor.os, "replace", fake)


def _stat_raises_for(monkeypatch, target: Path):
    """Make exactly one path unreadable; everything else stats normally."""
    real_stat = os.stat
    wanted = os.path.normcase(os.path.abspath(str(target)))

    def fake(path, *a, **kw):
        try:
            same = os.path.normcase(os.path.abspath(str(path))) == wanted
        except (TypeError, ValueError):
            same = False
        if same:
            raise OSError(5, "simulated I/O error")
        return real_stat(path, *a, **kw)
    monkeypatch.setattr(compressor.os, "stat", fake)


# ══ GROUPS A & B — a successful exit that produced no usable file ═══════════

@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize("payload", PHANTOM)
def test_ab_phantom_success_fails_and_leaves_nothing(env, fmt, payload):
    """Every container goes through the same gate. EMPTY is the realistic
    shape under an atomic reservation: the destination exists because Cove
    created it, not because ffmpeg filled it."""
    src, out_dir, fake = env
    fake.payloads = [payload]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert "output" not in result
    assert src.exists(), "the original is the only surviving copy"
    assert _names(out_dir) == [], \
        "an empty artifact Cove owns must not look like a result"


def test_b_failure_message_is_actionable_and_leaks_nothing(env):
    src, out_dir, fake = env
    fake.payloads = [EMPTY]

    result = _run(src, out_dir)

    msg = result.get("msg") or ""
    assert msg, "a failure the user can see must say something"
    for leak in ("st_size", "reservation", "inode", "artifact gate"):
        assert leak not in msg.lower(), f"implementation detail leaked: {leak}"


# ══ GROUP C — a successful exit that leaves real bytes ══════════════════════

@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_c1_one_byte_output_is_accepted(env, fmt):
    """The gate is structural, so a single unparseable byte is a file. The
    day this fails is the day a size heuristic or a media check crept in."""
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.suffix == EXT_OF[fmt]
    assert out.stat().st_size == 1
    assert result["new"] == 1


def test_c2_no_minimum_size_threshold_beyond_empty(env):
    """Sizes just above zero are ordinary successes, not suspicious ones."""
    src, out_dir, fake = env
    for size in (1, 2, 17, 512, 1023):
        fake.payloads = [b"v" * size]
        result = _run(src, out_dir)
        assert result["status"] == "ok", f"{size} bytes was rejected"
        assert result["new"] == size


# ══ GROUP D — source deletion happens only behind the gate ═════════════════

@pytest.mark.parametrize("payload", PHANTOM)
def test_d12_phantom_success_keeps_the_source(env, payload):
    src, out_dir, fake = env
    fake.payloads = [payload]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_d3_non_file_output_keeps_the_source(env, monkeypatch):
    src, out_dir, fake = env
    _replace_output_with_directory(monkeypatch)

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_d4_unreadable_output_keeps_the_source(env, monkeypatch):
    src, out_dir, fake = env
    _stat_raises_for(monkeypatch, out_dir / "Movie.mp4")

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_d5_valid_output_still_deletes_the_source(env):
    """The existing policy is untouched for conversions that really worked."""
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert not src.exists()
    assert result["source_deleted"] is True


def test_d6_deletion_stays_opt_in(env):
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = delete_source_if_eligible(_run(src, out_dir))

    assert src.exists()
    assert "source_deleted" not in result


# ══ GROUP E — a phantom success earns no fallback ══════════════════════════

@pytest.mark.parametrize("payload", PHANTOM)
def test_e_phantom_mp4_h265_success_does_not_retry_as_mkv(env, payload):
    """Tab 2b's retry is for an ordinary nonzero mux failure. A successful
    exit that produced nothing is a different anomaly and stays terminal
    rather than quietly widening what earns a second encode."""
    src, out_dir, fake = env
    fake.payloads = [payload]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 1, "a phantom success is not fallback-eligible"
    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"
    assert not result.get("fallback_used")
    assert src.exists()
    assert _names(out_dir) == []


# ══ GROUP F — the legitimate fallback is untouched ═════════════════════════

def test_f_eligible_mux_failure_still_retries_once_into_mkv(env):
    src, out_dir, fake = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, VALID]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert Path(result["output"]).suffix == ".mkv"
    assert Path(result["output"]).stat().st_size == 1


# ══ GROUP G — a fallback attempt faces the same gate ═══════════════════════

@pytest.mark.parametrize("payload", PHANTOM)
def test_g_phantom_fallback_success_fails_without_a_third_attempt(env, payload):
    src, out_dir, fake = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, payload]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 2, "two attempts is the ceiling, always"
    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert result["fallback_used"] is True
    assert src.exists()
    assert _names(out_dir) == [], "neither reservation may outlive the job"


# ══ GROUP H — two-pass verifies pass 2, and only pass 2 ════════════════════

def test_h1_pass_one_output_is_not_verified(env):
    """Pass 1 targets the null muxer and leaves nothing on purpose. A gate
    applied there would fail every two-pass job ever run."""
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = _two_pass(src, out_dir)

    assert len(fake.pass1_cmds) == 1
    assert _muxer_of(fake.pass1_cmds[0]) == "null"
    assert result["status"] == "ok"


@pytest.mark.parametrize("payload", PHANTOM)
def test_h23_phantom_pass_two_output_fails(env, payload):
    src, out_dir, fake = env
    fake.payloads = [payload]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert src.exists()
    assert _names(out_dir) == []


def test_h4_non_empty_pass_two_output_succeeds(env):
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = _two_pass(src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).stat().st_size == 1


@pytest.mark.parametrize("payload", [MISSING, EMPTY, VALID])
def test_h5_two_pass_is_always_exactly_two_subprocesses(env, payload):
    src, out_dir, fake = env
    fake.payloads = [payload]

    _two_pass(src, out_dir)

    assert len(fake.all_cmds) == 2, "verification must not run a subprocess"
    assert len(fake.pass1_cmds) == 1
    assert fake.attempts == 1


# ══ GROUP I — non-regular and unreadable outputs fail closed ═══════════════

def test_i1_directory_at_the_output_path_is_a_failure(env, monkeypatch):
    src, out_dir, fake = env
    _replace_output_with_directory(monkeypatch)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert src.exists()


def test_i2_unreadable_output_is_a_failure_not_an_exception(env, monkeypatch):
    src, out_dir, fake = env
    _stat_raises_for(monkeypatch, out_dir / "Movie.mp4")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert src.exists()


def test_i2b_a_non_regular_path_reporting_bytes_is_still_a_failure(
        env, monkeypatch):
    """Size alone is not enough. Windows reports a directory as zero bytes,
    which hides behind the size check - POSIX reports 4096, which sails
    straight through it. Only asking whether the path is a regular file
    answers the question on both, so this fakes the POSIX shape to keep that
    clause honest on every platform the suite runs on."""
    src, out_dir, fake = env
    target = out_dir / "Movie.mp4"
    real_stat = os.stat
    wanted = os.path.normcase(os.path.abspath(str(target)))
    directory_with_bytes = os.stat_result(
        (stat_mod.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 4096, 0, 0, 0))

    def fake_stat(path, *a, **kw):
        try:
            same = os.path.normcase(os.path.abspath(str(path))) == wanted
        except (TypeError, ValueError):
            same = False
        return directory_with_bytes if same else real_stat(path, *a, **kw)
    monkeypatch.setattr(compressor.os, "stat", fake_stat)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert src.exists()


def test_i3_a_directory_at_the_output_path_is_not_deleted(env, monkeypatch):
    """Cove only cleans up the empty stub it created; a directory is not
    that, whoever put it there."""
    src, out_dir, fake = env
    _replace_output_with_directory(monkeypatch)

    _run(src, out_dir)

    assert (out_dir / "Movie.mp4").is_dir(), \
        "a non-regular path is not Cove's to remove"


# ══ GROUP J — only Cove's own reservation is cleaned up ════════════════════

@pytest.mark.parametrize("neighbor_bytes", [b"the user's own file", b""])
def test_j_neighboring_pre_existing_file_survives(env, neighbor_bytes):
    """`Movie.mp4` already belongs to the user, so this job reserved
    `Movie_1.mp4`. Failing the gate may only reclaim the latter - and an
    empty neighbor is the nastiest shape, because Cove's claim is the
    reserved path, not every empty file in the folder."""
    src, out_dir, fake = env
    neighbor = out_dir / "Movie.mp4"
    neighbor.write_bytes(neighbor_bytes)
    fake.payloads = [EMPTY]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert neighbor.read_bytes() == neighbor_bytes
    assert _names(out_dir) == ["Movie.mp4"], "only the reserved stub goes"


# ══ GROUP K — sidecars are not finalized for a video that failed ═══════════

@pytest.mark.parametrize("payload", PHANTOM)
def test_k1_phantom_success_does_not_finalize_sidecars(env, monkeypatch,
                                                      payload):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [payload]

    result = delete_source_if_eligible(
        _run(src, out_dir, extract_english_subtitles=True), enabled=True)

    assert result["status"] == "error"
    assert "subtitles_extracted" not in result
    assert _names(out_dir, "*.srt") == [], \
        "a sidecar without its video is worse than no sidecar"
    assert src.exists()


def test_k2_valid_output_still_finalizes_sidecars(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [VALID]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert len(result["subtitles_extracted"]) == 1
    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]


# ══ GROUPS L & M — more specific verdicts are never overwritten ════════════

def test_l1_cancellation_is_still_reported_as_cancellation(env):
    src, out_dir, fake = env
    fake.finals = [(-2, "cancelled")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"


def test_l2_a_stall_is_still_reported_as_a_timeout(env):
    src, out_dir, fake = env
    fake.finals = [(-3, "no progress for 300s")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir)

    assert result["status"] == "timeout"
    assert result["msg"] == "no progress for 300s"


def test_l3_a_failed_launch_is_still_a_launch_failure(env):
    src, out_dir, fake = env
    fake.finals = [(-1, "ffmpeg not found")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert "ffmpeg not found" in result["msg"]
    assert fake.attempts == 1, "a launch failure earns no fallback"


def test_m1_no_space_failure_keeps_its_own_message(env):
    src, out_dir, fake = env
    fake.finals = [(1, "av_interleaved_write_frame(): No space left on device")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert "no space left on device" in result["msg"].lower()
    assert fake.attempts == 1, "a full disk is not a container problem"


def test_m2_an_ordinary_nonzero_exit_keeps_its_stderr(env):
    src, out_dir, fake = env
    fake.finals = [(1, "Unknown encoder 'libx265'")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    assert "Unknown encoder 'libx265'" in result["msg"]


# ══ GROUP N — stream policy and target accounting are untouched ════════════

def test_n1_mp4_mapping_is_unchanged_by_the_gate(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")], audio_count=2)
    fake.payloads = [VALID]

    _run(src, out_dir, fmt="MP4 (H.264)")

    cmd = fake.final_cmds[0]
    assert cmd.count("-map") == 3
    assert "0:v:0" in cmd and "0:a?" in cmd and "0:2" in cmd


def test_n2_mkv_mapping_is_unchanged_by_the_gate(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")], audio_count=2)
    fake.payloads = [VALID]

    _run(src, out_dir, fmt="MKV (H.265)")

    cmd = fake.final_cmds[0]
    assert "0:t?" in cmd, "attachments still ride along"
    assert "0:a?" in cmd and "0:2" in cmd


def test_n3_webm_mapping_is_unchanged_by_the_gate(env, monkeypatch):
    src, out_dir, fake = env
    _probe(monkeypatch, [_sub(2, "subrip")], audio_count=2)
    fake.payloads = [VALID]

    _run(src, out_dir, fmt="WebM (VP9)")

    cmd = fake.final_cmds[0]
    assert "0:v:0" in cmd and "0:a?" in cmd and "0:2" in cmd
    assert "0:t?" not in cmd, "WebM carries no attachments"
    assert cmd[cmd.index("-c:s") + 1] == "webvtt"


def test_n4_target_accounting_is_unchanged_by_the_gate(env, monkeypatch):
    """Two audio tracks still cost two tracks' worth of the budget."""
    src, out_dir, fake = env
    _probe(monkeypatch, [], audio_count=2)
    fake.payloads = [VALID]

    _two_pass(src, out_dir)

    cmd = fake.final_cmds[0]
    got = int(cmd[cmd.index("-b:v") + 1].rstrip("k"))
    assert got == compressor.calc_video_bitrate_kbps(1024 * 1024, 10.0, 128, 2)


# ══ GROUP O — nothing is remembered between files ══════════════════════════

def test_o_verification_state_does_not_leak_between_files(tmp_path,
                                                          monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sources = []
    for name in ("A", "B", "C"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)
    _fake_stack(monkeypatch, FakeFfmpeg(payloads=[MISSING, VALID, EMPTY]))

    statuses = [_run(s, out_dir)["status"] for s in sources]

    assert statuses == ["error", "ok", "error"]
    assert _names(out_dir) == ["B.mp4"], "only the real conversion survives"
    assert all(s.exists() for s in sources)


# ══ call counts — verification is filesystem metadata only ═════════════════

@pytest.mark.parametrize("payload", [MISSING, EMPTY, VALID])
def test_single_pass_runs_exactly_one_subprocess(env, payload):
    src, out_dir, fake = env
    fake.payloads = [payload]

    _run(src, out_dir)

    assert len(fake.all_cmds) == 1


def test_verification_never_probes_the_output(env, monkeypatch):
    """The gate reads filesystem metadata. Any probe of the output would be
    a media check, which this slice deliberately is not."""
    src, out_dir, fake = env
    probed: list[Path] = []

    def counted_inventory(path):
        probed.append(Path(path))
        return compressor.StreamInventory(subtitles=[], audio_count=1)

    def counted_duration(path):
        probed.append(Path(path))
        return 10.0
    monkeypatch.setattr(compressor, "ffprobe_stream_inventory",
                        counted_inventory)
    monkeypatch.setattr(compressor, "ffprobe_duration", counted_duration)

    for payload in (MISSING, EMPTY, VALID):
        probed.clear()
        fake.payloads = [payload]
        _run(src, out_dir)
        assert probed and all(p == src for p in probed), \
            "the output is never probed"


def test_result_shape_gains_no_verification_fields(env):
    src, out_dir, fake = env
    fake.payloads = [VALID]

    result = _run(src, out_dir)

    for leaked in ("output_verified", "output_size", "phantom_success",
                   "artifact_status", "mux_failed"):
        assert leaked not in result
