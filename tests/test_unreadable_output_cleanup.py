"""Ownership-safe removal of a final artifact ffprobe definitively refused.

Tab 18 made `ffprobe -v error <output>` the last gate before a conversion is
called a success, and deliberately stopped there: a rejected artifact stayed on
disk, because Cove's only failed-output cleanup reclaims the *zero-byte*
reservation it made and declines anything with bytes in it. That guard is
right in general - it cannot prove whose bytes those are - and wrong here,
where the proof exists. `reserve_output` claimed this exact path with
O_CREAT|O_EXCL, so no pre-existing file was ever at it; `os.replace` filled
that same claimed path from this job's own temp; and ownership never passed to
the user, because readability failed before the ok result, before sidecar
finalization, and before source deletion became eligible. What is left is
Cove's own file, named like a finished video, and not a video.

So it is removed - and only it, and only for one reason.

The reason matters more than the removal. "Not readable" collapses two facts
that are not the same fact:

    ffprobe ran, finished, and exited nonzero
        -> evidence about the bytes. The artifact is not media. Remove it.

    ffprobe could not be launched, timed out, or the user cancelled
        -> evidence about the probe. The artifact may well be a perfectly
           good video that Cove merely failed to verify. Keep it.

Both still fail the conversion closed and both still keep the source. Only the
first earns a deletion. Deleting output because the validator broke would turn
a verification bug into data loss, which is the larger of the two mistakes.

    A.  Definitive rejection: the owned artifact is gone, in every container.
    B.  Readable output is never touched.
    C.  A probe that cannot launch preserves the artifact.
    D.  A probe that times out preserves the artifact.
    E.  A cancellation preserves the artifact.
    F.  Tab 10's zero-byte cleanup is untouched and unprobed.
    G.  A failed encode keeps its own cleanup and its own verdict.
    H.  An unreadable apparent MP4 success is still terminal - cleanup buys
        no fallback.
    I.  The fallback MKV is cleaned; two attempts remains the ceiling.
    J/K/L. Only the exact reserved path is eligible. Collision neighbours,
        deep-collision neighbours and fallback-collision neighbours are
        byte-identical afterwards.
    M.  No sidecar is finalized behind a removed video.
    N.  The source survives every cleanup; readable deletion is unchanged.
    O.  A cleanup that itself fails stays secondary to the readability error.
    P.  Nothing about cleanup reaches the public result dict.
    Q.  Nothing leaks between files.

Real ffprobe, real media, real `reserve_output`: the encoder is faked and
writes the bytes it was scripted with, exactly as the Tab 10 and Tab 18 suites
do, but the readability gate, the reservation and the cleanup are all the
production ones.
"""
import os
import subprocess
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


# ── real media, real noise ───────────────────────────────────────────────────

SRC_BYTES = 4 * 1024 * 1024

# Non-empty and structurally fine by every Tab 10 criterion - a regular file
# well over zero bytes - and not media by any of them.
GARBAGE = bytes(range(256)) * 4

MISSING = None
EMPTY = b""

# The user's own bytes. Any test that finds these changed has watched Cove
# delete a file it does not own.
SENTINEL = b"the user's own file - not Cove's to touch"

CLEANABLE_FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"]
EXT_OF = {"MP4 (H.264)": ".mp4", "MP4 (H.265)": ".mp4",
          "MKV (H.265)": ".mkv", "WebM (VP9)": ".webm"}

_MEDIA_RECIPES = {
    ".mp4": ["-c:v", "libx264", "-c:a", "aac"],
    ".mkv": ["-c:v", "libx265", "-c:a", "aac"],
    ".webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
}


@pytest.fixture(scope="session")
def real_media(tmp_path_factory) -> dict:
    """Half a second of genuine video per container, built once."""
    d = tmp_path_factory.mktemp("cleanup_readable_media")
    out = {}
    for ext, codecs in _MEDIA_RECIPES.items():
        path = d / f"sample{ext}"
        cmd = [compressor.FFMPEG_BIN, "-v", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=0.5",
               "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
               *codecs, "-shortest", str(path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           env=compressor.clean_subprocess_env(),
                           **compressor.SUBPROCESS_FLAGS)
        assert r.returncode == 0 and path.exists() and path.stat().st_size > 0, \
            f"could not build a readable {ext} sample: {r.stderr[-400:]}"
        out[ext] = path.read_bytes()
    return out


# ── fakes ────────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _sub(index: int, codec: str) -> dict:
    return {"index": index, "codec_name": codec, "tags": {"language": "eng"}}


class FakeFfmpeg:
    """`run_ffmpeg` stand-in: scripts each invocation's exit code and the bytes
    it leaves behind. Same shape as the Tab 10 and Tab 18 suites' fakes."""

    def __init__(self, finals=None, pass1=None, subtitle=None, payloads=None,
                 cancel_after_final=None):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.subtitle = list(subtitle or [])
        self.payloads = list(payloads or [])
        self.cancel_after_final = cancel_after_final
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
        payload = self._next(self.payloads, GARBAGE)
        if rc == 0 and payload is not MISSING:
            out.write_bytes(payload)
        if rc == 0 and self.cancel_after_final is not None:
            self.cancel_after_final.set()
        return rc, err

    @property
    def final_muxers(self) -> list[str]:
        return [_muxer_of(c) for c in self.final_cmds]

    @property
    def attempts(self) -> int:
        return len(self.final_cmds)


class ProbeSpy:
    """Counts and records every ffprobe launched through `subprocess.run`.

    Source probes are faked out of the way by `_fake_stack`, so anything that
    reaches here is a probe of a finished artifact. The real ffprobe still runs
    by default; `result` and `raises` script an outcome without replacing the
    whole subprocess layer.
    """

    def __init__(self, real_run):
        self._real = real_run
        self.cmds: list[list] = []
        self.result: tuple | None = None      # (returncode, stdout, stderr)
        self.raises: BaseException | None = None

    def __call__(self, cmd, *args, **kwargs):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        if argv and str(argv[0]) == str(compressor.FFPROBE_BIN):
            self.cmds.append(argv)
            if self.raises is not None:
                raise self.raises
            if self.result is not None:
                rc, out, err = self.result
                return subprocess.CompletedProcess(argv, rc, out, err)
        return self._real(cmd, *args, **kwargs)

    @property
    def count(self) -> int:
        return len(self.cmds)

    @property
    def targets(self) -> list[Path]:
        return [Path(c[-1]) for c in self.cmds]


def _fake_stack(monkeypatch, fake) -> ProbeSpy:
    """Fake the encoder and the *source* probes; leave the final readability
    gate, the reservation and the cleanup entirely alone."""
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: 10.0)
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: compressor.StreamInventory(subtitles=[], audio_count=1))
    monkeypatch.setattr(compressor, "nvenc_available",
                        lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": False)
    spy = ProbeSpy(subprocess.run)
    monkeypatch.setattr(compressor.subprocess, "run", spy)
    return spy


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fake = FakeFfmpeg()
    spy = _fake_stack(monkeypatch, fake)
    return src, out_dir, fake, spy


def _probe(monkeypatch, streams, audio_count=1):
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda path: compressor.StreamInventory(
            subtitles=[dict(s) for s in streams], audio_count=audio_count))


def _run(src, out_dir, fmt="MP4 (H.264)", mode="Quality preset",
         mode_value="Balanced", cancel_flag=None, **kw):
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, "128",
        cancel_flag if cancel_flag is not None else threading.Event(), **kw)


def _two_pass(src, out_dir, fmt="MP4 (H.264)"):
    return _run(src, out_dir, fmt=fmt, mode="Target file size", mode_value=1)


def _names(out_dir, pattern="*") -> list[str]:
    return sorted(p.name for p in out_dir.glob(pattern))


def _video_names(out_dir) -> list[str]:
    return sorted(p.name for p in out_dir.glob("*")
                  if p.suffix.lower() in (".mp4", ".mkv", ".webm"))


# ══ GROUP A — a definitively rejected artifact is removed ═══════════════════

@pytest.mark.parametrize("fmt", CLEANABLE_FORMATS)
def test_a1_definitively_rejected_artifact_is_removed(env, fmt):
    """The Tab 18 residue. ffmpeg exits 0, the destination holds bytes that
    are not media, ffprobe runs to completion and refuses them - and the file
    Cove reserved, filled and never handed over does not survive the job."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert spy.count == 1, "exactly one probe, and it was conclusive"
    assert _video_names(out_dir) == [], \
        "a rejected artifact named like a finished video is worse than none"
    assert src.exists(), "the original is still the only readable copy"


def test_a2_the_removed_path_is_exactly_the_probed_path(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    _run(src, out_dir)

    assert spy.targets == [out_dir / "Movie.mp4"]
    assert not (out_dir / "Movie.mp4").exists()


def test_a3_rejection_by_a_scripted_nonzero_exit_also_cleans(env, real_media):
    """The bytes are genuinely readable; ffprobe is scripted to refuse them
    anyway. Cleanup follows the verdict, not the payload - a definitive
    nonzero exit is the whole trigger."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (1, "", "moov atom not found")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert "readable" in result["msg"].lower()
    assert not (out_dir / "Movie.mp4").exists()


def test_a4_two_pass_rejection_cleans_without_a_third_encode(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 2, "pass 1 and pass 2, and nothing more"
    assert spy.count == 1, "pass 1 is never probed"
    assert _video_names(out_dir) == []
    assert src.exists()


# ══ GROUP B — a readable artifact is never touched ═════════════════════════

@pytest.mark.parametrize("fmt", CLEANABLE_FORMATS)
def test_b1_readable_output_survives(env, real_media, fmt):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes() == real_media[EXT_OF[fmt]]


def test_b2_readable_output_keeps_its_sidecar_and_success_shape(env,
                                                                real_media,
                                                                monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert (out_dir / "Movie.mp4").exists()
    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]


# ══ GROUP C — a probe that cannot launch preserves the artifact ════════════

@pytest.mark.parametrize("boom", [
    OSError(13, "permission denied"),
    OSError(2, "no such file or directory"),
])
def test_c1_launch_failure_preserves_a_valid_artifact(env, real_media, boom):
    """ffprobe was there when the job started and is not there now. The file
    at the destination is genuine video. Cove cannot say so, fails closed, and
    must not delete what it merely failed to verify."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = boom

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"], \
        "a validator failure is not permission to delete a valid output"
    assert src.exists()
    assert "output" not in result, "unverified is never handed to a caller"


def test_c2_launch_failure_preserves_even_unverifiable_bytes(env):
    """Cove does not get to guess. The payload happens to be garbage, but the
    probe never ran, so nothing here is evidence about the bytes."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    spy.raises = OSError(13, "permission denied")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == GARBAGE
    assert src.exists()


@pytest.mark.parametrize("rc", [
    -9,             # POSIX SIGKILL - the OOM killer, or a supervisor
    -11,            # POSIX SIGSEGV - ffprobe itself crashed
    3,              # Windows CRT abort() - what av_assert0 lands on
    2,              # any other small status ffprobe does not use to refuse
    255,
    3221225477,     # Windows 0xC0000005 access violation
    3221225725,     # Windows 0xC00000FD stack overflow
    3221226505,     # Windows 0xC0000409 fast-fail, the other abort() shape
])
def test_c3_a_dead_probe_preserves_the_artifact(env, real_media, rc):
    """A crash is not a refusal. ffprobe died before it could deliver a
    verdict, which puts it in the same class as a launch failure and a timeout:
    the conversion fails closed, and the artifact is not Cove's to throw away.

    The parametrization is deliberately awkward because process death does not
    have one shape. POSIX signal death is negative. A Windows crash status is
    reported *unsigned*, so 0xC0000005 arrives as 3221225477. And an aborted
    CRT process - which is where ffmpeg's `av_assert0` lands - can exit with
    plain `3`, indistinguishable by size from a normal status. That last case
    is why the rule matches ffprobe's single refusal code instead of any range:
    every range rule, however carefully bounded, eventually deletes an artifact
    on a crash."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (rc, "", "")

    result = _run(src, out_dir)

    assert result["status"] == "error", "still fails closed"
    assert "readable" in result["msg"].lower()
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"], \
        "a probe that died said nothing about these bytes"
    assert src.exists()
    assert set(result) == {"file", "status", "msg"}


def test_c4_ffprobes_own_refusal_status_is_still_a_rejection(env, real_media):
    """The counterpart to c3, and the reason the rule is not simply "preserve
    unless rc == 0". Exit 1 is how ffprobe says no, measured against the real
    binary on garbage bytes, an empty file, ASCII text, 4 KB of zeros, a
    truncated MP4, a missing path and a directory - all 1, and only genuinely
    readable media exits 0. That verdict still earns cleanup."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (1, "", "moov atom not found")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()


@pytest.mark.parametrize("payload", [
    GARBAGE,
    b"",
    b"this is plainly not a video" * 50,
    b"\x00" * 4096,
])
def test_c5_the_refusal_constant_matches_the_real_binary(tmp_path, payload):
    """The whole rule rests on ffprobe having exactly one way to say no. Pin
    that against the installed binary instead of trusting a comment: if some
    future ffprobe starts refusing files with a status other than 1, this fails
    here rather than silently turning every rejection into a preserved
    artifact."""
    bad = tmp_path / "not-media.mp4"
    bad.write_bytes(payload)

    r = subprocess.run([compressor.FFPROBE_BIN, "-v", "error", str(bad)],
                       capture_output=True, text=True, timeout=60,
                       env=compressor.clean_subprocess_env(),
                       **compressor.SUBPROCESS_FLAGS)

    assert r.returncode == compressor._FFPROBE_REFUSAL_STATUS


def test_c6_real_media_still_exits_zero(tmp_path, real_media):
    """The other half of the same pin."""
    good = tmp_path / "media.mp4"
    good.write_bytes(real_media[".mp4"])

    r = subprocess.run([compressor.FFPROBE_BIN, "-v", "error", str(good)],
                       capture_output=True, text=True, timeout=60,
                       env=compressor.clean_subprocess_env(),
                       **compressor.SUBPROCESS_FLAGS)

    assert r.returncode == 0


# ══ GROUP D — a timeout preserves the artifact ═════════════════════════════

def test_d1_probe_timeout_preserves_the_artifact(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"]
    assert src.exists()


def test_d2_timeout_on_garbage_still_preserves(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    spy.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    _run(src, out_dir)

    assert (out_dir / "Movie.mp4").read_bytes() == GARBAGE


# ══ GROUP E — cancellation preserves the artifact ══════════════════════════

def test_e1_cancellation_during_validation_preserves_the_artifact(env,
                                                                  real_media):
    """The flag lands between ffmpeg exiting and the gate running. A cancelled
    job is not a successful conversion - and it is not a licence to delete the
    output the user just interrupted either."""
    src, out_dir, fake, spy = env
    cancel = threading.Event()
    fake.cancel_after_final = cancel
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir, cancel_flag=cancel)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"]
    assert src.exists()


def test_e2_cancellation_probes_nothing_and_deletes_nothing(env):
    src, out_dir, fake, spy = env
    cancel = threading.Event()
    fake.cancel_after_final = cancel
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, cancel_flag=cancel)

    assert result["msg"] == "cancelled"
    assert spy.count == 0, "a cancelled job asks ffprobe nothing"
    assert (out_dir / "Movie.mp4").read_bytes() == GARBAGE


# ══ GROUP F — Tab 10's zero-byte cleanup is untouched ══════════════════════

def test_f1_zero_byte_output_is_still_reclaimed_and_never_probed(env):
    src, out_dir, fake, spy = env
    fake.payloads = [EMPTY]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert "no valid output file" in result["msg"]
    assert spy.count == 0, "Tab 10 runs first and stops the job before a probe"
    assert not (out_dir / "Movie.mp4").exists(), \
        "the empty reservation is reclaimed exactly as it always was"
    assert src.exists()


def test_f2_missing_output_is_still_reclaimed_and_never_probed(env):
    src, out_dir, fake, spy = env
    fake.payloads = [MISSING]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0
    assert not (out_dir / "Movie.mp4").exists()


def test_f3_zero_byte_cleanup_still_spares_a_neighbour(env):
    src, out_dir, fake, spy = env
    sentinel = out_dir / "Movie.mp4"
    sentinel.write_bytes(SENTINEL)
    fake.payloads = [EMPTY]

    _run(src, out_dir)

    assert sentinel.read_bytes() == SENTINEL
    assert not (out_dir / "Movie_1.mp4").exists()


# ══ GROUP G — a failed encode keeps its own cleanup and verdict ════════════

@pytest.mark.parametrize("rc,err,status,needle", [
    (1, "Encoder error", "error", "Encoder error"),
    (-1, "could not start FFmpeg", "error", "could not start FFmpeg"),
    (-3, "stalled", "timeout", "stalled"),
])
def test_g1_failed_encode_is_unchanged(env, rc, err, status, needle):
    src, out_dir, fake, spy = env
    fake.finals = [(rc, err)]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == status, "each failure keeps its own verdict"
    assert needle in result["msg"]
    assert spy.count == 0, "a failed encode is never probed"
    assert not (out_dir / "Movie.mkv").exists(), \
        "the untouched reservation is reclaimed by the predecessor path"
    assert src.exists()


def test_g2_enospc_failure_is_unchanged(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "av_interleaved_write_frame(): No space left on device")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert fake.attempts == 1, "a full disk earns no container retry"
    assert spy.count == 0
    assert not (out_dir / "Movie.mp4").exists()


# ══ GROUP H — an unreadable MP4 success is still terminal ══════════════════

def test_h1_unreadable_mp4_h265_cleans_and_does_not_fall_back(env):
    """Removing the artifact says nothing about retrying. An apparent success
    whose bytes are not media is a different anomaly from the mux failure the
    MP4 -> MKV retry exists for, and it stays terminal."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 1
    assert fake.final_muxers == ["mp4"]
    assert not result.get("fallback_used")
    assert spy.count == 1
    assert result["status"] == "error"
    assert _video_names(out_dir) == [], "no MP4 left, and no MKV attempted"
    assert src.exists()


# ══ GROUP I — the fallback MKV is cleaned; two attempts is the ceiling ═════

def test_i1_unreadable_fallback_mkv_is_cleaned(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, GARBAGE]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 2, "two attempts is the ceiling, always"
    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "error"
    assert result["fallback_used"] is True
    assert spy.count == 1
    assert spy.targets[0].suffix == ".mkv"
    assert _video_names(out_dir) == [], \
        "the failed MP4 stub and the rejected MKV both go"
    assert src.exists()


def test_i2_readable_fallback_mkv_survives(env, real_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, real_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == real_media[".mkv"]
    assert _video_names(out_dir) == ["Movie.mkv"]


# ══ GROUPS J/K/L — only the exact reserved path is eligible ════════════════

def test_j1_one_collision_the_neighbour_is_byte_identical(env):
    """`Movie.mp4` is the user's, so this job reserved `Movie_1.mp4`. Exactly
    one of those two files may be removed."""
    src, out_dir, fake, spy = env
    sentinel = out_dir / "Movie.mp4"
    sentinel.write_bytes(SENTINEL)
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.targets == [out_dir / "Movie_1.mp4"]
    assert not (out_dir / "Movie_1.mp4").exists()
    assert sentinel.read_bytes() == SENTINEL, "never touched"
    assert _video_names(out_dir) == ["Movie.mp4"]


def test_k1_deep_collision_only_the_reserved_path_goes(env):
    src, out_dir, fake, spy = env
    sentinels = {}
    for name in ("Movie.mp4", "Movie_1.mp4", "Movie_2.mp4"):
        p = out_dir / name
        p.write_bytes(SENTINEL + name.encode())
        sentinels[name] = p.read_bytes()
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.targets == [out_dir / "Movie_3.mp4"]
    assert not (out_dir / "Movie_3.mp4").exists()
    for name, blob in sentinels.items():
        assert (out_dir / name).read_bytes() == blob, f"{name} was touched"
    assert _video_names(out_dir) == ["Movie.mp4", "Movie_1.mp4", "Movie_2.mp4"]


def test_l1_fallback_collision_only_the_reserved_mkv_goes(env):
    src, out_dir, fake, spy = env
    sentinel = out_dir / "Movie.mkv"
    sentinel.write_bytes(SENTINEL)
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, GARBAGE]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert spy.targets == [out_dir / "Movie_1.mkv"]
    assert not (out_dir / "Movie_1.mkv").exists()
    assert sentinel.read_bytes() == SENTINEL
    assert _video_names(out_dir) == ["Movie.mkv"]


def test_l2_cleanup_never_globs_stem_alikes(env):
    """Nothing is removed by resembling the output name."""
    src, out_dir, fake, spy = env
    decoys = {}
    for name in ("Movie.mp4.bak", "Movie-final.mp4", "Movie.txt",
                 "Movie_1.mp4"):
        p = out_dir / name
        p.write_bytes(SENTINEL + name.encode())
        decoys[name] = p.read_bytes()
    fake.payloads = [GARBAGE]

    _run(src, out_dir)

    # `Movie.mp4` was free, so that is what was reserved, filled and rejected.
    assert not (out_dir / "Movie.mp4").exists()
    for name, blob in decoys.items():
        assert (out_dir / name).read_bytes() == blob, f"{name} was touched"


# ══ GROUP M — no sidecar is finalized behind a removed video ═══════════════

def test_m1_removed_video_finalizes_no_sidecar(env, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert "subtitles_extracted" not in result
    assert _names(out_dir, "*.srt") == [], \
        "a sidecar without a playable video is worse than no sidecar"
    assert not (out_dir / "Movie.mp4").exists()
    assert not list(out_dir.glob("cove_*")), "no temp directory outlives the job"


def test_m2_a_pre_existing_sidecar_is_not_collateral(env, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    existing = out_dir / "Movie.eng.srt"
    existing.write_bytes(SENTINEL)
    fake.payloads = [GARBAGE]

    _run(src, out_dir, extract_english_subtitles=True)

    assert existing.read_bytes() == SENTINEL, \
        "cleanup removes one video path, not a stem's worth of files"


# ══ GROUP N — the source survives every cleanup ════════════════════════════

def test_n1_cleaned_artifact_still_keeps_the_source(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists(), "deleting the only readable copy is the whole risk"
    assert result.get("source_deleted") is not True
    assert not (out_dir / "Movie.mp4").exists()


def test_n2_readable_output_still_deletes_the_source(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] == "ok"
    assert not src.exists()
    assert result["source_deleted"] is True
    assert (out_dir / "Movie.mp4").exists()


def test_n3_preserved_artifact_still_keeps_the_source(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = OSError(13, "permission denied")

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True
    assert (out_dir / "Movie.mp4").exists()


# ══ GROUP O — a cleanup that fails stays secondary ═════════════════════════

def _unlink_refusing(target: Path):
    """A real `os.unlink` that refuses exactly one path."""
    real = os.unlink

    def fake(path, *a, **kw):
        if Path(path) == target:
            raise OSError(13, "the file is open in another program")
        return real(path, *a, **kw)
    return fake


def test_o1_cleanup_oserror_keeps_the_readability_verdict(env, monkeypatch):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    monkeypatch.setattr(compressor.os, "unlink",
                        _unlink_refusing(out_dir / "Movie.mp4"))

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert "readable" in result["msg"].lower(), \
        "the user's answer is why the conversion failed, not why a file stayed"
    assert "delete" not in result["msg"].lower()
    assert "unlink" not in result["msg"].lower()
    assert src.exists()
    # The removal genuinely could not happen, so the file is still there.
    assert (out_dir / "Movie.mp4").read_bytes() == GARBAGE


def test_o2_cleanup_oserror_raises_nothing(env, monkeypatch):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    monkeypatch.setattr(compressor.os, "unlink",
                        _unlink_refusing(out_dir / "Movie.mp4"))

    result = _run(src, out_dir)  # must not raise

    assert isinstance(result, dict)
    assert result["status"] == "error"


def test_o3_a_vanished_artifact_is_not_an_error(env, monkeypatch):
    """Something else removed it between the probe and the cleanup."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    real_unlink = os.unlink

    def racing(path, *a, **kw):
        real_unlink(path, *a, **kw)
        raise FileNotFoundError(2, "gone")
    monkeypatch.setattr(compressor.os, "unlink", racing)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert "readable" in result["msg"].lower()


# ══ GROUP P — cleanup reaches nothing public ═══════════════════════════════

def test_p1_failure_result_shape_is_unchanged(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert set(result) == {"file", "status", "msg"}
    for leaked in ("artifact_deleted", "cleanup_failed", "probe_definitive",
                   "artifact_removed", "cleanup_error", "rejected"):
        assert leaked not in result


def test_p2_failed_cleanup_adds_no_keys_either(env, monkeypatch):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    monkeypatch.setattr(compressor.os, "unlink",
                        _unlink_refusing(out_dir / "Movie.mp4"))

    result = _run(src, out_dir)

    assert set(result) == {"file", "status", "msg"}


def test_p3_preserved_artifact_adds_no_keys(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = OSError(13, "permission denied")

    result = _run(src, out_dir)

    assert set(result) == {"file", "status", "msg"}


def test_p4_success_result_shape_is_unchanged(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert set(result) == {"file", "output", "status", "original", "new",
                           "encoder"}


# ══ GROUP Q — nothing leaks between files ══════════════════════════════════

def test_q1_four_files_each_get_their_own_verdict(tmp_path, monkeypatch,
                                                  real_media):
    """A unreadable and cleaned, B readable, C unreadable with cleanup
    refused, D readable. No file's outcome touches another's."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sources = []
    for name in ("A", "B", "C", "D"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)

    fake = FakeFfmpeg(payloads=[GARBAGE, real_media[".mp4"],
                                GARBAGE, real_media[".mp4"]])
    spy = _fake_stack(monkeypatch, fake)
    monkeypatch.setattr(compressor.os, "unlink",
                        _unlink_refusing(out_dir / "C.mp4"))

    results = [_run(s, out_dir) for s in sources]

    assert [r["status"] for r in results] == ["error", "ok", "error", "ok"]
    assert spy.count == 4, "one probe per finished artifact, every time"
    assert all(s.exists() for s in sources), "no source is deleted by this path"
    assert not (out_dir / "A.mp4").exists(), "A was cleaned"
    assert (out_dir / "B.mp4").read_bytes() == real_media[".mp4"]
    assert (out_dir / "C.mp4").read_bytes() == GARBAGE, "C's cleanup was refused"
    assert (out_dir / "D.mp4").read_bytes() == real_media[".mp4"]


def test_q2_a_cleaned_run_does_not_shift_the_next_reservation(tmp_path,
                                                              monkeypatch,
                                                              real_media):
    """A's path is freed by cleanup. B is a different file and must still get
    its own name - nothing about A's removal is remembered."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    a = tmp_path / "A.mov"
    a.write_bytes(b"s" * SRC_BYTES)
    b = tmp_path / "B.mov"
    b.write_bytes(b"s" * SRC_BYTES)

    fake = FakeFfmpeg(payloads=[GARBAGE, real_media[".mp4"]])
    spy = _fake_stack(monkeypatch, fake)

    first = _run(a, out_dir)
    second = _run(b, out_dir)

    assert first["status"] == "error"
    assert second["status"] == "ok"
    assert Path(second["output"]).name == "B.mp4"
    assert _video_names(out_dir) == ["B.mp4"]


def test_q4_every_rejected_file_is_cleaned_not_just_the_first(tmp_path,
                                                              monkeypatch):
    """Two rejections in a row. The second is cleaned because it is the second
    file's own reserved path - a cleanup that remembered the first file's path
    would quietly stop working after one file."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sources = []
    for name in ("A", "B"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)

    fake = FakeFfmpeg(payloads=[GARBAGE, GARBAGE])
    spy = _fake_stack(monkeypatch, fake)

    results = [_run(s, out_dir) for s in sources]

    assert [r["status"] for r in results] == ["error", "error"]
    assert spy.count == 2
    assert _video_names(out_dir) == [], "both reservations were reclaimed"
    assert all(s.exists() for s in sources)


def test_q3_the_same_source_twice_reuses_the_freed_name(tmp_path, monkeypatch,
                                                        real_media):
    """First run is rejected and its reservation removed, so the second run
    gets the base name back rather than bumping to `_1`."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)

    fake = FakeFfmpeg(payloads=[GARBAGE, real_media[".mp4"]])
    spy = _fake_stack(monkeypatch, fake)

    first = _run(src, out_dir)
    second = _run(src, out_dir)

    assert first["status"] == "error"
    assert second["status"] == "ok"
    assert Path(second["output"]).name == "Movie.mp4"
    assert _video_names(out_dir) == ["Movie.mp4"]


# ══ cleanup is one unlink of one path, and nothing else ═══════════════════

def test_cleanup_never_recurses_into_a_directory(env, monkeypatch):
    """Tab 10 rejects a directory at the output path long before a probe runs,
    so this can only fire on a future bug. It must fire as 'nothing happened',
    never as a recursive removal."""
    src, out_dir, fake, spy = env
    victim = out_dir / "Movie.mp4"

    def fake_replace(a, b):
        Path(b).unlink()
        Path(b).mkdir()
        (Path(b) / "precious.txt").write_bytes(SENTINEL)
    monkeypatch.setattr(compressor.os, "replace", fake_replace)
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0, "a directory is not probed"
    assert victim.is_dir()
    assert (victim / "precious.txt").read_bytes() == SENTINEL


def test_rejection_costs_no_extra_subprocess(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    _run(src, out_dir)

    assert len(fake.all_cmds) == 1, "one encode"
    assert spy.count == 1, "one probe - cleanup launches nothing"


def test_the_source_is_never_a_cleanup_target(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    _run(src, out_dir)

    assert src.exists() and src.stat().st_size == SRC_BYTES
