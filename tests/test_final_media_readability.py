"""Final media readability: a file with bytes in it is not a video.

Tab 10 asked whether an encode that exited 0 actually left a file, and that
question has one honest answer available from filesystem metadata alone. It
is also the last question Cove asked. A successful exit that leaves 1 KB of
noise at the destination passes that gate intact: Cove reports the conversion
as ok, finalizes its sidecars, and - with delete-source enabled - removes the
original, which was the only readable copy.

So one more question gets asked, and only one: can ffprobe open the finished
file as media? Not how long it is, not what codecs are in it, not how many
streams it carries - those are semantic policies with their own failure
modes, and every one of them would eventually reject a file that plays. The
gate here is `ffprobe -v error <output>` and its exit code, nothing else.

That makes ffprobe a hard dependency of every video conversion rather than a
target-mode one, in both places that can start an encode: the application
start boundary blocks before a worker thread exists, and `compress_video`
itself fails closed so a direct caller - a test, a script, a future GUI bug -
cannot route around it.

  A.   The Tab 17 scenario: success + non-empty garbage, in every container.
  B.   Real media through the real gate still succeeds.
  C.   Tab 10 runs first: nothing missing, empty, non-regular or unstatable
       is ever probed.
  D.   A failed encode is never probed, and keeps its own verdict.
  E.   Two-pass validates pass 2 and only pass 2.
  F.   The MP4 -> MKV fallback validates the MKV it actually produced; an
       unreadable apparent MP4 success is terminal and earns no retry.
  G.   Source deletion is impossible until readability passes.
  H.   Sidecars are not finalized behind an unreadable video.
  I/J. The collision-resolved path is what gets probed, never the requested
       one.
  K.   Readability is the exit code and nothing else.
  L.   A missing ffprobe fails the core closed, before any encode.
  M.   Nothing is remembered between files.
  N.   The result dict grows no probe internals.

Real ffprobe, real media: the encoder is faked (it writes the bytes the fake
was scripted with, exactly as the Tab 10 suite does) but the readability gate
is the production one, running the production subprocess against genuinely
readable and genuinely unreadable bytes. `ProbeSpy` wraps `subprocess.run` to
count and inspect those launches without replacing them.
"""
import os
import stat as stat_mod
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

ALL_FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"]
EXT_OF = {"MP4 (H.264)": ".mp4", "MP4 (H.265)": ".mp4",
          "MKV (H.265)": ".mkv", "WebM (VP9)": ".webm"}

_MEDIA_RECIPES = {
    ".mp4": ["-c:v", "libx264", "-c:a", "aac"],
    ".mkv": ["-c:v", "libx265", "-c:a", "aac"],
    ".webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
}


@pytest.fixture(scope="session")
def real_media(tmp_path_factory) -> dict:
    """Half a second of genuine video per container, built once.

    Small enough to be free and real enough that the production probe has to
    accept it for the right reason.
    """
    d = tmp_path_factory.mktemp("readable_media")
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
    """`run_ffmpeg` stand-in: scripts each invocation's exit code and the
    bytes it leaves behind. Identical in shape to the Tab 10 suite's fake so
    the two contracts are described the same way."""

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

    The source probes are faked out of the way by `_fake_stack`, so anything
    that reaches here is a probe of a finished artifact. By default the real
    ffprobe still runs; `result` and `raises` let a test script the outcome
    without pretending to be the whole subprocess layer.
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
        """What each probe was pointed at - the final positional argument."""
        return [Path(c[-1]) for c in self.cmds]


def _fake_stack(monkeypatch, fake) -> ProbeSpy:
    """Fake the encoder and the *source* probes; leave the final readability
    gate entirely alone."""
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


def _no_ffprobe(monkeypatch, tmp_path):
    """Point Cove at an ffprobe that is not there, the way Tab 17 did."""
    monkeypatch.setattr(compressor, "FFPROBE_BIN",
                        str(tmp_path / "definitely-not-ffprobe.exe"))


# ══ GROUP A — the Tab 17 scenario ═══════════════════════════════════════════

@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_a1_success_with_unreadable_output_is_a_failure(env, fmt):
    """ffmpeg exits 0, the destination holds 1 KB of noise, Tab 10 is
    satisfied - and the conversion did not happen."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert "output" not in result
    assert src.exists(), "the original is the only readable copy"
    assert spy.count == 1, "exactly one probe of the finished artifact"


def test_a2_failure_message_is_honest_and_leaks_nothing(env):
    """Tab 18 knows that ffprobe rejected the file. It does not know why, and
    must not claim to."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    msg = (_run(src, out_dir).get("msg") or "").lower()

    assert msg
    assert "readable" in msg
    for overclaim in ("corrupt", "damaged", "truncated"):
        assert overclaim not in msg
    for leak in ("returncode", "ffprobe -v", "stderr", "exit code"):
        assert leak not in msg


def test_a3_unreadable_output_leaves_no_success_side_effects(env, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [GARBAGE]

    result = delete_source_if_eligible(
        _run(src, out_dir, extract_english_subtitles=True), enabled=True)

    assert result["status"] == "error"
    assert src.exists()
    assert result.get("source_deleted") is not True
    assert "subtitles_extracted" not in result
    assert _names(out_dir, "*.srt") == []


def test_a4_the_unreadable_artifact_is_removed(env):
    """Tab 18 left this artifact on disk: the generic failed-output cleanup
    reclaims only the *empty* reservation it made and declines anything with
    bytes in it, because it cannot prove those bytes are its own.

    Tab 19 supplies that proof for this one case. `reserve_output` claimed
    this exact path with O_CREAT|O_EXCL, the encode filled that same claimed
    path, and ownership never passed to the user - so a probe that ran to
    completion and refused the file is licence to remove it. The generic
    guard is unchanged; see `test_unreadable_output_cleanup.py` for the full
    contract, including the validator failures that are *not* licence."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()
    assert "output" not in result, "removed, and never handed to a caller"


def test_a5_a_neighbouring_file_survives_an_unreadable_conversion(env):
    """`Movie.mp4` is the user's, so this job reserved `Movie_1.mp4`. Failing
    the readability gate may touch only the second of those - the one Cove
    claimed, filled, and never handed over."""
    src, out_dir, fake, spy = env
    neighbor = out_dir / "Movie.mp4"
    neighbor.write_bytes(b"the user's own file")
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert neighbor.read_bytes() == b"the user's own file"
    assert _names(out_dir) == ["Movie.mp4"], \
        "the reserved artifact goes; the neighbour it collided with stays"


# ══ GROUP B — real media through the real gate ══════════════════════════════

@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_b1_readable_output_still_succeeds(env, real_media, fmt):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.suffix == EXT_OF[fmt]
    assert out.exists()
    assert spy.count == 1
    assert spy.targets == [out]


def test_b2_a_tiny_valid_encode_is_still_valid(env, real_media):
    """No size heuristic crept in behind the readability check: half a second
    of 160x120 video is a conversion like any other."""
    src, out_dir, fake, spy = env
    payload = real_media[".mp4"]
    fake.payloads = [payload]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert result["new"] == len(payload)


# ══ GROUP C — Tab 10 runs first, and unreadable paths are never probed ══════

@pytest.mark.parametrize("fmt", ALL_FORMATS)
@pytest.mark.parametrize("payload", [MISSING, EMPTY])
def test_c1_phantom_success_fails_without_a_probe(env, fmt, payload):
    src, out_dir, fake, spy = env
    fake.payloads = [payload]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert spy.count == 0, "nothing that failed Tab 10 is worth probing"
    assert src.exists()
    assert _names(out_dir) == []


def test_c2_a_directory_at_the_output_path_is_not_probed(env, monkeypatch):
    src, out_dir, fake, spy = env

    def fake_replace(s, d, *a, **kw):
        os.unlink(s)
        if os.path.exists(d):
            os.unlink(d)
        os.mkdir(d)
    monkeypatch.setattr(compressor.os, "replace", fake_replace)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0
    assert (out_dir / "Movie.mp4").is_dir(), "not Cove's to remove"


def test_c3_an_unstatable_output_is_not_probed(env, monkeypatch):
    src, out_dir, fake, spy = env
    target = out_dir / "Movie.mp4"
    real_stat = os.stat
    wanted = os.path.normcase(os.path.abspath(str(target)))

    def fake_stat(path, *a, **kw):
        try:
            same = os.path.normcase(os.path.abspath(str(path))) == wanted
        except (TypeError, ValueError):
            same = False
        if same:
            raise OSError(5, "simulated I/O error")
        return real_stat(path, *a, **kw)
    monkeypatch.setattr(compressor.os, "stat", fake_stat)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0
    assert src.exists()


def test_c4_a_non_regular_path_reporting_bytes_is_not_probed(env, monkeypatch):
    """POSIX reports a directory as 4096 bytes; the size clause alone would
    wave it through and hand it to ffprobe."""
    src, out_dir, fake, spy = env
    target = out_dir / "Movie.mp4"
    real_stat = os.stat
    wanted = os.path.normcase(os.path.abspath(str(target)))
    dir_with_bytes = os.stat_result(
        (stat_mod.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 4096, 0, 0, 0))

    def fake_stat(path, *a, **kw):
        try:
            same = os.path.normcase(os.path.abspath(str(path))) == wanted
        except (TypeError, ValueError):
            same = False
        return dir_with_bytes if same else real_stat(path, *a, **kw)
    monkeypatch.setattr(compressor.os, "stat", fake_stat)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0


# ══ GROUP D — a failed encode is never probed ══════════════════════════════

@pytest.mark.parametrize("rc,err,status,needle", [
    (1, "Unknown encoder 'libx265'", "error", "Unknown encoder 'libx265'"),
    (1, "av_interleaved_write_frame(): No space left on device",
     "error", "no space left on device"),
    (-1, "ffmpeg not found", "error", "ffmpeg not found"),
    (-2, "cancelled", "error", "cancelled"),
    (-3, "no progress for 300s", "timeout", "no progress for 300s"),
])
def test_d1_failed_encodes_are_never_probed(env, rc, err, status, needle):
    src, out_dir, fake, spy = env
    fake.finals = [(rc, err)]
    fake.payloads = [MISSING]

    result = _run(src, out_dir)

    assert result["status"] == status
    assert needle in result["msg"].lower() or needle in result["msg"]
    assert spy.count == 0, "a failed encode has no artifact to validate"


def test_d2_enospc_keeps_its_classification_and_its_single_attempt(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "av_interleaved_write_frame(): No space left on device")]
    fake.payloads = [MISSING]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert "no space left on device" in result["msg"].lower()
    assert fake.attempts == 1
    assert spy.count == 0


def test_d3_cancellation_during_validation_keeps_the_source(env, real_media):
    """The flag is set between ffmpeg exiting 0 and the gate running. A
    cancelled job is not a successful conversion, whatever ffprobe would have
    said about the bytes."""
    src, out_dir, fake, spy = env
    cancel = threading.Event()
    fake.cancel_after_final = cancel
    fake.payloads = [real_media[".mp4"]]

    result = delete_source_if_eligible(
        _run(src, out_dir, cancel_flag=cancel), enabled=True)

    assert result["status"] != "ok"
    assert src.exists()
    assert result.get("source_deleted") is not True


# ══ GROUP E — two-pass validates pass 2 and only pass 2 ════════════════════

def test_e1_pass_one_is_never_validated(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _two_pass(src, out_dir)

    assert result["status"] == "ok"
    assert len(fake.pass1_cmds) == 1
    assert len(fake.all_cmds) == 2, "two encode subprocesses"
    assert spy.count == 1, "one readability probe, of the pass 2 output"
    assert spy.targets == [Path(result["output"])]


def test_e2_pass_one_failure_probes_nothing(env):
    src, out_dir, fake, spy = env
    fake.pass1 = [(1, "pass 1 exploded")]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 1
    assert fake.attempts == 0
    assert spy.count == 0


def test_e3_pass_two_failure_probes_nothing(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "pass 2 exploded")]
    fake.payloads = [MISSING]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 2
    assert spy.count == 0


def test_e4_unreadable_pass_two_output_fails(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 2
    assert spy.count == 1
    assert src.exists()


# ══ GROUP F — the fallback validates what it actually produced ═════════════

def test_f1_readable_fallback_mkv_succeeds(env, real_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, real_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.final_muxers == ["mp4", "matroska"]
    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert spy.count == 1, "the failed MP4 is not validated"
    assert spy.targets == [Path(result["output"])]
    assert Path(result["output"]).suffix == ".mkv"


def test_f2_unreadable_fallback_mkv_is_terminal(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, GARBAGE]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 2, "two attempts is the ceiling, always"
    assert result["status"] == "error"
    assert result["fallback_used"] is True
    assert spy.count == 1
    assert spy.targets[0].suffix == ".mkv"
    assert src.exists()


def test_f3_unreadable_mp4_success_earns_no_fallback(env):
    """An apparent success whose bytes are not media is not the mux failure
    Tab 2b retries. Widening the retry to cover it would double every failed
    encode for no reason."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert fake.attempts == 1
    assert fake.final_muxers == ["mp4"]
    assert result["status"] == "error"
    assert not result.get("fallback_used")
    assert spy.count == 1
    assert spy.targets[0].suffix == ".mp4"
    assert src.exists()


# ══ GROUP G — source deletion is impossible before readability ═════════════

def test_g1_readable_output_still_deletes_the_source(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] == "ok"
    assert not src.exists()
    assert result["source_deleted"] is True


def test_g2_unreadable_output_keeps_the_source(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_g3_missing_ffprobe_keeps_the_source(env, tmp_path, monkeypatch):
    src, out_dir, fake, spy = env
    _no_ffprobe(monkeypatch, tmp_path)

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] != "ok"
    assert src.exists()
    assert result.get("source_deleted") is not True


@pytest.mark.parametrize("boom", [
    OSError(13, "permission denied"),
    subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
])
def test_g4_a_probe_that_cannot_run_keeps_the_source(env, real_media, boom):
    """Launch failure and timeout are both 'the gate did not pass', not
    'the gate does not apply'."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = boom

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] == "error"
    assert src.exists()
    assert result.get("source_deleted") is not True


# ══ GROUP H — sidecars stay behind the gate ═══════════════════════════════

def test_h1_readable_video_still_finalizes_sidecars(env, real_media,
                                                    monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert len(result["subtitles_extracted"]) == 1
    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]


def test_h2_unreadable_video_finalizes_no_sidecar(env, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert "subtitles_extracted" not in result
    assert _names(out_dir, "*.srt") == [], \
        "a sidecar without a playable video is worse than no sidecar"


def test_h3_sidecar_temps_are_still_cleaned_up(env, monkeypatch, tmp_path):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [GARBAGE]

    _run(src, out_dir, extract_english_subtitles=True)

    assert not list(out_dir.glob("cove_*")), "no temp directory outlives the job"


# ══ GROUPS I & J — the collision-resolved path is what gets probed ═════════

def test_i_collision_resolved_output_is_what_is_probed(env, real_media):
    """`Movie.mp4` is the user's. This job reserved `Movie_1.mp4`, and that
    is the file whose readability decides the job."""
    src, out_dir, fake, spy = env
    sentinel = out_dir / "Movie.mp4"
    sentinel.write_bytes(b"the user's own file")
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert Path(result["output"]).name == "Movie_1.mp4"
    assert spy.targets == [out_dir / "Movie_1.mp4"]
    assert sentinel.read_bytes() == b"the user's own file", "never touched"


def test_j_fallback_collision_resolved_output_is_what_is_probed(env,
                                                                real_media):
    src, out_dir, fake, spy = env
    sentinel = out_dir / "Movie.mkv"
    sentinel.write_bytes(b"the user's own mkv")
    fake.finals = [(1, "Could not write header: Invalid argument")]
    fake.payloads = [MISSING, real_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).name == "Movie_1.mkv"
    assert spy.targets == [out_dir / "Movie_1.mkv"]
    assert sentinel.read_bytes() == b"the user's own mkv"


# ══ GROUP K — readability is the exit code and nothing else ═══════════════

def test_k1_a_probe_that_exits_zero_is_accepted_whatever_it_printed(env):
    """Zero streams, zero duration, a format name nobody recognizes: if
    ffprobe opened the file, the file opened. Every semantic check beyond
    that is a separate policy with its own false rejections, and none of them
    are in this slice."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]
    spy.result = (0, '{"streams": [], "format": {"duration": "0.000000", '
                     '"format_name": "utterly unexpected"}}', "")

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert spy.count == 1


def test_k2_a_probe_that_exits_nonzero_is_rejected_however_quiet(env,
                                                                real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (1, "", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert src.exists()


def test_k3_the_probe_asks_for_no_output_semantics(env, real_media):
    """The command itself is the contract: anything that requests stream,
    format or frame data is a semantic check waiting to grow teeth."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert spy.count == 1
    argv = spy.cmds[0]
    for banned in ("-show_entries", "-show_streams", "-show_format",
                   "-show_packets", "-show_frames", "-count_frames",
                   "-count_packets", "-print_format", "-of"):
        assert banned not in argv, f"{banned} is a semantic check"
    assert argv[-1] == str(result["output"])


def test_k4_the_probe_is_bounded(env, real_media):
    """A validation step that can hang forever is a worse bug than the one it
    fixes, so the launch carries a timeout."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    seen = {}

    real_run = subprocess.run

    def capture(cmd, *a, **kw):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        if argv and str(argv[0]) == str(compressor.FFPROBE_BIN):
            seen["timeout"] = kw.get("timeout")
            spy.cmds.append(argv)
        return real_run(cmd, *a, **kw)
    import unittest.mock as _m
    with _m.patch.object(compressor.subprocess, "run", capture):
        result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert isinstance(seen.get("timeout"), (int, float)) and seen["timeout"] > 0


# ══ GROUP L — a missing ffprobe fails the core closed ═════════════════════

def test_l1_direct_call_without_ffprobe_fails_before_encoding(env, tmp_path,
                                                              monkeypatch):
    """No app.py in sight. The core does not delegate its own safety to the
    GUI that usually calls it."""
    src, out_dir, fake, spy = env
    _no_ffprobe(monkeypatch, tmp_path)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert "ffprobe" in result["msg"].lower()
    assert "output" not in result
    assert src.exists()
    assert fake.attempts == 0, "nothing to encode if nothing can verify it"
    assert _names(out_dir) == [], "no reservation outlives the refusal"


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_l2_no_format_is_exempt(env, tmp_path, monkeypatch, fmt):
    src, out_dir, fake, spy = env
    _no_ffprobe(monkeypatch, tmp_path)

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert fake.attempts == 0


@pytest.mark.parametrize("mode,value", [
    ("Quality preset", "Balanced"),
    ("Target file size", 1),
    ("Target reduction", 50),
])
def test_l3_no_mode_is_exempt(env, tmp_path, monkeypatch, mode, value):
    """Quality preset used to be the mode that could run without ffprobe.
    It is not any more."""
    src, out_dir, fake, spy = env
    _no_ffprobe(monkeypatch, tmp_path)

    result = _run(src, out_dir, mode=mode, mode_value=value)

    assert result["status"] == "error"
    assert "ffprobe" in result["msg"].lower()
    assert fake.attempts == 0


# ══ GROUP M — nothing is remembered between files ═════════════════════════

def test_m_validation_state_does_not_leak_between_files(tmp_path, monkeypatch,
                                                        real_media):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sources = []
    for name in ("A", "B", "C"):
        p = tmp_path / f"{name}.mov"
        p.write_bytes(b"s" * SRC_BYTES)
        sources.append(p)
    fake = FakeFfmpeg(payloads=[real_media[".mp4"], GARBAGE,
                                real_media[".mp4"]])
    spy = _fake_stack(monkeypatch, fake)

    statuses = [_run(s, out_dir)["status"] for s in sources]

    assert statuses == ["ok", "error", "ok"]
    assert spy.count == 3, "one probe per finished artifact, every time"
    assert all(s.exists() for s in sources), \
        "no source is deleted by this path anyway, and B's least of all"


# ══ GROUP N — the result dict grows no probe internals ════════════════════

def test_n1_success_result_shape_is_unchanged(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert set(result) == {"file", "output", "status", "original", "new",
                           "encoder"}


def test_n2_failure_result_shape_is_unchanged(env):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert set(result) == {"file", "status", "msg"}
    for leaked in ("ffprobe_validated", "readability_status", "probe_rc",
                   "probe_stderr", "readable", "mux_failed"):
        assert leaked not in result


# ══ call-count contract ═══════════════════════════════════════════════════

def test_single_pass_success_costs_one_ffmpeg_and_one_ffprobe(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    _run(src, out_dir)

    assert len(fake.all_cmds) == 1
    assert spy.count == 1


def test_the_source_is_never_the_validation_target(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    _run(src, out_dir)

    assert all(p != src for p in spy.targets)
