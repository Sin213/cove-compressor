"""A readable media file is not a successful video conversion without video.

Tab 18 made `ffprobe -v error <output>` the last gate before a conversion is
called a success, and Tab 19 made a definitive refusal remove the artifact Cove
owns. Both read one number: the exit code. That number answers "are these bytes
media?" and nothing else - and for a *video compressor* the two questions come
apart at exactly one place. An audio-only MP4 is media. ffprobe opens it, parses
it, and exits 0. It is also not the thing the user asked Cove to produce, and
every success-only consequence fires behind that 0: the ok result, the sidecars,
and the offer to delete the original that was the only copy of the picture.

So the final probe is asked for one more fact, in the same invocation:

    ffprobe -v error -select_streams v:0 -show_entries stream=index -of csv=p=0

    rc 0, stdout non-empty  -> media, and it has a video stream. Success.
    rc 0, stdout empty      -> media, and no video stream at all. A verdict
                               about the artifact: reject it, keep the source,
                               remove the file Cove owns.
    rc 1                    -> Tab 19, unchanged. Not media.
    anything else, launch
    failure, timeout,
    cancel                  -> the probe failed, not the file. Preserve.

One probe still, not two. The stream index printed on stdout is the whole
addition, and the only rule read off it is "at least one". Not exactly one, not
the same count as the source, not the codec, not the resolution, not the
duration - each of those is a separate policy with its own false rejections,
and a legitimately odd but playable file must not be thrown away here.

    A.  Real audio-only media, every container: rejected, source kept, the
        owned artifact removed.
    B.  Real video: still a success, still one probe.
    C.  The enriched command is the same single invocation.
    D.  Tab 19's rc==1 behaviour is untouched.
    E.  Launch failure, timeout, cancel and abnormal exits still preserve.
    F.  Tab 10 precedence: nothing structural reaches a probe.
    G.  A failed encode is never probed.
    H.  No-video MP4 (H.265) is terminal - it earns no MKV fallback.
    I.  A no-video fallback MKV is cleaned; two attempts remains the ceiling.
    J.  Collision neighbours - shallow, deep and fallback - are untouched.
    K.  Source deletion is impossible behind a no-video output.
    L.  No sidecar is finalized behind a rejected video.
    M.  What Tab 20 explicitly does NOT validate: video count, audio count,
        codec, duration.
    N.  Public result shape unchanged; nothing leaks between files.

Real ffprobe, real media, real `reserve_output`: the encoder is faked and
writes the bytes it was scripted with - exactly as the Tab 10, Tab 18 and
Tab 19 suites do - but the readability gate, the reservation and the cleanup
are all the production ones. The audio-only artifacts are genuine ffmpeg
output; only the encoder's *success* is controlled.
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

# Non-empty and structurally fine by every Tab 10 criterion, and not media by
# any of them: ffprobe runs to completion and exits 1.
GARBAGE = bytes(range(256)) * 4

MISSING = None
EMPTY = b""

# The user's own bytes. Any test that finds these changed has watched Cove
# delete a file it does not own.
SENTINEL = b"the user's own file - not Cove's to touch"

FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"]
EXT_OF = {"MP4 (H.264)": ".mp4", "MP4 (H.265)": ".mp4",
          "MKV (H.265)": ".mkv", "WebM (VP9)": ".webm"}

_VIDEO_RECIPES = {
    ".mp4": ["-c:v", "libx264", "-c:a", "aac"],
    ".mkv": ["-c:v", "libx265", "-c:a", "aac"],
    ".webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
}

# The gap, as real files. Genuine ffmpeg output, genuine containers, genuine
# audio - and no video stream anywhere in them.
_AUDIO_ONLY_RECIPES = {
    ".mp4": ["-c:a", "aac"],
    ".mkv": ["-c:a", "aac"],
    ".webm": ["-c:a", "libopus"],
}


def _ffmpeg(args):
    return subprocess.run([compressor.FFMPEG_BIN, "-v", "error", "-y", *args],
                          capture_output=True, text=True, timeout=180,
                          env=compressor.clean_subprocess_env(),
                          **compressor.SUBPROCESS_FLAGS)


def _ffprobe(args):
    return subprocess.run([compressor.FFPROBE_BIN, *args],
                          capture_output=True, text=True, timeout=120,
                          env=compressor.clean_subprocess_env(),
                          **compressor.SUBPROCESS_FLAGS)


def _video_stream_count(path: Path) -> int:
    r = _ffprobe(["-v", "error", "-select_streams", "v",
                  "-show_entries", "stream=index", "-of", "csv=p=0",
                  str(path)])
    assert r.returncode == 0, f"{path} is not readable media: {r.stderr[-300:]}"
    return len([ln for ln in r.stdout.splitlines() if ln.strip()])


@pytest.fixture(scope="session")
def real_media(tmp_path_factory) -> dict:
    """Half a second of genuine video per container, built once."""
    d = tmp_path_factory.mktemp("presence_video_media")
    out = {}
    for ext, codecs in _VIDEO_RECIPES.items():
        path = d / f"sample{ext}"
        r = _ffmpeg(["-f", "lavfi",
                     "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                     "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                     *codecs, "-shortest", str(path)])
        assert r.returncode == 0 and path.stat().st_size > 0, \
            f"could not build a readable {ext} sample: {r.stderr[-400:]}"
        assert _video_stream_count(path) >= 1
        out[ext] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def audio_only_media(tmp_path_factory) -> dict:
    """Half a second of genuine *audio-only* media per container.

    Built by the real encoder and asserted at build time to be exactly what
    the gap needs: ffprobe opens them (rc 0) and finds no video stream at all.
    """
    d = tmp_path_factory.mktemp("presence_audio_only_media")
    out = {}
    for ext, codecs in _AUDIO_ONLY_RECIPES.items():
        path = d / f"audio_only{ext}"
        r = _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                     *codecs, str(path)])
        assert r.returncode == 0 and path.stat().st_size > 0, \
            f"could not build an audio-only {ext} sample: {r.stderr[-400:]}"
        # The two facts that make this file the gap and not merely a fixture.
        plain = _ffprobe(["-v", "error", str(path)])
        assert plain.returncode == 0, \
            f"audio-only {ext} must be readable media, got {plain.returncode}"
        assert _video_stream_count(path) == 0, \
            f"audio-only {ext} must carry no video stream"
        out[ext] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def two_video_media(tmp_path_factory) -> bytes:
    """An MKV carrying two video streams.

    Tab 20 asks for `>= 1`, deliberately not `== 1`: stream-count parity is a
    separate policy and must not arrive here by accident.
    """
    d = tmp_path_factory.mktemp("presence_two_video_media")
    path = d / "two_video.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-f", "lavfi",
                 "-i", "testsrc2=size=160x120:rate=10:duration=0.5",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                 "-map", "0:v", "-map", "1:v", "-map", "2:a",
                 "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)])
    assert r.returncode == 0, f"could not build a 2-video sample: {r.stderr[-400:]}"
    assert _video_stream_count(path) == 2
    return path.read_bytes()


@pytest.fixture(scope="session")
def odd_codec_media(tmp_path_factory) -> bytes:
    """Video in a codec Cove never produces, in a container it does.

    Tab 20 proves a video stream exists and stops. Codec, profile and pixel
    format are explicitly not its business.
    """
    d = tmp_path_factory.mktemp("presence_odd_codec_media")
    path = d / "odd.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-c:v", "mpeg4", "-an", str(path)])
    assert r.returncode == 0, f"could not build an mpeg4 sample: {r.stderr[-400:]}"
    assert _video_stream_count(path) == 1
    return path.read_bytes()


@pytest.fixture(scope="session")
def single_frame_media(tmp_path_factory) -> bytes:
    """One frame, no audio: a duration a duration-policy would hate.

    Duration and truncation detection are a separate slice. A video stream is
    present, so Tab 20 is satisfied and says nothing about how long it runs.
    """
    d = tmp_path_factory.mktemp("presence_single_frame_media")
    path = d / "oneframe.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-frames:v", "1", "-c:v", "libx264", "-an", str(path)])
    assert r.returncode == 0, f"could not build a 1-frame sample: {r.stderr[-400:]}"
    assert _video_stream_count(path) == 1
    return path.read_bytes()


# ── fakes ────────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _sub(index: int, codec: str) -> dict:
    return {"index": index, "codec_name": codec, "tags": {"language": "eng"}}


class FakeFfmpeg:
    """`run_ffmpeg` stand-in: scripts each invocation's exit code and the bytes
    it leaves behind. Same shape as the Tab 10, Tab 18 and Tab 19 fakes."""

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
    """Fake the encoder and the *source* probes; leave the final validation
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


# ══ GROUP A — real readable audio-only output is not a video conversion ═════

@pytest.mark.parametrize("fmt", FORMATS)
def test_a1_readable_audio_only_output_fails(env, audio_only_media, fmt):
    """The dominant gap. CONTROLLED ENCODER SUCCESS + REAL AUDIO-ONLY MEDIA
    ARTIFACT: ffmpeg is scripted to exit 0 and the bytes it leaves are genuine
    ffmpeg-built media that ffprobe opens without complaint - and that contain
    no picture. Tab 18 sees rc 0 and calls it a conversion. It is not one."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error", \
        "a media file with no video stream is not a compressed video"
    assert "output" not in result, "a rejected artifact is never handed over"
    assert src.exists(), "the original is still the only copy of the picture"


@pytest.mark.parametrize("fmt", FORMATS)
def test_a2_readable_audio_only_artifact_is_removed(env, audio_only_media, fmt):
    """Readable-but-no-video is a verdict about the artifact, not about the
    probe, so Tab 19's ownership-safe cleanup applies exactly as it does to a
    definitive refusal."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[EXT_OF[fmt]]]

    _run(src, out_dir, fmt=fmt)

    assert _video_names(out_dir) == [], \
        "an audio-only file named like a finished video is worse than none"


def test_a3_the_removed_path_is_exactly_the_probed_path(env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    _run(src, out_dir)

    assert spy.targets == [out_dir / "Movie.mp4"]
    assert not (out_dir / "Movie.mp4").exists()


def test_a4_audio_only_rejection_costs_exactly_one_probe(env, audio_only_media):
    """Tab 20 adds a fact to the existing probe, not a probe."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    _run(src, out_dir)

    assert spy.count == 1
    assert fake.attempts == 1


def test_a5_two_pass_audio_only_rejection_needs_no_third_encode(
        env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 2, "pass 1 and pass 2, and nothing more"
    assert spy.count == 1, "pass 1 is never probed"
    assert _video_names(out_dir) == []
    assert src.exists()


def test_a6_rejection_by_a_scripted_empty_video_list_also_cleans(
        env, real_media):
    """The bytes are genuinely video; the probe is scripted to report no video
    stream anyway. The decision follows the probe's answer, not the payload."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()
    assert src.exists()


def test_a7_whitespace_only_probe_output_is_not_a_video_stream(env, real_media):
    """A trailing newline is what an *empty* result looks like on stdout. Only
    a printed stream index counts as evidence that a picture exists."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "\n  \r\n", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()


# ══ GROUP B — real video still converts ════════════════════════════════════

@pytest.mark.parametrize("fmt", FORMATS)
def test_b1_readable_video_output_still_succeeds(env, real_media, fmt):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.exists() and out.read_bytes() == real_media[EXT_OF[fmt]]
    assert spy.count == 1, "one probe proves both facts"


def test_b2_two_pass_video_output_still_succeeds(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _two_pass(src, out_dir)

    assert result["status"] == "ok"
    assert len(fake.all_cmds) == 2, "two encodes"
    assert spy.count == 1, "one final probe"


def test_b3_success_keeps_its_sidecar_and_result_shape(env, real_media,
                                                       monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]
    assert set(result) >= {"file", "output", "status", "original", "new",
                           "encoder"}


# ══ GROUP C — the enriched command is still one command ════════════════════

def test_c1_the_final_probe_asks_for_a_video_stream_index(env, real_media):
    """The whole of Tab 20's addition, on the wire."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    _run(src, out_dir)

    assert spy.count == 1
    argv = [str(a) for a in spy.cmds[0]]
    assert argv[0] == str(compressor.FFPROBE_BIN)
    assert argv[-1] == str(out_dir / "Movie.mp4")
    assert "-select_streams" in argv
    assert argv[argv.index("-select_streams") + 1].startswith("v")
    assert "-show_entries" in argv
    assert "-v" in argv and argv[argv.index("-v") + 1] == "error"


def test_c2_no_extra_subprocess_is_spawned_on_success(env, real_media,
                                                      monkeypatch):
    """Every subprocess of a successful single-pass job: one faked encode and
    one real probe. Nothing else may appear."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert fake.attempts == 1
    assert spy.count == 1


def test_c3_no_extra_subprocess_is_spawned_on_rejection(env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    _run(src, out_dir)

    assert fake.attempts == 1
    assert spy.count == 1


def test_c4_the_probe_targets_the_output_not_the_source(env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    _run(src, out_dir)

    assert spy.targets == [out_dir / "Movie.mp4"]
    assert src not in spy.targets


# ══ GROUP D — Tab 19's definitive refusal is unchanged ═════════════════════

@pytest.mark.parametrize("fmt", FORMATS)
def test_d1_non_media_is_still_refused_and_removed(env, fmt):
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error"
    assert spy.count == 1
    assert _video_names(out_dir) == []
    assert src.exists()


def test_d2_a_scripted_refusal_still_cleans(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (1, "", "moov atom not found")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()


def test_d3_refusal_stdout_is_never_read_as_evidence(env, real_media):
    """A refusing probe that printed something anyway is still a refusal: the
    exit code decides whether the file is media at all, and stdout is only
    consulted once it is."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (1, "0\n", "moov atom not found")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()


# ══ GROUP E — an inconclusive probe still preserves the artifact ═══════════

@pytest.mark.parametrize("boom", [
    OSError(13, "permission denied"),
    OSError(2, "no such file or directory"),
])
def test_e1_launch_failure_preserves_the_artifact(env, real_media, boom):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = boom

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"]
    assert src.exists()
    assert "output" not in result


def test_e2_timeout_preserves_the_artifact(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"]
    assert src.exists()


def test_e3_timeout_on_an_audio_only_artifact_still_preserves(
        env, audio_only_media):
    """No verdict was reached, so nothing about the bytes was established -
    not even that they carry no video. Preserve."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]
    spy.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == audio_only_media[".mp4"]
    assert src.exists()


def test_e4_cancel_preserves_the_artifact(env, audio_only_media):
    src, out_dir, fake, spy = env
    flag = threading.Event()
    fake.cancel_after_final = flag
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir, cancel_flag=flag)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"
    assert (out_dir / "Movie.mp4").exists(), \
        "an interrupted job's output is not Cove's to throw away"
    assert src.exists()


@pytest.mark.parametrize("rc", [3, 2, 255, -9, 3221225477])
def test_e5_abnormal_exit_preserves_the_artifact(env, audio_only_media, rc):
    """A crash is a number too. Anything that is neither 0 nor 1 is the probe
    dying rather than answering, and says nothing about the file."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]
    spy.result = (rc, "", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").exists()
    assert src.exists()


# ══ GROUP F — Tab 10 precedence: structural failures never reach a probe ═══

@pytest.mark.parametrize("payload", [MISSING, EMPTY])
def test_f1_structural_failure_is_never_probed(env, payload):
    src, out_dir, fake, spy = env
    fake.payloads = [payload]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0, "nothing structural earns a probe"
    assert src.exists()


def test_f2_zero_byte_reservation_cleanup_is_untouched(env):
    src, out_dir, fake, spy = env
    fake.payloads = [EMPTY]

    _run(src, out_dir)

    assert spy.count == 0
    assert _video_names(out_dir) == []


# ══ GROUP G — a failed encode is never probed ══════════════════════════════

def test_g1_failed_encode_is_never_probed(env):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "Error while opening encoder")]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0
    assert src.exists()


def test_g2_failed_pass_one_is_never_probed(env):
    src, out_dir, fake, spy = env
    fake.pass1 = [(1, "pass 1 failed")]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 0


# ══ GROUP H — no-video MP4 (H.265) is terminal ═════════════════════════════

def test_h1_audio_only_h265_mp4_earns_no_mkv_fallback(env, audio_only_media):
    """The MP4 -> MKV retry exists for a mux failure. An apparent success whose
    bytes carry no picture is a different anomaly and stays terminal."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert fake.attempts == 1, "no second encode"
    assert "mp4" in fake.final_muxers[0]
    assert spy.count == 1
    assert _video_names(out_dir) == []
    assert src.exists()


def test_h2_audio_only_h264_mp4_earns_no_fallback_either(env,
                                                         audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    assert result["status"] == "error"
    assert fake.attempts == 1


# ══ GROUP I — a no-video fallback MKV is cleaned, and two is the ceiling ═══

def test_i1_fallback_mkv_with_no_video_fails_after_two_attempts(
        env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, audio_only_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert fake.attempts == 2, "two encode attempts is the ceiling"
    assert spy.count == 1, "only the successful final artifact is probed"
    assert spy.targets == [out_dir / "Movie.mkv"]
    assert _video_names(out_dir) == []
    assert src.exists()


def test_i2_fallback_mkv_with_video_still_succeeds(env, real_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, real_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).name == "Movie.mkv"
    assert fake.attempts == 2
    assert spy.count == 1


# ══ GROUP J — only the exact owned path is eligible ════════════════════════

def test_j1_collision_neighbour_survives_an_audio_only_rejection(
        env, audio_only_media):
    src, out_dir, fake, spy = env
    neighbour = out_dir / "Movie.mp4"
    neighbour.write_bytes(SENTINEL)
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert neighbour.read_bytes() == SENTINEL, \
        "the user's own file is not Cove's to touch"
    assert not (out_dir / "Movie_1.mp4").exists()
    assert spy.targets == [out_dir / "Movie_1.mp4"]
    assert src.exists()


def test_j2_deep_collision_neighbours_all_survive(env, audio_only_media):
    src, out_dir, fake, spy = env
    kept = {}
    for name in ("Movie.mp4", "Movie_1.mp4", "Movie_2.mp4", "Movie_3.mp4"):
        p = out_dir / name
        p.write_bytes(SENTINEL + name.encode())
        kept[name] = p.read_bytes()
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    for name, data in kept.items():
        assert (out_dir / name).read_bytes() == data
    assert not (out_dir / "Movie_4.mp4").exists()
    assert spy.targets == [out_dir / "Movie_4.mp4"]


def test_j3_fallback_collision_neighbour_survives(env, audio_only_media):
    src, out_dir, fake, spy = env
    neighbour = out_dir / "Movie.mkv"
    neighbour.write_bytes(SENTINEL)
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, audio_only_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert neighbour.read_bytes() == SENTINEL
    assert not (out_dir / "Movie_1.mkv").exists()
    assert spy.targets == [out_dir / "Movie_1.mkv"]


# ══ GROUP K — the source survives ══════════════════════════════════════════

@pytest.mark.parametrize("fmt", FORMATS)
def test_k1_source_deletion_is_blocked_behind_a_no_video_output(
        env, audio_only_media, fmt):
    """The whole point. Deletion is opt-in and asked for here; a conversion
    that produced no picture must not be allowed to consume the only copy."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[EXT_OF[fmt]]]

    result = delete_source_if_eligible(_run(src, out_dir, fmt=fmt),
                                       enabled=True)

    assert result["status"] == "error"
    assert src.exists(), "the source outlives a videoless conversion"
    assert result.get("source_deleted") is not True


def test_k2_source_survives_an_inconclusive_probe(env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]
    spy.raises = OSError(13, "permission denied")

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_k3_source_deletion_still_works_behind_real_video(env, real_media):
    """Tab 20 blocks one case and changes nothing else."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] == "ok"
    assert result["source_deleted"] is True
    assert not src.exists()
    assert Path(result["output"]).exists()


# ══ GROUP L — no sidecar behind a rejected video ═══════════════════════════

def test_l1_no_sidecar_is_finalized_behind_an_audio_only_output(
        env, audio_only_media, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert _names(out_dir, "*.srt") == [], \
        "a subtitle track beside no video is not a delivered sidecar"
    assert "subtitles" not in result


# ══ GROUP M — what Tab 20 explicitly does NOT validate ═════════════════════

def test_m1_two_video_streams_satisfy_the_gate(env, two_video_media):
    """`>= 1`, deliberately not `== 1`. Stream-count parity is a separate
    policy and must not arrive here by accident."""
    src, out_dir, fake, spy = env
    fake.payloads = [two_video_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == two_video_media


def test_m2_a_scripted_multi_index_result_satisfies_the_gate(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mkv"]]
    spy.result = (0, "0\n1\n", "")

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"


def test_m3_an_unusual_codec_satisfies_the_gate(env, odd_codec_media):
    """mpeg4 video Cove would never emit, and no audio track at all. A video
    stream exists; codec and audio count are not Tab 20's business."""
    src, out_dir, fake, spy = env
    fake.payloads = [odd_codec_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == odd_codec_media


def test_m4_a_single_frame_duration_satisfies_the_gate(env,
                                                       single_frame_media):
    """Duration and truncation detection remain a separate slice."""
    src, out_dir, fake, spy = env
    fake.payloads = [single_frame_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"


def test_m5_a_nonzero_stream_index_still_counts(env, real_media):
    """A video stream that is not stream 0 is still a video stream."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mkv"]]
    spy.result = (0, "3\n", "")

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"


# ══ GROUP N — result shape and isolation ═══════════════════════════════════

def test_n1_the_rejection_result_shape_is_the_ordinary_error_shape(
        env, audio_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media[".mp4"]]

    result = _run(src, out_dir)

    assert set(result) == {"file", "status", "msg"}
    assert result["file"] == src
    assert result["status"] == "error"
    assert isinstance(result["msg"], str) and result["msg"]
    assert "mux_failed" not in result, "a videoless output earns no retry"


def test_n2_the_success_result_shape_is_unchanged(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert set(result) == {"file", "output", "status", "original", "new",
                           "encoder"}


def test_n3_nothing_leaks_between_files(env, real_media, audio_only_media,
                                        tmp_path):
    """Three jobs through one process: reject, succeed, reject. No validation
    state may survive a file boundary in either direction."""
    src, out_dir, fake, spy = env
    second = tmp_path / "Second.mov"
    second.write_bytes(b"s" * SRC_BYTES)
    third = tmp_path / "Third.mov"
    third.write_bytes(b"s" * SRC_BYTES)
    fake.payloads = [audio_only_media[".mp4"], real_media[".mp4"],
                     audio_only_media[".mp4"]]

    first_r = _run(src, out_dir)
    second_r = _run(second, out_dir)
    third_r = _run(third, out_dir)

    assert first_r["status"] == "error"
    assert second_r["status"] == "ok"
    assert third_r["status"] == "error"
    assert _video_names(out_dir) == ["Second.mp4"]
    assert spy.count == 3
    assert src.exists() and second.exists() and third.exists()


def test_n4_a_success_after_a_rejection_still_probes_its_own_path(
        env, real_media, audio_only_media, tmp_path):
    src, out_dir, fake, spy = env
    second = tmp_path / "Second.mov"
    second.write_bytes(b"s" * SRC_BYTES)
    fake.payloads = [audio_only_media[".mp4"], real_media[".mp4"]]

    _run(src, out_dir)
    _run(second, out_dir)

    assert spy.targets == [out_dir / "Movie.mp4", out_dir / "Second.mp4"]
