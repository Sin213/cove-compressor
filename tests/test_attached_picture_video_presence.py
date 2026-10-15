"""Attached cover art is not a successful video conversion.

Tab 20 made the final probe ask one extra question - is there a video stream? -
and rejected a readable output that had none. It read the answer off a printed
stream index, and an index is printed for *every* video stream, including the
one a container carries purely to show a thumbnail. So an audio track with a
picture glued to it walks the gate:

    ffprobe -v error -select_streams v:0 -show_entries stream=index -of csv=p=0
        audio + attached cover art  ->  rc 0, stdout "1"   ->  accepted

That file is media, it has a codec_type=video stream, and it is not a video.
Behind that acceptance sit the ok result, the sidecars and the offer to delete
the original that was the only copy of the moving picture.

What tells the two apart is one field ffprobe already knows and the old probe
simply did not ask for, `disposition.attached_pic`, so the same single
invocation is re-pointed at it:

    ffprobe -v error -select_streams v
            -show_entries stream_disposition=attached_pic -of csv=p=1

    one line per video stream, `stream,0` or `stream,1`.

    any `0`                 -> a non-attached-picture video stream exists.
                               Success, exactly as before.
    lines, all `1`          -> every picture in the file is artwork. A verdict
                               about the artifact: reject it, keep the source,
                               remove the file Cove owns.
    no lines at all         -> Tab 20's zero-video case, unchanged.
    a line that is neither  -> the payload is not the contract. The probe told
                               us nothing usable, so this is inconclusive like
                               a timeout is: fail closed, preserve the bytes.
    rc 1 / launch / timeout
    / cancel / abnormal     -> Tab 19, unchanged.

One probe still, and no new subprocess: `attached_pic` is the only new
discriminator. Codec, duration, resolution, video/audio/subtitle counts and
every other disposition flag remain out of scope, and nothing strips artwork
from an output that also carries real picture - a video *with* a thumbnail is
a perfectly good video and must keep succeeding.

Real ffprobe representation, measured on the installed FFmpeg 8.1.1 build and
pinned by GROUP G:

    MP4   audio + `-disposition:v:0 attached_pic`  -> video, attached_pic 1
    MP4   video + artwork                          -> attached_pic 0 and 1
    MKV   `-attach cover.jpg`                      -> video, attached_pic 1
    MKV   video track + `-disposition attached_pic`-> attached_pic 0; Matroska
          does not carry that disposition on an ordinary video track, so such a
          file is indistinguishable from real video and is *not* rejected. That
          is the honest limit of this slice, not a hidden failure.
    WebM  no cover-art fixture is constructible: the muxer admits only VP8/VP9/
          AV1 video, so mjpeg artwork cannot be written at all. Ordinary WebM
          must keep succeeding, and that is all WebM can be asked here.

    A.  Playable-video presence: the rule and its four corners.
    B.  Probe failures: rc 1, launch, timeout, cancel, malformed payload.
    C.  Containers, including the MKV and WebM evidence above.
    D.  Two-pass and the MP4 -> MKV fallback ceiling.
    E.  Source deletion and sidecars stay behind the gate.
    F.  Collision neighbours are untouched.
    G.  The real ffprobe contract and the payload vocabulary.
    H.  What this slice still does not validate.

Real ffprobe, real media, real `reserve_output`: the encoder is faked and
writes the bytes it was scripted with - exactly as the Tab 10, Tab 18, Tab 19
and Tab 20 suites do - but the readability gate, the reservation and the
cleanup are all the production ones. The cover-art artifacts are genuine ffmpeg
output; only the encoder's *success* is controlled.
"""
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

GARBAGE = bytes(range(256)) * 4

MISSING = None

# The user's own bytes. Any test that finds these changed has watched Cove
# delete a file it does not own.
SENTINEL = b"the user's own file - not Cove's to touch"

FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)", "WebM (VP9)"]
EXT_OF = {"MP4 (H.264)": ".mp4", "MP4 (H.265)": ".mp4",
          "MKV (H.265)": ".mkv", "WebM (VP9)": ".webm"}

# Containers in which real cover art is constructible on this FFmpeg build.
# WebM is absent by measurement, not by choice - see GROUP C.
COVER_ART_EXTS = [".mp4", ".mkv"]
COVER_ART_FORMATS = ["MP4 (H.264)", "MP4 (H.265)", "MKV (H.265)"]

_VIDEO_RECIPES = {
    ".mp4": ["-c:v", "libx264", "-c:a", "aac"],
    ".mkv": ["-c:v", "libx265", "-c:a", "aac"],
    ".webm": ["-c:v", "libvpx-vp9", "-c:a", "libopus"],
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


def _dispositions(path: Path) -> list[str]:
    """`attached_pic` for every video stream, in order, straight off ffprobe.

    The fixtures assert against this rather than against Cove, so a build whose
    ffmpeg stops producing the file the gap needs fails loudly at construction
    instead of quietly passing every test.
    """
    r = _ffprobe(["-v", "error", "-select_streams", "v",
                  "-show_entries", "stream_disposition=attached_pic",
                  "-of", "csv=p=0", str(path)])
    assert r.returncode == 0, f"{path} is not readable media: {r.stderr[-300:]}"
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


@pytest.fixture(scope="session")
def cover_jpeg(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("attached_pic_cover")
    path = d / "cover.jpg"
    r = _ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=160x120:d=1",
                 "-frames:v", "1", str(path)])
    assert r.returncode == 0 and path.stat().st_size > 0, \
        f"could not build cover art: {r.stderr[-400:]}"
    return path


@pytest.fixture(scope="session")
def real_media(tmp_path_factory) -> dict:
    """Half a second of genuine video per container, built once."""
    d = tmp_path_factory.mktemp("attached_pic_video_media")
    out = {}
    for ext, codecs in _VIDEO_RECIPES.items():
        path = d / f"sample{ext}"
        r = _ffmpeg(["-f", "lavfi",
                     "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                     "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                     *codecs, "-shortest", str(path)])
        assert r.returncode == 0 and path.stat().st_size > 0, \
            f"could not build a readable {ext} sample: {r.stderr[-400:]}"
        assert _dispositions(path) == ["0"], \
            f"a plain {ext} video must be one non-attached-picture stream"
        out[ext] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def cover_art_only_media(tmp_path_factory, cover_jpeg) -> dict:
    """The gap, as real files: audio plus artwork, and no moving picture.

    MP4 carries the artwork as a video stream flagged `attached_pic`; Matroska
    will not take that flag on an ordinary track, so the MKV fixture uses the
    mechanism Matroska actually has - a file attachment - which ffprobe reports
    as a video stream with `attached_pic=1`. Both are asserted at build time to
    be exactly what the gap needs: readable media, a codec_type=video stream
    present, and every one of those streams artwork.
    """
    d = tmp_path_factory.mktemp("attached_pic_cover_only")
    audio = d / "audio.m4a"
    r = _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                 "-c:a", "aac", str(audio)])
    assert r.returncode == 0, f"could not build audio: {r.stderr[-400:]}"

    out = {}
    recipes = {
        ".mp4": ["-i", str(audio), "-i", str(cover_jpeg),
                 "-map", "0:a", "-map", "1:v", "-c:a", "copy",
                 "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"],
        ".mkv": ["-i", str(audio), "-c:a", "copy",
                 "-attach", str(cover_jpeg),
                 "-metadata:s:t", "mimetype=image/jpeg"],
    }
    for ext, args in recipes.items():
        path = d / f"cover_only{ext}"
        r = _ffmpeg([*args, str(path)])
        assert r.returncode == 0 and path.stat().st_size > 0, \
            f"could not build a cover-art-only {ext}: {r.stderr[-400:]}"
        plain = _ffprobe(["-v", "error", str(path)])
        assert plain.returncode == 0, \
            f"cover-art-only {ext} must be readable media, got {plain.returncode}"
        # The three facts that make this file the gap and not merely a fixture:
        # a video stream exists, so Tab 20 accepts it, and every one of them is
        # artwork, so Tab 21 must not.
        assert _dispositions(path) == ["1"], \
            f"cover-art-only {ext} must be exactly one attached picture"
        out[ext] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def video_plus_cover_media(tmp_path_factory, cover_jpeg, real_media) -> dict:
    """Real video *and* artwork - the file an over-broad rejection would eat."""
    d = tmp_path_factory.mktemp("attached_pic_video_plus_cover")
    out = {}
    for ext in COVER_ART_EXTS:
        base = d / f"base{ext}"
        base.write_bytes(real_media[ext])
        path = d / f"video_plus_cover{ext}"
        if ext == ".mp4":
            args = ["-i", str(base), "-i", str(cover_jpeg),
                    "-map", "0:v", "-map", "0:a", "-map", "1:v", "-c", "copy",
                    "-disposition:v:1", "attached_pic"]
        else:
            args = ["-i", str(base), "-c", "copy", "-attach", str(cover_jpeg),
                    "-metadata:s:t", "mimetype=image/jpeg"]
        r = _ffmpeg([*args, str(path)])
        assert r.returncode == 0 and path.stat().st_size > 0, \
            f"could not build a video+cover {ext}: {r.stderr[-400:]}"
        assert sorted(_dispositions(path)) == ["0", "1"], \
            f"video+cover {ext} must carry one real picture and one artwork"
        out[ext] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def multi_video_plus_cover_media(tmp_path_factory, cover_jpeg) -> bytes:
    """Two real video streams and artwork.

    `>= 1` playable, deliberately not `== 1`: stream-count parity is a separate
    policy and must not arrive here by accident.
    """
    d = tmp_path_factory.mktemp("attached_pic_multi_video")
    path = d / "multi_plus_cover.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-f", "lavfi",
                 "-i", "testsrc2=size=160x120:rate=10:duration=0.5",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                 "-map", "0:v", "-map", "1:v", "-map", "2:a",
                 "-c:v", "libx264", "-c:a", "aac", "-shortest",
                 "-attach", str(cover_jpeg),
                 "-metadata:s:t", "mimetype=image/jpeg", str(path)])
    assert r.returncode == 0, f"could not build a 2-video+cover: {r.stderr[-400:]}"
    assert _dispositions(path) == ["0", "0", "1"]
    return path.read_bytes()


@pytest.fixture(scope="session")
def audio_only_media(tmp_path_factory) -> bytes:
    """Tab 20's case: media, and no video stream of any kind."""
    d = tmp_path_factory.mktemp("attached_pic_audio_only")
    path = d / "audio_only.mp4"
    r = _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                 "-c:a", "aac", str(path)])
    assert r.returncode == 0, f"could not build audio-only: {r.stderr[-400:]}"
    assert _dispositions(path) == []
    return path.read_bytes()


@pytest.fixture(scope="session")
def odd_codec_media(tmp_path_factory) -> bytes:
    """Video in a codec Cove never produces, and no artwork anywhere."""
    d = tmp_path_factory.mktemp("attached_pic_odd_codec")
    path = d / "odd.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-c:v", "mpeg4", "-an", str(path)])
    assert r.returncode == 0, f"could not build an mpeg4 sample: {r.stderr[-400:]}"
    assert _dispositions(path) == ["0"]
    return path.read_bytes()


@pytest.fixture(scope="session")
def single_frame_media(tmp_path_factory) -> bytes:
    """One frame of real video: a duration a duration-policy would hate, and a
    file a naive "one frame means thumbnail" rule would wrongly eat."""
    d = tmp_path_factory.mktemp("attached_pic_single_frame")
    path = d / "oneframe.mkv"
    r = _ffmpeg(["-f", "lavfi",
                 "-i", "testsrc=size=160x120:rate=10:duration=0.5",
                 "-frames:v", "1", "-c:v", "libx264", "-an", str(path)])
    assert r.returncode == 0, f"could not build a 1-frame sample: {r.stderr[-400:]}"
    assert _dispositions(path) == ["0"]
    return path.read_bytes()


# ── fakes ────────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _sub(index: int, codec: str) -> dict:
    return {"index": index, "codec_name": codec, "tags": {"language": "eng"}}


class FakeFfmpeg:
    """`run_ffmpeg` stand-in: scripts each invocation's exit code and the bytes
    it leaves behind. Same shape as the Tab 10, Tab 18, Tab 19 and Tab 20
    fakes."""

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


# ══ GROUP A — a playable video stream, not merely a video stream ════════════

@pytest.mark.parametrize("fmt", FORMATS)
def test_a1_real_video_output_still_succeeds(env, real_media, fmt):
    """The neighbour that must not move. Nothing about artwork may cost an
    ordinary conversion its success."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.exists() and out.read_bytes() == real_media[EXT_OF[fmt]]
    assert spy.count == 1, "one probe proves both facts"


@pytest.mark.parametrize("fmt", COVER_ART_FORMATS)
def test_a2_cover_art_only_output_fails(env, cover_art_only_media, fmt):
    """The dominant gap. CONTROLLED ENCODER SUCCESS + REAL COVER-ART MEDIA
    ARTIFACT: ffmpeg is scripted to exit 0 and the bytes it leaves are genuine
    ffmpeg-built media that ffprobe opens without complaint, containing a
    codec_type=video stream that is nothing but a thumbnail. Tab 20 sees a
    printed index and calls it a conversion. It is not one."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[EXT_OF[fmt]]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "error", \
        "artwork glued to an audio track is not a compressed video"
    assert "output" not in result, "a rejected artifact is never handed over"
    assert src.exists(), "the original is still the only copy of the picture"


@pytest.mark.parametrize("fmt", COVER_ART_FORMATS)
def test_a3_cover_art_only_artifact_is_removed(env, cover_art_only_media, fmt):
    """A verdict about the artifact, not about the probe, so Tab 19's
    ownership-safe cleanup applies exactly as it does to a definitive
    refusal."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[EXT_OF[fmt]]]

    _run(src, out_dir, fmt=fmt)

    assert _video_names(out_dir) == [], \
        "an audio file with a thumbnail named like a video is worse than none"


def test_a4_the_removed_path_is_exactly_the_probed_path(env,
                                                        cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]

    _run(src, out_dir)

    assert spy.targets == [out_dir / "Movie.mp4"]
    assert not (out_dir / "Movie.mp4").exists()


@pytest.mark.parametrize("ext,fmt", [(".mp4", "MP4 (H.264)"),
                                     (".mkv", "MKV (H.265)")])
def test_a5_real_video_with_cover_art_still_succeeds(env, video_plus_cover_media,
                                                     ext, fmt):
    """Load-bearing. The rule is "at least one non-attached-picture video",
    never "no attached picture anywhere" - a video that also carries a
    thumbnail is an ordinary, good output and nothing here may eat it."""
    src, out_dir, fake, spy = env
    fake.payloads = [video_plus_cover_media[ext]]

    result = _run(src, out_dir, fmt=fmt)

    assert result["status"] == "ok"
    out = Path(result["output"])
    assert out.exists() and out.read_bytes() == video_plus_cover_media[ext]


def test_a6_multiple_real_video_streams_with_cover_art_succeed(
        env, multi_video_plus_cover_media):
    src, out_dir, fake, spy = env
    fake.payloads = [multi_video_plus_cover_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == multi_video_plus_cover_media


def test_a7_zero_video_rejection_is_unchanged(env, audio_only_media):
    """Tab 20's case survives intact: no video stream at all is still a
    definitive rejection, and still removes the owned artifact."""
    src, out_dir, fake, spy = env
    fake.payloads = [audio_only_media]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert _video_names(out_dir) == []
    assert src.exists()
    assert spy.count == 1


def test_a8_cover_art_rejection_costs_exactly_one_probe(env,
                                                        cover_art_only_media):
    """This slice adds a fact to the existing probe, not a probe."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]

    _run(src, out_dir)

    assert spy.count == 1
    assert fake.attempts == 1


def test_a9_success_costs_exactly_one_probe(env, video_plus_cover_media):
    src, out_dir, fake, spy = env
    fake.payloads = [video_plus_cover_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert spy.count == 1
    assert fake.attempts == 1


# ══ GROUP B — the probe failing is not the file failing ════════════════════

def test_b1_non_media_is_still_refused_and_removed(env):
    """Tab 19's rc==1 path, untouched."""
    src, out_dir, fake, spy = env
    fake.payloads = [GARBAGE]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert spy.count == 1
    assert _video_names(out_dir) == []
    assert src.exists()


def test_b2_scripted_refusal_of_cover_art_bytes_still_cleans(
        env, cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    spy.result = (1, "stream,0\n", "moov atom not found")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists(), \
        "rc 1 is a refusal whatever the probe managed to print"


@pytest.mark.parametrize("boom", [
    OSError(13, "permission denied"),
    OSError(2, "no such file or directory"),
])
def test_b3_launch_failure_preserves_the_artifact(env, cover_art_only_media,
                                                  boom):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    spy.raises = boom

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == cover_art_only_media[".mp4"]
    assert src.exists()
    assert "output" not in result


def test_b4_timeout_preserves_the_artifact(env, cover_art_only_media):
    """No verdict was reached, so nothing about the bytes was established - not
    even that their only picture is artwork. Preserve."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    spy.raises = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == cover_art_only_media[".mp4"]
    assert src.exists()


def test_b5_cancel_preserves_the_artifact(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    flag = threading.Event()
    fake.cancel_after_final = flag
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir, cancel_flag=flag)

    assert result["status"] == "error"
    assert result["msg"] == "cancelled"
    assert (out_dir / "Movie.mp4").exists(), \
        "an interrupted job's output is not Cove's to throw away"
    assert src.exists()


@pytest.mark.parametrize("rc", [3, 2, 255, -9, 3221225477])
def test_b6_abnormal_exit_preserves_the_artifact(env, cover_art_only_media, rc):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    spy.result = (rc, "stream,1\n", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").exists()
    assert src.exists()


@pytest.mark.parametrize("payload", [
    "stream,\n",                     # the field the contract is built on, absent
    "stream,3\n",                    # a value the vocabulary does not contain
    "attached_pic=0\n",              # a different output format entirely
    "{\"streams\": []}\n",           # json where csv was asked for
    "stream,yes\n",
    # A whole record has to be the contract, not merely its last field. Reading
    # only the text after the final comma would let every line below decide the
    # fate of the file: the first three would vouch for a playable stream that
    # was never reported, and the last two would count as artwork and take a
    # definitive rejection - deleting an artifact off a payload this code does
    # not understand.
    "0\n",                           # the *old* index vocabulary, unprefixed
    "format,0\n",                    # a section that is not a stream
    "extra,stream,0\n",              # the contract with something ahead of it
    "garbage,1\n",
    "stream,0,1\n",                  # more fields than were asked for
])
def test_b7_malformed_payload_preserves_the_artifact(env, real_media, payload):
    """rc 0 with a payload that is not the contract is the *validator* failing,
    not the file. Nothing about the bytes was established, so they are kept -
    the same rule that keeps a timed-out probe from deleting good video."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, payload, "")

    result = _run(src, out_dir)

    assert result["status"] == "error", "an unparseable answer is not a pass"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"], \
        "deleting media the probe never judged is the larger mistake"
    assert src.exists()


def test_b7b_an_unparseable_record_never_causes_a_deletion(env, real_media):
    """The sharp edge of the rule above, on its own.

    A line the parser cannot read must not be allowed to *count* - not as a
    playable stream and not as artwork. Counting it as artwork is the dangerous
    direction: it completes a "every picture here is a thumbnail" verdict, and
    that verdict removes the file.
    """
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "garbage,1\ngarbage,1\n", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert (out_dir / "Movie.mp4").read_bytes() == real_media[".mp4"], \
        "an unreadable payload is not a licence to delete"


def test_b8_a_malformed_line_beside_a_playable_one_still_succeeds(env,
                                                                  real_media):
    """A printed `0` is positive proof a playable stream exists, and noise
    printed after it does not retract that proof. Only the *absence* of proof
    has to be classified."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "stream,0\nformat_name=utterly unexpected\n", "")

    result = _run(src, out_dir)

    assert result["status"] == "ok"


# ══ GROUP C — containers, as measured ══════════════════════════════════════

def test_c1_mp4_h264_cover_art_only_fails(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    assert result["status"] == "error"
    assert fake.attempts == 1
    assert _video_names(out_dir) == []


def test_c2_mp4_h265_cover_art_only_is_terminal_without_fallback(
        env, cover_art_only_media):
    """The MP4 -> MKV retry exists for a mux failure. An apparent success whose
    only picture is artwork is a different anomaly and stays terminal."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert fake.attempts == 1, "no second encode"
    assert "mp4" in fake.final_muxers[0]
    assert "mux_failed" not in result
    assert spy.count == 1
    assert _video_names(out_dir) == []
    assert src.exists()


def test_c3_mkv_attachment_cover_art_only_fails(env, cover_art_only_media):
    """Matroska's own cover-art mechanism is a file attachment, and ffprobe
    reports it as a video stream with `attached_pic=1` - so the same rule
    catches it, on evidence rather than on assumption."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mkv"]]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "error"
    assert _video_names(out_dir) == []
    assert src.exists()


def test_c4_mkv_video_track_disposition_is_not_carried_by_matroska(
        tmp_path, cover_jpeg):
    """The honest limit of this slice, pinned as evidence rather than hidden.

    Asking ffmpeg to flag an ordinary Matroska *video track* as an attached
    picture does not survive the muxer: ffprobe reports `attached_pic=0`, so
    such a file is indistinguishable from real video and is accepted. Nothing
    in Cove produces it, and reaching it would need a stream-level attachment
    model this slice deliberately does not build.
    """
    audio = tmp_path / "audio.m4a"
    assert _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                    "-c:a", "aac", str(audio)]).returncode == 0
    path = tmp_path / "disposition_track.mkv"
    r = _ffmpeg(["-i", str(audio), "-i", str(cover_jpeg),
                 "-map", "0:a", "-map", "1:v", "-c:a", "copy",
                 "-c:v", "mjpeg", "-disposition:v:0", "attached_pic",
                 str(path)])
    assert r.returncode == 0, r.stderr[-400:]

    assert _dispositions(path) == ["0"], \
        "if Matroska ever starts carrying this disposition, widen the suite"


def test_c5_webm_cover_art_is_not_constructible(tmp_path, cover_jpeg):
    """WebM, audited honestly. The muxer admits only VP8/VP9/AV1 video, so
    mjpeg artwork cannot be written into it at all and there is no WebM
    cover-art case for this slice to have an opinion about."""
    audio = tmp_path / "audio.opus"
    assert _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                    "-c:a", "libopus", str(audio)]).returncode == 0
    path = tmp_path / "cover_only.webm"
    r = _ffmpeg(["-i", str(audio), "-i", str(cover_jpeg),
                 "-map", "0:a", "-map", "1:v", "-c:a", "copy",
                 "-c:v", "mjpeg", "-disposition:v:0", "attached_pic",
                 str(path)])

    assert r.returncode != 0, \
        "a constructible WebM cover-art fixture would need its own coverage"


def test_c6_ordinary_webm_still_succeeds(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".webm"]]

    result = _run(src, out_dir, fmt="WebM (VP9)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == real_media[".webm"]


# ══ GROUP D — two-pass and the fallback ceiling ════════════════════════════

def test_d1_two_pass_cover_art_only_final_fails(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _two_pass(src, out_dir)

    assert result["status"] == "error"
    assert len(fake.all_cmds) == 2, "pass 1 and pass 2, and nothing more"
    assert spy.count == 1, "pass 1 is never probed"
    assert _video_names(out_dir) == []
    assert src.exists()


def test_d2_two_pass_video_with_cover_art_still_succeeds(
        env, video_plus_cover_media):
    src, out_dir, fake, spy = env
    fake.payloads = [video_plus_cover_media[".mp4"]]

    result = _two_pass(src, out_dir)

    assert result["status"] == "ok"
    assert len(fake.all_cmds) == 2
    assert spy.count == 1


def test_d3_fallback_mkv_with_cover_art_only_fails_after_two_attempts(
        env, cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, cover_art_only_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert fake.attempts == 2, "two encode attempts is the ceiling"
    assert spy.count == 1, "only the successful final artifact is probed"
    assert spy.targets == [out_dir / "Movie.mkv"]
    assert _video_names(out_dir) == []
    assert src.exists()


def test_d4_fallback_mkv_with_video_and_cover_art_still_succeeds(
        env, video_plus_cover_media):
    src, out_dir, fake, spy = env
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, video_plus_cover_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).name == "Movie.mkv"
    assert fake.attempts == 2
    assert spy.count == 1


# ══ GROUP E — the source and the sidecars stay behind the gate ═════════════

@pytest.mark.parametrize("fmt", COVER_ART_FORMATS)
def test_e1_source_deletion_is_blocked_behind_cover_art_only(
        env, cover_art_only_media, fmt):
    """The whole point. Deletion is opt-in and asked for here; a conversion
    that produced only a thumbnail must not consume the only copy."""
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[EXT_OF[fmt]]]

    result = delete_source_if_eligible(_run(src, out_dir, fmt=fmt),
                                       enabled=True)

    assert result["status"] == "error"
    assert src.exists(), "the source outlives an artwork-only conversion"
    assert result.get("source_deleted") is not True


def test_e2_source_survives_an_inconclusive_probe(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    spy.raises = OSError(13, "permission denied")

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert src.exists()
    assert result.get("source_deleted") is not True


def test_e3_source_deletion_still_works_behind_video_with_cover_art(
        env, video_plus_cover_media):
    """One case is blocked and nothing else changes."""
    src, out_dir, fake, spy = env
    fake.payloads = [video_plus_cover_media[".mp4"]]

    result = delete_source_if_eligible(_run(src, out_dir), enabled=True)

    assert result["status"] == "ok"
    assert result["source_deleted"] is True
    assert not src.exists()
    assert Path(result["output"]).exists()


def test_e4_no_sidecar_is_finalized_behind_cover_art_only(
        env, cover_art_only_media, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert _names(out_dir, "*.srt") == [], \
        "a subtitle track beside a thumbnail is not a delivered sidecar"
    assert "subtitles" not in result


def test_e5_sidecars_still_finalize_behind_video_with_cover_art(
        env, video_plus_cover_media, monkeypatch):
    src, out_dir, fake, spy = env
    _probe(monkeypatch, [_sub(2, "subrip")])
    fake.payloads = [video_plus_cover_media[".mp4"]]

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert _names(out_dir, "*.srt") == ["Movie.eng.srt"]


# ══ GROUP F — only the exact owned path is eligible ════════════════════════

def test_f1_collision_neighbour_survives_a_cover_art_rejection(
        env, cover_art_only_media):
    src, out_dir, fake, spy = env
    neighbour = out_dir / "Movie.mp4"
    neighbour.write_bytes(SENTINEL)
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert neighbour.read_bytes() == SENTINEL, \
        "the user's own file is not Cove's to touch"
    assert not (out_dir / "Movie_1.mp4").exists()
    assert spy.targets == [out_dir / "Movie_1.mp4"]
    assert src.exists()


def test_f2_deep_collision_neighbours_all_survive(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    kept = {}
    for name in ("Movie.mp4", "Movie_1.mp4", "Movie_2.mp4", "Movie_3.mp4"):
        p = out_dir / name
        p.write_bytes(SENTINEL + name.encode())
        kept[name] = p.read_bytes()
    fake.payloads = [cover_art_only_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "error"
    for name, data in kept.items():
        assert (out_dir / name).read_bytes() == data
    assert not (out_dir / "Movie_4.mp4").exists()
    assert spy.targets == [out_dir / "Movie_4.mp4"]


def test_f3_fallback_collision_neighbour_survives(env, cover_art_only_media):
    src, out_dir, fake, spy = env
    neighbour = out_dir / "Movie.mkv"
    neighbour.write_bytes(SENTINEL)
    fake.finals = [(1, "muxer error"), (0, "")]
    fake.payloads = [MISSING, cover_art_only_media[".mkv"]]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "error"
    assert neighbour.read_bytes() == SENTINEL
    assert not (out_dir / "Movie_1.mkv").exists()
    assert spy.targets == [out_dir / "Movie_1.mkv"]


# ══ GROUP G — the real ffprobe contract, pinned ════════════════════════════

def test_g1_real_ffprobe_reports_attached_pic_for_cover_art(
        cover_art_only_media, video_plus_cover_media, real_media,
        tmp_path):
    """The representation this slice is built on, straight off the installed
    binary. If a future FFmpeg stops reporting it this way, this fails first
    and says so, instead of the whole gate quietly changing meaning."""
    d = tmp_path
    for ext in COVER_ART_EXTS:
        art = d / f"art{ext}"
        art.write_bytes(cover_art_only_media[ext])
        assert _dispositions(art) == ["1"], \
            f"cover art in {ext} must report attached_pic=1"

        both = d / f"both{ext}"
        both.write_bytes(video_plus_cover_media[ext])
        assert sorted(_dispositions(both)) == ["0", "1"]

    for ext in (".mp4", ".mkv", ".webm"):
        plain = d / f"plain{ext}"
        plain.write_bytes(real_media[ext])
        assert _dispositions(plain) == ["0"], \
            f"ordinary {ext} video must report attached_pic=0, not a missing field"


def test_g2_the_final_probe_asks_for_attached_pic(env, real_media):
    """The whole of this slice's addition, on the wire - and still one probe,
    one `-show_entries`, and nothing that dumps streams, formats or packets."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert spy.count == 1
    argv = [str(a) for a in spy.cmds[0]]
    assert argv[0] == str(compressor.FFPROBE_BIN)
    assert argv[-1] == str(out_dir / "Movie.mp4")
    assert "-v" in argv and argv[argv.index("-v") + 1] == "error"
    assert argv[argv.index("-select_streams") + 1] == "v", \
        "every video stream, because any one of them may be the playable one"
    assert argv.count("-show_entries") == 1
    assert "attached_pic" in argv[argv.index("-show_entries") + 1]
    for banned in ("-show_streams", "-show_format", "-show_packets",
                   "-show_frames", "-count_frames", "-count_packets",
                   "-show_data", "-show_chapters", "-show_programs"):
        assert banned not in argv, f"{banned} is a further semantic check"


def test_g3_a_scripted_zero_passes(env, real_media):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "stream,0\n", "")

    assert _run(src, out_dir)["status"] == "ok"


def test_g4_a_scripted_one_alone_fails_and_cleans(env, real_media):
    """The bytes are genuinely video; the probe is scripted to report artwork
    only. The decision follows the probe's answer, not the payload."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, "stream,1\n", "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()
    assert src.exists()


@pytest.mark.parametrize("payload", ["stream,1\nstream,0\n",
                                     "stream,0\nstream,1\n",
                                     "stream,1\nstream,1\nstream,0\n"])
def test_g5_artwork_beside_a_playable_stream_passes(env, real_media, payload):
    """Never `if any attached_pic: reject`."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, payload, "")

    assert _run(src, out_dir)["status"] == "ok"


@pytest.mark.parametrize("payload", ["stream,1\nstream,1\n",
                                     "stream,1\nstream,1\nstream,1\n"])
def test_g6_artwork_only_fails_however_many_pictures(env, real_media, payload):
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, payload, "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()


@pytest.mark.parametrize("payload", ["", "\n", "  \r\n \n"])
def test_g7_an_empty_payload_is_still_the_zero_video_verdict(env, real_media,
                                                             payload):
    """rc 0 and nothing printed is a valid, complete answer: ffprobe found no
    video streams. Definitive, and Tab 20's case unchanged."""
    src, out_dir, fake, spy = env
    fake.payloads = [real_media[".mp4"]]
    spy.result = (0, payload, "")

    result = _run(src, out_dir)

    assert result["status"] == "error"
    assert not (out_dir / "Movie.mp4").exists()
    assert src.exists()


# ══ GROUP H — what this slice still does not validate ══════════════════════

def test_h1_an_unusual_codec_satisfies_the_gate(env, odd_codec_media):
    """mpeg4 video Cove would never emit, and no audio track at all. A playable
    video stream exists; codec is not this slice's business."""
    src, out_dir, fake, spy = env
    fake.payloads = [odd_codec_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"
    assert Path(result["output"]).read_bytes() == odd_codec_media


def test_h2_a_single_frame_duration_satisfies_the_gate(env,
                                                       single_frame_media):
    """One frame of real video is not artwork. Duration and truncation
    detection remain a separate slice, and "short" is not "attached"."""
    src, out_dir, fake, spy = env
    fake.payloads = [single_frame_media]

    result = _run(src, out_dir, fmt="MKV (H.265)")

    assert result["status"] == "ok"


def test_h3_no_exact_playable_count_is_required(env,
                                               multi_video_plus_cover_media):
    """Two playable streams and one artwork stream: `>= 1`, never `== 1`."""
    src, out_dir, fake, spy = env
    fake.payloads = [multi_video_plus_cover_media]

    assert _run(src, out_dir, fmt="MKV (H.265)")["status"] == "ok"


def test_h4_the_result_shapes_are_unchanged(env, cover_art_only_media,
                                            video_plus_cover_media):
    src, out_dir, fake, spy = env
    fake.payloads = [cover_art_only_media[".mp4"]]
    rejected = _run(src, out_dir)
    assert set(rejected) == {"file", "status", "msg"}
    assert rejected["file"] == src
    assert "mux_failed" not in rejected, "an artwork-only output earns no retry"

    fake.payloads = [video_plus_cover_media[".mp4"]]
    ok = _run(src, out_dir)
    assert set(ok) >= {"file", "output", "status", "original", "new",
                       "encoder"}


def test_h5_nothing_leaks_between_files(env, cover_art_only_media,
                                        video_plus_cover_media, real_media):
    """Per-file isolation: a rejection must not poison the next job, and an
    acceptance must not vouch for the one after it."""
    src, out_dir, fake, spy = env
    second = src.parent / "Second.mov"
    second.write_bytes(b"s" * SRC_BYTES)
    third = src.parent / "Third.mov"
    third.write_bytes(b"s" * SRC_BYTES)

    fake.payloads = [cover_art_only_media[".mp4"],
                     video_plus_cover_media[".mp4"],
                     cover_art_only_media[".mp4"]]

    first_r = _run(src, out_dir)
    second_r = _run(second, out_dir)
    third_r = _run(third, out_dir)

    assert first_r["status"] == "error"
    assert second_r["status"] == "ok"
    assert Path(second_r["output"]).name == "Second.mp4"
    assert third_r["status"] == "error"
    assert _video_names(out_dir) == ["Second.mp4"]
    assert spy.count == 3
    assert spy.targets == [out_dir / "Movie.mp4", out_dir / "Second.mp4",
                           out_dir / "Third.mp4"]
