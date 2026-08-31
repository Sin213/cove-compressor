"""AMD AMF capability-probe geometry.

`_probe_amf()` proves an AMF encoder really initializes on this machine by
running one tiny synthetic encode through it. The workload it used to run was
320x240, and that geometry is below what AMD's HEVC encoder will accept: on a
Radeon RX 9070 XT with a working driver,

    ffmpeg -f lavfi -i color=c=black:s=320x240:r=10:d=0.3 -c:v hevc_amf -f null NUL
    [hevc_amf] encoder->Init() failed with error 5

while the very same machine encodes 720p HEVC through AMF without complaint.
The probe therefore reported `amf_available("hevc_amf") is False` on hardware
that demonstrably supports it, and Cove silently fell back to libx265.

The repair is a workload change and nothing else: one representative
1280x720 frame instead of a 320x240 thumbnail. Everything that made the
probe trustworthy stays exactly as it was - the encoder-listing fast path,
the single initialization attempt, the "nonzero or raised means False" rule,
the null sink, and the per-encoder cache.

Locked down here:

  A. The initialization command carries the representative geometry, exactly
     one frame, the requested encoder, and no output file.
  B. An encoder missing from `ffmpeg -encoders` short-circuits before any
     initialization encode runs.
  C. Every failure route - listing raises, listing rejects, init raises, init
     exits nonzero - still yields False, and only a clean exit yields True.
  D. `amf_available()` probes once per encoder and then serves the cache, with
     h264_amf and hevc_amf kept independent.
  E-F. The public `compress_video()` entry point selects `hevc_amf` from the
     repaired probe under both a forced AMF preference and Automatic, and
     still yields to NVENC when NVIDIA is present.
  G-I. Tab 5's rate control, Tab 4's multi-audio budget, Tab 3's attachments
     and Tab 2b's two-attempt fallback are untouched by the workload change.
  J. No probing ladder: one geometry, one initialization attempt, no
     `ffmpeg -h encoder=...`, no retry at a second size.

No AMD GPU is needed here - `subprocess.run` is faked, so the tests assert on
the command Cove *builds*. The proof that the built command actually
initializes hevc_amf on a Radeon, and that the predecessor one did not, is
real-hardware evidence recorded in the handoff.
"""
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    StreamInventory, _probe_amf, amf_available, compress_video,
)

PROBE_GEOMETRY = "1280x720"
PREDECESSOR_GEOMETRY = "320x240"
LISTING = "V....D h264_amf   AMD AMF H.264\nV....D hevc_amf   AMD AMF HEVC\n"


# ── subprocess fake ──────────────────────────────────────────────────────────

class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRun:
    """Stands in for `subprocess.run`, splitting probe calls by kind.

    `listing` calls are the `ffmpeg -encoders` query; `init` calls are the
    synthetic encode. Anything else is recorded as `other` so a newly
    introduced subprocess kind (an `ffmpeg -h encoder=...` capability query,
    say) cannot slip in unnoticed.
    """

    def __init__(self, listing_out=LISTING, listing_rc=0, init_rc=0,
                 listing_exc=None, init_exc=None):
        self.listing_out = listing_out
        self.listing_rc = listing_rc
        self.init_rc = init_rc
        self.listing_exc = listing_exc
        self.init_exc = init_exc
        self.listing_cmds: list[list] = []
        self.init_cmds: list[list] = []
        self.other_cmds: list[list] = []

    def __call__(self, cmd, *a, **kw):
        cmd = list(cmd)
        if "-encoders" in cmd:
            self.listing_cmds.append(cmd)
            if self.listing_exc:
                raise self.listing_exc
            return FakeCompleted(self.listing_rc, self.listing_out)
        if "-c:v" in cmd:
            self.init_cmds.append(cmd)
            if self.init_exc:
                raise self.init_exc
            return FakeCompleted(self.init_rc)
        self.other_cmds.append(cmd)
        return FakeCompleted(0)

    @property
    def init_cmd(self) -> list:
        assert len(self.init_cmds) == 1, (
            f"expected exactly one initialization encode, got "
            f"{len(self.init_cmds)}: {self.init_cmds}")
        return self.init_cmds[0]


@pytest.fixture
def probe(monkeypatch):
    """A faked subprocess layer with a cleared AMF cache around each test."""
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    compressor._amf_cache.clear()
    yield fake
    compressor._amf_cache.clear()


def _joined(cmd) -> str:
    return " ".join(str(t) for t in cmd)


def _value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _lavfi_size(cmd) -> str:
    """The exact `s=` geometry of the probe's synthetic source.

    Parsed out as a whole field rather than matched as a substring, so a
    superset like `1280x7200` cannot masquerade as the proven geometry.
    """
    source = _value_after(cmd, "-i")
    assert source is not None, "probe carried no -i source"
    fields = dict(
        part.split("=", 1) for part in source.split(":") if "=" in part)
    assert "s" in fields, f"probe source has no size field: {source}"
    return fields["s"]


# ── A. probe command shape ───────────────────────────────────────────────────

class TestProbeCommandShape:
    def test_h264_initialization_uses_representative_geometry(self, probe):
        assert _probe_amf("h264_amf") is True
        assert _lavfi_size(probe.init_cmd) == PROBE_GEOMETRY
        assert PREDECESSOR_GEOMETRY not in _joined(probe.init_cmd)

    def test_hevc_initialization_uses_representative_geometry(self, probe):
        assert _probe_amf("hevc_amf") is True
        assert _lavfi_size(probe.init_cmd) == PROBE_GEOMETRY
        assert PREDECESSOR_GEOMETRY not in _joined(probe.init_cmd)

    def test_probe_encodes_exactly_one_frame(self, probe):
        _probe_amf("hevc_amf")
        assert _value_after(probe.init_cmd, "-frames:v") == "1"

    def test_probe_uses_the_requested_encoder(self, probe):
        _probe_amf("h264_amf")
        assert _value_after(probe.init_cmd, "-c:v") == "h264_amf"

        probe.init_cmds.clear()
        _probe_amf("hevc_amf")
        assert _value_after(probe.init_cmd, "-c:v") == "hevc_amf"

    def test_probe_writes_only_to_the_null_sink(self, probe):
        import os

        _probe_amf("hevc_amf")
        cmd = probe.init_cmd
        assert cmd[-3:] == ["-f", "null", os.devnull]
        assert not any(str(t).endswith((".mp4", ".mkv", ".webm"))
                       for t in cmd)

    def test_probe_input_is_synthetic_not_a_user_file(self, probe):
        _probe_amf("hevc_amf")
        cmd = probe.init_cmd
        assert "lavfi" in cmd
        assert _value_after(cmd, "-i").startswith("color=")
        assert _lavfi_size(cmd) == PROBE_GEOMETRY


# ── B. encoder-listing fast path ─────────────────────────────────────────────

class TestListingFastPath:
    def test_encoder_absent_from_listing_returns_false_without_init(self, probe):
        probe.listing_out = "V....D libx265   H.265\n"
        assert _probe_amf("hevc_amf") is False
        assert len(probe.listing_cmds) == 1
        assert probe.init_cmds == []

    def test_encoder_present_proceeds_to_initialization(self, probe):
        assert _probe_amf("hevc_amf") is True
        assert len(probe.listing_cmds) == 1
        assert len(probe.init_cmds) == 1


# ── C. failure semantics ─────────────────────────────────────────────────────

class TestFailureSemantics:
    def test_listing_oserror_returns_false(self, probe):
        probe.listing_exc = OSError("ffmpeg is not on this machine")
        assert _probe_amf("hevc_amf") is False
        assert probe.init_cmds == []

    def test_listing_subprocess_error_returns_false(self, probe):
        probe.listing_exc = subprocess.TimeoutExpired("ffmpeg", 15)
        assert _probe_amf("hevc_amf") is False
        assert probe.init_cmds == []

    def test_initialization_oserror_returns_false(self, probe):
        probe.init_exc = OSError("cannot spawn")
        assert _probe_amf("hevc_amf") is False

    def test_initialization_subprocess_error_returns_false(self, probe):
        probe.init_exc = subprocess.TimeoutExpired("ffmpeg", 30)
        assert _probe_amf("hevc_amf") is False

    def test_nonzero_initialization_returns_false(self, probe):
        probe.init_rc = 1
        assert _probe_amf("hevc_amf") is False

    def test_clean_initialization_returns_true(self, probe):
        probe.init_rc = 0
        assert _probe_amf("hevc_amf") is True


# ── D. cache semantics ───────────────────────────────────────────────────────

class TestCacheSemantics:
    def test_first_availability_call_probes(self, probe):
        assert amf_available("hevc_amf") is True
        assert len(probe.init_cmds) == 1
        assert compressor._amf_cache["hevc_amf"] is True

    def test_second_availability_call_is_served_from_cache(self, probe):
        assert amf_available("hevc_amf") is True
        assert amf_available("hevc_amf") is True
        assert len(probe.listing_cmds) == 1
        assert len(probe.init_cmds) == 1

    def test_h264_and_hevc_are_cached_independently(self, monkeypatch):
        """A working h264_amf must not vouch for hevc_amf, and vice versa."""
        def by_encoder(cmd, *a, **kw):
            cmd = list(cmd)
            if "-encoders" in cmd:
                return FakeCompleted(0, LISTING)
            rc = 0 if _value_after(cmd, "-c:v") == "h264_amf" else 1
            return FakeCompleted(rc)

        monkeypatch.setattr(subprocess, "run", by_encoder)
        compressor._amf_cache.clear()
        try:
            assert amf_available("h264_amf") is True
            assert amf_available("hevc_amf") is False
            assert compressor._amf_cache == {
                "h264_amf": True, "hevc_amf": False}
        finally:
            compressor._amf_cache.clear()

    def test_clearing_the_cache_forces_a_fresh_probe(self, probe):
        assert amf_available("hevc_amf") is True
        compressor._amf_cache.clear()
        assert amf_available("hevc_amf") is True
        assert len(probe.init_cmds) == 2


# ── compress_video harness ───────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, recording each encode invocation."""

    def __init__(self, finals=None, audio_streams=1, attachments=1):
        self.finals = list(finals or [])
        self.audio_streams = audio_streams
        self.attachments = attachments
        self.final_cmds: list[list] = []

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        cmd = list(cmd)
        if _muxer_of(cmd) == "null":
            return 0, ""
        self.final_cmds.append(cmd)
        rc, err = self.finals.pop(0) if self.finals else (0, "")
        if rc == 0:
            Path(cmd[-1]).write_bytes(b"v" * 10)
        return rc, err

    @property
    def encoders(self) -> list[str]:
        return [_value_after(c, "-c:v") for c in self.final_cmds]

    @property
    def cmd(self) -> list:
        assert self.final_cmds, "no encode invocation was recorded"
        return self.final_cmds[-1]


SRC_BYTES = 8 * 1024 * 1024
DURATION = 10.0


@pytest.fixture
def amd_box(tmp_path, monkeypatch):
    """A real-probe AMD machine: `subprocess.run` is faked so the *production*
    `_probe_amf` runs and succeeds, but nothing above it is stubbed."""
    src = tmp_path / "Movie.mov"
    src.write_bytes(b"s" * SRC_BYTES)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    fake_run = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake_run)
    fake = FakeFfmpeg()
    monkeypatch.setattr(compressor, "run_ffmpeg", fake)
    monkeypatch.setattr(compressor, "ffprobe_duration", lambda p: DURATION)
    monkeypatch.setattr(
        compressor, "ffprobe_stream_inventory",
        lambda p: StreamInventory(subtitles=[], audio_count=1))
    monkeypatch.setattr(compressor, "nvenc_available",
                        lambda e="hevc_nvenc": False)
    compressor._amf_cache.clear()
    yield src, out_dir, fake, fake_run
    compressor._amf_cache.clear()


def _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="4", audio_kbps="128", encoder_pref="amf"):
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, audio_kbps,
        threading.Event(), encoder_pref=encoder_pref)


# ── E. forced AMF H.265 selection through the real probe ─────────────────────

class TestForcedAmfSelection:
    def test_forced_amf_h265_selects_hevc_amf(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir)
        assert fake.encoders == ["hevc_amf"]

    def test_forced_amf_h265_keeps_tab5_target_rate_control(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir)
        assert _value_after(fake.cmd, "-rc") == "vbr_peak"

    def test_forced_amf_mkv_h265_selects_hevc_amf(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir, fmt="MKV (H.265)")
        assert fake.encoders == ["hevc_amf"]

    def test_selection_flows_from_a_real_initialization_probe(self, amd_box):
        """The encoder is chosen because ffmpeg was actually asked to
        initialize it - not because availability was stubbed True."""
        src, out_dir, fake, fake_run = amd_box
        _run(src, out_dir)
        assert len(fake_run.init_cmds) == 1
        assert _lavfi_size(fake_run.init_cmds[0]) == PROBE_GEOMETRY

    def test_failing_initialization_falls_back_to_cpu(self, amd_box):
        src, out_dir, fake, fake_run = amd_box
        fake_run.init_rc = 1
        _run(src, out_dir)
        assert fake.encoders == ["libx265"]


# ── F. automatic selection ───────────────────────────────────────────────────

class TestAutomaticSelection:
    def test_auto_picks_hevc_amf_when_nvenc_is_absent(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir, encoder_pref="auto")
        assert fake.encoders == ["hevc_amf"]

    def test_auto_still_prefers_nvenc_when_nvidia_is_present(
            self, amd_box, monkeypatch):
        src, out_dir, fake, _ = amd_box
        monkeypatch.setattr(compressor, "nvenc_available",
                            lambda e="hevc_nvenc": True)
        _run(src, out_dir, encoder_pref="auto")
        assert fake.encoders == ["hevc_nvenc"]


# ── G. Tab 5 rate control survives ───────────────────────────────────────────

class TestRateControlSurvives:
    def test_target_mode_amf_still_uses_vbr_peak(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir, mode="Target reduction", mode_value="50")
        assert _value_after(fake.cmd, "-c:v") == "hevc_amf"
        assert _value_after(fake.cmd, "-rc") == "vbr_peak"

    def test_quality_mode_amf_still_uses_cqp(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir, mode="Quality preset", mode_value="Balanced")
        assert _value_after(fake.cmd, "-c:v") == "hevc_amf"
        assert _value_after(fake.cmd, "-rc") == "cqp"


# ── H. Tab 4 multi-audio survives ────────────────────────────────────────────

class TestMultiAudioSurvives:
    def test_amf_encode_still_maps_every_audio_track(self, amd_box, monkeypatch):
        src, out_dir, fake, _ = amd_box
        monkeypatch.setattr(
            compressor, "ffprobe_stream_inventory",
            lambda p: StreamInventory(subtitles=[], audio_count=3))
        _run(src, out_dir)
        assert _value_after(fake.cmd, "-c:v") == "hevc_amf"
        assert "0:a?" in _maps(fake.cmd)

    def test_audio_budget_still_scales_with_track_count(
            self, amd_box, monkeypatch):
        src, out_dir, fake, _ = amd_box
        monkeypatch.setattr(
            compressor, "ffprobe_stream_inventory",
            lambda p: StreamInventory(subtitles=[], audio_count=3))
        _run(src, out_dir)
        one_track_kbps = compressor.calc_video_bitrate_kbps(
            int(4 * 1024 * 1024), DURATION, 128, 1)
        three_track_kbps = compressor.calc_video_bitrate_kbps(
            int(4 * 1024 * 1024), DURATION, 128, 3)
        assert three_track_kbps < one_track_kbps
        assert _value_after(fake.cmd, "-b:v") == f"{three_track_kbps}k"


# ── I. Tab 3 attachments / Tab 2b fallback survive ───────────────────────────

class TestNeighbouringContractsSurvive:
    def test_direct_amf_mkv_still_carries_attachments(self, amd_box):
        src, out_dir, fake, _ = amd_box
        _run(src, out_dir, fmt="MKV (H.265)")
        cmd = fake.cmd
        assert "0:t?" in _maps(cmd)
        assert _value_after(cmd, "-c:t") == "copy"

    def test_mp4_h265_amf_fallback_is_still_two_attempts_at_most(self, amd_box):
        src, out_dir, fake, _ = amd_box
        fake.finals = [(1, "muxer error"), (1, "muxer error"), (1, "nope")]
        _run(src, out_dir)
        assert len(fake.final_cmds) == 2
        assert fake.encoders == ["hevc_amf", "hevc_amf"]
        assert [_muxer_of(c) for c in fake.final_cmds] == ["mp4", "matroska"]


# ── J. no generalized or dynamic probing ─────────────────────────────────────

class TestNoGeneralizedProbing:
    def test_successful_probe_runs_exactly_one_initialization_encode(self, probe):
        _probe_amf("hevc_amf")
        assert len(probe.init_cmds) == 1

    def test_a_failed_initialization_is_not_retried_at_another_geometry(
            self, probe):
        probe.init_rc = 1
        assert _probe_amf("hevc_amf") is False
        assert len(probe.init_cmds) == 1
        assert probe.other_cmds == []

    def test_production_never_asks_ffmpeg_for_encoder_help(self, probe):
        _probe_amf("hevc_amf")
        for cmd in probe.listing_cmds + probe.init_cmds + probe.other_cmds:
            assert not any(str(t).startswith("encoder=") for t in cmd)
            assert "-h" not in cmd
        assert probe.other_cmds == []

    def test_probe_call_counts_are_exactly_one_listing_and_one_init(self, probe):
        amf_available("hevc_amf")
        assert (len(probe.listing_cmds), len(probe.init_cmds)) == (1, 1)
        amf_available("hevc_amf")
        assert (len(probe.listing_cmds), len(probe.init_cmds)) == (1, 1)
