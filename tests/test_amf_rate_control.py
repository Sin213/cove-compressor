"""AMD AMF target-mode rate control.

ffmpeg's AMF encoders never accepted a bare `-rc vbr`. Their rate-control
vocabulary is `cqp | cbr | vbr_peak | vbr_latency | qvbr | hqvbr | hqcbr`, and
handing them `vbr` fails at option parsing before a single frame is encoded:

    [h264_amf] Unable to parse "rc" option value "vbr"
    [h264_amf] Error setting option rc to value vbr.

Cove's target-size and target-reduction paths emitted exactly that, so every
AMD GPU target-mode job died on argument parsing. The peak-constrained mode
`vbr_peak` is the one that expresses what Cove already asks for - a `-b:v`
average with a capped `-maxrate` and a `-bufsize` - so that is what the AMF
bitrate branch now emits.

Locked down here:

  A-B. The builder emits `-rc vbr_peak` for both AMF codecs with the bitrate
     envelope and single-pass behaviour untouched, and quality-preset mode
     still uses `-rc cqp -qp`.
  C-D. NVENC keeps its own perfectly valid `-rc vbr` and `-multipass fullres`
     (a blind global replacement would break the NVIDIA path), and the CPU
     encoders carry no rate-control token at all.
  E-G. The public `compress_video()` entry point emits the corrected mode for
     H.264 and H.265, under `Target file size` and `Target reduction` alike.
  H-I. Tab 4's per-track audio accounting, zero-audio arithmetic and
     impossible-target gate compute exactly what they computed before: only
     the rate-control token changed, never the bitrate it is applied to.
  J-K. Matroska direct and the MP4 -> MKV fallback inherit the corrected mode,
     with the attachment, subtitle and two-attempt contracts intact.
  L. Automatic encoder selection - not just the forced-AMF UI path - gets the
     fix, and still prefers NVENC (with NVENC's `vbr`) when NVIDIA is present.
  M. No new probing: the option vocabulary is a fixed compile-time fact, not
     something production re-discovers by running `ffmpeg -h encoder=...`.

No ffmpeg and no AMD GPU are needed here; `run_ffmpeg`, the availability
probes and the stream inventory are faked. The real-hardware proof that
`vbr` is rejected and `vbr_peak` encodes lives in the handoff, not in a unit
test that would fail on any machine without a Radeon in it.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    StreamInventory,
    build_video_encoder_args,
    calc_video_bitrate_kbps,
    compress_video,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _muxer_of(cmd) -> str:
    return cmd[cmd.index("-f") + 1] if "-f" in cmd else ""


def _maps(cmd) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]


def _value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _rc_values(cmd) -> list[str]:
    """Every `-rc` token in the command, so a second one cannot hide."""
    return [cmd[i + 1] for i, tok in enumerate(cmd[:-1]) if tok == "-rc"]


def _rc(cmd):
    """The single rate-control value, asserting there is exactly one."""
    values = _rc_values(cmd)
    assert len(values) == 1, f"expected exactly one -rc, got {values}"
    return values[0]


def _video_kbps_of(cmd) -> int:
    v = _value_after(cmd, "-b:v")
    assert v is not None, "invocation carried no -b:v"
    return int(str(v).rstrip("k"))


def _args(encoder, video_kbps=None, crf=None, vf=None, use_two_pass=False,
          pass_num=None, amf_quality="balanced"):
    return build_video_encoder_args(
        encoder=encoder, vf=vf, use_two_pass=use_two_pass, pass_num=pass_num,
        video_kbps=video_kbps, crf=crf, speed_preset="medium",
        nvenc_preset="p6", amf_quality=amf_quality,
    )


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`, recording and classifying each invocation."""

    def __init__(self, finals=None, pass1=None,
                 audio_streams=1, attachments=1, encode_bytes=b"v" * 10):
        self.finals = list(finals or [])
        self.pass1 = list(pass1 or [])
        self.audio_streams = audio_streams
        self.attachments = attachments
        self.encode_bytes = encode_bytes
        self.pass1_cmds: list[list] = []
        self.final_cmds: list[list] = []

    @staticmethod
    def _next(q):
        return q.pop(0) if q else (0, "")

    def _unsatisfied_map(self, cmd):
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
    def final_encoders(self) -> list[str]:
        return [_value_after(c, "-c:v") for c in self.final_cmds]

    @property
    def mux_cmd(self) -> list:
        assert self.final_cmds, "no final muxing invocation was recorded"
        return self.final_cmds[-1]


def _inventory(monkeypatch, audio_count=1, subtitles=(), calls=None):
    def fake(path):
        if calls is not None:
            calls.append(Path(path))
        return StreamInventory(
            subtitles=[dict(s) for s in subtitles], audio_count=audio_count)

    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", fake)
    return fake


SRC_BYTES = 8 * 1024 * 1024
DURATION = 10.0


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder/probe stack with AMF
    reachable and NVENC absent - the AMD machine this slice is about."""
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
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": True)
    _inventory(monkeypatch, audio_count=1)
    return src, out_dir, fake


def _run(src, out_dir, fmt="MP4 (H.265)", mode="Target file size",
         mode_value="4", audio_kbps="128", encoder_pref="amf",
         cancel_flag=None, **kw):
    return compress_video(
        src, out_dir, mode, mode_value, fmt, None, audio_kbps,
        cancel_flag if cancel_flag is not None else threading.Event(),
        encoder_pref=encoder_pref, **kw)


def _expected_kbps(target_mb=4.0, audio_kbps=128, audio_count=1):
    return calc_video_bitrate_kbps(
        int(target_mb * 1024 * 1024), DURATION, audio_kbps, audio_count)


# ══ GROUP A — pure argument contract ═════════════════════════════════════════

def test_a1_h264_amf_target_bitrate_uses_vbr_peak():
    args = _args("h264_amf", video_kbps=2000)
    assert _rc(args) == "vbr_peak"
    assert "vbr" not in _rc_values(args)


def test_a2_hevc_amf_target_bitrate_uses_vbr_peak():
    args = _args("hevc_amf", video_kbps=2000)
    assert _rc(args) == "vbr_peak"
    assert "vbr" not in _rc_values(args)


@pytest.mark.parametrize("encoder", ["h264_amf", "hevc_amf"])
def test_a3_bitrate_envelope_survives_unchanged(encoder):
    args = _args(encoder, video_kbps=2000)
    assert _value_after(args, "-b:v") == "2000k"
    assert _value_after(args, "-maxrate") == "2800k"   # 2000 * 1.4
    assert _value_after(args, "-bufsize") == "4000k"   # 2000 * 2


@pytest.mark.parametrize("encoder", ["h264_amf", "hevc_amf"])
def test_a4_amf_target_path_stays_single_pass(encoder):
    args = _args(encoder, video_kbps=2000)
    assert "-multipass" not in args
    assert "-pass" not in args
    assert "-passlogfile" not in args


@pytest.mark.parametrize("encoder", ["h264_amf", "hevc_amf"])
def test_a5_amf_quality_and_usage_args_survive(encoder):
    args = _args(encoder, video_kbps=2000, amf_quality="quality")
    assert _value_after(args, "-c:v") == encoder
    assert _value_after(args, "-quality") == "quality"
    assert _value_after(args, "-usage") == "transcoding"


def test_a6_amf_target_bitrate_keeps_scale_filter():
    args = _args("hevc_amf", video_kbps=1500, vf="scale=1280:-2")
    assert _value_after(args, "-vf") == "scale=1280:-2"
    assert _rc(args) == "vbr_peak"


# ══ GROUP B — AMF quality mode is untouched ══════════════════════════════════

@pytest.mark.parametrize("encoder", ["h264_amf", "hevc_amf"])
def test_b1_b2_amf_quality_mode_remains_cqp(encoder):
    args = _args(encoder, crf=28)
    assert _rc(args) == "cqp"
    assert _value_after(args, "-qp") == "28"
    assert "vbr_peak" not in args
    assert "-b:v" not in args


def test_b3_amf_quality_dial_mapping_is_unchanged():
    for dial in ("speed", "balanced", "quality"):
        quality_args = _args("hevc_amf", crf=24, amf_quality=dial)
        target_args = _args("hevc_amf", video_kbps=2000, amf_quality=dial)
        assert _value_after(quality_args, "-quality") == dial
        assert _value_after(target_args, "-quality") == dial


# ══ GROUP C — NVENC isolation ════════════════════════════════════════════════

@pytest.mark.parametrize("encoder", ["h264_nvenc", "hevc_nvenc"])
def test_c1_c2_nvenc_target_mode_keeps_bare_vbr(encoder):
    args = _args(encoder, video_kbps=2000)
    assert _rc(args) == "vbr"
    assert "vbr_peak" not in args


@pytest.mark.parametrize("encoder", ["h264_nvenc", "hevc_nvenc"])
def test_c3_nvenc_multipass_survives(encoder):
    args = _args(encoder, video_kbps=2000)
    assert _value_after(args, "-multipass") == "fullres"
    assert _value_after(args, "-b:v") == "2000k"


def test_c4_nvenc_quality_mode_keeps_bare_vbr():
    args = _args("hevc_nvenc", crf=26)
    assert _rc(args) == "vbr"
    assert _value_after(args, "-cq") == "26"


# ══ GROUP D — CPU isolation ══════════════════════════════════════════════════

@pytest.mark.parametrize("encoder", ["libx264", "libx265"])
def test_d1_d2_cpu_target_mode_carries_no_rate_control_token(encoder):
    args = _args(encoder, video_kbps=2000, use_two_pass=True, pass_num=2)
    assert _rc_values(args) == []
    assert _value_after(args, "-b:v") == "2000k"
    assert _value_after(args, "-pass") == "2"


@pytest.mark.parametrize("encoder", ["libx264", "libx265"])
def test_d3_cpu_quality_mode_still_uses_crf(encoder):
    args = _args(encoder, crf=23)
    assert _rc_values(args) == []
    assert _value_after(args, "-crf") == "23"


# ══ GROUP E — public H.264 AMF target-size path ══════════════════════════════

def test_e1_public_mp4_h264_amf_target_emits_vbr_peak(env):
    src, out_dir, fake = env

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    assert result["status"] == "ok"
    cmd = fake.mux_cmd
    assert _value_after(cmd, "-c:v") == "h264_amf"
    assert _rc(cmd) == "vbr_peak"
    assert _video_kbps_of(cmd) == _expected_kbps()


def test_e2_public_h264_amf_target_preserves_mp4_contracts(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2)

    result = _run(src, out_dir, fmt="MP4 (H.264)")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    maps = _maps(cmd)
    assert "0:a?" in maps
    assert "0:t?" not in maps                      # attachments are MKV-only
    assert _value_after(cmd, "-movflags") == "+faststart"
    assert len(fake.final_cmds) == 1               # one AMF encode attempt
    assert fake.pass1_cmds == []                   # never the software 2-pass


# ══ GROUP F — public H.265 AMF target-size path ══════════════════════════════

def test_f1_public_mp4_h265_amf_target_emits_vbr_peak(env):
    src, out_dir, fake = env

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _value_after(cmd, "-c:v") == "hevc_amf"
    assert _rc(cmd) == "vbr_peak"
    assert _video_kbps_of(cmd) == _expected_kbps()
    assert "0:a?" in _maps(cmd)
    assert "-multipass" not in cmd


# ══ GROUP G — target reduction ═══════════════════════════════════════════════

@pytest.mark.parametrize("fmt,encoder", [("MP4 (H.264)", "h264_amf"),
                                         ("MP4 (H.265)", "hevc_amf")])
def test_g1_g2_target_reduction_amf_emits_vbr_peak(env, fmt, encoder):
    src, out_dir, fake = env

    result = _run(src, out_dir, fmt=fmt, mode="Target reduction",
                  mode_value="50")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _value_after(cmd, "-c:v") == encoder
    assert _rc(cmd) == "vbr_peak"


def test_g3_target_reduction_math_is_unchanged(env):
    src, out_dir, fake = env

    _run(src, out_dir, fmt="MP4 (H.265)", mode="Target reduction",
         mode_value="50")

    expected = calc_video_bitrate_kbps(
        int(SRC_BYTES * 50 / 100.0), DURATION, 128, 1)
    assert _video_kbps_of(fake.mux_cmd) == expected


# ══ GROUP H — multi-audio accounting survives ════════════════════════════════

def test_h1_three_audio_tracks_reserve_three_times_the_bitrate(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3)

    result = _run(src, out_dir, fmt="MP4 (H.265)", audio_kbps="128")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _rc(cmd) == "vbr_peak"
    assert _video_kbps_of(cmd) == _expected_kbps(audio_count=3)
    # 3 x 128 kbps reserved, not 128 shared.
    assert (_expected_kbps(audio_count=1) - _expected_kbps(audio_count=3)
            == 256)
    assert "0:a?" in _maps(cmd)
    assert _value_after(cmd, "-b:a") == "128k"


# ══ GROUP I — zero audio and impossible targets ══════════════════════════════

def test_i1_zero_audio_amf_target_reserves_nothing(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 0
    _inventory(monkeypatch, audio_count=0)

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _rc(cmd) == "vbr_peak"
    assert _video_kbps_of(cmd) == _expected_kbps(audio_count=0)


def test_i2_impossible_multi_audio_target_never_reaches_the_encoder(
        env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 6
    _inventory(monkeypatch, audio_count=6)

    result = _run(src, out_dir, fmt="MP4 (H.265)", mode_value="0.4",
                  audio_kbps="192")

    assert result["status"] == "error"
    assert "audio" in result["msg"]
    assert fake.final_cmds == []


# ══ GROUP J — Matroska direct ════════════════════════════════════════════════

def test_j1_mkv_h265_amf_target_keeps_matroska_fidelity(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 2
    _inventory(monkeypatch, audio_count=2,
               subtitles=[{"index": 3, "codec_name": "subrip",
                           "codec_type": "subtitle",
                           "tags": {"language": "eng"}}])

    result = _run(src, out_dir, fmt="MKV (H.265)")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _muxer_of(cmd) == "matroska"
    assert _value_after(cmd, "-c:v") == "hevc_amf"
    assert _rc(cmd) == "vbr_peak"
    maps = _maps(cmd)
    assert "0:a?" in maps
    assert "0:t?" in maps
    assert _value_after(cmd, "-c:t") == "copy"


# ══ GROUP K — MP4 -> MKV fallback with AMF ═══════════════════════════════════

def test_k1_fallback_attempts_both_containers_with_vbr_peak(env, monkeypatch):
    src, out_dir, fake = env
    fake.audio_streams = 3
    _inventory(monkeypatch, audio_count=3, calls=(probe_calls := []))
    fake.finals = [(1, "Could not write header: Invalid argument")]

    result = _run(src, out_dir, fmt="MP4 (H.265)")

    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert fake.final_muxers == ["mp4", "matroska"]
    assert fake.final_encoders == ["hevc_amf", "hevc_amf"]
    assert [_rc(c) for c in fake.final_cmds] == ["vbr_peak", "vbr_peak"]
    # Same budget both times, and the inventory was probed once for the file.
    kbps = {_video_kbps_of(c) for c in fake.final_cmds}
    assert kbps == {_expected_kbps(audio_count=3)}
    assert len(probe_calls) == 1
    assert "0:a?" in _maps(fake.final_cmds[1])
    assert "0:t?" in _maps(fake.final_cmds[1])


# ══ GROUP L — automatic encoder selection ════════════════════════════════════

def test_l1_automatic_selection_picks_amf_with_vbr_peak(env, monkeypatch):
    src, out_dir, fake = env
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": False)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": True)

    result = _run(src, out_dir, fmt="MP4 (H.265)", encoder_pref="auto")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _value_after(cmd, "-c:v") == "hevc_amf"
    assert _rc(cmd) == "vbr_peak"


def test_l2_automatic_selection_still_prefers_nvenc_and_its_own_vbr(
        env, monkeypatch):
    src, out_dir, fake = env
    monkeypatch.setattr(compressor, "nvenc_available", lambda e="hevc_nvenc": True)
    monkeypatch.setattr(compressor, "amf_available", lambda e="hevc_amf": True)

    result = _run(src, out_dir, fmt="MP4 (H.265)", encoder_pref="auto")

    cmd = fake.mux_cmd
    assert result["status"] == "ok"
    assert _value_after(cmd, "-c:v") == "hevc_nvenc"
    assert _rc(cmd) == "vbr"
    assert _value_after(cmd, "-multipass") == "fullres"


# ══ GROUP M — no new probing, no new invocations ═════════════════════════════

def test_m1_production_never_asks_ffmpeg_for_encoder_help(env, monkeypatch):
    src, out_dir, fake = env
    seen: list[list] = []
    real_run = compressor.run_ffmpeg

    def recording(cmd, *a, **kw):
        seen.append(list(cmd))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(compressor, "run_ffmpeg", recording)
    calls: list[list] = []
    monkeypatch.setattr(compressor.subprocess, "run",
                        lambda *a, **kw: calls.append(list(a[0])))

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"

    for cmd in seen + calls:
        joined = " ".join(str(t) for t in cmd)
        assert "-h" not in cmd
        assert "encoder=" not in joined


def test_m2_m3_m4_amf_target_job_probes_each_thing_once(env, monkeypatch):
    src, out_dir, fake = env
    probe_calls: list[Path] = []
    _inventory(monkeypatch, audio_count=2, calls=probe_calls)
    fake.audio_streams = 2
    amf_calls: list[str] = []
    duration_calls: list[Path] = []
    monkeypatch.setattr(compressor, "amf_available",
                        lambda e="hevc_amf": (amf_calls.append(e) or True))
    monkeypatch.setattr(compressor, "ffprobe_duration",
                        lambda p: (duration_calls.append(Path(p)) or DURATION))

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"

    assert len(probe_calls) == 1
    assert len(duration_calls) == 1
    assert amf_calls == ["hevc_amf"]


def test_m5_successful_amf_target_encode_is_one_invocation(env):
    src, out_dir, fake = env

    assert _run(src, out_dir, fmt="MP4 (H.265)")["status"] == "ok"

    assert len(fake.final_cmds) == 1
    assert fake.pass1_cmds == []
