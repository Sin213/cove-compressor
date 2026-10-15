"""Tab 2b - English subtitle discovery + sidecar extraction.

Everything here is deterministic: ffprobe and ffmpeg are faked, no real media
is decoded, and every path lives under pytest's tmp_path. The one thing these
tests never fake is the Tab 2a deletion gate - the combined safety cases call
the real `delete_source_if_eligible`.
"""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import (  # noqa: E402
    SubtitleProbeError,
    build_subtitle_extract_cmd,
    compress_video,
    delete_source_if_eligible,
    ffprobe_stream_inventory,
    is_english_subtitle_stream,
    subtitle_sidecar_name,
    subtitle_sidecar_target,
)


# ── fixtures / fakes ────────────────────────────────────────────────────────

def _stream(index, codec_name="subrip", language=None, title=None,
            forced=0, hearing_impaired=0, default=0):
    tags = {}
    if language is not None:
        tags["language"] = language
    if title is not None:
        tags["title"] = title
    return {
        "index": index,
        "codec_name": codec_name,
        "tags": tags,
        "disposition": {"default": default, "forced": forced,
                        "hearing_impaired": hearing_impaired},
    }


def _has_pair(cmd, a, b) -> bool:
    return any(cmd[i] == a and cmd[i + 1] == b for i in range(len(cmd) - 1))


class FakeFfmpeg:
    """Stands in for `run_ffmpeg`. Subtitle extraction invocations are the ones
    that strip both other stream kinds — `-vn` *and* `-an` together, which is
    the shape only `build_subtitle_extract_cmd` produces. Everything else is a
    video encode. (`-map` alone is not a discriminator: normal MP4 encodes now
    carry explicit stream maps too, and two-pass pass 1 carries `-an`.) Both
    write their output file, exactly like the real thing, so callers see a
    success *shape*."""

    @staticmethod
    def _is_subtitle_extraction(cmd) -> bool:
        return "-vn" in cmd and "-an" in cmd and "-map" in cmd

    def __init__(self, encode_bytes=b"v" * 10, sub_bytes=b"1\nhello\n",
                 sub_rc=0, encode_rc=0, sub_rc_by_index=None,
                 sub_writes=True):
        self.encode_bytes = encode_bytes
        self.sub_bytes = sub_bytes
        self.sub_rc = sub_rc
        self.encode_rc = encode_rc
        self.sub_rc_by_index = sub_rc_by_index or {}
        self.sub_writes = sub_writes
        self.subtitle_cmds: list[list] = []
        self.encode_cmds: list[list] = []

    def __call__(self, cmd, cancel_flag, duration=None,
                 on_progress=None, on_start=None):
        out = Path(cmd[-1])
        if self._is_subtitle_extraction(cmd):
            self.subtitle_cmds.append(list(cmd))
            idx = cmd[cmd.index("-map") + 1].split(":")[-1]
            rc = self.sub_rc_by_index.get(int(idx), self.sub_rc)
            if rc == 0 and self.sub_writes:
                out.write_bytes(self.sub_bytes)
            return rc, "" if rc == 0 else "subtitle extraction failed"
        self.encode_cmds.append(list(cmd))
        if self.encode_rc == 0:
            out.write_bytes(self.encode_bytes)
        return self.encode_rc, "" if self.encode_rc == 0 else "encode failed"


@pytest.fixture
def video_env(tmp_path, monkeypatch):
    """A source file, an output dir, and a faked encoder stack."""
    src = tmp_path / "Episode 01.mkv"
    src.write_bytes(b"s" * 100)
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
    return src, out_dir, fake


def _probe(monkeypatch, streams, calls=None, audio_count=1):
    """Fake the one stream inventory probe; subtitles are what this suite reads."""
    def fake(path):
        if calls is not None:
            calls.append(Path(path))
        return compressor.StreamInventory(
            subtitles=list(streams), audio_count=audio_count)
    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", fake)
    return fake


def _run(src, out_dir, **kw):
    return compress_video(
        src, out_dir, "Quality preset", "Balanced", "MKV (H.265)",
        None, "192", threading.Event(), **kw)


# ── Group 1 - setting OFF performs zero extraction ──────────────────────────
#
# These run as MKV, which maps its streams explicitly and so classifies
# subtitles once for that purpose alone (see `tests/test_mkv_attachments.py`).
# What must stay at zero with the setting off is the *extraction* work: no
# sidecar invocation, no sidecar keys, no `subtitles_failed`.

def test_default_argument_never_extracts(video_env, monkeypatch):
    src, out_dir, fake = video_env
    calls: list[Path] = []
    _probe(monkeypatch, [_stream(2, language="eng")], calls)

    result = _run(src, out_dir)

    assert result["status"] == "ok"
    assert len(calls) == 1          # the mapping probe, never a second one
    assert fake.subtitle_cmds == []
    assert "subtitles_extracted" not in result
    assert not result.get("subtitles_failed")


def test_explicit_false_never_extracts(video_env, monkeypatch):
    src, out_dir, fake = video_env
    calls: list[Path] = []
    _probe(monkeypatch, [_stream(2, language="eng")], calls)

    result = _run(src, out_dir, extract_english_subtitles=False)

    assert result["status"] == "ok"
    assert len(calls) == 1
    assert fake.subtitle_cmds == []


# ── Group 2 - English detection policy ──────────────────────────────────────

@pytest.mark.parametrize("stream,expected", [
    (_stream(0, language="eng"), True),
    (_stream(0, language="en"), True),
    (_stream(0, language="en-US"), True),
    (_stream(0, language="en_GB"), True),
    (_stream(0, language="ENG"), True),
    (_stream(0, language="  eng  "), True),
    (_stream(0, language="fra"), False),
    (_stream(0, language="jpn"), False),
    (_stream(0, language="spa"), False),
    (_stream(0, language="und", title="English"), True),
    (_stream(0, title="English SDH"), True),
    (_stream(0, language="", title="Signs & Songs - English"), True),
    (_stream(0, language="und", title="Commentary"), False),
    (_stream(0), False),
    (_stream(0, language="und"), False),
    # A non-English language tag is authoritative: no title rescue.
    (_stream(0, language="fra", title="English"), False),
    # Being the default track is not evidence of English.
    (_stream(0, language="und", default=1), False),
])
def test_english_detection(stream, expected):
    assert is_english_subtitle_stream(stream) is expected


# ── Group 3 - absolute stream index (mandatory) ─────────────────────────────

def test_extraction_uses_absolute_stream_index(video_env, monkeypatch):
    src, out_dir, fake = video_env
    # 0 video, 1+2 audio, 5 is the only (English) subtitle stream.
    _probe(monkeypatch, [_stream(5, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert len(fake.subtitle_cmds) == 1
    cmd = fake.subtitle_cmds[0]
    assert _has_pair(cmd, "-map", "0:5")
    assert not _has_pair(cmd, "-map", "0:0")
    assert "0:s:0" not in cmd


def test_build_cmd_maps_absolute_index_only():
    cmd = build_subtitle_extract_cmd(
        Path("in.mkv"), 5, "subrip", Path("out.srt"))
    assert cmd.count("-map") == 1
    assert _has_pair(cmd, "-map", "0:5")
    assert "-vn" in cmd and "-an" in cmd


# ── Group 4 - every English track is preserved ──────────────────────────────

def test_all_english_tracks_extracted_french_ignored(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [
        _stream(2, language="eng"),
        _stream(3, language="eng", forced=1),
        _stream(4, language="eng", hearing_impaired=1),
        _stream(5, language="fra"),
    ])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert len(fake.subtitle_cmds) == 3
    mapped = {c[c.index("-map") + 1] for c in fake.subtitle_cmds}
    assert mapped == {"0:2", "0:3", "0:4"}
    assert "0:5" not in mapped

    dests = sorted(p.name for p in result["subtitles_extracted"])
    assert dests == ["Episode 01.eng.forced.srt",
                     "Episode 01.eng.sdh.srt",
                     "Episode 01.eng.srt"]
    assert len(set(dests)) == 3
    for p in result["subtitles_extracted"]:
        assert p.is_file() and p.stat().st_size > 0
    assert not result.get("subtitles_failed")


def test_identical_english_tracks_get_distinct_sidecars(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [
        _stream(2, language="eng"),
        _stream(3, language="eng"),
    ])

    result = _run(src, out_dir, extract_english_subtitles=True)

    paths = result["subtitles_extracted"]
    assert len(paths) == 2
    assert len({str(p) for p in paths}) == 2
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


# ── Group 5 - codec / sidecar format policy ─────────────────────────────────

@pytest.mark.parametrize("codec,ext,scodec,muxer", [
    ("subrip", ".srt", "srt", "srt"),
    ("mov_text", ".srt", "srt", "srt"),
    ("text", ".srt", "srt", "srt"),
    ("webvtt", ".srt", "srt", "srt"),
    ("ass", ".ass", "ass", "ass"),
    ("ssa", ".ass", "ass", "ass"),
    ("hdmv_pgs_subtitle", ".mks", "copy", "matroska"),
    ("dvd_subtitle", ".mks", "copy", "matroska"),
    ("dvb_subtitle", ".mks", "copy", "matroska"),
    ("some_future_codec", ".mks", "copy", "matroska"),
    (None, ".mks", "copy", "matroska"),
])
def test_sidecar_target_policy(codec, ext, scodec, muxer):
    got_ext, codec_args, got_muxer = subtitle_sidecar_target(codec)
    assert got_ext == ext
    assert codec_args == ["-c:s", scodec]
    assert got_muxer == muxer


@pytest.mark.parametrize("codec,ext,scodec,muxer", [
    ("subrip", ".srt", "srt", "srt"),
    ("ass", ".ass", "ass", "ass"),
    ("hdmv_pgs_subtitle", ".mks", "copy", "matroska"),
])
def test_extract_cmd_states_codec_and_muxer_explicitly(codec, ext, scodec, muxer):
    out = Path(f"x{ext}")
    cmd = build_subtitle_extract_cmd(Path("in.mkv"), 3, codec, out)
    assert _has_pair(cmd, "-c:s", scodec)
    # The muxer is named outright, never inferred from the filename.
    assert _has_pair(cmd, "-f", muxer)
    assert cmd[-1] == str(out)


def test_pgs_stream_lands_in_mks_sidecar(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [_stream(4, codec_name="hdmv_pgs_subtitle",
                                 language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert [p.name for p in result["subtitles_extracted"]] == \
        ["Episode 01.eng.mks"]
    cmd = fake.subtitle_cmds[0]
    assert _has_pair(cmd, "-c:s", "copy")
    assert _has_pair(cmd, "-f", "matroska")


def test_ass_stream_preserves_ass_formatting(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [_stream(4, codec_name="ass", language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert [p.name for p in result["subtitles_extracted"]] == \
        ["Episode 01.eng.ass"]
    assert _has_pair(fake.subtitle_cmds[0], "-c:s", "ass")


def test_sidecar_name_markers():
    assert subtitle_sidecar_name("Ep", _stream(1), ".srt") == "Ep.eng.srt"
    assert subtitle_sidecar_name("Ep", _stream(1, forced=1), ".srt") == \
        "Ep.eng.forced.srt"
    assert subtitle_sidecar_name("Ep", _stream(1, hearing_impaired=1), ".ass") == \
        "Ep.eng.sdh.ass"
    assert subtitle_sidecar_name(
        "Ep", _stream(1, forced=1, hearing_impaired=1), ".mks") == \
        "Ep.eng.forced.sdh.mks"


def test_sidecar_name_ignores_title_text(video_env, monkeypatch):
    src, out_dir, _ = video_env
    _probe(monkeypatch, [_stream(2, language="eng",
                                 title="../../evil name; rm -rf /")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    name = result["subtitles_extracted"][0].name
    assert name == "Episode 01.eng.srt"
    assert "evil" not in name


# ── Group 6 - collision safety ──────────────────────────────────────────────

def test_existing_sidecar_is_never_overwritten(video_env, monkeypatch):
    src, out_dir, _ = video_env
    squatter = out_dir / "Episode 01.eng.srt"
    squatter.write_bytes(b"PRE-EXISTING USER SUBTITLES")
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert squatter.read_bytes() == b"PRE-EXISTING USER SUBTITLES"
    produced = result["subtitles_extracted"]
    assert len(produced) == 1
    assert produced[0] != squatter
    assert produced[0].is_file() and produced[0].stat().st_size > 0
    assert not result.get("subtitles_failed")


def test_second_collision_also_avoided(video_env, monkeypatch):
    src, out_dir, _ = video_env
    first = out_dir / "Episode 01.eng.srt"
    first.write_bytes(b"USER ONE")
    second = out_dir / "Episode 01.eng_1.srt"
    second.write_bytes(b"USER TWO")
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert first.read_bytes() == b"USER ONE"
    assert second.read_bytes() == b"USER TWO"
    produced = result["subtitles_extracted"][0]
    assert produced not in (first, second)
    assert produced.stat().st_size > 0


# ── Group 7 - no English subtitles is not a failure ─────────────────────────

def test_no_english_streams_is_not_a_failure(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [
        _stream(2, language="fra"),
        _stream(3, language="jpn"),
        _stream(4, language="spa"),
    ])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert fake.subtitle_cmds == []
    assert result["subtitles_extracted"] == []
    assert not result.get("subtitles_failed")

    delete_source_if_eligible(result, enabled=True)
    assert result["source_deleted"] is True
    assert not src.exists()
    assert Path(result["output"]).exists()


def test_zero_subtitle_streams_is_not_a_failure(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert fake.subtitle_cmds == []
    assert not result.get("subtitles_failed")


# ── Group 8 - probe failure fails closed ────────────────────────────────────

def test_probe_failure_blocks_deletion(video_env, monkeypatch):
    src, out_dir, fake = video_env

    def boom(path):
        raise SubtitleProbeError("ffprobe crashed")

    monkeypatch.setattr(compressor, "ffprobe_stream_inventory", boom)

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True
    assert fake.subtitle_cmds == []
    assert result.get("subtitles_extracted") == []
    assert any("probe" in e for e in result["subtitle_errors"])

    delete_source_if_eligible(result, enabled=True)
    assert src.exists()
    assert Path(result["output"]).exists()
    assert "source_deleted" not in result


def test_probe_raises_on_bad_json(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    with pytest.raises(SubtitleProbeError):
        ffprobe_stream_inventory(tmp_path / "x.mkv")


def test_probe_raises_on_nonzero_exit(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    with pytest.raises(SubtitleProbeError):
        ffprobe_stream_inventory(tmp_path / "x.mkv")


def test_probe_raises_when_ffprobe_missing(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    with pytest.raises(SubtitleProbeError):
        ffprobe_stream_inventory(tmp_path / "x.mkv")


def test_probe_returns_absolute_indexes(monkeypatch, tmp_path):
    payload = {"streams": [
        {"index": 5, "codec_name": "subrip", "codec_type": "subtitle",
         "tags": {"language": "eng"},
         "disposition": {"forced": 0, "hearing_impaired": 0}},
    ]}

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    inventory = ffprobe_stream_inventory(tmp_path / "x.mkv")

    assert [s["index"] for s in inventory.subtitles] == [5]
    # The probe inventories every stream now - audio has to be counted from
    # the same answer - so the classification happens here, not in ffprobe's
    # stream selector.
    assert "-select_streams" not in seen["cmd"]
    assert _has_pair(seen["cmd"], "-of", "json")


def test_probe_empty_stream_list_is_not_an_error(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    assert ffprobe_stream_inventory(tmp_path / "x.mkv") == ([], 0)


def test_probe_rejects_stream_without_index(monkeypatch, tmp_path):
    payload = {"streams": [{"codec_name": "subrip", "codec_type": "subtitle"}]}

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(compressor.subprocess, "run", fake_run)
    with pytest.raises(SubtitleProbeError):
        ffprobe_stream_inventory(tmp_path / "x.mkv")


# ── Group 9 - extraction failure fails closed ───────────────────────────────

def test_extraction_failure_blocks_deletion(video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.sub_rc = 1
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True
    assert result["subtitles_extracted"] == []
    assert result["subtitle_errors"]
    out = Path(result["output"])
    assert out.is_file()
    # No half-written sidecar was left in the output folder.
    assert [p.name for p in out_dir.iterdir()] == [out.name]

    delete_source_if_eligible(result, enabled=True)
    assert src.exists()
    assert out.exists()


# ── Group 10 - partial multi-track failure ──────────────────────────────────

def test_partial_failure_keeps_good_sidecars_and_blocks_deletion(
        video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.sub_rc_by_index = {3: 1}
    _probe(monkeypatch, [
        _stream(2, language="eng"),
        _stream(3, language="eng", forced=1),
    ])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True
    kept = result["subtitles_extracted"]
    assert [p.name for p in kept] == ["Episode 01.eng.srt"]
    assert kept[0].is_file() and kept[0].stat().st_size > 0
    assert not (out_dir / "Episode 01.eng.forced.srt").exists()

    delete_source_if_eligible(result, enabled=True)
    assert src.exists()
    assert Path(result["output"]).exists()
    assert kept[0].exists()


# ── Group 11 - non-success conversion cleans subtitle temps ─────────────────

def test_skipped_conversion_leaves_no_sidecars_or_temps(video_env, monkeypatch):
    src, out_dir, fake = video_env
    # Quality-preset mode skips when the encode grows the file.
    fake.encode_bytes = b"v" * 5000
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "skipped"
    assert fake.subtitle_cmds, "extraction must run before the encode"
    assert list(out_dir.iterdir()) == []
    assert src.exists()


def test_failed_encode_leaves_no_sidecars_or_temps(video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.encode_rc = 1
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert list(out_dir.iterdir()) == []
    assert src.exists()


def test_cancelled_extraction_does_not_start_an_encode(video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.sub_rc = -2
    _probe(monkeypatch, [_stream(2, language="eng")])

    cancel = threading.Event()

    def cancelling(cmd, cancel_flag, **kw):
        cancel_flag.set()
        return fake(cmd, cancel_flag, **kw)

    monkeypatch.setattr(compressor, "run_ffmpeg", cancelling)

    result = compress_video(
        src, out_dir, "Quality preset", "Balanced", "MKV (H.265)",
        None, "192", cancel, extract_english_subtitles=True)

    assert result["status"] == "error"
    assert fake.encode_cmds == []
    assert list(out_dir.iterdir()) == []
    assert src.exists()


# ── Group 12 - rc==0 is not proof of preservation ───────────────────────────

def test_missing_extraction_output_is_a_failure(video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.sub_writes = False  # rc 0, but nothing on disk
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True
    assert result["subtitles_extracted"] == []


def test_empty_extraction_output_is_a_failure(video_env, monkeypatch):
    src, out_dir, fake = video_env
    fake.sub_bytes = b""  # rc 0, zero-byte sidecar
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert result["subtitles_failed"] is True
    assert result["subtitles_extracted"] == []
    assert not (out_dir / "Episode 01.eng.srt").exists()


# ── Group 13 - combined success contract ────────────────────────────────────

def test_success_extraction_plus_deletion(video_env, monkeypatch):
    src, out_dir, fake = video_env
    _probe(monkeypatch, [
        _stream(2, language="eng"),
        _stream(3, language="fra"),
    ])

    result = _run(src, out_dir, extract_english_subtitles=True)

    assert result["status"] == "ok"
    assert not result.get("subtitles_failed")
    sidecar = result["subtitles_extracted"][0]
    assert sidecar.name == "Episode 01.eng.srt"
    assert sidecar.stat().st_size > 0

    delete_source_if_eligible(result, enabled=True)

    assert result["source_deleted"] is True
    assert not src.exists()
    assert Path(result["output"]).is_file()
    assert sidecar.is_file() and sidecar.stat().st_size > 0


def test_success_extraction_with_deletion_off_keeps_source(video_env, monkeypatch):
    src, out_dir, _ = video_env
    _probe(monkeypatch, [_stream(2, language="eng")])

    result = _run(src, out_dir, extract_english_subtitles=True)
    delete_source_if_eligible(result)

    assert src.exists()
    assert result["subtitles_extracted"][0].is_file()
    assert "source_deleted" not in result


def test_subtitles_off_with_delete_on_is_unchanged(video_env, monkeypatch):
    src, out_dir, fake = video_env
    calls: list[Path] = []
    _probe(monkeypatch, [_stream(2, language="eng")], calls)

    result = _run(src, out_dir)
    delete_source_if_eligible(result, enabled=True)

    assert len(calls) == 1          # mapping only; extraction stayed off
    assert fake.subtitle_cmds == []
    assert result["source_deleted"] is True
    assert not src.exists()
