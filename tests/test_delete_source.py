"""Tab 2a - safe post-success source deletion.

These tests exercise the destructive safety gate directly. No ffmpeg, no real
media, no user files: every path lives under pytest's tmp_path.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402
from cove_compressor.compressor import delete_source_if_eligible  # noqa: E402


def _pair(tmp_path: Path, src_name="in.png", out_name="out.webp",
          src_data=b"source-bytes", out_data=b"output-bytes"):
    src = tmp_path / src_name
    out = tmp_path / out_name
    src.write_bytes(src_data)
    out.write_bytes(out_data)
    return src, out


def _ok(src: Path, out: Path, **extra) -> dict:
    r = {"file": src, "output": out, "status": "ok",
         "original": 100, "new": 50}
    r.update(extra)
    return r


# ── Group 1 - opt-in gate ────────────────────────────────────────────────────

def test_default_argument_never_deletes(tmp_path):
    src, out = _pair(tmp_path)
    result = delete_source_if_eligible(_ok(src, out))
    assert src.exists()
    assert out.exists()
    assert "source_deleted" not in result


def test_disabled_never_deletes(tmp_path):
    src, out = _pair(tmp_path)
    result = delete_source_if_eligible(_ok(src, out), enabled=False)
    assert src.exists()
    assert "source_deleted" not in result


def test_enabled_ok_deletes_source_only(tmp_path):
    src, out = _pair(tmp_path)
    result = delete_source_if_eligible(_ok(src, out), enabled=True)
    assert not src.exists()
    assert out.exists()
    assert out.read_bytes() == b"output-bytes"
    assert result["source_deleted"] is True
    assert result["status"] == "ok"
    assert "delete_error" not in result


# ── Group 2 - status gate ───────────────────────────────────────────────────

@pytest.mark.parametrize("status,msg", [
    ("skipped", "compression would increase size"),
    ("error", "ffmpeg failed"),
    ("error", "cancelled"),
    ("timeout", "no progress"),
])
def test_non_ok_status_never_deletes(tmp_path, status, msg):
    src, out = _pair(tmp_path)
    result = {"file": src, "output": out, "status": status, "msg": msg}
    result = delete_source_if_eligible(result, enabled=True)
    assert src.exists()
    assert out.exists()
    assert "source_deleted" not in result


# ── Group 3 - output validity ───────────────────────────────────────────────

def test_missing_output_key_never_deletes(tmp_path):
    src, _ = _pair(tmp_path)
    result = delete_source_if_eligible(
        {"file": src, "status": "ok", "original": 100, "new": 50}, enabled=True)
    assert src.exists()
    assert "source_deleted" not in result


def test_output_path_absent_never_deletes(tmp_path):
    src, out = _pair(tmp_path)
    out.unlink()
    delete_source_if_eligible(_ok(src, out), enabled=True)
    assert src.exists()


def test_zero_byte_output_never_deletes(tmp_path):
    src, out = _pair(tmp_path, out_data=b"")
    delete_source_if_eligible(_ok(src, out), enabled=True)
    assert src.exists()


def test_directory_output_never_deletes(tmp_path):
    src, _ = _pair(tmp_path)
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    delete_source_if_eligible(_ok(src, out_dir), enabled=True)
    assert src.exists()
    assert out_dir.is_dir()


# ── Group 4 - source validity / alias safety ────────────────────────────────

def test_missing_source_never_deletes(tmp_path):
    src, out = _pair(tmp_path)
    src.unlink()
    result = delete_source_if_eligible(_ok(src, out), enabled=True)
    assert out.exists()
    assert result.get("source_deleted") is not True


def test_directory_source_never_deleted(tmp_path):
    src_dir = tmp_path / "queued_folder"
    src_dir.mkdir()
    (src_dir / "keep.txt").write_bytes(b"keep")
    out = tmp_path / "out.webp"
    out.write_bytes(b"output-bytes")
    delete_source_if_eligible(_ok(src_dir, out), enabled=True)
    assert src_dir.is_dir()
    assert (src_dir / "keep.txt").exists()


def test_identical_literal_path_never_deletes(tmp_path):
    src = tmp_path / "same.png"
    src.write_bytes(b"data")
    delete_source_if_eligible(_ok(src, src), enabled=True)
    assert src.exists()


def test_symlink_alias_never_deletes(tmp_path):
    real = tmp_path / "real.png"
    real.write_bytes(b"data")
    link = tmp_path / "alias.png"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this filesystem")
    delete_source_if_eligible(_ok(real, link), enabled=True)
    assert real.exists()


def test_hardlink_alias_never_deletes(tmp_path):
    real = tmp_path / "real.png"
    real.write_bytes(b"data")
    link = tmp_path / "hard.png"
    try:
        os.link(real, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links unsupported on this filesystem")
    delete_source_if_eligible(_ok(real, link), enabled=True)
    assert real.exists()
    assert link.exists()


def test_stat_failure_fails_closed(tmp_path):
    src, out = _pair(tmp_path)

    def boom(*a, **kw):
        raise PermissionError("stat denied")

    original_stat = compressor.os.stat

    def fake_stat(path, *a, **kw):
        if str(path) == str(src):
            boom()
        return original_stat(path, *a, **kw)

    compressor.os.stat = fake_stat
    try:
        result = delete_source_if_eligible(_ok(src, out), enabled=True)
    finally:
        compressor.os.stat = original_stat

    assert src.exists()
    assert out.exists()
    assert result["status"] == "ok"
    assert result["source_deleted"] is False
    assert result["delete_error"]


# ── Group 5 - subtitle safety seam (forward contract for Tab 2b) ────────────

def test_subtitles_failed_blocks_deletion(tmp_path):
    src, out = _pair(tmp_path)
    result = delete_source_if_eligible(
        _ok(src, out, subtitles_failed=True), enabled=True)
    assert src.exists()
    assert "source_deleted" not in result


def test_subtitles_failed_false_still_deletes(tmp_path):
    src, out = _pair(tmp_path)
    result = delete_source_if_eligible(
        _ok(src, out, subtitles_failed=False), enabled=True)
    assert not src.exists()
    assert result["source_deleted"] is True


# ── Group 6 - deletion failure ──────────────────────────────────────────────

def test_unlink_failure_keeps_output_and_status(tmp_path, monkeypatch):
    src, out = _pair(tmp_path)

    def deny(path, *a, **kw):
        raise PermissionError("unlink denied")

    monkeypatch.setattr(compressor.os, "unlink", deny)
    result = delete_source_if_eligible(_ok(src, out), enabled=True)

    assert src.exists()
    assert out.exists()
    assert result["status"] == "ok"
    assert result["source_deleted"] is False
    assert "unlink denied" in result["delete_error"]
    assert "Traceback" not in result["delete_error"]


# ── Group 7 - per-file independence ─────────────────────────────────────────

def test_batch_results_decided_independently(tmp_path):
    a_src, a_out = _pair(tmp_path, "a.png", "a.webp")
    b_src, b_out = _pair(tmp_path, "b.png", "b.webp")
    c_src, c_out = _pair(tmp_path, "c.png", "c.webp")

    results = [
        _ok(a_src, a_out),
        {"file": b_src, "output": b_out, "status": "skipped", "msg": "no gain"},
        {"file": c_src, "output": c_out, "status": "error", "msg": "boom"},
    ]
    for r in results:
        delete_source_if_eligible(r, enabled=True)

    assert not a_src.exists()
    assert b_src.exists()
    assert c_src.exists()
    assert a_out.exists() and b_out.exists() and c_out.exists()


def test_helper_returns_same_result_object(tmp_path):
    src, out = _pair(tmp_path)
    r = _ok(src, out)
    assert delete_source_if_eligible(r, enabled=True) is r
