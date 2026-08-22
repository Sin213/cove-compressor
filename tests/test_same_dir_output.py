"""Opt-in: output to the same directory as the input file.

Exercises output_dir_for (the per-file directory resolver) plus the
unique_path / reserve_output collision behaviour that keeps results landing
beside their originals safe. No ffmpeg, no real media, no Qt: every path
lives under pytest's tmp_path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor.compressor import (  # noqa: E402
    output_dir_for, reserve_output, unique_path,
)


def _src(tmp_path: Path, name="in.png") -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"source-bytes")
    return p


# ── Group 1 - opt-in resolution ──────────────────────────────────────────────

def test_same_as_source_uses_input_parent(tmp_path):
    shared = tmp_path / "shared"
    src = _src(tmp_path / "nested")
    assert output_dir_for(src, shared, True) == src.parent


def test_off_uses_shared_dir(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    src = _src(tmp_path)
    assert output_dir_for(src, shared, False) == shared


def test_default_argument_is_shared_dir(tmp_path):
    shared = tmp_path / "shared"
    src = _src(tmp_path)
    assert output_dir_for(src, shared) == shared


def test_recursive_folder_drops_resolve_per_file(tmp_path):
    shared = tmp_path / "shared"
    a = _src(tmp_path / "a", "a.png")
    b = _src(tmp_path / "a" / "sub", "b.png")
    c = _src(tmp_path / "b", "c.png")
    assert {output_dir_for(f, shared, True) for f in (a, b, c)} == {
        tmp_path / "a", tmp_path / "a" / "sub", tmp_path / "b",
    }


# ── Group 2 - collisions beside originals stay safe ─────────────────────────

def test_unique_path_numbers_beside_existing_sibling(tmp_path):
    src = _src(tmp_path, "photo.png")
    existing_out = src.with_suffix(".webp")
    existing_out.write_bytes(b"earlier-result")

    claimed = unique_path(existing_out)

    assert claimed != existing_out
    assert claimed.parent == src.parent
    assert claimed.name == "photo_1.webp"


def test_reserve_output_never_claims_existing_source_name(tmp_path):
    src = _src(tmp_path, "clip.mkv")

    first_output, first_tmp = reserve_output(src.with_suffix(".mp4"))
    second_output, second_tmp = reserve_output(src.with_suffix(".mp4"))

    assert first_output == tmp_path / "clip.mp4"
    assert second_output.name == "clip_1.mp4"
    for out in (first_output, second_output):
        assert out.exists() and out != src
        out.unlink()
    assert src.exists()
