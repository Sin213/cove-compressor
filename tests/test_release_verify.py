"""The release verifier is the last thing between a broken build and a release.

`scripts/release_verify.py` is run by both build jobs and by the final
assembly job. It answers three questions about the files a release is made of:

    assets   - is this exactly the set of binaries + `.sha256` files promised
               for this version, and does every digest match its binary?
    deb      - is the .deb a real package of this version that carries both
               ffmpeg and ffprobe?
    tree     - does an unpacked AppImage / installed Setup tree carry both
               tools, and do they actually start?
    listing  - does the portable exe's PyInstaller archive carry both tools?

A verifier that approves a broken release is worse than none, so every
refusal is pinned here, one cause per test, against the exact failure it
exists to catch.
"""
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import release_verify as rv  # noqa: E402

V = "2.6.0"
LINUX = [f"Cove-Compressor-{V}-x86_64.AppImage", f"cove-compressor_{V}_amd64.deb"]
WINDOWS = [f"cove-compressor-{V}-Setup.exe", f"cove-compressor-{V}-Portable.exe"]


def _sidecar(directory: Path, name: str, payload: bytes) -> None:
    (directory / name).write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (directory / f"{name}.sha256").write_text(f"{digest}  {name}\n")


def _assets(directory: Path, names) -> None:
    for n in names:
        _sidecar(directory, n, b"payload of " + n.encode())


def _ar(members) -> bytes:
    out = b"!<arch>\n"
    for name, data in members:
        header = (
            f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(data):<10}"
        ).encode() + b"`\n"
        out += header + data + (b"\n" if len(data) % 2 else b"")
    return out


def _tar_xz(entries) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tf:
        for name, data, mode in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _deb(path: Path, version=V, tools=("ffmpeg", "ffprobe"), mode=0o755) -> Path:
    control = (
        f"Package: cove-compressor\nVersion: {version}\nArchitecture: amd64\n"
    ).encode()
    data = [("usr/lib/cove-compressor/cove-compressor", b"app", 0o755)]
    data += [(f"usr/lib/cove-compressor/_internal/{t}", b"tool", mode) for t in tools]
    path.write_bytes(
        _ar(
            [
                ("debian-binary", b"2.0\n"),
                ("control.tar.xz", _tar_xz([("./control", control, 0o644)])),
                ("data.tar.xz", _tar_xz(data)),
            ]
        )
    )
    return path


# --- assets ---------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,names",
    [("all", LINUX + WINDOWS), ("linux", LINUX), ("windows", WINDOWS)],
)
def test_a1_the_complete_set_verifies(tmp_path, platform, names):
    _assets(tmp_path, names)
    assert rv.verify_assets(tmp_path, V, platform) == []


def test_a3_a_missing_binary_is_refused(tmp_path):
    _assets(tmp_path, LINUX + WINDOWS)
    (tmp_path / WINDOWS[1]).unlink()
    errors = rv.verify_assets(tmp_path, V)
    assert any(WINDOWS[1] in e and "missing" in e for e in errors)


def test_a4_a_missing_checksum_is_refused(tmp_path):
    _assets(tmp_path, LINUX + WINDOWS)
    (tmp_path / f"{LINUX[1]}.sha256").unlink()
    errors = rv.verify_assets(tmp_path, V)
    assert any(f"{LINUX[1]}.sha256" in e and "missing" in e for e in errors)


def test_a5_a_digest_that_does_not_match_the_binary_is_refused(tmp_path):
    _assets(tmp_path, LINUX)
    (tmp_path / LINUX[0]).write_bytes(b"changed after the checksum was taken")
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert any(LINUX[0] in e and "mismatch" in e for e in errors)


def test_a6_an_unexpected_extra_file_is_refused(tmp_path):
    _assets(tmp_path, LINUX)
    (tmp_path / "notes.txt").write_text("stray")
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert any("notes.txt" in e and "unexpected" in e for e in errors)


def test_a7_an_empty_binary_is_refused(tmp_path):
    _assets(tmp_path, LINUX)
    _sidecar(tmp_path, LINUX[0], b"")
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert any(LINUX[0] in e and "empty" in e for e in errors)


def test_a8_artifacts_of_another_version_are_refused(tmp_path):
    _assets(tmp_path, [n.replace(V, "2.5.0") for n in LINUX])
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert errors and any("missing" in e for e in errors)


def test_a9_a_sidecar_that_names_another_file_is_refused(tmp_path):
    _assets(tmp_path, LINUX)
    digest = hashlib.sha256((tmp_path / LINUX[0]).read_bytes()).hexdigest()
    (tmp_path / f"{LINUX[0]}.sha256").write_text(f"{digest}  {LINUX[1]}\n")
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert any(f"{LINUX[0]}.sha256" in e and "names" in e for e in errors)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not-a-digest  {name}\n",
        "{digest}\n",
        "{digest} {name}\n",
        "{digest}  {name}\n{digest}  {name}\n",
        "{upper}  {name}\n",
    ],
    ids=["empty", "not-hex", "no-name", "one-space", "two-lines", "uppercase"],
)
def test_a10_a_malformed_sidecar_is_refused(tmp_path, text):
    _assets(tmp_path, LINUX)
    name = LINUX[0]
    digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / f"{name}.sha256").write_text(
        text.format(name=name, digest=digest, upper=digest.upper())
    )
    errors = rv.verify_assets(tmp_path, V, "linux")
    assert any(f"{name}.sha256" in e for e in errors)


def test_a11_a_missing_directory_is_refused(tmp_path):
    assert rv.verify_assets(tmp_path / "nope", V)


def test_a12_digest_is_computed_from_the_bytes_not_trusted_from_the_file(tmp_path):
    _assets(tmp_path, LINUX)
    expected = hashlib.sha256((tmp_path / LINUX[0]).read_bytes()).hexdigest()
    assert rv.sha256_of(tmp_path / LINUX[0]) == expected


# --- deb ------------------------------------------------------------------


def test_b1_a_complete_deb_verifies(tmp_path):
    assert rv.verify_deb(_deb(tmp_path / "a.deb"), V) == []


def test_b2_a_deb_of_another_version_is_refused(tmp_path):
    errors = rv.verify_deb(_deb(tmp_path / "a.deb", version="2.5.0"), V)
    assert any("Version" in e for e in errors)


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_b3_a_deb_without_either_tool_is_refused(tmp_path, missing):
    tools = tuple(t for t in ("ffmpeg", "ffprobe") if t != missing)
    errors = rv.verify_deb(_deb(tmp_path / "a.deb", tools=tools), V)
    assert any(missing in e and "missing" in e for e in errors)


def test_b4_a_tool_that_is_not_executable_is_refused(tmp_path):
    errors = rv.verify_deb(_deb(tmp_path / "a.deb", mode=0o644), V)
    assert any("not executable" in e for e in errors)


def test_b5_something_that_is_not_a_deb_is_refused(tmp_path):
    p = tmp_path / "a.deb"
    p.write_bytes(b"this is not an ar archive")
    assert rv.verify_deb(p, V)


def test_b6_a_deb_missing_a_member_is_refused(tmp_path):
    p = tmp_path / "a.deb"
    p.write_bytes(_ar([("debian-binary", b"2.0\n")]))
    assert rv.verify_deb(p, V)


# --- tree -----------------------------------------------------------------


def _tree(root: Path, names) -> None:
    sub = root / "usr" / "lib" / "cove-compressor" / "_internal"
    sub.mkdir(parents=True)
    for n in names:
        (sub / n).write_bytes(b"tool")


def test_c1_a_tree_with_both_tools_verifies(tmp_path):
    _tree(tmp_path, ["ffmpeg", "ffprobe"])
    assert rv.verify_tree(tmp_path) == []


def test_c2_the_windows_names_are_the_exe_names(tmp_path):
    _tree(tmp_path, ["ffmpeg.exe", "ffprobe.exe"])
    assert rv.verify_tree(tmp_path, windows=True) == []
    assert rv.verify_tree(tmp_path, windows=False)


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_c3_a_tree_without_either_tool_is_refused(tmp_path, missing):
    _tree(tmp_path, [t for t in ("ffmpeg", "ffprobe") if t != missing])
    errors = rv.verify_tree(tmp_path)
    assert any(missing in e and "missing" in e for e in errors)


def test_c4_an_empty_tool_is_refused(tmp_path):
    _tree(tmp_path, ["ffmpeg", "ffprobe"])
    (tmp_path / "usr/lib/cove-compressor/_internal/ffprobe").write_bytes(b"")
    assert any("empty" in e for e in rv.verify_tree(tmp_path))


def test_c5_a_tool_that_will_not_start_is_refused(tmp_path):
    _tree(tmp_path, ["ffmpeg", "ffprobe"])

    def runner(cmd):
        return subprocess.CompletedProcess(cmd, 0 if "ffmpeg" in cmd[0] else 1, "", "")

    errors = rv.verify_tree(tmp_path, runner=runner)
    assert any("ffprobe" in e and "-version" in e for e in errors)


def test_c6_a_runner_is_asked_for_each_tool_exactly_once(tmp_path):
    _tree(tmp_path, ["ffmpeg", "ffprobe"])
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "ffmpeg version 1", "")

    assert rv.verify_tree(tmp_path, runner=runner) == []
    assert [Path(c[0]).name for c in seen] == ["ffmpeg", "ffprobe"]
    assert all(c[1:] == ["-version"] for c in seen)


def test_c7_a_missing_root_is_refused(tmp_path):
    assert rv.verify_tree(tmp_path / "nope")


# --- listing --------------------------------------------------------------

_LISTING = """\
Contents of 'x-Portable.exe' (PKG/CArchive):
 position, length, uncompressed_length, is_compressed, typecode, name
 0, 233, 289, 1, 'm', 'struct'
 44230102, 36993301, 101897728, 1, 'b', 'ffmpeg.exe'
 81223403, 36888607, 101692928, 1, 'b', 'ffprobe.exe'
"""


def test_d1_a_listing_with_both_tools_verifies():
    assert rv.verify_listing(_LISTING) == []


@pytest.mark.parametrize("missing", ["ffmpeg.exe", "ffprobe.exe"])
def test_d2_a_listing_without_either_tool_is_refused(missing):
    text = "\n".join(l for l in _LISTING.splitlines() if missing not in l)
    errors = rv.verify_listing(text)
    assert any(missing in e and "missing" in e for e in errors)


def test_d3_a_tool_listed_as_something_other_than_a_binary_is_refused():
    errors = rv.verify_listing(_LISTING.replace("'b', 'ffmpeg.exe'", "'d', 'ffmpeg.exe'"))
    assert any("ffmpeg.exe" in e for e in errors)


def test_d4_a_similar_name_does_not_count():
    errors = rv.verify_listing(_LISTING.replace("'ffprobe.exe'", "'ffprobe.exe.bak'"))
    assert any("ffprobe.exe" in e and "missing" in e for e in errors)


def test_d5_an_empty_listing_is_refused():
    assert rv.verify_listing("")


# --- command line ---------------------------------------------------------


def test_e1_the_cli_exits_zero_on_success_and_nonzero_on_failure(tmp_path, capsys):
    _assets(tmp_path, LINUX)
    assert rv.main(["assets", str(tmp_path), V, "--platform", "linux"]) == 0
    (tmp_path / LINUX[0]).write_bytes(b"tampered")
    assert rv.main(["assets", str(tmp_path), V, "--platform", "linux"]) == 1
    assert "mismatch" in capsys.readouterr().err


def test_e2_the_cli_checks_a_deb(tmp_path):
    p = _deb(tmp_path / "a.deb")
    assert rv.main(["deb", str(p), V]) == 0
    assert rv.main(["deb", str(p), "9.9.9"]) == 1


def test_e3_the_cli_checks_a_listing_file(tmp_path):
    p = tmp_path / "listing.txt"
    p.write_text(_LISTING)
    assert rv.main(["listing", str(p)]) == 0
    p.write_text("")
    assert rv.main(["listing", str(p)]) == 1


def test_e4_the_cli_checks_a_tree(tmp_path):
    _tree(tmp_path, ["ffmpeg", "ffprobe"])
    assert rv.main(["tree", str(tmp_path)]) == 0
    assert rv.main(["tree", str(tmp_path / "nope")]) == 1
