#!/usr/bin/env python3
"""Verify the files a Cove Compressor release is made of.

Run by both build jobs and by the final assembly job in release.yml, so a
release is only ever assembled from files that passed the same checks.
Standard library only; works the same on Linux and Windows.

    release_verify.py assets  DIR VERSION [--platform linux|windows|all]
    release_verify.py deb     FILE VERSION
    release_verify.py tree    DIR [--windows] [--run]
    release_verify.py listing FILE

Every command prints one line per problem to stderr and exits 1 if there was
any, 0 otherwise.
"""
import argparse
import hashlib
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

PLATFORMS = ("all", "linux", "windows")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TOOLS = ("ffmpeg", "ffprobe")


def expected_binaries(version: str, platform: str = "all") -> list:
    linux = [
        f"Cove-Compressor-{version}-x86_64.AppImage",
        f"cove-compressor_{version}_amd64.deb",
    ]
    windows = [
        f"cove-compressor-{version}-Setup.exe",
        f"cove-compressor-{version}-Portable.exe",
    ]
    if platform == "linux":
        return linux
    if platform == "windows":
        return windows
    return linux + windows


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sidecar(path: Path):
    """Return (digest, name, None) or (None, None, problem).

    The one accepted shape is what `sha256sum` writes: a lowercase hex digest,
    two spaces, the file name, one newline - and nothing else.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, None, f"{path.name}: unreadable ({exc})"
    lines = text.splitlines()
    if len(lines) != 1 or not text.endswith("\n"):
        return None, None, f"{path.name}: malformed - expected exactly one line"
    digest, sep, name = lines[0].partition("  ")
    if not sep or not name or not _DIGEST.fullmatch(digest):
        return None, None, f"{path.name}: malformed - expected '<sha256>  <name>'"
    return digest, name, None


def verify_assets(directory, version: str, platform: str = "all") -> list:
    """Exactly the promised binaries + one matching .sha256 each, nothing else."""
    directory = Path(directory)
    if not directory.is_dir():
        return [f"{directory}: not a directory"]
    binaries = expected_binaries(version, platform)
    expected = set(binaries) | {f"{b}.sha256" for b in binaries}
    present = {p.name for p in directory.iterdir()}
    errors = []
    for name in sorted(expected - present):
        errors.append(f"{name}: missing")
    for name in sorted(present - expected):
        errors.append(f"{name}: unexpected file")
    for binary in binaries:
        path = directory / binary
        if not path.is_file():
            continue
        if path.stat().st_size == 0:
            errors.append(f"{binary}: empty")
            continue
        sidecar = directory / f"{binary}.sha256"
        if not sidecar.is_file():
            continue
        digest, named, problem = _read_sidecar(sidecar)
        if problem:
            errors.append(problem)
            continue
        if named != binary:
            errors.append(f"{sidecar.name}: names {named!r}, not {binary!r}")
            continue
        actual = sha256_of(path)
        if actual != digest:
            errors.append(f"{binary}: sha256 mismatch (file {actual}, {sidecar.name} {digest})")
    return errors


def _ar_members(data: bytes) -> dict:
    if data[:8] != b"!<arch>\n":
        raise ValueError("not an ar archive")
    members, pos = {}, 8
    while pos < len(data):
        header = data[pos : pos + 60]
        if len(header) < 60 or header[58:60] != b"`\n":
            raise ValueError("corrupt ar header")
        name = header[:16].decode("ascii").strip().rstrip("/")
        size = int(header[48:58].decode("ascii").strip())
        start = pos + 60
        if start + size > len(data):
            raise ValueError(f"ar member {name!r} is truncated")
        members[name] = data[start : start + size]
        pos = start + size + (size % 2)
    return members


def verify_deb(path, version: str) -> list:
    """A real ar-format .deb of this version carrying executable ffmpeg+ffprobe."""
    try:
        members = _ar_members(Path(path).read_bytes())
        for needed in ("debian-binary", "control.tar.xz", "data.tar.xz"):
            if needed not in members:
                return [f"{Path(path).name}: member {needed} missing"]
        with tarfile.open(fileobj=io.BytesIO(members["control.tar.xz"]), mode="r:*") as tf:
            control_member = next(
                (m for m in tf.getmembers() if m.name in ("control", "./control")), None
            )
            if control_member is None:
                return [f"{Path(path).name}: control file missing"]
            control = tf.extractfile(control_member).read().decode("utf-8")
        with tarfile.open(fileobj=io.BytesIO(members["data.tar.xz"]), mode="r:*") as tf:
            files = {
                m.name.removeprefix("./"): m for m in tf.getmembers() if m.isfile()
            }
    except (ValueError, OSError, tarfile.TarError, UnicodeDecodeError) as exc:
        return [f"{Path(path).name}: not a valid .deb ({exc})"]
    errors = []
    fields = dict(
        line.split(": ", 1) for line in control.splitlines() if ": " in line and line[0] != " "
    )
    if fields.get("Package") != "cove-compressor":
        errors.append(f"deb Package is {fields.get('Package')!r}, expected 'cove-compressor'")
    if fields.get("Version") != version:
        errors.append(f"deb Version is {fields.get('Version')!r}, expected {version!r}")
    if fields.get("Architecture") != "amd64":
        errors.append(f"deb Architecture is {fields.get('Architecture')!r}, expected 'amd64'")
    for tool in _TOOLS:
        hits = [m for n, m in files.items() if n.rsplit("/", 1)[-1] == tool]
        if not hits:
            errors.append(f"deb: {tool} missing from data archive")
        elif not any(m.mode & 0o111 and m.size > 0 for m in hits):
            errors.append(f"deb: {tool} is empty or not executable")
    return errors


def _default_runner(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def verify_tree(root, windows: bool = False, runner=None) -> list:
    """Both tools exist somewhere under an unpacked/installed tree.

    With a runner, each is also asked for `-version` - the only honest proof
    that a bundled binary starts on the machine that is about to ship it.
    """
    root = Path(root)
    if not root.is_dir():
        return [f"{root}: not a directory"]
    suffix = ".exe" if windows else ""
    errors = []
    for tool in _TOOLS:
        name = tool + suffix
        found = sorted(p for p in root.rglob(name) if p.is_file())
        if not found:
            errors.append(f"{name}: missing under {root}")
            continue
        path = found[0]
        if path.stat().st_size == 0:
            errors.append(f"{path}: empty")
            continue
        if runner is not None:
            try:
                result = runner([str(path), "-version"])
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{path} -version: could not run ({exc})")
                continue
            if result.returncode != 0:
                errors.append(f"{path} -version: exit {result.returncode}")
    return errors


_LISTING_ROW = re.compile(r"^\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*'(\w+)',\s*'(.*)'\s*$")


def verify_listing(text: str) -> list:
    """`pyi-archive_viewer -l -r` output shows both tools as bundled binaries."""
    binaries = set()
    for line in text.splitlines():
        row = _LISTING_ROW.match(line)
        if row and row.group(1) == "b":
            binaries.add(row.group(2))
    return [
        f"{tool}.exe: missing from the archive's binaries"
        for tool in _TOOLS
        if f"{tool}.exe" not in binaries
    ]


def _report(errors: list) -> int:
    for e in errors:
        print(f"FAIL {e}", file=sys.stderr)
    return 1 if errors else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("assets")
    p.add_argument("directory")
    p.add_argument("version")
    p.add_argument("--platform", choices=PLATFORMS, default="all")
    p = sub.add_parser("deb")
    p.add_argument("file")
    p.add_argument("version")
    p = sub.add_parser("tree")
    p.add_argument("directory")
    p.add_argument("--windows", action="store_true")
    p.add_argument("--run", action="store_true")
    p = sub.add_parser("listing")
    p.add_argument("file")
    args = ap.parse_args(argv)

    if args.cmd == "assets":
        errors = verify_assets(args.directory, args.version, args.platform)
    elif args.cmd == "deb":
        errors = verify_deb(args.file, args.version)
    elif args.cmd == "tree":
        errors = verify_tree(
            args.directory, args.windows, _default_runner if args.run else None
        )
    else:
        try:
            text = Path(args.file).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors = [f"{args.file}: unreadable ({exc})"]
        else:
            errors = verify_listing(text)
    code = _report(errors)
    if code == 0:
        print(f"OK {args.cmd}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
