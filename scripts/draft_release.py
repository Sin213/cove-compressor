#!/usr/bin/env python3
"""Create (or replace a leftover draft of) the DRAFT GitHub Release for a tag.

Called by the final job of release.yml with the verified asset directory.
This script never publishes: a release it creates or touches is a draft, and
a release that is already published is refused, not modified.

    draft_release.py create REPO TAG VERSION ASSETS_DIR NOTES_FILE
    draft_release.py check  REPO TAG VERSION ASSETS_DIR

`check` re-reads the release from GitHub and exits 1 unless it is a single,
unpublished, non-prerelease release with exactly the eight expected assets,
each uploaded, with the size (and, when GitHub reports one, the sha256) of the
local verified file.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_verify import expected_binaries, sha256_of  # noqa: E402

_TAG = re.compile(r"v\d+\.\d+\.\d+")
_REPO = re.compile(r"[\w.-]+/[\w.-]+")


class ReleaseError(Exception):
    pass


def gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()}")
    return result.stdout


def title_for(version: str) -> str:
    return f"Cove Compressor v{version}"


def expected_assets(version: str) -> list:
    return sorted(b for n in expected_binaries(version) for b in (n, f"{n}.sha256"))


def find_releases(repo: str, tag: str, run=None) -> list:
    """Every release - drafts included - whose tag is exactly `tag`."""
    run = run or gh
    pages = json.loads(run("api", "--paginate", "--slurp", f"repos/{repo}/releases?per_page=100"))
    return [r for page in pages for r in page if r.get("tag_name") == tag]


def _validate(repo: str, tag: str, version: str) -> None:
    if not _REPO.fullmatch(repo):
        raise ReleaseError(f"unexpected repository {repo!r}")
    if not _TAG.fullmatch(tag) or tag != f"v{version}":
        raise ReleaseError(f"tag {tag!r} does not match version {version!r}")


def create(repo, tag, version, assets_dir, notes, run=None) -> None:
    run = run or gh
    _validate(repo, tag, version)
    files = sorted(str(p) for p in Path(assets_dir).iterdir())
    if sorted(Path(f).name for f in files) != expected_assets(version):
        raise ReleaseError("asset directory is not exactly the eight expected files")
    if not Path(notes).is_file() or not Path(notes).read_text(encoding="utf-8").strip():
        raise ReleaseError(f"release notes missing or empty: {notes}")
    title = title_for(version)
    found = find_releases(repo, tag, run)
    if len(found) > 1:
        raise ReleaseError(f"{len(found)} releases already exist for {tag}")
    if found:
        release = found[0]
        if not release["draft"]:
            raise ReleaseError(f"{tag} is already published; refusing to modify it")
        # A leftover draft from an earlier attempt at this same tag is replaced,
        # never joined by a second release.
        run("api", "--method", "DELETE", f"repos/{repo}/releases/{release['id']}")
    run("release", "create", tag, *files, "--repo", repo, "--draft",
        "--verify-tag", "--title", title, "--notes-file", str(notes))


def check(repo, tag, version, assets_dir, run=None) -> list:
    _validate(repo, tag, version)
    found = find_releases(repo, tag, run)
    if len(found) != 1:
        return [f"expected exactly one release for {tag}, found {len(found)}"]
    release, errors = found[0], []
    if not release["draft"]:
        errors.append("release is published, it must remain a draft")
    if release["prerelease"]:
        errors.append("release is marked prerelease")
    if release["name"] != title_for(version):
        errors.append(f"title is {release['name']!r}, expected {title_for(version)!r}")
    if not (release.get("body") or "").strip():
        errors.append("release notes are empty")
    remote = {a["name"]: a for a in release["assets"]}
    names = expected_assets(version)
    for name in names:
        if name not in remote:
            errors.append(f"{name}: not attached")
    for name in sorted(set(remote) - set(names)):
        errors.append(f"{name}: unexpected asset")
    for name in names:
        asset, local = remote.get(name), Path(assets_dir) / name
        if asset is None or not local.is_file():
            continue
        if asset.get("state") != "uploaded":
            errors.append(f"{name}: state is {asset.get('state')!r}")
        if asset["size"] != local.stat().st_size:
            errors.append(f"{name}: size {asset['size']} != local {local.stat().st_size}")
        digest = asset.get("digest")
        if digest and digest != f"sha256:{sha256_of(local)}":
            errors.append(f"{name}: GitHub digest {digest} != local sha256")
    return errors


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if len(argv) == 6 and argv[0] == "create":
            create(*argv[1:])
            errors = check(*argv[1:5])
        elif len(argv) == 5 and argv[0] == "check":
            errors = check(*argv[1:])
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except ReleaseError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    for e in errors:
        print(f"FAIL {e}", file=sys.stderr)
    if not errors:
        print(f"OK {argv[0]}: {argv[2]} is a draft with all {len(expected_assets(argv[3]))} assets")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
