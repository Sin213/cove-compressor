"""The release job may create a draft. It may never publish anything.

`scripts/draft_release.py` is the only code in the pipeline that talks to the
Releases API with write access, so these tests drive it against a fake `gh`
that keeps a little world of releases and records every call:

    A.  A first run creates exactly one DRAFT with the agreed title, tag,
        notes and the eight assets - and nothing else.
    B.  A published release (v2.5.0, or the target itself) is never touched,
        and a duplicate is never created.
    C.  A leftover draft is replaced, not joined by a second release.
    D.  `check` reads the release back and refuses every way it can be wrong.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import draft_release as dr  # noqa: E402
import release_verify as rv  # noqa: E402

V = "2.6.0"
TAG = f"v{V}"
REPO = "Sin213/cove-compressor"
TITLE = f"Cove Compressor v{V}"


def _assets_dir(tmp_path) -> Path:
    d = tmp_path / "assets"
    d.mkdir()
    for name in dr.expected_assets(V):
        (d / name).write_bytes(b"bytes of " + name.encode())
    return d


def _notes(tmp_path) -> Path:
    p = tmp_path / "notes.md"
    p.write_text("## Changes since 2.5.0\n- something\n")
    return p


def _release(tag, *, draft, id_, assets=(), name=None, prerelease=False, body="notes"):
    return {
        "id": id_,
        "tag_name": tag,
        "name": name if name is not None else f"Cove Compressor {tag}",
        "draft": draft,
        "prerelease": prerelease,
        "body": body,
        "assets": list(assets),
    }


def _remote_assets(directory: Path, **override):
    out = []
    for i, p in enumerate(sorted(directory.iterdir())):
        a = {
            "id": 1000 + i,
            "name": p.name,
            "size": p.stat().st_size,
            "state": "uploaded",
            "digest": f"sha256:{rv.sha256_of(p)}",
        }
        a.update(override.get(p.name, {}))
        out.append(a)
    return out


class FakeGh:
    """Just enough of `gh` to list releases and record what would be changed."""

    def __init__(self, releases):
        self.releases = releases
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("api", "--paginate"):
            return json.dumps([self.releases])
        return ""

    @property
    def mutations(self):
        return [c for c in self.calls if c[:2] != ("api", "--paginate")]


def _create(tmp_path, gh, *, assets=None, notes=None):
    dr.create(
        REPO,
        TAG,
        V,
        assets or _assets_dir(tmp_path),
        notes or _notes(tmp_path),
        run=gh,
    )


# --- A. first run ---------------------------------------------------------


def test_a1_a_first_run_makes_one_draft_with_the_agreed_title_and_tag(tmp_path):
    gh = FakeGh([_release("v2.5.0", draft=False, id_=1)])
    _create(tmp_path, gh)
    assert len(gh.mutations) == 1
    call = gh.mutations[0]
    assert call[:3] == ("release", "create", TAG)
    assert "--draft" in call and "--verify-tag" in call
    assert call[call.index("--title") + 1] == TITLE
    assert call[call.index("--repo") + 1] == REPO
    assert call[call.index("--notes-file") + 1].endswith("notes.md")


def test_a2_the_draft_carries_exactly_the_eight_expected_files(tmp_path):
    gh = FakeGh([])
    _create(tmp_path, gh)
    call = gh.mutations[0]
    flags = call.index("--repo")
    attached = sorted(Path(a).name for a in call[3:flags])
    assert attached == dr.expected_assets(V)
    assert len(attached) == 8


def test_a3_nothing_in_the_call_can_publish_or_promote(tmp_path):
    gh = FakeGh([])
    _create(tmp_path, gh)
    flat = " ".join(gh.mutations[0])
    for forbidden in ("--prerelease", "--latest", "--draft=false", "edit"):
        assert forbidden not in flat


def test_a4_a_missing_asset_stops_before_anything_is_changed(tmp_path):
    d = _assets_dir(tmp_path)
    (d / f"cove-compressor-{V}-Portable.exe").unlink()
    gh = FakeGh([])
    with pytest.raises(dr.ReleaseError):
        _create(tmp_path, gh, assets=d)
    assert gh.calls == []


def test_a5_an_extra_asset_stops_before_anything_is_changed(tmp_path):
    d = _assets_dir(tmp_path)
    (d / "stray.txt").write_text("x")
    gh = FakeGh([])
    with pytest.raises(dr.ReleaseError):
        _create(tmp_path, gh, assets=d)
    assert gh.calls == []


@pytest.mark.parametrize("content", [None, "  \n"], ids=["absent", "blank"])
def test_a6_empty_or_missing_notes_stop_before_anything_is_changed(tmp_path, content):
    notes = tmp_path / "n.md"
    if content is not None:
        notes.write_text(content)
    gh = FakeGh([])
    with pytest.raises(dr.ReleaseError):
        _create(tmp_path, gh, notes=notes)
    assert gh.calls == []


@pytest.mark.parametrize(
    "tag,version",
    [("v2.5.0", V), ("2.6.0", V), ("v2.6.0-rc1", "2.6.0-rc1"), ("v2.6", "2.6")],
)
def test_a7_a_tag_that_is_not_this_versions_is_refused(tmp_path, tag, version):
    gh = FakeGh([])
    with pytest.raises(dr.ReleaseError):
        dr.create(REPO, tag, version, _assets_dir(tmp_path), _notes(tmp_path), run=gh)
    assert gh.calls == []


def test_a8_a_repo_string_that_is_not_owner_slash_name_is_refused(tmp_path):
    gh = FakeGh([])
    with pytest.raises(dr.ReleaseError):
        dr.create("Sin213/x; rm -rf", TAG, V, _assets_dir(tmp_path), _notes(tmp_path), run=gh)
    assert gh.calls == []


# --- B. published releases ------------------------------------------------


def test_b1_the_existing_v2_5_0_release_is_never_touched(tmp_path):
    gh = FakeGh([_release("v2.5.0", draft=False, id_=7)])
    _create(tmp_path, gh)
    assert all("v2.5.0" not in " ".join(c) and "/7" not in " ".join(c) for c in gh.mutations)


def test_b2_a_published_release_for_the_target_tag_is_refused_untouched(tmp_path):
    gh = FakeGh([_release(TAG, draft=False, id_=9)])
    with pytest.raises(dr.ReleaseError, match="already published"):
        _create(tmp_path, gh)
    assert gh.mutations == []


def test_b3_duplicate_releases_for_the_tag_are_refused_untouched(tmp_path):
    gh = FakeGh([_release(TAG, draft=True, id_=1), _release(TAG, draft=True, id_=2)])
    with pytest.raises(dr.ReleaseError):
        _create(tmp_path, gh)
    assert gh.mutations == []


# --- C. leftover draft ----------------------------------------------------


def test_c1_a_leftover_draft_is_replaced_by_exactly_one_new_draft(tmp_path):
    gh = FakeGh([_release(TAG, draft=True, id_=42)])
    _create(tmp_path, gh)
    assert gh.mutations[0] == ("api", "--method", "DELETE", f"repos/{REPO}/releases/42")
    assert gh.mutations[1][:3] == ("release", "create", TAG)
    assert "--draft" in gh.mutations[1]
    assert len(gh.mutations) == 2


def test_c2_only_the_leftover_drafts_own_id_is_ever_deleted(tmp_path):
    gh = FakeGh([_release("v2.5.0", draft=False, id_=7), _release(TAG, draft=True, id_=42)])
    _create(tmp_path, gh)
    deletes = [c for c in gh.mutations if "DELETE" in c]
    assert deletes == [("api", "--method", "DELETE", f"repos/{REPO}/releases/42")]


# --- D. read-back ---------------------------------------------------------


def _good(tmp_path):
    d = _assets_dir(tmp_path)
    return d, _release(TAG, draft=True, id_=5, name=TITLE, assets=_remote_assets(d))


def test_d1_a_correct_draft_passes(tmp_path):
    d, rel = _good(tmp_path)
    assert dr.check(REPO, TAG, V, d, run=FakeGh([rel])) == []


def test_d2_a_published_release_fails_the_check(tmp_path):
    d, rel = _good(tmp_path)
    rel["draft"] = False
    assert any("published" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d3_a_prerelease_fails_the_check(tmp_path):
    d, rel = _good(tmp_path)
    rel["prerelease"] = True
    assert any("prerelease" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d4_the_wrong_title_fails_the_check(tmp_path):
    d, rel = _good(tmp_path)
    rel["name"] = "v2.6.0"
    assert any("title" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d5_empty_notes_fail_the_check(tmp_path):
    d, rel = _good(tmp_path)
    rel["body"] = "  "
    assert any("notes" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d6_seven_assets_fail_the_check(tmp_path):
    d, rel = _good(tmp_path)
    gone = rel["assets"].pop()
    errors = dr.check(REPO, TAG, V, d, run=FakeGh([rel]))
    assert any(gone["name"] in e and "not attached" in e for e in errors)


def test_d7_a_ninth_asset_fails_the_check(tmp_path):
    d, rel = _good(tmp_path)
    rel["assets"].append({"id": 1, "name": "extra.zip", "size": 1, "state": "uploaded"})
    assert any("extra.zip" in e and "unexpected" in e
               for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d8_an_asset_that_is_not_uploaded_fails_the_check(tmp_path):
    d = _assets_dir(tmp_path)
    name = dr.expected_assets(V)[0]
    rel = _release(TAG, draft=True, id_=5, name=TITLE,
                   assets=_remote_assets(d, **{name: {"state": "starter"}}))
    assert any(name in e and "state" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d9_a_size_that_differs_from_the_verified_file_fails_the_check(tmp_path):
    d = _assets_dir(tmp_path)
    name = dr.expected_assets(V)[1]
    rel = _release(TAG, draft=True, id_=5, name=TITLE,
                   assets=_remote_assets(d, **{name: {"size": 1}}))
    assert any(name in e and "size" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d10_a_digest_that_differs_from_the_verified_file_fails_the_check(tmp_path):
    d = _assets_dir(tmp_path)
    name = dr.expected_assets(V)[2]
    rel = _release(TAG, draft=True, id_=5, name=TITLE,
                   assets=_remote_assets(d, **{name: {"digest": "sha256:" + "0" * 64}}))
    assert any(name in e and "digest" in e for e in dr.check(REPO, TAG, V, d, run=FakeGh([rel])))


def test_d11_no_release_and_two_releases_both_fail_the_check(tmp_path):
    d, rel = _good(tmp_path)
    assert dr.check(REPO, TAG, V, d, run=FakeGh([]))
    assert dr.check(REPO, TAG, V, d, run=FakeGh([rel, dict(rel, id=6)]))


def test_d12_the_check_never_mutates_anything(tmp_path):
    d, rel = _good(tmp_path)
    gh = FakeGh([rel])
    dr.check(REPO, TAG, V, d, run=gh)
    assert gh.mutations == []


# --- the command line -----------------------------------------------------


def test_e1_the_cli_rejects_unknown_commands_and_wrong_arity():
    assert dr.main([]) == 2
    assert dr.main(["publish", REPO, TAG, V, "x"]) == 2
    assert dr.main(["check", REPO, TAG]) == 2


def test_e2_a_gh_failure_is_reported_as_a_failure_not_a_traceback(tmp_path, monkeypatch, capsys):
    def boom(*args):
        raise dr.ReleaseError("gh api failed: HTTP 403")

    monkeypatch.setattr(dr, "gh", boom)
    d = _assets_dir(tmp_path)
    assert dr.main(["check", REPO, TAG, V, str(d)]) == 1
    assert "403" in capsys.readouterr().err


def test_e3_gh_itself_raises_on_a_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "nope"),
    )
    with pytest.raises(dr.ReleaseError, match="nope"):
        dr.gh("api", "x")
