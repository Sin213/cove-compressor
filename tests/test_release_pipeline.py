"""The CI and release pipeline, pinned as data.

v2.5.0's first hosted CI run failed on both runners because neither had ffmpeg
or ffprobe, and the same release workflow uploaded its artifacts *before*
writing their checksums and let each build job attach files to the release on
its own. Nothing here runs a workflow; it reads the YAML and pins the shape
that makes those failures impossible to reintroduce:

    A.  The version is one number everywhere it is written down.
    B.  tests.yml installs both tools on both runners, proves they are on PATH
        before pytest, and still runs the whole suite on both.
    C.  The build jobs verify, checksum and upload binaries + checksums
        together, and neither of them touches the release.
    D.  One final job, gated on tests and both builds, verifies all eight files
        and creates a draft - never a published release.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(doc: dict) -> dict:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return doc.get("on", doc.get(True))


def _steps(doc: dict, job: str) -> list:
    return doc["jobs"][job]["steps"]


def _index(steps: list, predicate, what: str) -> int:
    for i, step in enumerate(steps):
        if predicate(step):
            return i
    raise AssertionError(f"no step: {what}")


def _runs(step: dict) -> str:
    return step.get("run", "")


def _uses(step: dict) -> str:
    return step.get("uses", "")


def _all_runs(doc: dict, job: str) -> str:
    return "\n".join(_runs(s) for s in _steps(doc, job))


@pytest.fixture(scope="module")
def tests_yml():
    return _load("tests.yml")


@pytest.fixture(scope="module")
def release_yml():
    return _load("release.yml")


@pytest.fixture(scope="module")
def release_text():
    return (WORKFLOWS / "release.yml").read_text(encoding="utf-8")


# --- A. one version -------------------------------------------------------


def _version() -> str:
    text = (ROOT / "src" / "cove_compressor" / "__init__.py").read_text()
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def test_a1_the_version_is_a_plain_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _version())


def test_a2_the_readme_badge_and_screenshot_alt_follow_the_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    v = _version()
    assert f"badge/release-v{v}-" in readme
    assert f"![Cove Compressor v{v}]" in readme
    assert re.findall(r"badge/release-v([\d.]+)-", readme) == [v]


@pytest.mark.parametrize(
    "path,pattern",
    [
        ("scripts/build-release.sh", r'VERSION="\$\{VERSION:-([\d.]+)\}"'),
        ("scripts/build-release.sh", r"default ([\d.]+)\)"),
        ("scripts/build-windows-wine.sh", r'VERSION="\$\{VERSION:-([\d.]+)\}"'),
        ("scripts/build-windows-wine.sh", r"VERSION=([\d.]+)\s+# artifact version"),
        ("build.ps1", r'\[string\]\$Version = "([\d.]+)"'),
        ("build.ps1", r"\.\\build\.ps1 -Version ([\d.]+)"),
    ],
)
def test_a3_every_build_script_defaults_to_the_current_version(path, pattern):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert re.findall(pattern, text) == [_version()]


# --- B. tests.yml ---------------------------------------------------------


def test_b1_the_reusable_workflow_and_both_runners_are_preserved(tests_yml):
    assert "workflow_call" in _triggers(tests_yml)
    assert "push" in _triggers(tests_yml) and "pull_request" in _triggers(tests_yml)
    job = tests_yml["jobs"]["tests"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert job["runs-on"] == "${{ matrix.os }}"
    assert job["strategy"]["fail-fast"] is False


def _ffmpeg_install(step: dict, os_name: str) -> bool:
    label = _runs(step) + step.get("name", "")
    return os_name in step.get("if", "") and "ffmpeg" in label and "nstall" in label


@pytest.mark.parametrize("os_name", ["Linux", "Windows"])
def test_b2_each_runner_installs_ffmpeg(tests_yml, os_name):
    steps = _steps(tests_yml, "tests")
    _index(steps, lambda s: _ffmpeg_install(s, os_name), f"{os_name} ffmpeg install")


def test_b3_linux_installs_the_distro_package_that_carries_both_tools(tests_yml):
    steps = _steps(tests_yml, "tests")
    step = steps[_index(steps, lambda s: _ffmpeg_install(s, "Linux"), "linux")]
    assert re.search(r"apt-get install[^\n]*\bffmpeg\b", _runs(step))


def test_b4_windows_puts_the_tools_on_PATH_for_later_steps(tests_yml):
    steps = _steps(tests_yml, "tests")
    step = steps[_index(steps, lambda s: _ffmpeg_install(s, "Windows"), "windows")]
    assert "GITHUB_PATH" in _runs(step)


def test_b5_both_tools_are_verified_on_PATH_before_the_tests_run(tests_yml):
    steps = _steps(tests_yml, "tests")
    verify = _index(
        steps,
        lambda s: "ffmpeg -version" in _runs(s) and "ffprobe -version" in _runs(s),
        "verify",
    )
    run_tests = _index(steps, lambda s: "python -m pytest" in _runs(s), "pytest")
    installs = [
        _index(steps, lambda s, o=o: _ffmpeg_install(s, o), o)
        for o in ("Linux", "Windows")
    ]
    assert max(installs) < verify < run_tests
    assert "shutil.which" in _runs(steps[verify]), "must prove what pytest will resolve"
    assert "continue-on-error" not in steps[verify]


def test_b6_the_verify_step_runs_on_every_runner(tests_yml):
    steps = _steps(tests_yml, "tests")
    verify = steps[_index(steps, lambda s: "ffprobe -version" in _runs(s), "verify")]
    assert "if" not in verify


def test_b7_the_verify_step_requires_the_encoders_the_tests_use(tests_yml):
    steps = _steps(tests_yml, "tests")
    verify = steps[_index(steps, lambda s: "ffprobe -version" in _runs(s), "verify")]
    for encoder in ("libx264", "libx265", "libvpx-vp9"):
        assert encoder in _runs(verify)


def test_b8_the_whole_suite_runs_unweakened_on_both_runners(tests_yml):
    steps = _steps(tests_yml, "tests")
    step = steps[_index(steps, lambda s: "python -m pytest" in _runs(s), "pytest")]
    cmd = _runs(step).strip()
    assert re.fullmatch(r"python -m pytest -q -p no:cacheprovider tests", cmd), cmd
    assert "if" not in step and "continue-on-error" not in step
    assert "continue-on-error" not in tests_yml["jobs"]["tests"]
    text = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    for weakener in ("--deselect", "--ignore", "|| true", "xfail"):
        assert weakener not in text
    assert not re.search(r"pytest[^\n]* -(k|m)\s", text), "no selection by keyword/marker"


def test_b9_pyyaml_is_installed_for_the_pipeline_tests(tests_yml):
    steps = _steps(tests_yml, "tests")
    step = steps[_index(steps, lambda s: "pip install" in _runs(s), "pip")]
    assert "pyyaml" in _runs(step).lower()


# --- C. build jobs --------------------------------------------------------

BUILDS = {
    "build-linux": ("linux-release", ("AppImage", ".deb")),
    "build-windows": ("windows-release", ("-Setup.exe", "-Portable.exe")),
}


def _is_checksum(step: dict) -> bool:
    return "sha256sum" in _runs(step) or "Get-FileHash" in _runs(step)


def test_c1_the_release_keeps_its_test_gate(release_yml):
    assert release_yml["jobs"]["verify-tests"]["uses"] == "./.github/workflows/tests.yml"
    for name in BUILDS:
        assert "verify-tests" in release_yml["jobs"][name]["needs"]


@pytest.mark.parametrize("name", sorted(BUILDS))
def test_c2_checksums_are_written_before_anything_is_uploaded(release_yml, name):
    steps = _steps(release_yml, name)
    checksum = _index(steps, _is_checksum, "checksum generation")
    uploads = [i for i, s in enumerate(steps) if "upload-artifact" in _uses(s)]
    assert uploads and checksum < min(uploads)


@pytest.mark.parametrize("name", sorted(BUILDS))
def test_c3_the_verifier_runs_after_checksums_and_before_upload(release_yml, name):
    steps = _steps(release_yml, name)
    verifies = [i for i, s in enumerate(steps) if "release_verify.py assets" in _runs(s)]
    uploads = [i for i, s in enumerate(steps) if "upload-artifact" in _uses(s)]
    checksum = _index(steps, _is_checksum, "checksum")
    assert verifies and checksum < min(verifies) < min(uploads)


@pytest.mark.parametrize("name", sorted(BUILDS))
def test_c4_binaries_and_checksums_travel_in_one_workflow_artifact(release_yml, name):
    artifact, patterns = BUILDS[name]
    uploads = [s for s in _steps(release_yml, name) if "upload-artifact" in _uses(s)]
    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert upload["name"] == artifact
    assert upload["if-no-files-found"] == "error"
    assert ".sha256" in upload["path"]
    for pattern in patterns:
        assert pattern in upload["path"]


@pytest.mark.parametrize("name", sorted(BUILDS))
def test_c5_neither_build_job_touches_the_release(release_yml, name):
    for step in _steps(release_yml, name):
        assert "action-gh-release" not in _uses(step)
        assert "gh release" not in _runs(step)
    perms = release_yml["jobs"][name].get("permissions", {})
    assert perms.get("contents", "read") != "write"
    assert release_yml.get("permissions", {}).get("contents") != "write"


def test_c6_both_tools_are_proven_to_be_inside_every_distribution(release_yml):
    linux = _all_runs(release_yml, "build-linux")
    windows = _all_runs(release_yml, "build-windows")
    assert "release_verify.py deb" in linux
    assert "--appimage-extract" in linux and "release_verify.py tree" in linux
    assert "release_verify.py tree" in windows
    assert "pyi-archive_viewer" in windows and "release_verify.py listing" in windows
    assert "/VERYSILENT" in windows, "the Setup installer must be installed for real"


# --- D. the final release job --------------------------------------------


def test_d1_the_release_job_waits_for_tests_and_both_builds(release_yml):
    needs = release_yml["jobs"]["release"]["needs"]
    assert {"verify-tests", "build-linux", "build-windows", "resolve-version"} <= set(needs)


def test_d2_only_a_tag_run_can_reach_the_release_job(release_yml):
    cond = release_yml["jobs"]["release"]["if"]
    assert "github.ref_type == 'tag'" in cond
    assert "always()" not in cond and "failure()" not in cond


def test_d3_only_the_release_job_can_write(release_yml):
    for name, job in release_yml["jobs"].items():
        wants_write = job.get("permissions", {}).get("contents") == "write"
        assert wants_write == (name == "release"), name
    assert release_yml.get("permissions", {}).get("contents") != "write"


def test_d4_the_release_job_downloads_both_artifacts_and_verifies_all_eight(release_yml):
    steps = _steps(release_yml, "release")
    download = _index(steps, lambda s: "download-artifact" in _uses(s), "download")
    verify = _index(steps, lambda s: "release_verify.py assets" in _runs(s), "verify")
    create = _index(steps, lambda s: "draft_release.py create" in _runs(s), "create")
    assert download < verify < create
    assert "--platform" not in _runs(steps[verify]), "the final check is of all eight"
    assert "sha256sum -c" in _runs(steps[verify])


def test_d5_nothing_in_the_workflow_can_publish(release_text):
    for publish in ("--draft=false", "draft: false", "action-gh-release",
                    "gh release edit", "--prerelease", "--latest"):
        assert publish not in release_text, publish


def test_d6_the_release_is_made_only_by_the_tested_draft_script(release_yml):
    for name, job in release_yml["jobs"].items():
        for step in job.get("steps", []):
            if "gh release" in _runs(step) or "releases" in _runs(step):
                raise AssertionError(f"{name} talks to the Releases API directly")
    assert "draft_release.py create" in _all_runs(release_yml, "release")


def test_d7_the_script_that_writes_the_release_only_ever_asks_for_a_draft():
    text = (ROOT / "scripts" / "draft_release.py").read_text(encoding="utf-8")
    assert '"--draft"' in text and '"--verify-tag"' in text
    for publish in ("--draft=false", "draft=false", "--prerelease", "--latest",
                    '"edit"', "PATCH"):
        assert publish not in text, publish


def test_d8_the_draft_is_read_back_after_it_is_created(release_yml):
    steps = _steps(release_yml, "release")
    create = _index(steps, lambda s: "draft_release.py create" in _runs(s), "create")
    readback = _index(steps, lambda s: "draft_release.py check" in _runs(s), "check")
    assert create < readback


def test_d9_a_tag_must_match_the_version_in_the_source(release_yml):
    steps = _steps(release_yml, "resolve-version")
    assert any("checkout" in _uses(s) for s in steps)
    script = _all_runs(release_yml, "resolve-version")
    assert "src/cove_compressor/__init__.py" in script
    assert re.search(
        r'if \[ "\$v" != "0\.0\.0-dev" \] && \[ "\$v" != "\$src" \]; then\n'
        r'\s*echo "::error::[^\n]*\n\s*exit 1\n\s*fi',
        script,
    ), "a version that differs from __version__ must fail the run"


def test_d10_release_notes_come_from_a_reviewed_file(release_yml):
    assert ".github/release-notes/v${VERSION}.md" in _all_runs(release_yml, "release")


def test_d11_a_second_run_cannot_race_the_first(release_yml):
    assert release_yml["concurrency"]["cancel-in-progress"] is False


def test_d12_the_notes_file_for_the_current_version_exists():
    notes = ROOT / ".github" / "release-notes" / f"v{_version()}.md"
    text = notes.read_text(encoding="utf-8")
    assert text.strip()
    assert "2.5.0" in text, "notes must say what changed since the previous release"
