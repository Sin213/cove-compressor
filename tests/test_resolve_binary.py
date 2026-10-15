"""Frozen tool selection and real consumer execution regressions.

Windows-name simulations exercise naming only, not native Windows execution.
"""

import os
import shlex
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "src"), *sys.path]):
    from cove_compressor import compressor


class ResolveBinaryTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary).resolve()
        self.checkout = self.root / "checkout"
        self.bundle = self.root / "bundle"
        self.adjacent = self.root / "app"
        self.host = self.root / "path"
        for directory in (self.checkout, self.bundle, self.adjacent, self.host):
            directory.mkdir()
        self.enterContext(patch.object(
            compressor, "__file__", str(self.checkout / "src" / "cove_compressor" / "compressor.py")
        ))
        self.enterContext(patch.object(sys, "executable", str(self.adjacent / "Cove")))

    def tool(self, directory, name, platform="linux", executable=True):
        path = directory / (name + ".exe" if platform == "win32" else name)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755 if executable else 0o644)
        return path

    def frozen(self, platform="linux", bundle=True):
        self.enterContext(patch.object(sys, "frozen", True, create=True))
        self.enterContext(patch.object(sys, "platform", platform))
        if bundle:
            self.enterContext(patch.object(sys, "_MEIPASS", str(self.bundle), create=True))
        else:
            self.enterContext(patch.dict(sys.__dict__, {}))
            sys.__dict__.pop("_MEIPASS", None)

    def layout(self, platform, onefile):
        self.frozen(platform)
        # Onedir resources can live in the bundle's internal resource root.
        # Onefile extraction is separate from the launcher directory.
        self.bundle = self.root / ("extraction" if onefile else "app/_internal")
        self.bundle.mkdir(parents=True)
        with patch.object(sys, "_MEIPASS", str(self.bundle)):
            for name in ("ffmpeg", "ffprobe"):
                with self.subTest(name=name):
                    bundled = self.tool(self.bundle, name, platform)
                    self.tool(self.adjacent, name, platform)
                    self.tool(self.checkout, name, platform)
                    host = self.tool(self.host, name, platform)
                    with patch.object(compressor.shutil, "which", return_value=str(host)) as which:
                        self.assertEqual(compressor._resolve_binary(name), str(bundled))
                        which.assert_not_called()

    def test_linux_onedir_bundle_beats_adjacent_and_path(self):
        self.layout("linux", False)

    def test_linux_onefile_bundle_beats_adjacent_and_path(self):
        self.layout("linux", True)

    def test_windows_name_onedir_bundle_beats_adjacent_and_path(self):
        self.layout("win32", False)

    def test_windows_name_onefile_bundle_beats_adjacent_and_path(self):
        self.layout("win32", True)

    def test_bundle_beats_path_without_adjacent(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                bundled = self.tool(self.bundle, name)
                host = self.tool(self.host, name)
                with patch.object(compressor.shutil, "which", return_value=str(host)) as which:
                    self.assertEqual(compressor._resolve_binary(name), str(bundled))
                    which.assert_not_called()

    def test_frozen_never_consults_checkout_ancestor(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                self.tool(self.checkout, name)
                host = self.tool(self.host, name)
                with patch.object(compressor.shutil, "which", return_value=str(host)) as which:
                    self.assertEqual(compressor._resolve_binary(name), str(host))
                    which.assert_called_once_with(name)

    def test_missing_meipass_uses_adjacent(self):
        self.frozen(bundle=False)
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                adjacent = self.tool(self.adjacent, name)
                with patch.object(compressor.shutil, "which") as which:
                    self.assertEqual(compressor._resolve_binary(name), str(adjacent))
                    which.assert_not_called()

    def test_missing_meipass_skips_checkout_for_path(self):
        self.frozen(bundle=False)
        self.tool(self.checkout, "ffmpeg")
        with patch.object(compressor.shutil, "which", return_value="host-ffmpeg") as which:
            self.assertEqual(compressor._resolve_binary("ffmpeg"), "host-ffmpeg")
            which.assert_called_once_with("ffmpeg")

    def test_missing_bundle_candidate_uses_adjacent(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                adjacent = self.tool(self.adjacent, name)
                with patch.object(compressor.shutil, "which") as which:
                    self.assertEqual(compressor._resolve_binary(name), str(adjacent))
                    which.assert_not_called()

    def test_bundle_directory_uses_adjacent(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                (self.bundle / name).mkdir()
                adjacent = self.tool(self.adjacent, name)
                with patch.object(compressor.shutil, "which") as which:
                    self.assertEqual(compressor._resolve_binary(name), str(adjacent))
                    which.assert_not_called()

    @unittest.skipIf(os.name == "nt", reason="POSIX executable permissions")
    def test_posix_nonexecutable_bundle_uses_adjacent(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                self.tool(self.bundle, name, executable=False)
                adjacent = self.tool(self.adjacent, name)
                with patch.object(compressor.shutil, "which") as which:
                    self.assertEqual(compressor._resolve_binary(name), str(adjacent))
                    which.assert_not_called()

    @unittest.skipIf(os.name == "nt", reason="POSIX executable permissions")
    def test_posix_nonexecutable_adjacent_uses_path(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                self.tool(self.adjacent, name, executable=False)
                with patch.object(compressor.shutil, "which", return_value="host-tool") as which:
                    self.assertEqual(compressor._resolve_binary(name), "host-tool")
                    which.assert_called_once_with(name)

    def test_invalid_local_directories_use_path(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                (self.bundle / name).mkdir()
                (self.adjacent / name).mkdir()
                with patch.object(compressor.shutil, "which", return_value="host-tool"):
                    self.assertEqual(compressor._resolve_binary(name), "host-tool")

    def test_windows_name_regular_file_needs_no_posix_exec_bit(self):
        self.frozen("win32")
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                bundled = self.tool(self.bundle, name, "win32", executable=False)
                with patch.object(compressor.shutil, "which") as which:
                    self.assertEqual(compressor._resolve_binary(name), str(bundled))
                    which.assert_not_called()

    def test_frozen_missing_everywhere_returns_bare_name(self):
        self.frozen()
        for name in ("ffmpeg", "ffprobe"):
            with self.subTest(name=name):
                with patch.object(compressor.shutil, "which", return_value=None) as which:
                    self.assertEqual(compressor._resolve_binary(name), name)
                    which.assert_called_once_with(name)

    def test_nonfrozen_checkout_keeps_is_file_only_precedence(self):
        with patch.object(sys, "frozen", False, create=True):
            for platform in ("linux", "win32"):
                with patch.object(sys, "platform", platform):
                    for name in ("ffmpeg", "ffprobe"):
                        with self.subTest(platform=platform, name=name):
                            checkout = self.tool(self.checkout, name, platform, executable=False)
                            self.tool(self.bundle, name, platform)
                            self.tool(self.adjacent, name, platform)
                            with patch.object(compressor.shutil, "which") as which:
                                self.assertEqual(compressor._resolve_binary(name), str(checkout))
                                which.assert_not_called()

    def test_nonfrozen_path_then_bare_name(self):
        with patch.object(sys, "frozen", False, create=True):
            for name in ("ffmpeg", "ffprobe"):
                with self.subTest(name=name):
                    with patch.object(compressor.shutil, "which", return_value="host-tool") as which:
                        self.assertEqual(compressor._resolve_binary(name), "host-tool")
                        which.assert_called_once_with(name)
                    with patch.object(compressor.shutil, "which", return_value=None):
                        self.assertEqual(compressor._resolve_binary(name), name)

    def test_empty_or_relative_meipass_uses_adjacent_without_path_lookup(self):
        self.frozen()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        for bundle_root in ("", "rel-bundle"):
            decoy_dir = self.root / bundle_root
            decoy_dir.mkdir(exist_ok=True)
            with patch.object(sys, "_MEIPASS", bundle_root):
                for name in ("ffmpeg", "ffprobe"):
                    with self.subTest(bundle_root=bundle_root, name=name):
                        marker = self.root / "resolver-marker"
                        self.fake(decoy_dir, name, "cwd-decoy", marker, 1)
                        adjacent = self.fake(self.adjacent, name, "adjacent", marker, 7)
                        host = self.fake(self.host, name, "path", marker, 99)
                        with patch.object(compressor.shutil, "which", return_value=str(host)) as which:
                            self.assertEqual(compressor._resolve_binary(name), str(adjacent))
                            which.assert_not_called()

    @unittest.skipIf(sys.platform == "win32", "Real fake-tool execution requires POSIX /bin/sh")
    def test_empty_meipass_real_consumers_run_adjacent_identity(self):
        self.frozen()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)
        with patch.object(sys, "_MEIPASS", ""), patch.dict(
            os.environ, {"PATH": str(self.host) + os.pathsep + os.environ.get("PATH", "")}
        ):
            for name in ("ffmpeg", "ffprobe"):
                with self.subTest(name=name):
                    marker = self.root / (name + "-adjacent-marker")
                    self.fake(self.root, name, "cwd-decoy", marker, 1)
                    adjacent = self.fake(self.adjacent, name, "adjacent", marker, 7.25)
                    self.fake(self.host, name, "path", marker, 99)
                    resolved = compressor._resolve_binary(name)
                    constant = "FFPROBE_BIN" if name == "ffprobe" else "FFMPEG_BIN"
                    with patch.object(compressor, constant, resolved):
                        if name == "ffprobe":
                            duration = compressor.ffprobe_duration(self.root / "input.mp4")
                        else:
                            progress = []
                            result = compressor.run_ffmpeg(
                                [compressor.FFMPEG_BIN, "-i", "input.mp4", "output.mp4"],
                                threading.Event(), duration=2, on_progress=progress.append,
                            )
                    self.assertEqual(marker.read_text(), "adjacent\n")
                    self.assertEqual(resolved, str(adjacent))
                    if name == "ffprobe":
                        self.assertEqual(duration, 7.25)
                    else:
                        self.assertEqual(result[0], 0)
                        self.assertEqual(progress, [50.0])

    def fake(self, directory, name, identity, marker, duration):
        path = self.tool(directory, name)
        output = (
            f"printf '%s\\n' '{duration}'\n" if name == "ffprobe"
            else "printf '%s\\n' 'frame=1 time=00:00:01.00 bitrate=1kbits/s' >&2\n"
        )
        path.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' '{identity}' >> {shlex.quote(str(marker))}\n"
            + output + "exit 0\n"
        )
        return path

    @unittest.skipIf(sys.platform == "win32", "Real fake-tool execution requires POSIX /bin/sh")
    def test_real_ffprobe_runs_bundled_identity(self):
        self.frozen()
        marker = self.root / "probe-marker"
        bundled = self.fake(self.bundle, "ffprobe", "bundled-probe", marker, 7.25)
        self.fake(self.host, "ffprobe", "path-probe", marker, 99.0)
        with patch.dict(os.environ, {"PATH": str(self.host) + os.pathsep + os.environ.get("PATH", "")}):
            resolved = compressor._resolve_binary("ffprobe")
            with patch.object(compressor, "FFPROBE_BIN", resolved):
                duration = compressor.ffprobe_duration(self.root / "input.mp4")
        self.assertEqual(duration, 7.25)
        self.assertEqual(marker.read_text(), "bundled-probe\n")
        self.assertEqual(resolved, str(bundled))

    @unittest.skipIf(sys.platform == "win32", "Real fake-tool execution requires POSIX /bin/sh")
    def test_real_ffmpeg_runs_bundled_identity_and_progress(self):
        self.frozen()
        marker = self.root / "encode-marker"
        bundled = self.fake(self.bundle, "ffmpeg", "bundled-encoder", marker, 0)
        self.fake(self.host, "ffmpeg", "path-encoder", marker, 0)
        progress = []
        with patch.dict(os.environ, {"PATH": str(self.host) + os.pathsep + os.environ.get("PATH", "")}):
            resolved = compressor._resolve_binary("ffmpeg")
            with patch.object(compressor, "FFMPEG_BIN", resolved):
                result = compressor.run_ffmpeg(
                    [compressor.FFMPEG_BIN, "-i", "input.mp4", "output.mp4"],
                    threading.Event(), duration=2, on_progress=progress.append,
                )
        self.assertEqual(result[0], 0)
        self.assertEqual(progress, [50.0])
        self.assertEqual(marker.read_text(), "bundled-encoder\n")
        self.assertEqual(resolved, str(bundled))
