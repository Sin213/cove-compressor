"""Regression tests: every ffmpeg/ffprobe spawn must run with a scrubbed env.

Inside an AppImage/PyInstaller bundle, LD_LIBRARY_PATH points at the bundle's
own libs. If that leaks into a spawned system ffmpeg, the bundle's stale
libfontconfig shadows the host one and libass fails its symbol lookup
(FcConfigSetDefaultSubstitute), breaking every encode. clean_subprocess_env()
strips those vars; these tests assert each spawn site actually uses it.
"""

import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402


class CleanSubprocessEnvTests(unittest.TestCase):
    def test_strips_bundle_lib_vars(self):
        fake = {
            "LD_LIBRARY_PATH": "/bundle/_internal",
            "QT_PLUGIN_PATH": "/bundle/qt",
            "PYTHONHOME": "/bundle",
            "PATH": "/usr/bin",
        }
        with patch.dict(os.environ, fake, clear=True):
            env = compressor.clean_subprocess_env()
        self.assertNotIn("LD_LIBRARY_PATH", env)
        self.assertNotIn("QT_PLUGIN_PATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_restores_host_lib_path(self):
        fake = {
            "LD_LIBRARY_PATH": "/bundle/_internal",
            "LD_LIBRARY_PATH_ORIG": "/usr/lib",
        }
        with patch.dict(os.environ, fake, clear=True):
            env = compressor.clean_subprocess_env()
        self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/lib")


class SpawnSiteEnvTests(unittest.TestCase):
    """Each ffmpeg/ffprobe spawn must pass env=clean_subprocess_env()."""

    def test_run_ffmpeg_scrubs_env(self):
        captured = {}
        real_popen = compressor.subprocess.Popen

        def spy(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            return real_popen(cmd, **kwargs)

        cmd = [
            sys.executable,
            "-c",
            "import sys; print('frame=1 time=00:00:01.00', "
            "file=sys.stderr, flush=True)",
        ]
        fake = {"LD_LIBRARY_PATH": "/bundle/_internal",
                "PATH": os.environ.get("PATH", "")}
        with patch.dict(os.environ, fake, clear=True), \
                patch.object(compressor.subprocess, "Popen", spy):
            compressor.run_ffmpeg(cmd, threading.Event(), duration=1)

        self.assertIsNotNone(captured["env"])
        self.assertNotIn("LD_LIBRARY_PATH", captured["env"])

    def _assert_run_calls_scrubbed(self, call):
        captured = []

        class _Result:
            returncode = 0
            stdout = "hevc_nvenc h264_nvenc hevc_amf h264_amf"
            stderr = ""

        def spy(cmd, **kwargs):
            captured.append(kwargs.get("env"))
            return _Result()

        fake = {"LD_LIBRARY_PATH": "/bundle/_internal",
                "PATH": os.environ.get("PATH", "")}
        with patch.dict(os.environ, fake, clear=True), \
                patch.object(compressor.subprocess, "run", spy):
            call()

        self.assertTrue(captured, "expected at least one subprocess.run call")
        for env in captured:
            self.assertIsNotNone(env)
            self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_probe_nvenc_scrubs_env(self):
        self._assert_run_calls_scrubbed(
            lambda: compressor._probe_nvenc("hevc_nvenc"))

    def test_probe_amf_scrubs_env(self):
        self._assert_run_calls_scrubbed(
            lambda: compressor._probe_amf("hevc_amf"))

    def test_ffprobe_duration_scrubs_env(self):
        with patch.object(compressor.shutil, "which", return_value="ffprobe"):
            self._assert_run_calls_scrubbed(
                lambda: compressor.ffprobe_duration(Path("dummy.mp4")))


if __name__ == "__main__":
    unittest.main()
