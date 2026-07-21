"""Regression tests for ffmpeg process supervision."""

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cove_compressor import compressor  # noqa: E402


class RunFfmpegTests(unittest.TestCase):
    @staticmethod
    def _silent_command():
        return [sys.executable, "-c", "import time; time.sleep(10)"]

    def test_silent_process_times_out(self):
        started_at = time.monotonic()
        with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 0.05):
            rc, message = compressor.run_ffmpeg(
                self._silent_command(), threading.Event()
            )

        self.assertEqual(rc, -3)
        self.assertIn("no encoding progress", message)
        self.assertLess(time.monotonic() - started_at, 3)

    def test_cancel_remains_responsive_while_process_is_silent(self):
        cancel_flag = threading.Event()
        timer = threading.Timer(0.05, cancel_flag.set)
        started_at = time.monotonic()
        timer.start()
        try:
            rc, message = compressor.run_ffmpeg(
                self._silent_command(), cancel_flag
            )
        finally:
            timer.cancel()

        self.assertEqual((rc, message), (-2, "cancelled"))
        self.assertLess(time.monotonic() - started_at, 3)

    def test_progress_fires_start_once(self):
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('frame=1 time=00:00:01.00', file=sys.stderr, flush=True); "
                "print('frame=2 time=00:00:02.00', file=sys.stderr, flush=True)"
            ),
        ]
        starts = []
        progress = []

        rc, _ = compressor.run_ffmpeg(
            command,
            threading.Event(),
            duration=2,
            on_progress=progress.append,
            on_start=lambda: starts.append(True),
        )

        self.assertEqual(rc, 0)
        self.assertEqual(starts, [True])
        self.assertEqual(progress, [50.0, 100.0])


if __name__ == "__main__":
    unittest.main()
