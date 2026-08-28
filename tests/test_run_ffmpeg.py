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


# ── Group D — MP4 faststart (moov relocation) finalization watchdog ──────────
#
# `-movflags +faststart` makes ffmpeg rewrite the finished file to move the
# moov atom to the front. That phase emits no `time=` progress and can run for
# minutes on a large file, so the ordinary encode stall watchdog would kill a
# perfectly healthy job. The finalization allowance covers exactly that phase —
# it is not a global relaxation, and entering it is not a success signal.

MOOV_LINE = ("[mp4 @ 000001d3] Starting second pass: moving the moov atom "
             "to the beginning of the file")


class MoovFinalizationWatchdogTests(unittest.TestCase):
    @staticmethod
    def _script(*steps: str) -> list:
        """A fake ffmpeg: `steps` are python statements run in order."""
        body = "import sys, time\n" + "\n".join(steps) + "\n"
        return [sys.executable, "-c", body]

    @staticmethod
    def _emit(text: str) -> str:
        return f"print({text!r}, file=sys.stderr, flush=True)"

    @staticmethod
    def _sleep(seconds: float) -> str:
        return f"time.sleep({seconds})"

    @staticmethod
    def _exit(code: int) -> str:
        return f"sys.exit({code})"

    def test_d1_moov_line_refreshes_watchdog_activity(self):
        """Both allowances are equal here, so only the refreshed timestamp can
        keep this job alive: the two silent gaps each fit, their sum does not.
        """
        command = self._script(
            self._emit("frame=1 time=00:00:01.00"),
            self._sleep(1.2),
            self._emit(MOOV_LINE),
            self._sleep(1.3),
            self._exit(0),
        )
        with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 1.5), \
             patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 1.5):
            rc, message = compressor.run_ffmpeg(command, threading.Event())

        self.assertEqual(rc, 0, message)

    def test_d2_moov_line_enters_the_extended_finalization_allowance(self):
        """A moov relocation far longer than the encode allowance survives."""
        command = self._script(
            self._emit("frame=1 time=00:00:01.00"),
            self._emit(MOOV_LINE),
            self._sleep(1.2),
            self._exit(0),
        )
        with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 0.05), \
             patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 30):
            rc, message = compressor.run_ffmpeg(command, threading.Event())

        self.assertEqual(rc, 0, message)

    def test_d2_finalization_allowance_is_bounded_not_infinite(self):
        started_at = time.monotonic()
        command = self._script(
            self._emit(MOOV_LINE),
            self._sleep(10),
        )
        with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 0.05), \
             patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 0.05):
            rc, message = compressor.run_ffmpeg(command, threading.Event())

        self.assertEqual(rc, -3)
        self.assertIn("finaliz", message)
        self.assertLess(time.monotonic() - started_at, 5)

    def test_d3_moov_line_never_reports_100_percent(self):
        progress = []
        command = self._script(
            self._emit("frame=1 time=00:00:01.00"),
            self._emit(MOOV_LINE),
            self._exit(0),
        )
        rc, _ = compressor.run_ffmpeg(
            command, threading.Event(), duration=10,
            on_progress=progress.append)

        self.assertEqual(rc, 0)
        self.assertEqual(progress, [10.0])
        self.assertNotIn(100.0, progress)

    def test_d4_nonzero_exit_after_moov_line_is_still_an_error(self):
        command = self._script(
            self._emit(MOOV_LINE),
            self._emit("Error writing trailer: No space left on device"),
            self._exit(1),
        )
        with patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 30):
            rc, message = compressor.run_ffmpeg(command, threading.Event())

        self.assertEqual(rc, 1)
        self.assertIn("No space left", message)

    def test_d5_successful_exit_after_moov_line_follows_the_success_path(self):
        command = self._script(
            self._emit("frame=1 time=00:00:05.00"),
            self._emit(MOOV_LINE),
            self._exit(0),
        )
        rc, _ = compressor.run_ffmpeg(command, threading.Event())

        self.assertEqual(rc, 0)

    def test_d6_ordinary_encode_still_uses_the_encode_stall_timeout(self):
        """The finalization allowance must not leak into normal encoding."""
        started_at = time.monotonic()
        with patch.object(compressor, "ENCODE_STALL_TIMEOUT", 0.05), \
             patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 600):
            rc, message = compressor.run_ffmpeg(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                threading.Event())

        self.assertEqual(rc, -3)
        self.assertIn("no encoding progress", message)
        self.assertLess(time.monotonic() - started_at, 5)

    def test_d_cancellation_stays_responsive_during_finalization(self):
        cancel_flag = threading.Event()
        command = self._script(
            self._emit(MOOV_LINE),
            self._sleep(10),
        )
        timer = threading.Timer(0.2, cancel_flag.set)
        started_at = time.monotonic()
        timer.start()
        try:
            with patch.object(compressor, "FINALIZE_STALL_TIMEOUT", 600):
                rc, message = compressor.run_ffmpeg(command, cancel_flag)
        finally:
            timer.cancel()

        self.assertEqual((rc, message), (-2, "cancelled"))
        self.assertLess(time.monotonic() - started_at, 5)

    def test_d_default_finalization_allowance_exceeds_the_encode_allowance(self):
        self.assertGreater(compressor.FINALIZE_STALL_TIMEOUT,
                           compressor.ENCODE_STALL_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
