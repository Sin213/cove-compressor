import faulthandler
import os
import sys

# PyInstaller `--windowed` builds detach from the console, which sets
# `sys.stderr` to None. `faulthandler.enable()` without args tries to
# register stderr's fd and raises `RuntimeError: sys.stderr is None`
# before the GUI even starts. Point it at a real file when there's no
# stderr, so crash tracebacks still land somewhere instead of nuking
# the app on startup.
if sys.stderr is not None and hasattr(sys.stderr, "fileno"):
    try:
        faulthandler.enable()
    except (RuntimeError, OSError, ValueError):
        pass
else:
    try:
        if sys.platform == "win32":
            log_dir = os.path.join(
                os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                "CoveCompressor",
            )
        else:
            log_dir = os.path.join(os.path.expanduser("~"), ".cove-compressor")
        os.makedirs(log_dir, exist_ok=True)
        _fault_log = open(os.path.join(log_dir, "faulthandler.log"), "a", buffering=1)
        faulthandler.enable(file=_fault_log)
    except (OSError, RuntimeError, ValueError):
        pass

from PySide6.QtWidgets import QApplication  # noqa: E402

from . import theme  # noqa: E402
from .app import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Cove Compressor")
    app.setOrganizationName("Cove")
    theme.apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
