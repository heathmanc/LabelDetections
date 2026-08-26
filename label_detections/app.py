"""Application entry point."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is not installed, so the desktop UI cannot start.\n"
            "  pip install -r requirements.txt\n\n"
            "The questionnaires can still be reviewed without it:\n"
            "  python -m label_detections.preview labels\n"
            "  python -m label_detections.preview recipe",
            file=sys.stderr,
        )
        return 1

    from .ui.launcher import Launcher

    app = QApplication(argv if argv is not None else sys.argv)
    window = Launcher()
    window.show()
    return app.exec()
