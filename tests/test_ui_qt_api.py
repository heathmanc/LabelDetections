"""Static guards against Qt API misuse that only fails at runtime.

PySide6 is not installed in the headless test environment, so these scan the
source rather than importing it. Each pattern here has actually broken the app.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_UI = Path(__file__).resolve().parents[1] / "bung_labeler" / "ui"


def _sources():
    return sorted(_UI.glob("*.py"))


def test_findchildren_is_never_passed_a_tuple():
    # QObject.findChildren takes a single type, not a tuple like isinstance.
    # Passing a tuple raises "called with wrong argument types" at startup.
    bad = []
    for path in _sources():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"findChildren\(\s*\(", line):
                bad.append(f"{path.name}:{i}: {line.strip()}")
    assert not bad, "findChildren() called with a tuple:\n" + "\n".join(bad)


def test_event_filter_uses_canonical_enum():
    # QEvent.Type.Wheel is valid across all PySide6 6.x releases.
    src = (_UI / "main_window.py").read_text()
    if "def eventFilter" in src:
        assert "QEvent.Type.Wheel" in src, "use QEvent.Type.Wheel, not the alias"


def test_ui_modules_parse():
    for path in _sources():
        ast.parse(path.read_text())


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
