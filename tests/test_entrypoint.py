"""Structural guards on main.py's startup order (headless, no Qt needed).

These encode ordering that is invisible at runtime from source but breaks the
packaged build badly, so they are checked statically rather than by import.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MAIN = Path(__file__).resolve().parents[1] / "main.py"


def _top_level():
    return ast.parse(_MAIN.read_text()).body


def _index_of(needle: str):
    for i, node in enumerate(_top_level()):
        if needle in ast.dump(node):
            return i
    return None


def test_freeze_support_is_called():
    # Without it, every DataLoader worker spawned during training re-executes
    # the frozen exe and opens another GUI while training stalls.
    assert _index_of("freeze_support") is not None


def test_freeze_support_runs_before_worker_dispatch():
    assert _index_of("freeze_support") < _index_of("maybe_run_worker")


def test_freeze_support_runs_before_qt_import():
    assert _index_of("freeze_support") < _index_of("main_window")


def test_nothing_executes_before_freeze_support():
    # freeze_support() must see a pristine interpreter: anything that runs
    # first would also run in every spawned child.
    for node in _top_level()[: _index_of("freeze_support")]:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)), \
            f"statement runs before freeze_support(): {ast.dump(node)[:80]}"


def test_worker_dispatch_precedes_qt_import():
    # A worker process must never construct a QApplication.
    assert _index_of("maybe_run_worker") < _index_of("main_window")


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


def test_the_launcher_takes_ultralytics_offline_before_it_is_imported():
    """A py-spy dump of a live session showed an Ultralytics telemetry thread
    stuck in an SSL handshake. It is not why anything hung, but a tool running
    production hardware should not have a thread whose job is to phone home.

    Order matters: YOLO_OFFLINE is read by ultralytics.utils at import time.
    """
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text()
    offline = source.index('YOLO_OFFLINE')
    imported = source.index('from label_detections')
    assert offline < imported, (
        "the switch is set after the import that reads it")
