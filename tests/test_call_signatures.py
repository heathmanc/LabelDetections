"""Static guard: every core function the UI calls, called correctly.

The bug this exists for shipped as a dialog reading "export_all_labels_yolo()
got an unexpected keyword argument 'class_mode'". main_window was still passing
an argument the rewritten exporter had never accepted, and nothing caught it
because the core tests called the exporter directly with the right arguments.
A signature drift between the two halves is invisible to a test that only ever
looks at one of them.

Parsed rather than executed, so it needs no Qt and covers call sites that only
run when an operator presses something.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_UI = pathlib.Path(__file__).resolve().parents[1] / "label_detections" / "ui"


def _imported_names(tree: ast.AST) -> dict[str, str]:
    """``{local name: module it came from}`` for core imports."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and "label_detections" in node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = node.module
    return out


def _mismatches(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = _imported_names(tree)
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in imported:
            continue
        try:
            target = getattr(importlib.import_module(imported[name]), name)
            signature = inspect.signature(target)
        except Exception:
            continue
        if inspect.isclass(target) or not callable(target):
            continue
        # Values do not matter, only arity and keyword names.
        try:
            signature.bind(*[None] * len(node.args),
                           **{kw.arg: None for kw in node.keywords if kw.arg})
        except TypeError as exc:
            problems.append(f"{path.name}:{node.lineno} {name}(...) -> {exc}")
    return problems


def test_every_core_call_from_the_ui_binds():
    problems: list[str] = []
    for path in sorted(_UI.glob("*.py")):
        problems.extend(_mismatches(path))
    assert not problems, "stale call site(s):\n" + "\n".join(problems)


def test_the_guard_actually_catches_a_stale_call(tmp_path):
    """Pin the guard itself: a checker that never fires protects nothing."""
    bad = tmp_path / "stale.py"
    bad.write_text(
        "from label_detections.core.yolo_export import export_all_labels_yolo\n"
        "export_all_labels_yolo(class_mode='label_names')\n",
        encoding="utf-8",
    )
    problems = _mismatches(bad)
    assert problems and "class_mode" in problems[0]
