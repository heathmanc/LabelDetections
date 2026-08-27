"""Every name the package uses resolves.

Four separate NameErrors shipped in one release, all the same shape: a rename
left a dead reference behind on a path no test walked, and the code sat there
looking fine until an operator pressed the button. Two of them were in status
messages -- the work succeeded and then the line describing it raised.

A test per path would not have found them; there are too many paths, and the
ones that broke are exactly the ones nobody thought to cover. This reads the
source instead, so a dead reference cannot reach a release regardless of
whether anything exercises it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _pyflakes(*targets: str) -> list[str]:
    pytest.importorskip("pyflakes", reason="pyflakes is the checker this test runs")
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        cwd=ROOT, capture_output=True, text=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_no_undefined_names_anywhere_in_the_package():
    """The bug class: `other_counts`, `battery_count` and `bung_count` in the
    auto-label summary, `items` in the review-queue scorer, and a
    `save_annotations` the module never imported."""
    findings = [line for line in _pyflakes("label_detections", "main.py")
                if "undefined name" in line]
    assert findings == []


def test_no_names_used_before_they_are_assigned():
    """The same failure with a different spelling -- a local read on a path
    that runs before the branch assigning it."""
    findings = [line for line in _pyflakes("label_detections", "main.py")
                if "local variable" in line and "referenced before assignment" in line]
    assert findings == []


def test_every_internal_import_target_exists():
    """pyflakes cannot see across modules, and the same rename that leaves a
    dead local reference leaves dead imports too -- `make_background_record`
    was imported inside two functions and defined nowhere, so marking an image
    background raised in the UI and came back as a per-file error on import.
    Function-local imports make these invisible until the line runs.
    """
    import ast
    import importlib

    problems: list[str] = []
    package = ROOT / "label_detections"
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = list(parts[:-1])
                for _ in range(node.level - 1):
                    base = base[:-1]
                target = ".".join(base + ([node.module] if node.module else []))
            elif (node.module or "").startswith("label_detections"):
                target = node.module
            else:
                continue
            try:
                module = importlib.import_module(target)
            except Exception as exc:  # a module that cannot even be imported
                problems.append(f"{path.name}:{node.lineno}: {target}: {exc}")
                continue
            for alias in node.names:
                if alias.name == "*" or hasattr(module, alias.name):
                    continue
                # `from . import submodule` before the submodule is loaded.
                try:
                    importlib.import_module(f"{target}.{alias.name}")
                except Exception:
                    problems.append(
                        f"{path.name}:{node.lineno}: {target} has no '{alias.name}'")
    assert problems == []
