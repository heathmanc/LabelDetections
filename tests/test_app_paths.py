"""Tests for frozen/installed data-directory resolution (headless, no cv2).

storage.py imports cv2/numpy, so the functions under test are extracted from
its AST rather than imported. That keeps these runnable in a bare environment
and still tests the real shipped code rather than a copy.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STORAGE = Path(__file__).resolve().parents[1] / "bung_labeler" / "core" / "storage.py"
_WANT = {"_is_writable", "_app_root", "APP_FOLDER_NAME"}


def _load():
    tree = ast.parse(_STORAGE.read_text())
    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in _WANT)
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in _WANT)
    ]
    ns = {"sys": sys, "os": os, "Path": Path, "__file__": str(_STORAGE)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "storage", "exec"), ns)
    return ns


class _FrozenEnv:
    """Temporarily present as a frozen app with a given executable path."""

    def __init__(self, exe: str, localappdata: str | None = None):
        self.exe, self.localappdata = exe, localappdata

    def __enter__(self):
        self._frozen = getattr(sys, "frozen", None)
        self._exe = sys.executable
        self._lad = os.environ.get("LOCALAPPDATA")
        sys.frozen = True
        sys.executable = self.exe
        if self.localappdata is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self.localappdata
        return self

    def __exit__(self, *exc):
        if self._frozen is None:
            if hasattr(sys, "frozen"):
                del sys.frozen
        else:
            sys.frozen = self._frozen
        sys.executable = self._exe
        if self._lad is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = self._lad
        return False


def test_source_mode_uses_repo_root():
    ns = _load()
    assert ns["_app_root"]() == _STORAGE.parents[2]


def test_portable_build_keeps_data_beside_exe():
    ns = _load()
    tmp = Path(tempfile.mkdtemp())
    with _FrozenEnv(str(tmp / "App.exe")):
        assert ns["_app_root"]() == tmp


def test_installed_build_falls_back_to_localappdata():
    # Stands in for C:\Program Files: the exe's parent is a regular file, so a
    # data/ subdirectory can never be created there.
    ns = _load()
    blocked = Path(tempfile.mkdtemp()) / "notadir"
    blocked.write_text("x")
    lad = tempfile.mkdtemp()
    with _FrozenEnv(str(blocked / "App.exe"), localappdata=lad):
        assert ns["_app_root"]() == Path(lad) / "BungVisionLabelStudio"


def test_is_writable_rejects_uncreatable_path():
    ns = _load()
    blocked = Path(tempfile.mkdtemp()) / "notadir"
    blocked.write_text("x")
    assert ns["_is_writable"](blocked / "data") is False


def test_is_writable_accepts_and_cleans_up():
    ns = _load()
    tmp = Path(tempfile.mkdtemp()) / "fresh"
    assert ns["_is_writable"](tmp) is True
    # The probe file must not be left behind.
    assert list(tmp.iterdir()) == []


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
