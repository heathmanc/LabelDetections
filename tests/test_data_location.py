"""Tests for the configurable image-library location (headless, no cv2).

storage.py imports cv2, so the resolution functions are extracted from its AST
rather than imported. That keeps these runnable in a bare environment while
still exercising the real shipped code.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_STORAGE = Path(__file__).resolve().parents[1] / "bung_labeler" / "core" / "storage.py"
_WANT = {
    "APP_FOLDER_NAME", "DATA_DIR_ENV", "SUBDIRS",
    "_is_writable", "_app_root", "config_dir", "data_location_file",
    "read_configured_data_dir", "write_configured_data_dir",
    "resolve_data_dir", "_ensure_data_dirs",
}


def _load():
    tree = ast.parse(_STORAGE.read_text())
    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in _WANT)
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in _WANT)
    ]
    ns = {"sys": sys, "os": os, "json": json, "Path": Path, "__file__": str(_STORAGE)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "storage", "exec"), ns)
    return ns


class _Env:
    """Point config/env at a scratch area so tests never touch real settings."""

    def __init__(self, localappdata: str, data_dir_env: str | None = None):
        self.localappdata, self.data_dir_env = localappdata, data_dir_env

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in ("LOCALAPPDATA", "XDG_CONFIG_HOME", "BUNGVISION_DATA_DIR")}
        os.environ["LOCALAPPDATA"] = self.localappdata
        os.environ.pop("XDG_CONFIG_HOME", None)
        if self.data_dir_env is None:
            os.environ.pop("BUNGVISION_DATA_DIR", None)
        else:
            os.environ["BUNGVISION_DATA_DIR"] = self.data_dir_env
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def test_default_when_nothing_configured():
    ns = _load()
    with _Env(tempfile.mkdtemp()):
        assert ns["resolve_data_dir"]() == ns["_app_root"]() / "data"


def test_configured_dir_is_used():
    ns = _load()
    cfg, lib = tempfile.mkdtemp(), tempfile.mkdtemp()
    with _Env(cfg):
        ns["write_configured_data_dir"](lib)
        assert ns["read_configured_data_dir"]() == Path(lib)
        assert ns["resolve_data_dir"]() == Path(lib)


def test_env_var_overrides_configured_dir():
    ns = _load()
    cfg, lib, override = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    with _Env(cfg, data_dir_env=override):
        ns["write_configured_data_dir"](lib)
        assert ns["resolve_data_dir"]() == Path(override)


def test_reset_clears_configuration():
    ns = _load()
    cfg, lib = tempfile.mkdtemp(), tempfile.mkdtemp()
    with _Env(cfg):
        ns["write_configured_data_dir"](lib)
        ns["write_configured_data_dir"](None)
        assert ns["read_configured_data_dir"]() is None
        assert ns["resolve_data_dir"]() == ns["_app_root"]() / "data"


def test_pointer_file_lives_outside_the_data_dir():
    # It must be readable before the data location is known, so it cannot be
    # stored inside the folder it points at.
    ns = _load()
    cfg, lib = tempfile.mkdtemp(), tempfile.mkdtemp()
    with _Env(cfg):
        ns["write_configured_data_dir"](lib)
        pointer = ns["data_location_file"]()
        assert pointer.is_file()
        assert Path(lib) not in pointer.parents


def test_unreadable_config_falls_back_rather_than_raising():
    ns = _load()
    cfg = tempfile.mkdtemp()
    with _Env(cfg):
        bad = ns["data_location_file"]()
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{ not json", encoding="utf-8")
        assert ns["read_configured_data_dir"]() is None
        assert ns["resolve_data_dir"]() == ns["_app_root"]() / "data"


def test_ensure_data_dirs_creates_subfolders():
    ns = _load()
    lib = Path(tempfile.mkdtemp()) / "library"
    assert ns["_ensure_data_dirs"](lib) is True
    for name in ns["SUBDIRS"]:
        assert (lib / name).is_dir()


def test_ensure_data_dirs_reports_unusable_location():
    # Stands in for an offline network share: the parent is a regular file, so
    # the subfolders can never be created there.
    ns = _load()
    blocked = Path(tempfile.mkdtemp()) / "notadir"
    blocked.write_text("x")
    assert ns["_ensure_data_dirs"](blocked / "library") is False


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
