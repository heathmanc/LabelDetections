"""Filesystem layout and JSON persistence.

The data-root resolution (env override, operator-chosen folder, network-share
fallback) is ported from BungVision Label Studio, where it was proven on
shared industrial workstations, and is kept behaviour-for-behaviour.

The layout below is not. This tool trains **one label at a time**: a dataset
is a folder of images of a single label, gathered from wherever they come
from, annotated and exported on its own schedule. Nothing here knows about
batteries, cameras or capture sets -- the vision program's recipe is what
assembles labels into an inspection, and it lives on the runtime side.

    captures/<label_id>/<image>.jpg
    labels/<label_id>/<image>.json
    library/labels.json          every label definition
    recipes/<recipe>.json        the vision program's bill of labels + ROIs

A dataset folder per label means a label can be added, trained and shipped
without touching any other label's data, and it makes "how many examples do I
have of this label" a directory listing.

Stdlib only: no cv2/numpy here, so the layout rules stay unit testable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

APP_FOLDER_NAME = "LabelVisionStudio"
DATA_DIR_ENV = "LABELVISION_DATA_DIR"

SUBDIRS = ("captures", "labels", "recipes", "exports", "library", "models")


def _is_writable(path: Path) -> bool:
    """True if we can actually create files in ``path``.

    Checked by writing, not by ``os.access()`` -- on Windows ``os.access()``
    reports success for directories Explorer/UAC will still refuse, and
    Program Files is exactly that case.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except Exception:
        return False


def _app_root() -> Path:
    """Fallback directory holding ``data`` when no location is configured.

    From source this is the repo root. When frozen, prefer beside the .exe so a
    portable extract stays one copyable folder, and fall back to LOCALAPPDATA
    when that is read-only (the Program Files install case). Never anchored to
    ``__file__`` when frozen -- that points inside the bundle, which an upgrade
    wipes.
    """
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parents[2]
    exe_dir = Path(sys.executable).resolve().parent
    if _is_writable(exe_dir / "data"):
        return exe_dir
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home()) / APP_FOLDER_NAME


def config_dir() -> Path:
    """Per-user directory for settings that must be read *before* the data root."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / APP_FOLDER_NAME
    return Path.home() / f".{APP_FOLDER_NAME.lower()}"


def data_location_file() -> Path:
    return config_dir() / "data_location.json"


def read_configured_data_dir() -> Path | None:
    try:
        path = data_location_file()
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8")).get("data_dir", "")
    except Exception:
        return None
    raw = str(raw).strip()
    return Path(raw) if raw else None


def resolve_data_dir() -> Path:
    """Active data folder: env var, then configured pointer, then the default."""
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    if env:
        return Path(env)
    configured = read_configured_data_dir()
    if configured is not None:
        return configured
    return _app_root() / "data"


def write_configured_data_dir(path: Path | str | None) -> None:
    """Persist the library location. ``None`` restores the default.

    Takes effect on next launch: the module-level paths below are imported
    directly all over the app.
    """
    target = data_location_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {}
    try:
        if target.is_file():
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
    except Exception:
        config = {}
    config["data_dir"] = "" if path is None else str(Path(path))
    target.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _ensure_data_dirs(base: Path) -> bool:
    try:
        for name in SUBDIRS:
            (base / name).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


# A configured library on a network share may simply be offline. Falling back
# keeps the app launchable; raising at import would leave it unusable until
# someone hand-edited a config file.
DATA_DIR_FALLBACK_REASON = ""
DATA_DIR = resolve_data_dir()
if not _ensure_data_dirs(DATA_DIR):
    _unusable = DATA_DIR
    DATA_DIR = _app_root() / "data"
    DATA_DIR_FALLBACK_REASON = (
        f"The configured data folder could not be opened:\n{_unusable}\n\n"
        f"Using the default location instead:\n{DATA_DIR}"
    )
    _ensure_data_dirs(DATA_DIR)

CAPTURE_DIR = DATA_DIR / "captures"
LABEL_DIR = DATA_DIR / "labels"
RECIPE_DIR = DATA_DIR / "recipes"
EXPORT_DIR = DATA_DIR / "exports"
LIBRARY_DIR = DATA_DIR / "library"
MODEL_DIR = DATA_DIR / "models"

LABEL_LIBRARY_PATH = LIBRARY_DIR / "labels.json"
CLASS_CONFIG_PATH = LIBRARY_DIR / "classes.json"


# --- name sanitising -------------------------------------------------------

def safe_token(text: str, fallback: str = "Unnamed") -> str:
    """Filesystem- and class-name-safe token."""
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text).strip())
    return cleaned.strip("_") or fallback


# --- per-label dataset paths -----------------------------------------------

def dataset_folder(label_id: str, root: Path | None = None) -> Path:
    """Where one label's training images live."""
    return (root or CAPTURE_DIR) / safe_token(label_id)


def label_folder(label_id: str, root: Path | None = None) -> Path:
    """Where one label's annotation sidecars live."""
    return (root or LABEL_DIR) / safe_token(label_id)


def annotation_path(label_id: str, image_name: str, root: Path | None = None) -> Path:
    """The sidecar beside a training image, matched by stem.

    Sidecars sit in a parallel tree rather than beside the image so a dataset
    folder can be handed to anything that reads a directory of images without
    it tripping over JSON.
    """
    stem = Path(str(image_name)).stem
    return label_folder(label_id, root) / f"{stem}.json"


def list_datasets(root: Path | None = None) -> list[str]:
    """Label ids that have a dataset folder, sorted."""
    base = root or CAPTURE_DIR
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(label_id: str, root: Path | None = None) -> list[Path]:
    """Training images for one label, sorted, hidden files skipped."""
    folder = dataset_folder(label_id, root)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_SUFFIXES
    )


def recipe_path(safe_name: str, root: Path | None = None) -> Path:
    return (root or RECIPE_DIR) / f"{safe_token(safe_name)}.json"


def list_recipe_files(root: Path | None = None) -> list[Path]:
    base = root or RECIPE_DIR
    return sorted(base.glob("*.json")) if base.is_dir() else []


# --- JSON helpers ----------------------------------------------------------

def read_json(path: Path) -> dict | None:
    """Parse a JSON file, or None when missing/unreadable/not an object.

    Sidecars can be half-written when a machine is powered off mid-shift. None
    here means "treat as unlabeled", which is always the safe reading.
    """
    try:
        if not Path(path).is_file():
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> Path:
    """Write JSON atomically, so an interrupted save cannot truncate a sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
