"""Filesystem layout and JSON persistence.

The data-root resolution (env override, operator-chosen folder, network-share
fallback) is ported from BungVision Label Studio, where it was proven on
shared industrial workstations, and is kept behaviour-for-behaviour.

The layout below is not. This tool trains **one label at a time**: a dataset
is a folder of images of a single label, gathered from wherever they come
from, annotated and exported on its own schedule. Nothing here knows about
batteries, cameras or recipes. Which labels a given battery must carry, and
where to look for each, is the front end's business -- authored and stored
there, never here. This tool's whole job is producing a trained label.

    captures/<label_id>/<image>.jpg
    labels/<label_id>/<image>.json
    library/labels.json          every label definition

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

SUBDIRS = ("captures", "labels", "exports", "library", "models")


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


MAX_RECENT_DATA_DIRS = 8


def _read_location_config() -> dict:
    try:
        path = data_location_file()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def read_recent_data_dirs() -> list[Path]:
    """Previously used library locations, most recent first.

    Lets an operator switch between site libraries from a list instead of
    re-browsing to a network path every time.
    """
    raw = _read_location_config().get("recent", [])
    out: list[Path] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Path(text))
    return out


def write_configured_data_dir(path: Path | str | None) -> None:
    """Persist the library location. ``None`` restores the default.

    Takes effect on next launch: the module-level paths below are imported
    directly all over the app.
    """
    config = _read_location_config()
    value = "" if path is None else str(Path(path))
    config["data_dir"] = value
    if value:
        recent = [str(p) for p in read_recent_data_dirs() if str(p) != value]
        config["recent"] = [value] + recent[: MAX_RECENT_DATA_DIRS - 1]
    target = data_location_file()
    target.parent.mkdir(parents=True, exist_ok=True)
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
EXPORT_DIR = DATA_DIR / "exports"
LIBRARY_DIR = DATA_DIR / "library"
MODEL_DIR = DATA_DIR / "models"

LABEL_LIBRARY_PATH = LIBRARY_DIR / "labels.json"
CLASS_CONFIG_PATH = LIBRARY_DIR / "classes.json"
CAMERA_SETTINGS_PATH = DATA_DIR / "camera_settings.json"
TRAINING_SETTINGS_PATH = DATA_DIR / "training_settings.json"
TEST_SETTINGS_PATH = DATA_DIR / "test_settings.json"


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


# --- detector families -----------------------------------------------------
# The classes the model is actually trained on. Deliberately few and visually
# distinct: this list is the thing that costs a retrain to change, so it holds
# *kinds* of label, never individual SKUs. Which exact label a detection is
# gets resolved afterwards from the library, by decoding and matching.

DEFAULT_FAMILY_CONFIG = [
    {"id": 0, "name": "battery_side", "default_tool": "OBB", "enabled": True},
    {"id": 1, "name": "spec_plate", "default_tool": "OBB", "enabled": True},
    {"id": 2, "name": "warning_label", "default_tool": "OBB", "enabled": True},
    {"id": 3, "name": "cert_mark", "default_tool": "OBB", "enabled": True},
    {"id": 4, "name": "trace_tag", "default_tool": "OBB", "enabled": True},
    {"id": 5, "name": "promo_label", "default_tool": "OBB", "enabled": True},
    {"id": 6, "name": "code_patch", "default_tool": "OBB", "enabled": True},
]


def load_class_config() -> list[dict[str, Any]]:
    """The detector families, from disk or the shipped defaults.

    A malformed file falls back to the defaults rather than raising: the app
    must still launch so an operator can fix the library location.
    """
    data = read_json(CLASS_CONFIG_PATH)
    if data is None:
        save_class_config(DEFAULT_FAMILY_CONFIG)
        return [dict(c) for c in DEFAULT_FAMILY_CONFIG]

    raw = data.get("classes", [])
    out: list[dict[str, Any]] = []
    used: set[int] = set()
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        cid = int(entry.get("id", len(out)))
        if cid in used:
            continue
        used.add(cid)
        name = str(entry.get("name", f"class_{cid}")).strip() or f"class_{cid}"
        tool = str(entry.get("default_tool", "OBB")).upper()
        out.append({
            "id": cid,
            "name": name,
            # OBB throughout: labels sit on curved, tilted battery faces and an
            # axis-aligned box around a rotated label swallows its neighbours.
            "default_tool": "BOX" if tool == "BOX" and entry.get("tool_locked") else "OBB",
            "enabled": bool(entry.get("enabled", True)),
            "tool_locked": bool(entry.get("tool_locked", False)),
        })
    if not out:
        return [dict(c) for c in DEFAULT_FAMILY_CONFIG]
    return sorted(out, key=lambda c: int(c["id"]))


def save_class_config(classes: list[dict[str, Any]]) -> Path:
    payload = {"classes": sorted(classes, key=lambda c: int(c.get("id", 0)))}
    return write_json(CLASS_CONFIG_PATH, payload)


def class_names_from_config(classes: list[dict[str, Any]]) -> list[str]:
    """Family names indexed by class id, with gaps filled, for YOLO export."""
    enabled = [c for c in sorted(classes, key=lambda c: int(c.get("id", 0)))
               if c.get("enabled", True)]
    max_id = max([int(c.get("id", 0)) for c in enabled], default=-1)
    names = [f"class_{i}" for i in range(max_id + 1)]
    for c in enabled:
        names[int(c["id"])] = str(c["name"])
    return names or ["battery_side", "spec_plate"]


# --- persisted UI settings -------------------------------------------------
# Ported unchanged: re-selecting a camera, a model path and an image size on
# every launch was pure friction.

DEFAULT_CAMERA_SETTINGS: dict[str, Any] = {
    "camera_source": "0",
    "camera_backend": "V4L2",
    "width": 2592,
    "height": 1944,
    "fps": 0,
    "preview_scale": "1/2",
    "exposure_auto": True,
    "exposure_us": 0,
    "force_v4l2": True,
    "low_latency": True,
    "threaded_camera": True,
    "mjpg": True,
    "skip_heavy_live": True,
}

def load_camera_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_CAMERA_SETTINGS)
    if CAMERA_SETTINGS_PATH.exists():
        try:
            data = json.loads(CAMERA_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update({k: data[k] for k in settings if k in data})
        except Exception:
            pass
    return settings


def save_camera_settings(settings: dict[str, Any]) -> Path:
    payload = dict(DEFAULT_CAMERA_SETTINGS)
    payload.update({k: settings[k] for k in payload if k in settings})
    CAMERA_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CAMERA_SETTINGS_PATH


def load_training_settings() -> dict[str, Any]:
    """Last-used YOLO training parameters, persisted between sessions."""
    if TRAINING_SETTINGS_PATH.exists():
        try:
            data = json.loads(TRAINING_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_training_settings(settings: dict[str, Any]) -> Path:
    TRAINING_SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return TRAINING_SETTINGS_PATH


def load_test_settings() -> dict[str, Any]:
    """Last-used Model Test settings, persisted between sessions.

    Re-selecting a model path, image size, confidence and class filters on every
    launch was pure friction, especially when iterating on a model.
    """
    if TEST_SETTINGS_PATH.exists():
        try:
            data = json.loads(TEST_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_test_settings(settings: dict[str, Any]) -> Path:
    TEST_SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return TEST_SETTINGS_PATH


# --- annotation sidecars ---------------------------------------------------
def image_label_json_path(image_path: Path) -> Path:
    """data/labels/<label_id>/<image_stem>.json for a capture path.

    The sidecar folder is named after the image's parent directory, which in
    this layout is the label id -- the same shape the per-recipe version had,
    so everything downstream of it ported unchanged.
    """
    label_id = image_path.parent.name
    folder = LABEL_DIR / label_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{image_path.stem}.json"


def save_annotations(
    image_path: Path,
    image_w: int,
    image_h: int,
    boxes: list[dict[str, Any]],
    class_names: list[str],
    review: dict[str, Any] | None = None,
    clear_review: bool = False,
    background: bool | None = None,
) -> Path:
    """Save Label Studio annotations.

    v0.9.28 adds an optional review marker so images imported from
    BungVision can remain visibly "needs review" until an operator
    explicitly saves or marks them reviewed.  When review is not supplied,
    existing review metadata is normally preserved.  v0.9.37 adds
    clear_review=True for the safety case where an already-reviewed image
    is edited into a quantity mismatch; normal Save must then remove the
    old review marker so only Force Review can include the mismatch in
    training/export.

    ``background=True`` records a deliberate negative image -- a conveyor,
    an empty fixture -- which exports as an empty label file. It is an explicit
    flag rather than "no boxes" so a half-finished annotation is never mistaken
    for a background sample. Drawing any box clears it; ``background=None``
    leaves an existing flag alone.
    """
    path = image_label_json_path(image_path)
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    payload = {
        "image": str(image_path),
        "width": image_w,
        "height": image_h,
        "classes": class_names,
        "boxes": boxes,
    }

    # Preserve useful source metadata from imported BungVision JSON.
    # Review metadata is preserved only for non-review-changing saves.  When
    # clear_review=True, stale Label Studio review markers are intentionally
    # removed so an image edited into a count mismatch cannot accidentally
    # remain eligible for reviewed-only export/training.
    source_keys = (
        "source",
        "origin",
        "imported_from",
        "imported_from_bungvision",
        "bungvision",
        "capture_source",
    )
    review_keys = (
        "review",
        "reviewed",
        "reviewed_at",
        "reviewed_by",
        "review_status",
        "review_source",
        "review_tool",
        "forced_review",
        "force_reviewed",
    )
    for key in source_keys:
        if key in previous:
            payload[key] = previous[key]

    # Background marker. A box on the image contradicts the flag outright, so
    # boxes always win -- otherwise an operator who marked a frame background
    # and then labelled something would export it as a negative anyway.
    if boxes:
        payload["background"] = False
    elif background is None:
        payload["background"] = bool(previous.get("background", False))
    else:
        payload["background"] = bool(background)
    if review is None and not clear_review:
        for key in review_keys:
            if key in previous:
                payload[key] = previous[key]
    elif review is None and clear_review:
        payload["reviewed"] = False
        payload["review_status"] = "needs_review"

    if review is not None:
        payload["review"] = review
        payload["reviewed"] = bool(review.get("reviewed", False))
        payload["review_source"] = review.get("source", "bungvision_label_studio")
        payload["review_tool"] = review.get("tool", "BungVision Label Studio")
        if review.get("reviewed_at"):
            payload["reviewed_at"] = review.get("reviewed_at")
        if review.get("reviewed_by"):
            payload["reviewed_by"] = review.get("reviewed_by")
        payload["review_status"] = review.get("review_status") or ("reviewed" if payload["reviewed"] else "needs_review")
        if review.get("forced_review") or review.get("force_reviewed"):
            payload["forced_review"] = True
            payload["force_reviewed"] = True

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_annotations(image_path: Path) -> dict[str, Any] | None:
    path = image_label_json_path(image_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
