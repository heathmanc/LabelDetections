from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_FOLDER_NAME = "BungVisionLabelStudio"


def _is_writable(path: Path) -> bool:
    """True if we can actually create files in ``path``.

    Checked by writing, not by os.access() -- on Windows os.access() reports
    success for directories that Explorer/UAC will still refuse, and
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
    """Fallback directory holding the ``data`` folder, when none is configured.

    From source this is the repo root.

    When frozen there are two deployment shapes and we must support both:
      * portable (zip extracted to Desktop/Documents) -- keep data beside the
        .exe so the whole folder can be copied around as one unit.
      * installed (Inno Setup, typically ``C:\\Program Files``) -- that folder is
        read-only for standard users, so writing there raises PermissionError at
        import. Fall back to ``%LOCALAPPDATA%\\BungVisionLabelStudio``.

    Never anchored to ``__file__`` when frozen: that points inside the bundle,
    which is wiped on upgrade.
    """
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parents[2]

    exe_dir = Path(sys.executable).resolve().parent
    if _is_writable(exe_dir / "data"):
        return exe_dir

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home()) / APP_FOLDER_NAME


# --- Configurable image-library location -------------------------------------
# The default locations above are per-machine or per-user, which is wrong for a
# shared workstation or a team library on a network share. The data folder can
# therefore be pointed anywhere, resolved in this order:
#
#   1. BUNGVISION_DATA_DIR environment variable  (per-launch override, scripting)
#   2. the configured pointer file               (set from Tools > Data folder)
#   3. the built-in default from _app_root()
#
# The pointer file deliberately lives in a per-user config directory rather than
# in the data folder itself -- it has to be readable *before* the data location
# is known, so it cannot be stored there.

DATA_DIR_ENV = "BUNGVISION_DATA_DIR"


def config_dir() -> Path:
    """Per-user directory for settings that must be found before the data root."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / APP_FOLDER_NAME
    return Path.home() / f".{APP_FOLDER_NAME.lower()}"


def data_location_file() -> Path:
    return config_dir() / "data_location.json"


def read_configured_data_dir() -> Path | None:
    """The data folder chosen by the operator, or None if unset."""
    path = data_location_file()
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8")).get("data_dir", "")
    except Exception:
        return None
    raw = str(raw).strip()
    return Path(raw) if raw else None


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
    """Persist the data folder choice. ``None`` clears it back to the default.

    The chosen folder is also remembered in the recent list so it can be
    switched back to without browsing again. Takes effect on the next launch:
    DATA_DIR and the paths derived from it are module-level constants that the
    whole app imports directly.
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


def resolve_data_dir() -> Path:
    """The active data folder, honouring the env var and configured override."""
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    if env:
        return Path(env)
    configured = read_configured_data_dir()
    if configured is not None:
        return configured
    return _app_root() / "data"


def _bundled_seed_dir() -> Path | None:
    """Starter recipes/class config shipped inside the frozen bundle, if any."""
    if not getattr(sys, "frozen", False):
        return None
    seed = Path(getattr(sys, "_MEIPASS", "")) / "seed"
    return seed if seed.is_dir() else None


SUBDIRS = ("captures", "labels", "recipes", "exports")


def _ensure_data_dirs(base: Path) -> bool:
    """Create the library's subfolders. False if the location is unusable."""
    try:
        for name in SUBDIRS:
            (base / name).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


# Set when a configured location could not be used, so the UI can say so
# instead of silently writing somewhere the operator does not expect.
DATA_DIR_FALLBACK_REASON = ""

DATA_DIR = resolve_data_dir()
if not _ensure_data_dirs(DATA_DIR):
    # A configured library on a network share may simply be offline. Falling
    # back keeps the app usable; crashing at import would make it unlaunchable
    # until someone edited a config file by hand.
    _unusable = DATA_DIR
    DATA_DIR = _app_root() / "data"
    DATA_DIR_FALLBACK_REASON = (
        f"The configured data folder could not be opened:\n{_unusable}\n\n"
        f"Using the default location instead:\n{DATA_DIR}\n\n"
        "If this is a network share, check that it is connected, then set the "
        "folder again from Tools > Data folder."
    )
    _ensure_data_dirs(DATA_DIR)

# ROOT stays the data folder's parent: it is only used as a working directory
# for child processes and to resolve run paths relative to the library.
ROOT = DATA_DIR.parent
CAPTURE_DIR = DATA_DIR / "captures"
LABEL_DIR = DATA_DIR / "labels"
RECIPE_DIR = DATA_DIR / "recipes"
EXPORT_DIR = DATA_DIR / "exports"
CLASS_CONFIG_PATH = DATA_DIR / "class_config.json"
CAMERA_SETTINGS_PATH = DATA_DIR / "camera_settings.json"
TRAINING_SETTINGS_PATH = DATA_DIR / "training_settings.json"
TEST_SETTINGS_PATH = DATA_DIR / "test_settings.json"


def _seed_user_data() -> None:
    """Copy starter recipes/class config out of the bundle on first run.

    Only fills in files that are missing, so an operator's edits are never
    overwritten on upgrade. Needed because an installed build writes its data to
    LOCALAPPDATA, which starts empty.
    """
    seed = _bundled_seed_dir()
    if seed is None:
        return
    try:
        src_cfg = seed / "class_config.json"
        if src_cfg.is_file() and not CLASS_CONFIG_PATH.exists():
            CLASS_CONFIG_PATH.write_bytes(src_cfg.read_bytes())
        src_recipes = seed / "recipes"
        if src_recipes.is_dir():
            for item in src_recipes.glob("*.json"):
                dest = RECIPE_DIR / item.name
                if not dest.exists():
                    dest.write_bytes(item.read_bytes())
    except Exception:
        # Seeding is a convenience; a failure must not stop the app launching.
        pass


_seed_user_data()

# Broad equipment category recipes fall under by default. The default category
# is special-cased so legacy recipes keep their original on-disk safe_name.
DEFAULT_CATEGORY = "General"


DEFAULT_CLASSES = [
    {"id": 0, "name": "battery", "default_tool": "OBB", "enabled": True, "role": "battery"},
    {"id": 1, "name": "bung", "default_tool": "OBB", "enabled": True, "role": "bung"},
    {"id": 2, "name": "retainer", "default_tool": "OBB", "enabled": True, "role": "retainer"},
]

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

def infer_role_and_layout(name: str) -> tuple[str, str]:
    """Infer a simple class role from the class name for old configs and exports.

    The second return value is retained as "none" only so older call sites
    and configs remain compatible.
    """
    n = str(name or "").strip().lower()
    if n == "battery" or n.startswith("battery"):
        return "battery", "none"
    if n == "bung" or n.startswith("bung"):
        return "bung", "none"
    if n == "retainer" or n.startswith("retainer"):
        return "retainer", "none"
    return "custom", "none"

def load_class_config() -> list[dict[str, Any]]:
    if not CLASS_CONFIG_PATH.exists():
        save_class_config(DEFAULT_CLASSES)
        return [dict(c) for c in DEFAULT_CLASSES]
    try:
        data = json.loads(CLASS_CONFIG_PATH.read_text(encoding="utf-8"))
        classes = data.get("classes", data if isinstance(data, list) else [])
        out = []
        used = set()
        for c in classes:
            cid = int(c.get("id", len(out)))
            name = str(c.get("name", f"class_{cid}")).strip() or f"class_{cid}"
            # Drop the old default layout helper classes from v0.9.21-v0.9.23.
            # Battery/bung labeling is now plain OBB, not layout-zone driven.
            if name.lower() in {"battery_6row", "battery_2x3"}:
                continue
            if cid in used:
                continue
            used.add(cid)
            role, _layout = infer_role_and_layout(name)
            default_tool = str(c.get("default_tool", "OBB")).upper()
            if default_tool not in {"OBB", "BOX"}:
                default_tool = "OBB"
            out.append({
                "id": cid,
                "name": name,
                "default_tool": default_tool,
                "enabled": bool(c.get("enabled", True)),
                "role": str(c.get("role", role)).lower(),
                # v0.9.38: old custom classes created before this version were
                # hardcoded as BOX.  tool_locked distinguishes a deliberate
                # v0.9.38+ Box fallback choice from that old default.
                "tool_locked": bool(c.get("tool_locked", False)),
            })
        if not out:
            return [dict(c) for c in DEFAULT_CLASSES]

        # OBB is now the normal workflow.  Built-in part classes always use
        # OBB, and old pre-v0.9.38 custom classes that were automatically
        # saved as BOX are migrated to OBB.  Deliberate v0.9.38+ Box fallback
        # choices are preserved with tool_locked=True.
        for c in out:
            role = str(c.get("role", "")).lower()
            name = str(c.get("name", "")).lower()
            if role in {"battery", "bung", "retainer"} or name in {"battery", "bung", "retainer"}:
                c["default_tool"] = "OBB"
            elif str(c.get("default_tool", "OBB")).upper() == "BOX" and not c.get("tool_locked", False):
                c["default_tool"] = "OBB"

        return sorted(out, key=lambda c: int(c["id"]))
    except Exception:
        return [dict(c) for c in DEFAULT_CLASSES]


def save_class_config(classes: list[dict[str, Any]]) -> Path:
    payload = {"classes": sorted(classes, key=lambda c: int(c.get("id", 0)))}
    CLASS_CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CLASS_CONFIG_PATH


def class_names_from_config(classes: list[dict[str, Any]]) -> list[str]:
    enabled = [c for c in sorted(classes, key=lambda c: int(c.get("id", 0))) if c.get("enabled", True)]
    max_id = max([int(c.get("id", 0)) for c in enabled], default=-1)
    names = [f"class_{i}" for i in range(max_id + 1)]
    for c in enabled:
        names[int(c["id"])] = str(c["name"])
    return names or ["battery", "bung"]


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


@dataclass
class Recipe:
    group: str
    model: str
    # Broad equipment category above group/model. Lets one install hold recipes
    # for several machines and load/browse them separately. The default category
    # keeps the legacy on-disk safe_name (group__model) so pre-category captures
    # and labels stay attached to their recipes.
    category: str = DEFAULT_CATEGORY
    expected_bungs: int = 6
    # When False, the recipe is unlocked from the battery/bung quantity check
    # so the tool can label arbitrary object classes (free-form labeling).
    constrained: bool = True
    # Sealed battery model: no bunsen valves at all. Correct product is a
    # battery with zero bungs, so a detected bung is a defect rather than a
    # missing-count failure. Written into the recipe JSON so the BungVision
    # runtime can read it and not reject a sealed battery for having no bungs.
    # Distinct from constrained=False, which disables the check entirely.
    sealed: bool = False
    brightness: int = 0
    contrast: int = 0
    gamma: float = 1.0
    clahe_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    sharpen: int = 0
    notes: str = ""

    @property
    def safe_name(self) -> str:
        def clean(s: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s.strip()) or "Unnamed"
        base = f"{clean(self.group)}__{clean(self.model)}"
        if str(self.category).strip() in ("", DEFAULT_CATEGORY):
            # Legacy form: keeps existing capture/label folders working.
            return base
        return f"{clean(self.category)}__{base}"


def recipe_path(group: str, model: str, category: str = DEFAULT_CATEGORY) -> Path:
    r = Recipe(group=group, model=model, category=category)
    return RECIPE_DIR / f"{r.safe_name}.json"


def save_recipe(recipe: Recipe) -> Path:
    path = RECIPE_DIR / f"{recipe.safe_name}.json"
    path.write_text(json.dumps(asdict(recipe), indent=2), encoding="utf-8")
    return path


def load_recipe(path: Path) -> Recipe:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Older builds stored app-wide camera settings in each recipe. Ignore those
    # keys so loading a recipe never changes the current camera setup.
    recipe_fields = set(Recipe.__dataclass_fields__)
    data = {k: v for k, v in data.items() if k in recipe_fields}
    return Recipe(**data)


def list_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    for p in sorted(RECIPE_DIR.glob("*.json")):
        try:
            recipes.append(load_recipe(p))
        except Exception:
            continue
    return recipes


def recipe_category(recipe: Recipe) -> str:
    """Category for a recipe, falling back to the default for legacy recipes."""
    cat = str(getattr(recipe, "category", "") or "").strip()
    return cat or DEFAULT_CATEGORY


def list_categories() -> list[str]:
    """Sorted, de-duplicated categories across all saved recipes (default first)."""
    cats = {recipe_category(r) for r in list_recipes()}
    cats.add(DEFAULT_CATEGORY)
    ordered = sorted(c for c in cats if c != DEFAULT_CATEGORY)
    return [DEFAULT_CATEGORY] + ordered


def capture_folder(recipe: Recipe) -> Path:
    p = CAPTURE_DIR / recipe.safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def label_folder(recipe: Recipe) -> Path:
    p = LABEL_DIR / recipe.safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_capture(
    recipe: Recipe,
    frame_bgr: np.ndarray,
    adjusted_bgr: np.ndarray | None = None,
    save_raw: bool = True,
) -> tuple[Path | None, Path | None]:
    """Write a capture to the recipe's folder.

    ``save_raw=False`` with an adjusted frame writes only the adjusted image.
    Capturing adjusted used to emit both files, which doubled the dataset and
    left an unadjusted twin of every frame to label or delete by hand.

    Returns (raw_path, adjusted_path); either may be None.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    folder = capture_folder(recipe)
    # Two captures inside the same millisecond produced the same name, and the
    # second silently overwrote the first -- image, and the label JSON keyed to
    # it. Rare by hand, but reachable by holding the capture shortcut down.
    base = f"{recipe.safe_name}_{ts}_{ms:03d}"
    suffix = 1
    while (folder / f"{base}.jpg").exists() or (folder / f"{base}_adjusted.jpg").exists():
        base = f"{recipe.safe_name}_{ts}_{ms:03d}_{suffix}"
        suffix += 1

    raw_path = None
    if save_raw or adjusted_bgr is None:
        raw_path = folder / f"{base}.jpg"
        cv2.imwrite(str(raw_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    adjusted_path = None
    if adjusted_bgr is not None:
        adjusted_path = folder / f"{base}_adjusted.jpg"
        cv2.imwrite(str(adjusted_path), adjusted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return raw_path, adjusted_path


IMPORT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_sidecar_json(image_src: Path, json_dir: Path | None = None) -> Path | None:
    """Locate a BungVision-style sidecar label JSON for a source image.

    If ``json_dir`` is given the JSON is looked up there (parallel-directory
    layout: images and labels live in separate sibling folders).  Otherwise
    the JSON is looked up next to the image file (co-located layout).

    Supports both ``foo.json`` (stem) and ``foo.jpg.json`` (full-name) naming.
    """
    if json_dir is not None:
        candidates = [
            json_dir / f"{image_src.stem}.json",
            json_dir / f"{image_src.name}.json",
        ]
    else:
        candidates = [
            image_src.with_suffix(".json"),
            Path(str(image_src) + ".json"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


def import_images(
    recipe: Recipe,
    paths: list[Path | str],
    json_dir: Path | None = None,
    as_background: bool = False,
) -> tuple[list[Path], list[str], int]:
    """Copy external images (and any sidecar label JSON) into a recipe.

    Each source image is decoded and re-encoded to JPEG under the recipe's
    normal naming convention so it shows up in the captured-image list.

    ``as_background=True`` marks every imported image as a deliberate negative
    (an empty conveyor, a bare fixture) instead of looking for sidecar labels.

    If ``json_dir`` is supplied, the matching ``.json`` label file is looked up
    there (parallel-directory layout).  Otherwise the JSON is expected to sit
    next to the image (co-located layout).  When a sidecar is found, its boxes
    and review/source metadata are written into the recipe's label folder under
    the new image name so imported labels appear immediately.

    Returns (imported_paths, errors, label_count).
    """
    folder = capture_folder(recipe)
    imported: list[Path] = []
    errors: list[str] = []
    label_count = 0
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, src in enumerate(paths):
        src = Path(src)
        try:
            img = cv2.imread(str(src))
            if img is None:
                errors.append(f"Could not read image: {src.name}")
                continue
            base = f"{recipe.safe_name}_import_{ts}_{i:04d}"
            dest = folder / f"{base}.jpg"
            cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            imported.append(dest)

            if as_background:
                # Bulk negatives: an empty conveyor has nothing to label, so the
                # annotation is written on import rather than making the operator
                # open and mark hundreds of images by hand.
                _write_background_label(dest, img)
                label_count += 1
                continue

            sidecar = find_sidecar_json(src, json_dir=json_dir)
            if sidecar is not None:
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    _write_imported_label(dest, img, data)
                    label_count += 1
                except Exception as exc:
                    errors.append(f"{src.name} label JSON: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{src.name}: {exc}")
    return imported, errors, label_count


def _write_background_label(image_path: Path, img_bgr: "np.ndarray") -> Path:
    """Write a reviewed, zero-box annotation marking an image as a negative."""
    from .review import make_background_record

    h, w = img_bgr.shape[:2]
    return save_annotations(
        image_path, int(w), int(h), [], [],
        review=make_background_record(),
        background=True,
    )


def _write_imported_label(image_path: Path, img_bgr: "np.ndarray", data: dict[str, Any]) -> Path:
    """Write a sidecar label JSON for an imported image, preserving its content.

    The full source payload is kept (boxes plus any review/source metadata) but
    the image path and dimensions are corrected to the newly imported file.
    """
    h, w = img_bgr.shape[:2]
    payload = dict(data) if isinstance(data, dict) else {}
    payload["image"] = str(image_path)
    try:
        payload["width"] = int(payload.get("width") or w)
        payload["height"] = int(payload.get("height") or h)
    except (TypeError, ValueError):
        payload["width"], payload["height"] = w, h
    payload["boxes"] = payload.get("boxes") or []
    # Record provenance so review tooling treats these as imported.
    payload.setdefault("imported_from", "image_import")
    path = image_label_json_path(image_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def image_label_json_path(image_path: Path) -> Path:
    # data/labels/<recipe>/<image_stem>.json
    recipe_name = image_path.parent.name
    folder = LABEL_DIR / recipe_name
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
