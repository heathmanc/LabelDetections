"""YOLO dataset export.

Rewritten from BungVision's per-recipe exporter for the per-label layout, with
two deliberate changes.

**The split.** The old ``_split_entries`` shuffled individual images. That was
correct for its single-camera case and is wrong here: label images arrive in
bursts -- several frames of the same physical label, a batch off one print run
-- and a per-image shuffle puts near-duplicates on both sides, so validation
measures memorisation. The split now goes through ``dataset.split_entries``,
which never separates a capture group and refuses to leave a class out of
validation.

**The classes.** Export writes the coarse detector **family** of each box, not
its library identity. That is the whole two-stage design: the model learns
``spec_plate``, and which spec plate a detection is gets resolved afterwards by
decoding and matching. Training per-SKU classes would put every new label back
on the retraining treadmill.

Reviewed-only is not optional, same as upstream: only images an operator
approved inside this tool may train a model.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import dataset as dataset_logic
from .review import annotation_reviewed as _annotation_reviewed
from .review import is_background_annotation as _is_background
from .storage import CAPTURE_DIR, EXPORT_DIR, LABEL_DIR, list_datasets, safe_token

DEFAULT_SPLIT_TRAIN = 0.8
DEFAULT_SEED = 0


def _family(box: dict) -> str:
    """The detector class for a box: its family, never its label id."""
    name = str(box.get("label", "") or "").strip()
    if name:
        return safe_token(name).lower()
    cls = box.get("class_id")
    return f"class_{int(cls)}" if cls is not None else "unknown"


def _obb_points(box: dict) -> list[list[float]] | None:
    """Four image-space corners, converting a legacy axis-aligned box.

    Older captures were drawn as plain rectangles. Exporting them as
    zero-rotation OBBs keeps them usable instead of silently dropping them.
    """
    pts = box.get("points") or box.get("obb") or []
    if len(pts) >= 4:
        try:
            return [[float(x), float(y)] for x, y in pts[:4]]
        except (TypeError, ValueError):
            return None
    try:
        x, y = float(box["x"]), float(box["y"])
        w, h = float(box["w"]), float(box["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _obb_line(box: dict, image_w: int, image_h: int, class_id: int) -> str | None:
    points = _obb_points(box)
    if not points or image_w <= 0 or image_h <= 0:
        return None
    values = []
    for x, y in points:
        values.append(max(0.0, min(1.0, x / image_w)))
        values.append(max(0.0, min(1.0, y / image_h)))
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in values)


def _detect_line(box: dict, image_w: int, image_h: int, class_id: int) -> str | None:
    """Axis-aligned fallback: the enclosing rect of the oriented box."""
    points = _obb_points(box)
    if not points or image_w <= 0 or image_h <= 0:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = ((min(xs) + max(xs)) / 2.0) / image_w
    cy = ((min(ys) + max(ys)) / 2.0) / image_h
    nw = (max(xs) - min(xs)) / image_w
    nh = (max(ys) - min(ys)) / image_h
    if nw <= 0 or nh <= 0:
        return None
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def collect_entries(label_id: str, reviewed_only: bool = True) -> list[dataset_logic.Entry]:
    """Exportable images from one label's dataset.

    Backgrounds are kept with no boxes: an empty ``.txt`` is exactly how YOLO
    consumes a negative, and negatives are what stop the model firing on bare
    fixtures.
    """
    image_dir = CAPTURE_DIR / safe_token(label_id)
    sidecar_dir = LABEL_DIR / safe_token(label_id)
    if not image_dir.is_dir() or not sidecar_dir.is_dir():
        return []

    entries: list[dataset_logic.Entry] = []
    for sidecar in sorted(sidecar_dir.glob("*.json")):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if reviewed_only and not _annotation_reviewed(data):
            continue

        image = Path(str(data.get("image", "")))
        if not image.exists():
            image = image_dir / (image.name or f"{sidecar.stem}.jpg")
        if not image.exists():
            candidates = list(image_dir.glob(f"{sidecar.stem}.*"))
            if not candidates:
                continue
            image = candidates[0]

        if not data.get("boxes") and not _is_background(data):
            continue
        entries.append(dataset_logic.entry_from_annotation(label_id, str(image), data))
    return entries


def _write_split(out: Path, entries: list[dataset_logic.Entry], class_index: dict[str, int],
                 split: str, task: str) -> list[str]:
    """Copy images and write label files for one split; returns manifest rows."""
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    line_for = _obb_line if task == "obb" else _detect_line
    rows: list[str] = []
    for entry in entries:
        image = Path(entry.image)
        data = entry.annotation
        # Prefix with the label id: two datasets can hold identically named
        # frames, and a plain copy would have one silently overwrite the other.
        out_name = f"{safe_token(entry.label_id)}__{image.name}"
        try:
            shutil.copy2(image, out / "images" / split / out_name)
        except OSError:
            continue

        width = int(data.get("width", 0) or 0)
        height = int(data.get("height", 0) or 0)
        lines: list[str] = []
        for box in data.get("boxes", []) or []:
            family = _family(box)
            if family not in class_index:
                continue
            line = line_for(box, width, height, class_index[family])
            if line:
                lines.append(line)

        (out / "labels" / split / f"{Path(out_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append(f"{split},{entry.label_id},{out_name},{len(lines)},"
                    f"{entry.session or entry.source or ''}")
    return rows


def write_dataset(out: Path, entries: list[dataset_logic.Entry], *, task: str = "obb",
                  split_train: float = DEFAULT_SPLIT_TRAIN, seed: int = DEFAULT_SEED,
                  reviewed_only: bool = True) -> Path:
    """Write a YOLO dataset from already-collected entries."""
    families: list[str] = []
    seen: set[str] = set()
    labeled = 0
    for entry in entries:
        boxes = entry.annotation.get("boxes") or []
        if boxes:
            labeled += 1
        for box in boxes:
            family = _family(box)
            if family not in seen:
                seen.add(family)
                families.append(family)

    # Backgrounds alone cannot train anything -- there would be no classes at
    # all -- so require at least one genuinely labeled image.
    if not labeled:
        raise FileNotFoundError(
            "No labels found for export. Draw labels, save them, then export again. "
            "Background-only images cannot train a model on their own."
        )

    families.sort()
    class_index = {name: i for i, name in enumerate(families)}

    train, val, report = dataset_logic.split_entries(
        entries, split_train, seed=seed)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = ["split,label_id,image,boxes,group"]
    rows += _write_split(out, train, class_index, "train", task)
    rows += _write_split(out, val, class_index, "val", task)

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(families))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(families)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    (out / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "task.txt").write_text(f"{task}\n", encoding="utf-8")
    (out / "review_filter.txt").write_text(
        "reviewed_only\n" if reviewed_only else "all\n", encoding="utf-8")
    # The split report ships with the dataset on purpose: a reviewer needs to
    # see that no capture group straddles train and val before trusting a
    # validation number.
    (out / "split_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    return out


def export_label_yolo(label_id: str, *, task: str = "obb", reviewed_only: bool = True,
                      split_train: float = DEFAULT_SPLIT_TRAIN,
                      seed: int = DEFAULT_SEED, out: Path | None = None) -> Path:
    """Export one label's dataset on its own.

    Useful for checking a single label in isolation. The normal training run
    uses ``export_all_labels_yolo``: one detector learns every family at once,
    and a model trained on a single class has nothing to tell them apart from.
    """
    entries = collect_entries(label_id, reviewed_only)
    if not entries:
        raise FileNotFoundError(
            f"No exportable images for '{label_id}'. Label some images and mark "
            "them reviewed first."
        )
    target = out or (EXPORT_DIR / f"{safe_token(label_id)}_{task}")
    return write_dataset(target, entries, task=task, split_train=split_train,
                         seed=seed, reviewed_only=reviewed_only)


def export_all_labels_yolo(*, task: str = "obb", reviewed_only: bool = True,
                           split_train: float = DEFAULT_SPLIT_TRAIN,
                           seed: int = DEFAULT_SEED, out: Path | None = None) -> Path:
    """Export every label's dataset into one training set.

    This is the normal export. Labels are *trained* one at a time in the sense
    that each gathers and is reviewed on its own schedule, but they are trained
    *together*: one detector over all the families, so it learns to tell a spec
    plate from a warning label rather than to find one thing everywhere.
    """
    entries: list[dataset_logic.Entry] = []
    for label_id in list_datasets():
        entries.extend(collect_entries(label_id, reviewed_only))
    if not entries:
        raise FileNotFoundError(
            "No exportable images in any dataset. Label some images and mark them "
            "reviewed first."
        )
    target = out or (EXPORT_DIR / f"all_labels_{task}")
    return write_dataset(target, entries, task=task, split_train=split_train,
                         seed=seed, reviewed_only=reviewed_only)
