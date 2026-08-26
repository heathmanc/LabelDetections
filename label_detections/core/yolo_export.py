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

from . import augment as augment_logic
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


def _write_label_file(out: Path, split: str, stem: str, data: dict,
                      class_index: dict[str, int], task: str) -> int:
    """Write one label file and return how many boxes it carried."""
    line_for = _obb_line if task == "obb" else _detect_line
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
    (out / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _write_split(out: Path, entries: list[dataset_logic.Entry], class_index: dict[str, int],
                 split: str, task: str) -> list[str]:
    """Copy images and write label files for one split; returns manifest rows."""
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for entry in entries:
        image = Path(entry.image)
        # Prefix with the label id: two datasets can hold identically named
        # frames, and a plain copy would have one silently overwrite the other.
        out_name = f"{safe_token(entry.label_id)}__{image.name}"
        try:
            shutil.copy2(image, out / "images" / split / out_name)
        except OSError:
            continue
        boxes = _write_label_file(out, split, Path(out_name).stem,
                                  entry.annotation, class_index, task)
        rows.append(f"{split},{entry.label_id},{out_name},{boxes},"
                    f"{entry.session or entry.source or ''},0")
    return rows


def _write_augmented(out: Path, entries: list[dataset_logic.Entry],
                     class_index: dict[str, int], task: str, library,
                     copies: int, seed: int) -> tuple[list[str], list[str]]:
    """Extra training images with each label's variable regions replaced.

    Train only, never validation: recombined images validate nothing except how
    well the model handles recombined images.

    Returns (manifest rows, notes for the report).
    """
    import random

    rng = random.Random(seed)
    rows: list[str] = []
    notes: list[str] = []
    if copies <= 0 or library is None:
        return rows, notes

    by_label: dict[str, list] = {}
    for entry in entries:
        by_label.setdefault(str(entry.label_id), []).append(entry)

    for label_id, group in sorted(by_label.items()):
        label = library.get(label_id)
        if label is None or not augment_logic.variable_regions(label):
            continue
        reports = augment_logic.scan_entries(group, library)
        planned = augment_logic.plan_copies(reports, copies)
        if not planned:
            notes.append(
                f"{label_id}: variable regions already vary across its images, so "
                "no extra copies were written.")
            continue

        donors = augment_logic.donor_pool(group, label)
        written = 0
        for entry in group:
            variants = augment_logic.augmented_variants(
                entry, label, planned, donors, rng)
            for index, image in enumerate(variants):
                stem = f"{safe_token(label_id)}__aug{index}__{Path(entry.image).stem}"
                target = out / "images" / "train" / f"{stem}.jpg"
                try:
                    import cv2
                    if not cv2.imwrite(str(target), image):
                        continue
                except Exception:
                    continue
                boxes = _write_label_file(out, "train", stem, entry.annotation,
                                          class_index, task)
                rows.append(f"train,{label_id},{target.name},{boxes},"
                            f"{entry.session or entry.source or ''},1")
                written += 1
        notes.append(
            f"{label_id}: {written} extra training image(s) written with variable "
            f"regions grafted from other images of the same label.")
    return rows, notes


def write_dataset(out: Path, entries: list[dataset_logic.Entry], *, task: str = "obb",
                  split_train: float = DEFAULT_SPLIT_TRAIN, seed: int = DEFAULT_SEED,
                  reviewed_only: bool = True, library=None,
                  augment: int = 0) -> Path:
    """Write a YOLO dataset from already-collected entries.

    ``augment`` asks for that many extra training copies per image of any label
    whose variable regions turn out to be constant across the dataset -- see
    ``core/augment.py``. Labels whose regions already vary get none, because
    recombining them teaches nothing and dilutes the real images.
    """
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

    rows = ["split,label_id,image,boxes,group,augmented"]
    rows += _write_split(out, train, class_index, "train", task)
    rows += _write_split(out, val, class_index, "val", task)

    augmented_rows, augment_notes = _write_augmented(
        out, train, class_index, task, library, augment, seed)
    rows += augmented_rows

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
    split_text = report.text()
    if augment_notes:
        split_text += "\n\nVariable regions:\n" + "\n".join(
            f"  {note}" for note in augment_notes)
    (out / "split_report.txt").write_text(split_text + "\n", encoding="utf-8")

    # The check ships with the dataset whether or not anything was augmented: a
    # region that is the same picture in every image is worth knowing about
    # before training, not after a month of drift.
    if library is not None:
        scan = augment_logic.scan_entries(entries, library)
        if scan:
            (out / "variable_regions.txt").write_text(
                augment_logic.scan_text(scan) + "\n", encoding="utf-8")
    return out


def export_label_yolo(label_id: str, *, task: str = "obb", reviewed_only: bool = True,
                      split_train: float = DEFAULT_SPLIT_TRAIN,
                      seed: int = DEFAULT_SEED, out: Path | None = None,
                      library=None, augment: int = 0) -> Path:
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
                         seed=seed, reviewed_only=reviewed_only,
                         library=library, augment=augment)


def export_all_labels_yolo(*, task: str = "obb", reviewed_only: bool = True,
                           split_train: float = DEFAULT_SPLIT_TRAIN,
                           seed: int = DEFAULT_SEED, out: Path | None = None,
                           library=None, augment: int = 0) -> Path:
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
                         seed=seed, reviewed_only=reviewed_only,
                         library=library, augment=augment)
