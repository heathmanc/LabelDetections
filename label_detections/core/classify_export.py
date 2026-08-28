"""Crops for a classification stage: detect where, classify which.

A single detector carrying one class per label has to do two jobs at once, and
the second one defeats it. Detection runs at 640-1024 px across the whole
frame, so a label 100 px wide is 100 px of evidence -- and the thing that
separates rev C from rev D, or the English warning from the French one, is a
line of text maybe 10 px tall at that scale. No amount of training fixes
information that was never sampled.

Cropping changes the arithmetic. The detector's quad is rectified and resized
to 224 px, so that same label arrives as 224 px of label and the deciding text
lands around 25-40 px. That is the difference between a discrimination the
model cannot make and one it makes easily.

The split is what makes the two stages comparable: crops inherit the detection
dataset's train/val split, seed and grouping, so a crop of a battery in
detection-val never turns up in classification-train. Split them separately and
the classifier is validated on labels it trained on, which reads as excellent
accuracy right up until the line.

Nothing here needs new annotation work. Every box an operator has already drawn
carries four corners and a label id, which is a crop and its class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import annotations as ann
from . import dataset as dataset_logic
from . import labels as labels_mod
from .imageio import rectify_quad
from .storage import EXPORT_DIR, safe_token
from .yolo_export import DEFAULT_SEED, DEFAULT_SPLIT_TRAIN

# 224 is the usual classification input. Kept explicit because it is the number
# the whole argument above turns on.
DEFAULT_CROP_PX = 224

# How much label to keep around the drawn quad. A detector's box at runtime
# will not sit exactly where an operator drew it, so training on perfectly
# tight crops teaches the classifier an alignment it will never see again.
DEFAULT_MARGIN = 0.06


def expand_quad(quad: list[list[float]], margin: float) -> list[list[float]]:
    """Push a quad outwards from its centre by a fraction of its size."""
    if margin <= 0 or len(quad) < 4:
        return quad
    pts = np.array(quad[:4], dtype=float)
    centre = pts.mean(axis=0)
    return (centre + (pts - centre) * (1.0 + float(margin))).tolist()


def letterbox(crop: np.ndarray, size: int) -> np.ndarray:
    """Fit a crop into a square without distorting it.

    Squashing to square would throw away aspect ratio, and aspect ratio is a
    real cue here -- a tall narrow trace tag and a wide spec plate are told
    apart by shape before any text is read. Padding keeps it.
    """
    import cv2

    h, w = crop.shape[:2]
    if not h or not w:
        return crop
    scale = float(size) / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=crop.dtype)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def crop_targets(data: dict[str, Any]) -> list[tuple[str, list[list[float]]]]:
    """``(label_id, quad)`` for every identified label box in one sidecar.

    Structural classes are skipped: ``battery_side`` is the face, not a label,
    and a classifier trained to call it one would report it on every battery.
    A box with no identity is skipped too rather than guessed at.
    """
    out: list[tuple[str, list[list[float]]]] = []
    for box in ann.boxes(data) or []:
        label_id = str(box.get("label_id", "") or "").strip()
        if not label_id:
            # Older sidecars put the identity in `label`; take it only when it
            # is not one of the structural classes.
            name = str(box.get("label", "") or "").strip()
            if name and name not in labels_mod.STRUCTURAL_CLASSES:
                label_id = name
        if not label_id or label_id in labels_mod.STRUCTURAL_CLASSES:
            continue
        quad = ann.box_polygon(box)
        if len(quad) >= 4:
            out.append((label_id, quad))
    return out


def _write_crops(out: Path, entries: list[dataset_logic.Entry], split: str,
                 size: int, margin: float, tick=None) -> list[str]:
    import cv2

    rows: list[str] = []
    for entry in entries:
        if tick is not None:
            tick(f"crops {split}: {Path(entry.image).name}")
        image_path = Path(entry.annotation.get("image") or entry.image)
        if not image_path.is_file():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        for index, (label_id, quad) in enumerate(crop_targets(entry.annotation)):
            patch = rectify_quad(frame, expand_quad(quad, margin))
            if patch is None or patch.size == 0:
                continue
            folder = out / split / safe_token(label_id)
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{image_path.stem}__{index}.jpg"
            cv2.imwrite(str(folder / name), letterbox(patch, size),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            rows.append(f"{split},{label_id},{name},{entry.group_key()}")
    return rows


def write_crop_dataset(out: Path, entries: list[dataset_logic.Entry], *,
                       size: int = DEFAULT_CROP_PX,
                       margin: float = DEFAULT_MARGIN,
                       split_train: float = DEFAULT_SPLIT_TRAIN,
                       seed: int = DEFAULT_SEED, progress=None) -> Path:
    """Write a YOLO classification dataset: ``train/<label_id>/`` crops.

    Uses the same group-aware split as the detection export, so passing the
    same entries, split and seed gives two datasets that agree about which
    batteries are held out.
    """
    import shutil

    train, val, report = dataset_logic.split_entries(entries, split_train, seed=seed)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    total = len(train) + len(val)
    done = 0

    def tick(message: str) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total, message)

    rows = ["split,label_id,crop,group"]
    rows += _write_crops(out, train, "train", size, margin, tick)
    rows += _write_crops(out, val, "val", size, margin, tick)

    classes = sorted({r.split(",")[1] for r in rows[1:]})
    if not classes:
        raise FileNotFoundError(
            "No identified label boxes to crop. A crop needs a box with a label "
            "id on it -- draw and save some, then export again."
        )
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "split_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    (out / "task.txt").write_text("classify\n", encoding="utf-8")
    return out


def _scaled(fn):
    """Turn a (done, total, msg) callback into one reporting a 0-100 fraction."""
    if fn is None:
        return None

    def report(done: int, total: int, message: str) -> None:
        fraction = (done / total) if total else 1.0
        fn(int(round(fraction * 100)), 100, message)

    return report


def export_crops(entries, *, out: Path | None = None, **kwargs) -> Path:
    return write_crop_dataset(out or (EXPORT_DIR / "all_labels_classify"),
                              entries, **kwargs)


def export_two_stage(*, task: str = "obb", reviewed_only: bool = True,
                     split_train: float = DEFAULT_SPLIT_TRAIN,
                     seed: int = DEFAULT_SEED, out: Path | None = None,
                     library=None, size: int | None = None,
                     imgsz: int = 640,
                     margin: float = DEFAULT_MARGIN,
                     progress=None) -> tuple[Path, Path]:
    """Both halves of a detect-then-classify pipeline, from one set of entries.

    Returns ``(detect_dir, classify_dir)``.

    The same entries, split fraction and seed go into both, which is the point:
    the two datasets hold out the same batteries. Export them separately and a
    crop of a battery the detector validates on can land in the classifier's
    training set, and the pipeline's measured accuracy stops meaning anything.

    ``size`` defaults to the size identity actually needs -- every crop is
    resized to it regardless of the label's native size, so the bar is the
    identity floor and not the detector's resolution. See
    ``scale_report.crop_for_identity``.
    """
    from . import yolo_export

    datasets, _orphans = yolo_export.exportable_datasets(library)
    entries: list[dataset_logic.Entry] = []
    for label_id in datasets:
        entries.extend(yolo_export.collect_entries(label_id, reviewed_only))
    if not entries:
        raise FileNotFoundError(
            "No exportable images in any dataset. Label some images and mark them "
            "reviewed first."
        )

    base = out or EXPORT_DIR
    # Two datasets, one bar: the halves are reported as one run because the
    # operator pressed one button, and a bar that fills and restarts reads as
    # a job that finished and then started again.
    def half(offset: int, span: float):
        if progress is None:
            return None
        return lambda done, total, message: progress(
            offset + int(span * done), 100, message)

    detect_dir = yolo_export.write_dataset(
        base / f"two_stage_detect_{task}", entries, task=task,
        split_train=split_train, seed=seed, reviewed_only=reviewed_only,
        library=library, class_mode="generic", progress=_scaled(half(0, 0.5)))
    if size is None:
        from . import scale_report
        # crop_for_identity, NOT recommend_crop. recommend_crop asks whether a
        # crop loses detail against the detector, which is the right question
        # only when the detector also identifies. Here it never does, so its
        # resolution is not the bar -- and using it made the export write 448 px
        # crops while every report recommended 320, two rules disagreeing about
        # one number in one pipeline.
        size = scale_report.crop_for_identity(scale_report.measure(entries))
    classify_dir = write_crop_dataset(
        base / "two_stage_classify", entries, size=int(size), margin=margin,
        split_train=split_train, seed=seed, progress=_scaled(half(50, 0.5)))
    return detect_dir, classify_dir


# --- region crops: the disambiguator for labels that differ in fine print ---
#
# Where two labels differ only by a revision letter or a language line, no
# detector input and no whole-label crop resolves the difference -- see
# core/scale_report for the arithmetic. What does resolve it is cropping the
# read-region itself out of the full-resolution frame, where its pixels were
# never downscaled at all.
#
# Foldering those crops by label id gives exactly the training set for a
# disambiguator: "given this revision block, which label is this?". The ground
# truth is free, because the label id already encodes the answer.

REGION_ROLES_TO_CROP = ("code", "text")


def region_crop_targets(data, library):
    """``(label_id, region_name, quad)`` for every read-region on every box.

    The region's four image-space corners come from the label's own quad by
    proportion, so nothing is measured and nothing is calibrated: the operator
    drew the label, the library knows where the region sits inside it.
    """
    from . import geometry as geo

    out = []
    for label_id, quad in crop_targets(data):
        label = library.get(label_id) if library is not None else None
        if label is None:
            continue
        for role, name, rect in label.regions():
            if role not in REGION_ROLES_TO_CROP:
                continue
            placed = geo.place_unit_rect(quad, rect)
            if placed:
                out.append((label_id, f"{role}_{name}", placed))
    return out


def write_region_dataset(out: Path, entries, library, *, size: int = DEFAULT_CROP_PX,
                         margin: float = DEFAULT_MARGIN,
                         split_train: float = DEFAULT_SPLIT_TRAIN,
                         seed: int = DEFAULT_SEED) -> Path:
    """Crops of the read-regions themselves, foldered by label id.

    Taken from the full-resolution frame, so a 120 px revision block arrives as
    120 px however the detector is configured. ``size`` here is an upscale
    target, not a downscale: these regions are small to begin with, which is
    the entire reason this works.
    """
    import cv2
    import shutil

    train, val, report = dataset_logic.split_entries(entries, split_train, seed=seed)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = ["split,label_id,region,crop,native_px,group"]
    for split, group in (("train", train), ("val", val)):
        for entry in group:
            image_path = Path(entry.annotation.get("image") or entry.image)
            if not image_path.is_file():
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            for i, (label_id, region_name, quad) in enumerate(
                    region_crop_targets(entry.annotation, library)):
                patch = rectify_quad(frame, expand_quad(quad, margin))
                if patch is None or patch.size == 0:
                    continue
                native = max(patch.shape[:2])
                folder = out / split / safe_token(label_id)
                folder.mkdir(parents=True, exist_ok=True)
                name = f"{image_path.stem}__{region_name}__{i}.jpg"
                cv2.imwrite(str(folder / name), letterbox(patch, size),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                rows.append(f"{split},{label_id},{region_name},{name},"
                            f"{native:.0f},{entry.group_key()}")

    classes = sorted({r.split(",")[1] for r in rows[1:]})
    if not classes:
        raise FileNotFoundError(
            "No read-regions to crop. Define regions on your labels first "
            "(Define Regions) -- this exports the regions, not the labels."
        )
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "split_report.txt").write_text(report.text() + "\n", encoding="utf-8")
    (out / "task.txt").write_text("classify\n", encoding="utf-8")
    return out


def export_region_crops(*, reviewed_only: bool = True, library=None,
                        out: Path | None = None, size: int = DEFAULT_CROP_PX,
                        split_train: float = DEFAULT_SPLIT_TRAIN,
                        seed: int = DEFAULT_SEED) -> Path:
    from . import yolo_export

    entries = []
    for label_id in yolo_export.list_datasets():
        entries.extend(yolo_export.collect_entries(label_id, reviewed_only))
    if not entries:
        raise FileNotFoundError(
            "No exportable images. Label some images and mark them reviewed first.")
    return write_region_dataset(out or (EXPORT_DIR / "region_crops"), entries,
                                library, size=size, split_train=split_train, seed=seed)
