"""Operator-readable diagnostics for an exported YOLO dataset.

Pure stdlib (csv + pathlib): reads back the files an export wrote (manifest.csv,
data.yaml) and summarizes what actually landed in the dataset. No Qt/OpenCV, so
it is unit testable headlessly.
"""
from __future__ import annotations

import csv
from pathlib import Path


def class_names(out: Path) -> list[str]:
    """Read the ordered class names from a dataset's data.yaml ``names:`` block."""
    data_yaml = Path(out) / "data.yaml"
    if not data_yaml.exists():
        return []
    names: list[str] = []
    in_names = False
    try:
        for raw in data_yaml.read_text(encoding="utf-8").splitlines():
            if raw.strip() == "names:":
                in_names = True
                continue
            if in_names:
                stripped = raw.strip()
                if not raw.startswith(" ") or not stripped:
                    break
                # Lines look like "  0: battery_model".
                _, _, name = stripped.partition(":")
                if name.strip():
                    names.append(name.strip())
    except Exception:
        return names
    return names


def _int(row: dict, key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def count_summary(out: Path) -> str:
    """An operator-readable breakdown of what an export actually wrote.

    Read back from the dataset's own manifest.csv rather than from whatever the
    exporter believed it was doing, so a mismatch between the two is visible
    before training rather than after.
    """
    out = Path(out)
    manifest = out / "manifest.csv"
    if not manifest.exists():
        return "No manifest.csv was written; cannot summarize export counts."
    try:
        rows = list(csv.DictReader(manifest.read_text(encoding="utf-8").splitlines()))
    except Exception as exc:
        return f"Could not read manifest.csv: {exc}"
    if not rows:
        return "No labeled images were written to this dataset."

    split_images = {"train": 0, "val": 0}
    per_label: dict[str, int] = {}
    groups: set[str] = set()
    total_boxes = 0
    empty_images = 0
    augmented = 0

    for row in rows:
        split = str(row.get("split", "")).strip()
        if split in split_images:
            split_images[split] += 1
        label_id = str(row.get("label_id", "")).strip() or "(unknown)"
        per_label[label_id] = per_label.get(label_id, 0) + 1
        group = str(row.get("group", "")).strip()
        if group:
            groups.add(group)
        boxes = _int(row, "boxes")
        total_boxes += boxes
        if boxes == 0:
            empty_images += 1
        if _int(row, "augmented"):
            augmented += 1

    lines = [
        f"Images written: {len(rows)}  (train {split_images['train']}, "
        f"val {split_images['val']})",
        f"Boxes written: {total_boxes}",
    ]
    if per_label:
        lines.append("Images per label:")
        for label_id in sorted(per_label):
            lines.append(f"  {label_id}: {per_label[label_id]}")
    if empty_images:
        # Backgrounds export as an empty label file on purpose, so this is a
        # count worth showing rather than a warning.
        lines.append(f"Images with no boxes (backgrounds): {empty_images}")
    if augmented:
        lines.append(
            f"Of those, {augmented} are variable-region copies (train only)")
    if groups:
        lines.append(f"Capture groups: {len(groups)} (never split across train/val)")

    classes = class_names(out)
    if classes:
        lines.append(f"Detector families ({len(classes)}): " + ", ".join(classes))

    split_report = out / "split_report.txt"
    if split_report.exists():
        try:
            warnings = [line for line in split_report.read_text(encoding="utf-8").splitlines()
                        if line.startswith("WARNING:")]
        except Exception:
            warnings = []
        if warnings:
            lines.append("")
            lines.extend(warnings)
    return "\n".join(lines)
