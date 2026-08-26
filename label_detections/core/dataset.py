"""Dataset assembly for a per-label training run, and the group-aware split.

This module exists because of one bug that a plain per-image random split
guarantees, and that no amount of model work can compensate for.

Label images arrive in bursts: several frames of the same physical label from
one fixture, a handful off one pallet, a batch pulled from one print run. Those
frames share lighting, wear, print drift and placement. Shuffle images
individually -- which is what BungVision's ``yolo_export._split_entries`` does,
and it was correct for its single-camera case -- and siblings land on both
sides of the split. Validation then measures memorisation and reports it as
accuracy.

So the split unit is the **group** (a capture session, a source, a pallet),
never the image. When an entry carries no grouping metadata it is its own
group, which degrades to the old behaviour rather than silently doing
something surprising.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class Entry:
    """One exportable image and the metadata the split reasons about."""
    label_id: str
    image: str
    annotation: dict[str, Any] = field(default_factory=dict)
    # Provenance, in the order the split prefers it. A capture session groups
    # most tightly; a source (line, pallet, print run) is the next best thing.
    session: str = ""
    source: str = ""
    view: str = ""
    status: str = "ready"

    def group_key(self) -> str:
        return self.session or self.source or self.image


def entry_from_annotation(label_id: str, image: str, data: dict[str, Any],
                          status: str = "ready") -> Entry:
    """Build an entry, taking provenance from the sidecar if it recorded any."""
    return Entry(
        label_id=label_id,
        image=image,
        annotation=data or {},
        session=str((data or {}).get("session", "") or ""),
        source=str((data or {}).get("source", "") or ""),
        view=str((data or {}).get("view", "") or ""),
        status=status,
    )


def group_entries(entries: Iterable[Entry],
                  group_by: Callable[[Entry], str] | None = None) -> dict[str, list[Entry]]:
    """Bucket entries by split group, preserving input order within a group."""
    key = group_by or (lambda e: e.group_key())
    groups: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        groups[key(entry)].append(entry)
    return dict(groups)


@dataclass
class SplitReport:
    """Everything a reviewer needs to trust -- or reject -- a split."""
    train_images: int = 0
    val_images: int = 0
    train_groups: int = 0
    val_groups: int = 0
    train_labels: dict[str, int] = field(default_factory=dict)
    val_labels: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            f"train: {self.train_images} images / {self.train_groups} groups",
            f"val:   {self.val_images} images / {self.val_groups} groups",
            "",
            "images per label:",
        ]
        for label in sorted(set(self.train_labels) | set(self.val_labels)):
            lines.append(f"  {label}: train {self.train_labels.get(label, 0)}, "
                         f"val {self.val_labels.get(label, 0)}")
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        return "\n".join(lines)


def split_entries(entries: list[Entry], split_train: float = 0.8, *, seed: int = 0,
                  group_by: Callable[[Entry], str] | None = None,
                  ) -> tuple[list[Entry], list[Entry], SplitReport]:
    """Split into train/val without ever separating a group.

    ``seed`` is explicit rather than global so a dataset is reproducible: the
    same images and the same seed give the same split, which is the only way
    two training runs are comparable.

    After the group split, any label missing from validation is repaired by
    moving the smallest group that contains it. A model validated without a
    single example of a label has said nothing about that label at all.
    """
    groups = group_entries(entries, group_by)
    report = SplitReport()
    if not groups:
        return [], [], report

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = sum(len(groups[k]) for k in keys)
    target_train = total * max(0.0, min(1.0, split_train))

    train_keys: list[str] = []
    val_keys: list[str] = []
    running = 0
    for key in keys:
        if running < target_train or not train_keys:
            train_keys.append(key)
            running += len(groups[key])
        else:
            val_keys.append(key)

    if not val_keys:
        if len(train_keys) > 1:
            val_keys.append(train_keys.pop())
        else:
            # A single-group dataset cannot be split. Training and validating on
            # the same images is at least honest about being a smoke test.
            report.warnings.append(
                "Only one capture group in the dataset: train and val are the same images."
            )
            val_keys = list(train_keys)

    train_keys, val_keys = _repair_label_coverage(groups, train_keys, val_keys, report)

    train = [e for k in train_keys for e in groups[k]]
    val = [e for k in val_keys for e in groups[k]]

    report.train_images = len(train)
    report.val_images = len(val)
    report.train_groups = len(train_keys)
    report.val_groups = len(val_keys)
    report.train_labels = dict(Counter(e.label_id for e in train))
    report.val_labels = dict(Counter(e.label_id for e in val))
    for label in sorted(set(report.train_labels) - set(report.val_labels)):
        report.warnings.append(
            f"Label '{label}' has no validation images, so its metrics mean nothing."
        )
    return train, val, report


def _repair_label_coverage(groups: dict[str, list[Entry]], train_keys: list[str],
                           val_keys: list[str], report: SplitReport,
                           ) -> tuple[list[str], list[str]]:
    """Move groups into val until every label is represented there."""
    train_keys = list(train_keys)
    val_keys = list(val_keys)
    all_labels = {e.label_id for entries in groups.values() for e in entries}

    for _ in range(len(train_keys)):
        val_labels = {e.label_id for k in val_keys for e in groups[k]}
        missing = all_labels - val_labels
        if not missing or len(train_keys) <= 1:
            break
        candidates = [k for k in train_keys if missing & {e.label_id for e in groups[k]}]
        if not candidates:
            break
        move = min(candidates, key=lambda k: len(groups[k]))
        train_keys.remove(move)
        val_keys.append(move)
        covered = sorted(missing & {e.label_id for e in groups[move]})
        report.warnings.append(
            f"Moved group '{move}' into validation so "
            + ", ".join(f"'{label}'" for label in covered)
            + " is represented there."
        )
    return train_keys, val_keys


# --- coverage reporting ----------------------------------------------------

def instance_counts(entries: Iterable[Entry]) -> dict[str, int]:
    """``{label_id: annotated instances}`` -- boxes, not images.

    Instances, because one image can carry two of the same label and an image
    count would understate the training signal.
    """
    counts: Counter = Counter()
    for entry in entries:
        for box in entry.annotation.get("boxes", []) or []:
            label_id = str(box.get("label_id", "") or "")
            if label_id:
                counts[label_id] += 1
    return dict(counts)


def family_counts(entries: Iterable[Entry]) -> dict[str, int]:
    """``{detector_family: instances}`` -- what the model is actually trained on."""
    counts: Counter = Counter()
    for entry in entries:
        for box in entry.annotation.get("boxes", []) or []:
            counts[str(box.get("label", "") or "unknown")] += 1
    return dict(counts)


def code_coverage(entries: Iterable[Entry]) -> dict[str, dict[str, int]]:
    """Per label, how many code regions were annotated and how many decoded.

    The decode rate is the number that separates "the model did not find it"
    from "the optics could not read it", and it settles that argument with
    data instead of opinion.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"regions": 0, "decoded": 0})
    for entry in entries:
        for box in entry.annotation.get("boxes", []) or []:
            label_id = str(box.get("label_id", "") or "")
            if not label_id:
                continue
            for region in box.get("regions", []) or []:
                if str(region.get("role", "")) != "code":
                    continue
                out[label_id]["regions"] += 1
                if region.get("decode_ok"):
                    out[label_id]["decoded"] += 1
    return {k: dict(v) for k, v in out.items()}


def thin_coverage(entries: Iterable[Entry], minimum: int = 150) -> list[str]:
    """Labels with too few instances to train on, thinnest first.

    Certification marks and region-specific warnings are always the thin ones,
    and they are exactly what a wrong-market shipment turns on.
    """
    totals = instance_counts(entries)
    thin = sorted((n, k) for k, n in totals.items() if n < minimum)
    return [f"{k}: {n} instances (want {minimum}+)" for n, k in thin]


def defect_mix(entries: Iterable[Entry]) -> dict[str, int]:
    """How many images are deliberate defect examples, by reason.

    A dataset of nothing but good labels trains a model that has never seen a
    torn one. This is the tally that shows whether the defect library is real
    or aspirational.
    """
    counts: Counter = Counter()
    for entry in entries:
        review = entry.annotation.get("review")
        if isinstance(review, dict) and review.get("forced_review"):
            counts[str(review.get("defect_reason", "other") or "other")] += 1
    return dict(counts)
