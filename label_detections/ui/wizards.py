"""The add-a-label wizard, and what happens when it finishes.

Thin on purpose: the dialog is the generic ``FlowDialog`` pointed at
``core.label_wizard.FLOW``, and the only real work here is persisting the
result and saying what the new label still needs.

There is deliberately no recipe wizard. Which labels a battery must carry, and
where each one belongs, is authored in the front end -- this tool's whole job
is producing a trained label.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from ..core import label_wizard, persistence
from ..core.labels import LabelDef
from .flow_dialog import FlowDialog


def _answers_from_label(label: LabelDef) -> dict:
    """Seed the wizard from an existing label, so editing reuses the same flow.

    A separate edit dialog would drift from the wizard within two releases;
    round-tripping through the same questions keeps them honest.
    """
    data = label.to_dict()
    answers = dict(label_wizard.FLOW.defaults())
    for question in label_wizard.FLOW.questions():
        if question.key in data:
            answers[question.key] = data[question.key]
    answers["codes"] = [dict(c) for c in data.get("codes", [])]
    answers["text_fields"] = [dict(t) for t in data.get("text_fields", [])]
    return answers


def add_label(parent=None, root: Path | None = None) -> LabelDef | None:
    """Run the wizard and save the result into the library."""
    library = persistence.load_library(root)
    dialog = FlowDialog(label_wizard.FLOW, library=library, parent=parent)
    if not dialog.exec():
        return None

    label: LabelDef = dialog.result_object
    if label.label_id in library:
        replace = QMessageBox.question(
            parent, "Label exists",
            f"'{label.label_id}' is already in the library.\n\n"
            "Replace its definition? Captured images and saved labels keep the "
            "same id, so they follow the new definition.",
        )
        if replace != QMessageBox.Yes:
            return None
        library.add(label, replace=True)
    else:
        library.add(label)
    persistence.save_library(library, root)

    QMessageBox.information(
        parent, "Label added",
        f"'{label.label_id}' is in the library.\n\n"
        f"Next: gather about {label.train_target} images of it into its dataset, "
        "label them, and train. No other label has to be retrained.",
    )
    return label


def edit_label(parent=None, label: LabelDef | None = None,
               root: Path | None = None) -> LabelDef | None:
    """Re-open an existing label's definition in the same wizard."""
    if label is None:
        return None
    library = persistence.load_library(root)
    dialog = FlowDialog(label_wizard.FLOW, answers=_answers_from_label(label),
                        library=library, parent=parent)
    if not dialog.exec():
        return None

    updated: LabelDef = dialog.result_object
    if updated.label_id != label.label_id:
        # Renaming would orphan the dataset folder, which is the one mistake
        # here that quietly loses work.
        QMessageBox.warning(
            parent, "Label id changed",
            f"The label id changed from '{label.label_id}' to '{updated.label_id}'.\n\n"
            f"'{label.label_id}' and its images are left alone; the new id starts "
            "with an empty dataset.",
        )
    library.add(updated, replace=True)
    persistence.save_library(library, root)
    return updated
