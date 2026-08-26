"""Entry points for the two wizards, and what happens when they finish.

Thin on purpose: both are the same ``FlowDialog`` pointed at a different flow,
and the only real work here is persisting the result and reporting what the
saved thing still needs before it can run.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from ..core import label_wizard, persistence, recipe_wizard
from ..core.labels import LabelDef
from ..core.recipes import Recipe
from .flow_dialog import FlowDialog


def add_label(parent=None, root: Path | None = None) -> LabelDef | None:
    """Run the add-a-label wizard and save the result into the library."""
    library = persistence.load_library(root)
    dialog = FlowDialog(label_wizard.FLOW, library=library, parent=parent)
    if not dialog.exec():
        return None

    label: LabelDef = dialog.result_object
    if label.label_id in library:
        replace = QMessageBox.question(
            parent, "Label exists",
            f"'{label.label_id}' is already in the library.\n\n"
            "Replace it? Existing annotations keep their label id, so they will "
            "follow the new definition.",
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
        "label them, and train. Nothing else has to be retrained.",
    )
    return label


def new_recipe(parent=None, existing: Recipe | None = None,
               root: Path | None = None) -> Recipe | None:
    """Run the recipe wizard, seeded from ``existing`` when editing."""
    library = persistence.load_library(root)
    answers = recipe_wizard.answers_from_recipe(existing) if existing else None
    flow = recipe_wizard.FLOW
    # Bind the library in so the summary page can say which labels are not
    # trained yet -- the one thing that stops a finished recipe from running.
    flow.review = lambda given: recipe_wizard.review_answers(given, library)

    dialog = FlowDialog(flow, answers=answers, library=library, parent=parent)
    if not dialog.exec():
        return None

    recipe: Recipe = dialog.result_object
    persistence.save_recipe(recipe, root)

    untrained = sorted(i for i in recipe.label_ids() if i not in library)
    message = f"Saved recipe '{recipe.safe_name}'."
    if untrained:
        message += ("\n\nThese labels are not in the library yet, so the recipe cannot "
                    "run until each has been added and trained:\n  "
                    + "\n  ".join(untrained))
    QMessageBox.information(parent, "Recipe saved", message)
    return recipe
