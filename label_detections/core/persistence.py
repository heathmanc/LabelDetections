"""Reading and writing the library, recipes and annotation sidecars.

Split out from the schema modules so those stay pure and unit testable without
a filesystem, and so every path in the app goes through one atomic writer.
Every function takes an optional root, which is what lets the tests run against
a temporary directory instead of the operator's real library.
"""
from __future__ import annotations

from pathlib import Path

from . import storage
from .labels import LabelDef, LabelLibrary
from .recipes import Recipe


# --- label library ---------------------------------------------------------

def library_path(root: Path | None = None) -> Path:
    return (root / "labels.json") if root else storage.LABEL_LIBRARY_PATH


def load_library(root: Path | None = None) -> LabelLibrary:
    """The label library, or an empty one on a first run or a bad file.

    An unreadable library returns empty rather than raising: the app must still
    launch so an operator can point it at the right data folder.
    """
    return LabelLibrary.from_dict(storage.read_json(library_path(root)))


def save_library(library: LabelLibrary, root: Path | None = None) -> Path:
    return storage.write_json(library_path(root), library.to_dict())


def add_label(label: LabelDef, root: Path | None = None, *, replace: bool = False) -> LabelLibrary:
    """Add one label and persist. Raises if it exists and ``replace`` is off."""
    library = load_library(root)
    library.add(label, replace=replace)
    save_library(library, root)
    return library


# --- recipes ---------------------------------------------------------------

def save_recipe(recipe: Recipe, root: Path | None = None) -> Path:
    return storage.write_json(storage.recipe_path(recipe.safe_name, root), recipe.to_dict())


def load_recipe(path: Path) -> Recipe | None:
    data = storage.read_json(path)
    return Recipe.from_dict(data) if data else None


def list_recipes(root: Path | None = None) -> list[Recipe]:
    """Every readable recipe, sorted by name. Unreadable files are skipped.

    Skipped rather than fatal: one corrupt recipe on a shared network library
    must not stop anyone opening the other forty.
    """
    out: list[Recipe] = []
    for path in storage.list_recipe_files(root):
        recipe = load_recipe(path)
        if recipe is not None:
            out.append(recipe)
    return sorted(out, key=lambda r: r.safe_name)


# --- annotation sidecars ---------------------------------------------------

def load_annotation(label_id: str, image: str | Path, root: Path | None = None) -> dict | None:
    return storage.read_json(storage.annotation_path(label_id, Path(image).name, root))


def save_annotation(label_id: str, image: str | Path, data: dict,
                    root: Path | None = None) -> Path:
    return storage.write_json(
        storage.annotation_path(label_id, Path(image).name, root), data)


def dataset_statuses(label_id: str, capture_root: Path | None = None,
                     label_root: Path | None = None) -> dict[str, str]:
    """``{image name: status}`` for one label's dataset.

    Walks the image folder rather than the sidecar folder, so an image with no
    annotation at all is reported as unlabeled instead of being invisible.
    """
    from . import review

    out: dict[str, str] = {}
    for image in storage.list_images(label_id, capture_root):
        data = storage.read_json(storage.annotation_path(label_id, image.name, label_root))
        out[image.name] = review.annotation_status(data, label_id)
    return out
