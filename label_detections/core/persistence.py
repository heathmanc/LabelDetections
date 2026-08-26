"""Reading and writing the label library and annotation sidecars.

Split out from the schema modules so those stay pure and unit testable without
a filesystem, and so every write in the app goes through one atomic writer.
Every function takes an optional root, which is what lets the tests run against
a temporary directory instead of the operator's real library.

No recipes here: which labels a battery must carry, and where each belongs, is
authored and stored by the front end.
"""
from __future__ import annotations

from pathlib import Path

from . import storage
from .labels import LabelDef, LabelLibrary


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
