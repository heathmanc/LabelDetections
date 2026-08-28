"""The rule that a label is its reference image.

Every read-region on a label -- a barcode, a part number, an anchor -- is
stored as a fraction of the label's outline, and that outline only means
anything against the artwork it was drawn on. The artwork is therefore not an
optional extra a label may acquire later; it is the coordinate system the rest
of the definition is written in. A label without one has regions that refer to
nothing, cannot be code-verified, cannot be text-verified, and cannot say where
anything sits.

So it is required, and it is immutable. Required because the alternative is a
label that looks finished in the list and quietly verifies nothing. Immutable
because every region, on every image already reviewed, is positioned against
it: re-flattening from a differently-drawn box moves all of them at once,
silently, against work that was already checked. Changing artwork is not an
edit -- it is deleting the label's coordinate system and drawing a new one, and
it has to cost what that costs.

The decisions live here; the dialog that captures one is in ``ui/reference_setup``.
"""
from __future__ import annotations

from pathlib import Path

# What a label cannot do without artwork. Not a style rule -- each of these
# genuinely has nowhere to put its answer without a coordinate system.
BLOCKED_WITHOUT_REFERENCE = (
    "drawing boxes against it",
    "defining read-regions",
    "verifying a code or printed text",
    "exporting or training",
)


def reference_path(label) -> str:
    """The label's artwork, if it has any still on disk.

    A reference whose file has gone counts as none. Recovering from a deleted
    file is not the same act as replacing artwork that exists, and it should
    not have to argue with a rule written for the second thing.
    """
    for reference in getattr(label, "reference_images", None) or []:
        if reference and Path(str(reference)).is_file():
            return str(reference)
    return ""


def has_reference(label) -> bool:
    return bool(reference_path(label))


def missing(labels) -> list[str]:
    """Every label id with no artwork, in the order they were given."""
    return [str(getattr(label, "label_id", "")) for label in labels or []
            if not has_reference(label)]


def block_reason(label, doing: str = "") -> str:
    """Why this label cannot be used, or "" when it can.

    One sentence and then the way out. A refusal that only says no is a wall;
    the operator needs to know that the fix is one capture away, and where.
    """
    if label is None:
        return ""
    if has_reference(label):
        return ""
    label_id = str(getattr(label, "label_id", "") or "this label")
    what = doing or "using it"
    return (f"{label_id} has no reference image, so {what} is not possible "
            f"yet.\n\n"
            f"Every read-region is stored as a fraction of the label's "
            f"outline, and that outline is drawn on the artwork -- without it "
            f"there is no coordinate system to put anything in.\n\n"
            f"Capture Reference sets it up: one photograph, the label's "
            f"outline, and its regions, in one go.")


def library_note(labels) -> str:
    """One line for the readout: how much of the library is unusable."""
    absent = missing(labels)
    total = len(list(labels or []))
    if not total:
        return ""
    if not absent:
        return f"All {total} label(s) have a reference image."
    return (f"{len(absent)} of {total} label(s) have NO reference image "
            f"({', '.join(absent[:4])}{', ...' if len(absent) > 4 else ''}). "
            f"Nothing can be drawn, verified or exported against those.")


def replace_warning(label) -> str:
    """What deleting a label's artwork costs, said before it happens."""
    label_id = str(getattr(label, "label_id", "") or "this label")
    codes = len(getattr(label, "codes", None) or [])
    texts = len(getattr(label, "text_fields", None) or [])
    regions = codes + texts
    lines = [f"Delete {label_id}'s reference image?",
             "",
             "The reference image is the coordinate system every region on "
             "this label is written in, which is why it is never edited in "
             "place -- only deleted and drawn again."]
    if regions:
        lines.append("")
        lines.append(
            f"{regions} region(s) are positioned against it and will have to "
            f"be drawn again on the new artwork. Until they are, this label "
            f"verifies nothing.")
    lines.append("")
    lines.append("Images already labelled and reviewed are not touched.")
    return "\n".join(lines)
