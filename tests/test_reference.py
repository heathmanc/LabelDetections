"""A label is its reference image.

Every read-region on a label is stored as a fraction of the label's outline,
and that outline only means anything against the artwork it was drawn on. The
artwork is not an extra a label picks up later; it is the coordinate system the
rest of the definition is written in.

So it is required, and it is immutable. Required because the alternative is a
label that looks finished in the list and verifies nothing. Immutable because
every region on every image already reviewed is positioned against it --
re-flattening from a differently-drawn box moves all of them at once, silently,
against work already checked.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import reference as ref


class _Label:
    def __init__(self, label_id="PC680", images=(), codes=(), texts=()):
        self.label_id = label_id
        self.reference_images = list(images)
        self.codes = list(codes)
        self.text_fields = list(texts)


def test_a_label_with_no_artwork_has_none(tmp_path):
    assert not ref.has_reference(_Label())
    assert ref.reference_path(_Label()) == ""


def test_a_reference_whose_file_has_gone_counts_as_none(tmp_path):
    """Recovering from a deleted file is not the same act as replacing artwork
    that exists, and should not have to argue with a rule written for that."""
    assert not ref.has_reference(_Label(images=[str(tmp_path / "never.png")]))


def test_a_reference_that_exists_counts(tmp_path):
    path = tmp_path / "art.png"
    path.write_bytes(b"x")
    label = _Label(images=[str(path)])
    assert ref.has_reference(label)
    assert ref.reference_path(label) == str(path)


def test_the_first_file_that_exists_wins(tmp_path):
    """Older libraries kept a list. One of them is the coordinate system."""
    good = tmp_path / "real.png"
    good.write_bytes(b"x")
    label = _Label(images=[str(tmp_path / "gone.png"), str(good)])
    assert ref.reference_path(label) == str(good)


# --- what a refusal has to say ----------------------------------------------

def test_a_usable_label_produces_no_refusal(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"x")
    assert ref.block_reason(_Label(images=[str(path)])) == ""


def test_a_refusal_names_the_label_the_act_and_the_way_out():
    """A refusal that only says no is a wall. The fix is one capture away and
    the operator needs to know where."""
    reason = ref.block_reason(_Label(), "labelling against it")
    assert "PC680" in reason
    assert "labelling against it" in reason
    assert "Capture Reference" in reason
    # And why, so it does not read as an arbitrary rule.
    assert "fraction of the label" in reason


def test_no_label_at_all_is_not_a_refusal():
    """Nothing is open; that is a different state and has its own message."""
    assert ref.block_reason(None) == ""


# --- what the library is worth ----------------------------------------------

def test_the_note_names_the_labels_that_cannot_be_used(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"x")
    note = ref.library_note([_Label("PC680", [str(path)]),
                             _Label("2220-9199"), _Label("ODX-Long")])
    assert "2 of 3" in note
    assert "2220-9199" in note and "ODX-Long" in note
    assert "PC680" not in note


def test_a_complete_library_says_so_rather_than_staying_silent():
    """Silence reads the same whether it was checked or not."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        note = ref.library_note([_Label("PC680", [handle.name])])
    assert "All 1 label(s)" in note


def test_an_empty_library_claims_nothing():
    assert ref.library_note([]) == ""


def test_missing_keeps_the_order_it_was_given():
    assert ref.missing([_Label("b"), _Label("a")]) == ["b", "a"]


# --- replacing is deleting ---------------------------------------------------

def test_the_replace_warning_counts_what_will_have_to_be_redrawn():
    warning = ref.replace_warning(_Label("PC680", codes=[1, 2], texts=[3]))
    assert "Delete PC680's reference image?" in warning
    assert "3 region(s)" in warning
    assert "verifies nothing" in warning


def test_it_says_labelled_images_are_untouched():
    """The fear this answers: that redoing the artwork throws away the dataset."""
    assert "not touched" in ref.replace_warning(_Label())


def test_a_label_with_no_regions_is_not_warned_about_losing_any():
    warning = ref.replace_warning(_Label("PC680"))
    assert "region(s)" not in warning
    assert "Delete PC680's reference image?" in warning
