"""Identifying a label from its printed part number.

The camera on this line has to take in a whole battery, which fixes the field
of view and leaves the UPC-A at 3.35 pixels per module -- under what a
photographed, rectified symbol survives. The part number printed beside it gets
a 40 pixel cap height, which is comfortable. So the barcode turns out to be the
hardest thing on the label to read rather than the easiest: it packs the same
characters into 95 narrow bars where the text spells them out across three
times the width.

What makes reading it workable is that this is not transcription. Nobody needs
to know what the label says, only which of a handful of enrolled part numbers
it is -- or that it is none of them, which is the failure the whole feature
exists to catch.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import text_read as tr

EXPECTED = {"PC680": ["ODS-AGM16L"], "sp_g31": ["NP16-12B"],
            "2220-9199": ["2220-9199"]}


class _Field:
    def __init__(self, policy="must_match_pattern", pattern="ODS-AGM16L",
                 region=(0.68, 0.05, 0.30, 0.10), name="part_number"):
        self.policy, self.pattern, self.name = policy, pattern, name
        self.region = list(region)


def _verdict(text, proposed="PC680", fields=None, expected=None):
    reads = [tr.Read(text=text, confidence=0.9)] if text else []
    return tr.verdict(proposed, fields or [_Field()], reads,
                      EXPECTED if expected is None else expected)


# --- what OCR actually does to a part number --------------------------------

@pytest.mark.parametrize("text,why", [
    ("ODS-AGM16L (PC680)", "a clean read"),
    ("ODS-AGM16L(PC680)", "the space between number and bracket dropped"),
    ("ODS-AGM16L （PC680)", "a full-width bracket, which low resolution invites"),
    ("0DS-AGM16L (PC680)", "one character misread, letter O as digit zero"),
    ("ODS-AGMI6L", "two characters misread and the bracket lost"),
    ("12V-16Ah ODS-AGM16L (PC680)", "the crop caught the line above as well"),
])
def test_a_damaged_read_still_identifies_its_label(text, why):
    """The bar is not transcription. Every one of these is unmistakably PC680
    against the other enrolled labels, however mangled."""
    assert _verdict(text).state == tr.CONFIRMED, why


def test_a_label_that_was_never_enrolled_is_refused():
    """The reason for all of it."""
    verdict = _verdict("NP18-99Z UNSEEN")
    assert verdict.state == tr.CONTRADICTED
    assert verdict.label_id == ""
    assert tr.resolve("PC680", verdict) == tr.UNKNOWN


def test_a_different_enrolled_label_relabels_rather_than_failing():
    verdict = _verdict("NP16-12B 12V 16Ah")
    assert verdict.state == tr.CONTRADICTED
    assert tr.resolve("PC680", verdict) == "sp_g31"


def test_a_crop_that_landed_on_the_wrong_line_is_refused():
    """"12V - 16Ah" is on every one of these labels and identifies none."""
    assert tr.resolve("PC680", _verdict("12V - 16Ah")) == tr.UNKNOWN


def test_reading_nothing_is_not_a_pass():
    verdict = _verdict("")
    assert verdict.state == tr.UNREADABLE and verdict.blocks


# --- refusing to guess ------------------------------------------------------

def test_two_labels_a_damaged_read_sits_between_are_not_guessed_at():
    """Part numbers one character apart are a real thing, and a read damaged in
    exactly that character is equally close to both. Taking the higher score
    would be choosing by noise."""
    expected = {"rev_c": ["ODS-AGM16C"], "rev_d": ["ODS-AGM16D"]}
    verdict = _verdict("ODS-AGM16", proposed="rev_c", expected=expected)
    assert verdict.state == tr.CONTRADICTED
    assert "too close to call" in verdict.detail
    assert tr.resolve("rev_c", verdict) == tr.UNKNOWN


def test_a_label_with_no_text_marked_for_inspection_is_left_alone():
    verdict = _verdict("anything", fields=[_Field(policy="ignore")])
    assert verdict.state == tr.NOT_CHECKED
    assert tr.resolve("PC680", verdict) == "PC680"


def test_text_read_with_nothing_to_compare_against_reports_presence():
    """Not confirmation: any text at all would satisfy it."""
    verdict = _verdict("ODS-AGM16L", expected={})
    assert verdict.state == tr.PRESENT
    assert not verdict.verified


def test_case_and_punctuation_are_ignored_but_digits_are_not_folded():
    """Mapping O to 0 would fix some misreads and make genuinely different part
    numbers collide, which is the failure this exists to prevent."""
    assert tr.normalise("ods-agm16l (pc680)") == "ODSAGM16LPC680"
    assert tr.similarity("ODS-AGM16L", "ods agm16l") == 1.0
    assert tr.similarity("PC680", "PC6BO") < 1.0


def test_a_short_expected_string_is_found_inside_a_longer_read():
    """The crop is a region, and will often carry more than the part number."""
    assert tr.similarity("ODS-AGM16L", "12V 16Ah ODS-AGM16L PC680") == 1.0


# --- what the operator sees -------------------------------------------------

def test_a_refusal_names_the_label_whose_text_was_expected():
    assert tr.plate_note(tr.Verdict(tr.UNREADABLE), "PC680") == "no text (PC680)"
    assert tr.plate_note(tr.Verdict(tr.CONFIRMED)) == "text ok"
    assert tr.plate_note(tr.Verdict(tr.CONTRADICTED, "sp_g31")) == "WRONG TEXT"


def test_the_expected_strings_come_off_the_library():
    class Label:
        label_id = "PC680"
        text_fields = [_Field(pattern="^ODS-AGM16L$")]

    # Anchors are a regex idiom; this compares text, so they are not part of it.
    assert tr.expected_for([Label()]) == {"PC680": ["ODS-AGM16L"]}


def test_a_field_nobody_inspects_contributes_no_expectation():
    class Label:
        label_id = "PC680"
        text_fields = [_Field(policy="ignore", pattern="ODS-AGM16L")]

    assert tr.expected_for([Label()]) == {}
