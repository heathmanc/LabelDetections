"""Reading printed text with a real recogniser.

The counterpart to test_code_decoding: everything else is tested against stub
reads, and this renders part numbers at the sizes the camera actually delivers
and puts them through the engine.

It exists because the barcode work taught the lesson the hard way -- the crop
was right, the wrapper worked on a clean render, and the failure lived entirely
in the gap between the two.

Skips where rapidocr-onnxruntime is not installed, which is a supported state
the readout names.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("rapidocr_onnxruntime",
                    reason="the recogniser itself is what is under test")
cv2 = pytest.importorskip("cv2")

from label_detections.core import text_read as tr
from label_detections.core import text_reader as trd

PART = "ODS-AGM16L (PC680)"
EXPECTED = {"PC680": ["ODS-AGM16L"], "sp_g31": ["NP16-12B"]}


def line(text: str = PART, cap_px: int = 40, blur: float = 0.0) -> np.ndarray:
    """One line of printed text at a given cap height."""
    scale = cap_px / 22.0
    w, h = int(620 * scale / 1.6), int(70 * scale)
    image = np.full((h, w, 3), 238, np.uint8)
    cv2.putText(image, text, (int(6 * scale), int(h * 0.72)),
                cv2.FONT_HERSHEY_DUPLEX, scale * 0.75, (25, 25, 25),
                max(1, int(scale * 1.4)))
    return cv2.GaussianBlur(image, (0, 0), blur) if blur else image


def test_the_recogniser_reads_the_part_number_at_the_size_the_camera_gives():
    """40 px is what this rig actually delivers, measured off the label."""
    reads = trd.recognise(line(cap_px=40))
    assert reads and "ODS-AGM16L" in tr.normalise(reads[0].text) or \
        tr.similarity("ODS-AGM16L", reads[0].text) > 0.9


@pytest.mark.parametrize("cap_px,blur", [(40, 0), (40, 1.2), (30, 0), (30, 1.2),
                                         (22, 1.2)])
def test_it_still_identifies_the_label_as_the_text_gets_smaller(cap_px, blur):
    """Not "reads it perfectly" -- identifies it. That is the bar, and it holds
    well below the size where transcription starts to fray."""
    reads = trd.recognise(line(cap_px=cap_px, blur=blur))
    verdict = tr.verdict("PC680", [_field()], reads, EXPECTED)
    assert verdict.state == tr.CONFIRMED, [r.text for r in reads]


def _field():
    class Field:
        policy, pattern, name = "must_match_pattern", "ODS-AGM16L", "part_number"
        region = [0.68, 0.05, 0.30, 0.10]
    return Field()


def test_an_unenrolled_part_number_is_refused_after_a_real_read():
    """The whole feature, on real pixels: a label nobody has taught it."""
    reads = trd.recognise(line("NP18-99Z UNSEEN"))
    assert reads, "the fixture must read, or this proves nothing"
    verdict = tr.verdict("PC680", [_field()], reads, EXPECTED)
    assert verdict.state == tr.CONTRADICTED
    assert tr.resolve("PC680", verdict) == tr.UNKNOWN


def test_recognition_skips_detection_and_stays_inside_a_frame_budget():
    """The region is drawn on the artwork, so "where is the text" was answered
    once by a person. Asking the engine again costs about a second, against
    fifteen milliseconds for reading the line it was handed."""
    import time

    image = line()
    trd.recognise(image)                       # warm up the session
    start = time.perf_counter()
    for _ in range(3):
        trd.recognise(image)
    per_call = (time.perf_counter() - start) / 3
    assert per_call < 0.20, f"{per_call*1000:.0f} ms is too slow for the live path"


def test_a_tiny_crop_is_scaled_up_before_it_is_read():
    """Recognition models want a line about 48 px tall, and letting their own
    preprocessing do it eats what little a 20 px cap height has."""
    small = line(cap_px=14)
    assert small.shape[0] < trd.TARGET_LINE_PX
    assert trd._scaled(small).shape[0] >= trd.TARGET_LINE_PX
    # And a crop that is already big enough is left alone.
    big = line(cap_px=60)
    assert trd._scaled(big) is big


def test_reading_a_blank_crop_returns_nothing_rather_than_noise():
    assert trd.recognise(np.full((60, 400, 3), 240, np.uint8)) == []


# --- through the geometry, from a whole frame -------------------------------

def test_the_field_is_cropped_out_of_the_full_resolution_frame_and_read():
    """End to end: a label inside a frame, the region mapped onto its quad, and
    the part number identified from the pixels that came back."""
    label_w = 2109
    label_h = int(label_w * 517 / 1600)
    label = np.full((label_h, label_w, 3), 236, np.uint8)
    cv2.rectangle(label, (0, 0), (label_w, int(label_h * 0.30)), (38, 38, 38), -1)
    cap = label_h * 0.058
    cv2.putText(label, PART, (int(label_w * 0.69), int(label_h * 0.115 + cap / 2)),
                cv2.FONT_HERSHEY_DUPLEX, cap / 22.0, (245, 245, 245), 2)

    frame = np.full((3672, 5496, 3), 200, np.uint8)
    frame[1400:1400 + label_h, 1600:1600 + label_w] = label
    quad = [[1600, 1400], [1600 + label_w, 1400],
            [1600 + label_w, 1400 + label_h], [1600, 1400 + label_h]]

    reads = trd.read_label(frame, quad, [_field()])
    verdict = tr.verdict("PC680", [_field()], reads, EXPECTED)
    assert verdict.state == tr.CONFIRMED, [r.text for r in reads]
    assert tr.plate_note(verdict, "PC680") == "text ok"
