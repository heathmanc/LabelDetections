"""Decoding real barcodes, with a real decoder.

Everything else about code verification is tested against stub reads, because
the decisions are the part that can be wrong in interesting ways. This file is
the other half: it renders genuine UPC-A symbols, degrades them the way the
pipeline does, and puts them through the actual decoder.

It exists because the first real barcode this tool was pointed at did not read,
and nothing in the test suite could have caught that -- the crop was perfect,
the wrapper worked on a clean render, and the failure lived entirely in the gap
between the two.

Skips where zxing-cpp is not installed. It is a real dependency, and running
without it is a supported state that the readout names.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("zxingcpp", reason="the decoder itself is what is under test")
cv2 = pytest.importorskip("cv2")

from label_detections.core import code_reader as cr
from label_detections.core import codes as cd

CODE = "635241140996"          # the UPC-A off a real Odyssey PC680

_L = ["0001101", "0011001", "0010011", "0111101", "0100011",
      "0110001", "0101111", "0111011", "0110111", "0001011"]
_R = ["".join("1" if c == "0" else "0" for c in p) for p in _L]


def render(digits: str = CODE, module_px: int = 3, height: int = 110,
           quiet_modules: int = 9) -> np.ndarray:
    """A real UPC-A: guards, parity, 95 modules."""
    bits = "101" + "".join(_L[int(d)] for d in digits[:6]) + "01010" \
        + "".join(_R[int(d)] for d in digits[6:]) + "101"
    assert len(bits) == 95
    quiet = quiet_modules * module_px
    image = np.full((height, quiet * 2 + 95 * module_px, 3), 255, np.uint8)
    x = quiet
    for bit in bits:
        if bit == "1":
            image[:, x:x + module_px] = 0
        x += module_px
    return image


def photographed(module_px: int, skew_px: int, blur: float) -> np.ndarray:
    """Through the real pipeline: seen at an angle, then rectified.

    The warp is what leaves the softness. A code the camera resolves cleanly
    still arrives at the decoder having been resampled twice.
    """
    image = render(module_px=module_px)
    h, w = image.shape[:2]
    flat = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    tilted = np.float32([[0, skew_px], [w, 0], [w, h], [0, h - skew_px]])
    seen = cv2.warpPerspective(image, cv2.getPerspectiveTransform(flat, tilted),
                               (w, h), borderValue=(255, 255, 255))
    back = cv2.warpPerspective(seen, cv2.getPerspectiveTransform(tilted, flat),
                               (w, h), borderValue=(255, 255, 255))
    return cv2.GaussianBlur(back, (0, 0), blur) if blur else back


# --- the decoder is wired up correctly --------------------------------------

def test_a_clean_symbol_reads():
    reads = cr.decode(render())
    assert reads and reads[0].text.endswith(CODE)


def test_a_clean_symbol_costs_one_attempt():
    """The ladder must not turn every good read into four decoder calls."""
    _reads, how = cr.decode(render(), detail=True)
    assert how == "as taken"


def test_the_decoder_returns_the_ean13_spelling_of_a_upca():
    """The trap that would have failed every genuine part. A UPC-A is twelve
    digits and the label prints twelve, so twelve is what anyone types -- and
    the decoder hands back thirteen."""
    reads = cr.decode(render())
    assert reads[0].text == "0" + CODE
    assert CODE in reads[0].candidates()


# --- the conditions the pipeline actually produces --------------------------

@pytest.mark.parametrize("module_px,skew,blur", [
    (4, 12, 1.2), (4, 20, 1.6), (3, 12, 1.2), (3, 20, 1.6), (2, 10, 1.0),
])
def test_a_rectified_photograph_reads(module_px, skew, blur):
    """Every one of these failed on the decoder's defaults. The ladder is not
    belt-and-braces; it is the difference between reading and not."""
    reads = cr.decode(photographed(module_px, skew, blur))
    assert reads and reads[0].text.endswith(CODE)


def test_the_default_binarizer_really_does_fail_on_these():
    """So the ladder cannot be quietly deleted as redundant."""
    import zxingcpp

    hard = photographed(3, 20, 1.6)
    assert not zxingcpp.read_barcodes(hard), (
        "the default settings now read this; the ladder may be unnecessary")
    assert cr.decode(hard)


def test_a_marginal_read_says_which_attempt_saved_it():
    """Worth knowing before the print or the optics drift any further."""
    _reads, how = cr.decode(photographed(3, 20, 1.6), detail=True)
    assert how and how != "as taken"


def test_a_missing_quiet_zone_is_not_what_breaks_it():
    """Stated as a test because it is the intuitive explanation and it is
    wrong -- chasing it would mean redrawing regions that were already right."""
    assert cr.decode(render(quiet_modules=0))


def test_nothing_is_read_out_of_a_picture_with_no_code_in_it():
    """A ladder that tries harder must not eventually hallucinate."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    assert cr.decode(noise) == []
    assert cr.decode(np.full((200, 400, 3), 255, np.uint8)) == []


# --- pixels to verdict ------------------------------------------------------

def test_a_photographed_code_confirms_the_label_it_belongs_to():
    """The whole feature, end to end, with no stubs between the bars and the
    verdict -- including the twelve-against-thirteen digit trap."""
    class Spec:
        policy, pattern = "must_decode", f"^{CODE}$"
        region, role, symbology = [0.84, 0.24, 0.15, 0.20], "serial", "upca"

    patterns = {"PC680": [f"^{CODE}$"], "2220-9199": [r"^2220-9199"]}
    reads = cr.decode(photographed(3, 12, 1.2))
    verdict = cd.verdict("PC680", [Spec()], reads, patterns)
    assert verdict.state == cd.CONFIRMED
    assert cd.resolve("PC680", verdict) == "PC680"
    assert cd.plate_note(verdict, "PC680") == "code ok"


def test_a_code_from_a_label_nobody_enrolled_is_refused():
    """The reason this exists. Same pipeline, a code that matches no pattern."""
    class Spec:
        policy, pattern = "must_decode", f"^{CODE}$"
        region, role, symbology = [0.84, 0.24, 0.15, 0.20], "serial", "upca"

    patterns = {"PC680": [f"^{CODE}$"]}
    stranger = render(digits="012345678905")      # a valid UPC-A, not ours
    reads = cr.decode(stranger)
    assert reads, "the fixture must decode, or this proves nothing"
    verdict = cd.verdict("PC680", [Spec()], reads, patterns)
    assert verdict.state == cd.CONTRADICTED
    assert cd.resolve("PC680", verdict) == cd.UNKNOWN
