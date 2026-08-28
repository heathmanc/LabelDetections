"""Turn one click into an oriented box.

Outlining a label is the whole cost of adding a new one. A hundred and fifty
images, four corner drags each, and the corners get sloppier as the operator
tires -- which matters more than it sounds, because read-regions are stored as
fractions of the box that was drawn, so a lazy corner moves every region on
every image after it.

A segmentation model prompted with a single point returns the label's actual
pixels. The tightest rotated rectangle around those pixels is the box, and it
is fitted to real edges rather than to where somebody's eye said the edge was.

This module is the half with no model in it: mask to quad, and the sanity
checks that decide whether the quad is worth handing back. Kept separate
because the geometry is where the mistakes are, and geometry can be tested
without a GPU, a checkpoint, or a picture of a battery.
"""
from __future__ import annotations

import numpy as np

# Canonical corner ordering. It lives in core.geometry because rectify_quad
# needs the same rule, and two copies of a convention is how they drift apart.
from .geometry import order_quad

__all__ = ["order_quad"]

# The frame is downscaled before the model sees it. A full frame is ~20 MP and
# the model's cost scales with pixels, while a label's outline does not need
# 20 MP to be found -- the corners come back within a pixel or two of where a
# person would put them, and the operator can still nudge them.
DEFAULT_ASSIST_PX = 1024

# Under this many pixels a "mask" is a speck: a click that landed on a screw
# head, a glare spot, a piece of dust. Fitting a rectangle to it produces a
# confident-looking box around nothing.
MIN_MASK_PX = 200

# A click that misses the label usually returns the *battery*, or the whole
# frame. Both come back as a huge plausible-looking rectangle, and a box around
# the entire battery is far worse than no box -- it looks right in the list and
# is wrong in the dataset.
MAX_FRAME_FRACTION = 0.55

# Labels are printed rectangles. A quad this far from any sensible aspect is
# a mask that leaked along an edge or a shadow.
MAX_ASPECT = 25.0


def assist_scale(width: int, height: int, max_px: int = DEFAULT_ASSIST_PX) -> float:
    """How much to shrink a frame before the model sees it.

    Never enlarges: upscaling a small frame costs time and invents no detail.
    """
    longest = max(int(width), int(height))
    if longest <= 0 or max_px <= 0 or longest <= max_px:
        return 1.0
    return float(max_px) / float(longest)


def quad_from_mask(mask) -> list[list[float]] | None:
    """The tightest rotated rectangle around a mask's largest blob.

    Largest blob rather than all of them: a segmentation of a label often comes
    back with a few stray pixels elsewhere in the frame, and a rectangle around
    the union of a label and a speck of glare is a rectangle around neither.
    """
    import cv2

    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.size == 0:
        return None
    binary = (array > 0.5).astype(np.uint8) if array.dtype != np.uint8 else (array > 0).astype(np.uint8)
    if int(binary.sum()) < MIN_MASK_PX:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_MASK_PX:
        return None

    points = cv2.boxPoints(cv2.minAreaRect(largest))
    return [[float(x), float(y)] for x, y in points]


def scale_quad(quad: list[list[float]], factor: float) -> list[list[float]]:
    """Map a quad back to full-resolution coordinates."""
    if not quad or factor == 0:
        return list(quad or [])
    return [[float(x) / factor, float(y) / factor] for x, y in quad]


def clamp_quad(quad: list[list[float]], width: int, height: int) -> list[list[float]]:
    """Keep every corner inside the frame.

    A mask that touches the edge gives a rectangle whose corner sits a pixel or
    two outside it. Stored that way it exports as an out-of-range coordinate.
    """
    if not quad:
        return []
    return [[max(0.0, min(float(width), float(x))),
             max(0.0, min(float(height), float(y)))] for x, y in quad]


def quad_area(quad: list[list[float]]) -> float:
    """Shoelace area, so a self-intersecting quad reads as small rather than big."""
    if len(quad) != 4:
        return 0.0
    total = 0.0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        total += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(total) / 2.0


def rejection(quad: list[list[float]], width: int, height: int) -> str:
    """Why this quad should not be offered, or "" when it should.

    Worth its own function and its own words: a click that misses the label by
    a few pixels does not fail, it succeeds at outlining the battery. Handing
    that back silently is the one outcome worse than handing back nothing --
    it looks outlined in the list, and it is wrong in the dataset.
    """
    if not quad or len(quad) != 4:
        return "Nothing was outlined there. Click on the label itself."

    frame_area = float(max(1, int(width) * int(height)))
    area = quad_area(quad)
    if area <= 0:
        return "Nothing was outlined there. Click on the label itself."
    if area / frame_area > MAX_FRAME_FRACTION:
        return ("That outlined most of the frame -- usually the battery rather "
                "than the label. Click nearer the middle of the label.")

    import math
    sides = [math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    long_side = max(sides)
    short_side = min(sides)
    if short_side <= 1.0:
        return "That outlined a sliver, not a label. Try clicking again."
    if long_side / short_side > MAX_ASPECT:
        return ("That outlined a long thin strip -- usually an edge or a "
                "shadow. Click nearer the middle of the label.")
    return ""


def outline_from_mask(mask, frame_w: int, frame_h: int,
                      factor: float = 1.0) -> tuple[list[list[float]], str]:
    """A mask at model scale to a full-resolution quad, or a reason it is not one.

    One function so the order can never drift: fit, then scale back, then clamp,
    then order, and only then judge it -- judging at model scale would compare
    a downscaled area against a full-resolution frame.
    """
    quad = quad_from_mask(mask)
    if quad is None:
        return [], "Nothing was outlined there. Click on the label itself."
    quad = clamp_quad(scale_quad(quad, factor), frame_w, frame_h)
    quad = order_quad(quad)
    why = rejection(quad, frame_w, frame_h)
    return ([], why) if why else (quad, "")
