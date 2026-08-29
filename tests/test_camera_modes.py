"""Sizes asked of the camera, rather than remembered and typed.

A Basler has no list of resolutions to enumerate. Width and Height are AOI
controls with a range and a step, and any value on that grid is valid -- so
there is nothing to look up, and a value typed off the step is silently rounded
to one that is. Which is the failure this replaces: a rig running at a size
nobody chose, because 1918 looks a lot like the 1920 that was asked for.

And on a sensor whose AOI is being set, a smaller size is a SMALLER VIEW rather
than a scaled-down picture of the same scene. Choosing half the width is
choosing to see half the battery, which the list has to say out loud.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import camera_modes as cm

# A Basler ace at 5 MP, and a bigger one whose axes step differently.
ACE = ((16, 2592, 4), (16, 1944, 4))
BIG = ((64, 5472, 32), (64, 3648, 2))


def _on_grid(size, width, height) -> bool:
    (w_min, _wmax, w_inc), (h_min, _hmax, h_inc) = width, height
    return (size[0] - w_min) % w_inc == 0 and (size[1] - h_min) % h_inc == 0


# --- what gets offered -------------------------------------------------------

def test_the_full_sensor_is_the_first_thing_offered():
    assert cm.offered(*ACE)[0] == (2592, 1944)


def test_nothing_offered_can_be_silently_rounded():
    """The whole point of choosing from a list: what is chosen is what is
    applied. A size off the step comes back as a different size, and nothing
    says so."""
    for limits in (ACE, BIG):
        for size in cm.offered(*limits):
            assert _on_grid(size, *limits), f"{size} is off the camera's grid"


def test_every_offered_size_is_inside_the_range():
    for limits in (ACE, BIG):
        (w_min, w_max, _wi), (h_min, h_max, _hi) = limits
        for w, h in cm.offered(*limits):
            assert w_min <= w <= w_max and h_min <= h <= h_max


def test_snapping_goes_down_never_up():
    """A size the camera rounds UP is bigger than the one asked for, and on an
    AOI control that means a wider view than was chosen."""
    assert cm.snap(1000, 16, 2592, 4) == 1000
    assert cm.snap(1001, 16, 2592, 4) == 1000
    assert cm.snap(1003, 16, 2592, 4) == 1000
    assert cm.snap(999999, 16, 2592, 4) <= 2592
    assert cm.snap(0, 16, 2592, 4) == 16


def test_a_step_of_one_offers_the_arithmetic_unchanged():
    sizes = cm.offered((1, 1920, 1), (1, 1080, 1))
    assert sizes[0] == (1920, 1080)
    assert (960, 540) in sizes


def test_a_camera_that_answers_nothing_offers_nothing():
    assert cm.offered((0, 0, 0), (0, 0, 0)) == []
    assert cm.offered((16, 2592, 4), (0, 0, 0)) == []


def test_duplicates_are_not_listed_twice():
    """A tiny sensor with a coarse step can snap two fractions to one size."""
    sizes = cm.offered((16, 64, 32), (16, 64, 32))
    assert len(sizes) == len(set(sizes))


# --- what the rows say -------------------------------------------------------

def test_a_smaller_size_says_it_is_less_of_the_sensor():
    """Not a scaled-down picture of the same scene -- less scene. Somebody
    choosing 1296x972 off a 2592x1944 sensor is choosing to see a quarter of
    the battery, and should find that out here rather than on the belt."""
    assert "full sensor" in cm.describe((2592, 1944), (2592, 1944))
    assert "50% of the sensor width" in cm.describe((1296, 972), (2592, 1944))


def test_a_row_without_a_sensor_size_still_reads():
    assert cm.describe((640, 480), (0, 0)) == "640 x 480"


def test_the_limits_are_stated_for_a_size_that_is_not_listed():
    note = cm.limits_note(*ACE)
    assert "16-2592" in note and "16-1944" in note
    assert "steps of 4 x 4" in note


def test_a_camera_with_no_step_does_not_mention_one():
    assert "steps of" not in cm.limits_note((1, 1920, 1), (1, 1080, 1))


def test_nothing_is_claimed_about_a_camera_that_said_nothing():
    assert cm.limits_note((0, 0, 0), (0, 0, 0)) == ""


# --- asking the camera -------------------------------------------------------

def test_limits_are_read_off_the_camera_not_a_table():
    """By what the handle can do, not by which backend was requested: a Basler
    request that fell back to OpenCV must not then be asked for Basler nodes,
    and that fallback is not always announced."""
    import inspect

    from label_detections.core.camera import CameraSource

    source = inspect.getsource(CameraSource.frame_limits)
    assert "GetMax" in source and "GetMin" in source and "GetInc" in source
    assert "_is_basler" in source
    assert 'hasattr(self.cap, "Width")' in inspect.getsource(CameraSource._is_basler)


def test_a_camera_that_is_not_open_reports_no_limits():
    from label_detections.core.camera import CameraSource

    camera = CameraSource()
    assert camera.frame_limits() == {}
    assert not camera._is_basler()


def test_a_plain_opencv_capture_is_not_asked_for_basler_nodes():
    from label_detections.core.camera import CameraSource

    class NotABasler:
        """An OpenCV VideoCapture has no Width node."""

    camera = CameraSource()
    camera.cap = NotABasler()
    try:
        assert not camera._is_basler()
        assert camera.frame_limits() == {}
    finally:
        camera.cap = None
