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


# --- exposure ----------------------------------------------------------------
#
# Exposure has no grid, so unlike size there is nothing to enumerate and no
# rounding to defend against. What there is instead is one number worth having:
# the value auto exposure settles on for the light actually in the room. Read
# it, take it, freeze it -- because auto exposure left running on a line is a
# variation nobody asked for, and stage 2 would be learning a lighting
# difference that says nothing about which label it is.

BASLER_EXPOSURE = (20.0, 10_000_000.0)      # (min, max) us, a Basler ace


def _labels(rows):
    return [text for text, _us in rows]


def test_the_reading_is_what_the_picker_is_built_around():
    """Not round numbers. 8340 us is not a nice value, it is the right one --
    and only the camera, pointed at this bench under these lights, knows it."""
    rows = cm.exposure_choices(*BASLER_EXPOSURE, 8340)
    assert 8340 in [us for _t, us in rows]
    assert any("what auto settled on" in t for t in _labels(rows))


def test_the_anchor_says_where_the_number_came_from():
    """A reading taken off the light is evidence; a number somebody typed
    earlier is not, and the row should not claim to be the first when it is
    the second."""
    auto = _labels(cm.exposure_choices(*BASLER_EXPOSURE, 8340, auto=True))
    manual = _labels(cm.exposure_choices(*BASLER_EXPOSURE, 8340, auto=False))
    assert any("auto settled on" in t for t in auto)
    assert not any("auto settled on" in t for t in manual)
    assert any("set to now" in t for t in manual)


def test_the_rows_either_side_bracket_the_reading():
    """For finding the edge of usable: wide enough to see a difference,
    narrow enough that every step is still a picture."""
    values = [us for _t, us in cm.exposure_choices(*BASLER_EXPOSURE, 8340)]
    assert min(values) < 8340 < max(values)
    assert sum(1 for v in values if v < 8340) >= 2
    assert sum(1 for v in values if v > 8340) >= 2


def test_rows_run_fastest_first():
    values = [us for _t, us in cm.exposure_choices(*BASLER_EXPOSURE, 8340)]
    assert values == sorted(values)


def test_nothing_offered_is_outside_what_the_camera_takes():
    low, high = BASLER_EXPOSURE
    for current in (0, 20, 5_000, 9_000_000, 99_000_000):
        for _t, us in cm.exposure_choices(low, high, current):
            assert low <= us <= high


def test_a_step_that_runs_past_the_end_does_not_steal_the_end_row():
    """2x a long exposure clamps onto the maximum, and labelling the camera's
    ceiling "2x that" loses the only row that says where the ceiling is."""
    rows = _labels(cm.exposure_choices(20, 10_000, 9_000))
    assert any("longest the camera takes" in t for t in rows)
    assert any("what auto settled on" in t for t in rows)


def test_the_reading_keeps_its_name_even_sitting_on_the_floor():
    """It is both the shortest the camera takes and the value auto chose, and
    of those two the second is the one worth saying."""
    rows = _labels(cm.exposure_choices(20, 10_000, 20))
    assert any("what auto settled on" in t for t in rows)


def test_the_ends_are_offered_even_with_no_reading_to_anchor_to():
    """A camera that answers the range but not the current value still tells
    somebody what a typed number will be clamped to."""
    rows = _labels(cm.exposure_choices(20, 10_000, 0))
    assert any("shortest" in t for t in rows) and any("longest" in t for t in rows)


def test_a_camera_that_answers_nothing_offers_nothing():
    assert cm.exposure_choices(0, 0, 0) == []


# --- what a long exposure costs ----------------------------------------------

def test_an_exposure_caps_the_frame_rate():
    """The sensor cannot start the next exposure until this one ends, so a
    50 ms exposure runs at 20/s however the rig is configured -- and asking
    for 30 then quietly gets 20 with nothing saying why."""
    assert cm.fps_ceiling(50_000) == 20.0
    assert cm.fps_ceiling(0) == 0.0


def test_the_cap_is_said_only_when_it_is_the_thing_that_binds():
    """Above a couple of hundred a second the exposure is not what limits the
    rate -- the interface, the model or the belt is -- so saying it would be
    noise on every row."""
    assert cm.rate_note(16_680) == ", max 60/s"
    assert cm.rate_note(100) == ""
    assert cm.rate_note(0) == ""


def test_a_very_long_exposure_does_not_report_as_a_fault():
    """A ten second exposure caps the camera at a tenth of a frame a second.
    Printing "max 0/s" reads as broken instead of as the correct consequence
    of asking for a ten second exposure."""
    assert cm.rate_note(10_000_000) == ", max 0.1/s"


def test_the_note_reports_the_range_and_where_auto_landed():
    note = cm.exposure_note({"min": 20, "max": 10_000_000,
                             "current": 8340, "auto": True})
    assert "20 to 10,000,000 us" in note
    assert "auto is on" in note and "8,340 us" in note


def test_the_note_does_not_call_a_frozen_value_an_auto_one():
    note = cm.exposure_note({"min": 20, "max": 10_000_000,
                             "current": 8340, "auto": False})
    assert "auto is on" not in note and "currently 8,340 us" in note


def test_an_exposure_that_undercuts_the_requested_rate_says_so():
    """The failure this catches: a rig set to 30 fps, an exposure that allows
    17, and a preview that just feels sluggish."""
    note = cm.exposure_note({"min": 20, "max": 10_000_000,
                             "current": 60_000, "auto": False}, wanted_fps=30)
    assert "17/s" in note and "30/s asked for" in note


def test_nothing_is_claimed_about_a_camera_that_said_nothing():
    assert cm.exposure_note({}) == ""
    assert cm.exposure_note({"min": 0, "max": 0, "current": 0}) == ""


# --- asking the camera -------------------------------------------------------

def test_exposure_limits_are_read_off_the_camera_not_a_table():
    """Two node names because older GigE models call it ExposureTimeAbs, and a
    rig with one of those would otherwise get an empty picker and no reason."""
    import inspect

    from label_detections.core.camera import CameraSource

    source = inspect.getsource(CameraSource.exposure_limits)
    assert "GetMax" in source and "GetMin" in source
    assert "ExposureAuto" in source
    assert "_is_basler" in source
    assert CameraSource.EXPOSURE_NODES == ("ExposureTime", "ExposureTimeAbs")


def test_a_camera_that_is_not_open_reports_no_exposure_limits():
    from label_detections.core.camera import CameraSource

    assert CameraSource().exposure_limits() == {}


def test_a_plain_opencv_capture_is_not_asked_for_exposure_nodes():
    from label_detections.core.camera import CameraSource

    class NotABasler:
        pass

    camera = CameraSource()
    camera.cap = NotABasler()
    try:
        assert camera.exposure_limits() == {}
    finally:
        camera.cap = None


# --- the picker in the window ------------------------------------------------

import pytest  # noqa: E402

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QLineEdit)

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


class _FakeBasler:
    """A camera that answers, without one being plugged in."""

    def __init__(self, limits):
        self._limits = limits

    def exposure_limits(self):
        return self._limits

    def is_open(self):
        return True


def _picker(limits, *, typed="0", auto=True):
    """Run a Detect press against a stubbed camera. Returns the widgets."""
    win = _window()
    combo, edit, check = QComboBox(), QLineEdit(typed), QCheckBox()
    check.setChecked(auto)
    saved = win.camera
    win.camera = _FakeBasler(limits)
    try:
        win._fill_camera_exposure(combo, edit, check)
    finally:
        win.camera = saved
    return win, combo, edit, check


def test_detect_fills_the_picker_from_the_camera():
    _w, combo, _e, _c = _picker({"min": 20, "max": 10_000_000,
                                 "current": 8340, "auto": True})
    assert combo.isEnabled()
    assert 8340 in [combo.itemData(i) for i in range(combo.count())]


def test_detect_lands_on_the_reading_rather_than_the_shortest_row():
    """It is the row somebody pressed Detect to see; making them hunt for it
    in a list sorted by microseconds defeats the point."""
    _w, combo, _e, _c = _picker({"min": 20, "max": 10_000_000,
                                 "current": 8340, "auto": True})
    assert combo.currentData() == 8340


def test_a_value_already_typed_is_the_one_selected():
    _w, combo, _e, _c = _picker({"min": 20, "max": 10_000_000,
                                 "current": 8340, "auto": True}, typed="16680")
    assert combo.currentData() == 16680


def test_choosing_an_exposure_freezes_it():
    """Not a side effect. An exposure chosen while auto is still running is a
    number the camera overwrites on the next frame -- and auto exposure left
    running is exactly the nuisance variation stage 2 must not be taught."""
    win, combo, edit, check = _picker({"min": 20, "max": 10_000_000,
                                       "current": 8340, "auto": True})
    index = combo.findData(16680)
    assert index >= 0
    combo.setCurrentIndex(index)
    win._apply_camera_exposure(combo, edit, check)
    assert edit.text() == "16680"
    assert not check.isChecked(), "auto exposure was left running"


def test_the_placeholder_row_does_not_overwrite_anything():
    win = _window()
    combo, edit, check = QComboBox(), QLineEdit("1234"), QCheckBox()
    check.setChecked(True)
    combo.addItem("Detect to fill this", None)
    win._apply_camera_exposure(combo, edit, check)
    assert edit.text() == "1234" and check.isChecked()


def test_a_camera_that_cannot_answer_says_so_instead_of_offering_nothing(monkeypatch):
    """An empty picker with no reason reads as a broken button."""
    from label_detections.ui import main_window as mw

    shown = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda parent, title, text, *a, **k: shown.update(
                            title=title, text=text))
    _w, combo, _e, _c = _picker({})
    assert not combo.isEnabled()
    assert "Basler" in shown.get("text", "")


def test_a_camera_that_raises_is_not_a_crash(monkeypatch):
    from label_detections.ui import main_window as mw

    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: None)
    win = _window()
    combo, edit, check = QComboBox(), QLineEdit("0"), QCheckBox()

    class Broken:
        def exposure_limits(self):
            raise RuntimeError("pylon fell over")

    saved, win.camera = win.camera, Broken()
    try:
        win._fill_camera_exposure(combo, edit, check)     # must not raise
    finally:
        win.camera = saved
    assert not combo.isEnabled()


def test_the_settings_dialog_offers_the_picker():
    """Both halves: the button that asks, and the list that answers."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow.open_camera_settings_dialog)
    assert "_fill_camera_exposure" in source
    assert "_apply_camera_exposure" in source
    assert "exposure_combo" in source
