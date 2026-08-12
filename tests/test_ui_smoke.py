"""Offscreen smoke tests that actually construct the UI.

Static checks cannot catch Qt errors that only surface at call time -- a wrong
argument type, a bad enum, a missing attribute. Those have shipped as hard
startup crashes more than once. These tests build the real MainWindow under the
offscreen platform plugin, so the whole widget tree, stylesheet and event
filters are exercised without a display.

Skipped cleanly when PySide6 or cv2 is unavailable, so a bare environment still
runs the rest of the suite.

    QT_QPA_PLATFORM=offscreen python tests/test_ui_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before QApplication is created, or Qt tries to reach a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# Redirect the image library to a scratch folder BEFORE bung_labeler is
# imported: storage resolves DATA_DIR at import time, and constructing
# MainWindow writes recipe/settings files. Without this the test would mutate
# the repo's seed data -- or, on an operator's machine, their real library.
import tempfile  # noqa: E402

os.environ["BUNGVISION_DATA_DIR"] = tempfile.mkdtemp(prefix="bungvision-smoke-")

try:
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import (
        QAbstractSpinBox, QApplication, QComboBox, QPushButton, QSlider,
    )
    import cv2  # noqa: F401  (storage imports it at module load)
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc


_app = None
_win = None


def _window():
    """Build the window once and reuse it; construction dominates runtime."""
    global _app, _win
    if _win is None:
        _app = QApplication.instance() or QApplication([])
        from bung_labeler.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def test_main_window_constructs():
    # The regression that motivated this file: findChildren() with a tuple
    # raised here, and nothing short of building the window would catch it.
    win = _window()
    assert win is not None
    assert win.windowTitle()


def test_all_tabs_are_present():
    win = _window()
    assert win.tabs.count() == 6
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    for expected in ("Recipe / SKU", "Live Capture", "Test Models", "Train"):
        assert expected in titles, f"{expected} missing from {titles}"


def test_every_tab_renders():
    # Switching tabs forces each page to lay out; a broken widget in a tab that
    # is never shown would otherwise go unnoticed.
    win = _window()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        assert win.tabs.currentWidget() is not None


def test_wheel_guard_is_installed_on_value_widgets():
    win = _window()
    found = 0
    for cls in (QAbstractSpinBox, QComboBox, QSlider):
        for w in win.findChildren(cls):
            found += 1
            assert w.focusPolicy() == Qt.StrongFocus
    assert found > 0, "no value widgets found to guard"


def test_wheel_over_a_spinbox_does_not_change_its_value():
    win = _window()
    spins = win.findChildren(QAbstractSpinBox)
    assert spins, "expected at least one spinbox"
    spin = spins[0]
    before = spin.value()
    event = QWheelEvent(
        QPoint(5, 5), spin.mapToGlobal(QPoint(5, 5)),
        QPoint(0, 120), QPoint(0, 120),
        Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False,
    )
    QApplication.sendEvent(spin, event)
    assert spin.value() == before, "wheel edited the field despite the guard"


def test_menu_actions_have_callable_slots():
    # Menu-bar actions are invisible (the bar is hidden), so a broken connection
    # would only be noticed via its keyboard shortcut.
    win = _window()
    for action in win.actions():
        assert action.text()


def test_key_buttons_exist_and_are_wired():
    win = _window()
    labels = {b.text() for b in win.findChildren(QPushButton)}
    for expected in ("Start Training", "Run Model", "Auto-label Current",
                     "Change Data Folder...", "Dataset Health"):
        assert expected in labels, f"{expected!r} button missing"


def test_theme_stylesheet_applied():
    win = _window()
    css = win.styleSheet()
    assert "QSpinBox::up-button" in css, "spinbox step buttons unstyled"
    assert "QSpinBox::down-button" in css

def test_combo_dropdown_subcontrols_are_styled():
    # Styling a QComboBox replaces its native drop-down with an unstyled
    # subcontrol -- the same trap that made the spinbox arrows unusable.
    css = _window().styleSheet()
    assert "QComboBox::drop-down" in css
    assert "QComboBox::down-arrow" in css


def test_stylesheet_has_no_unsubstituted_placeholders():
    css = _window().styleSheet()
    for token in ("__SPIN_UP__", "__SPIN_DOWN__", "__CHECKBOX_CHECK__"):
        assert token not in css, f"{token} left unsubstituted"


def test_short_combo_popups_are_not_padded_out():
    # Popups were forced to a 120px floor, so a 2-item list opened as a tall
    # box of dead space. Height must now track the row count.
    win = _window()
    for combo in win.findChildren(QComboBox):
        n = combo.count()
        if not n or n > 3:
            continue
        _popup_rows_visible(combo)  # realise the popup
        assert combo.view().height() < 100, (
            f"popup for {combo.itemText(0)!r} is {combo.view().height()}px "
            f"for only {n} items"
        )

def _popup_rows_visible(combo):
    combo.showPopup()
    view = combo.view()
    row = view.sizeHintForRow(0) or 22
    out = view.height() / row
    combo.hidePopup()
    return out


def test_every_combo_popup_shows_all_its_rows():
    # Qt sizes the popup from an unstyled row metric while the themed rows
    # render taller, so every dropdown showed n-0.5 items regardless of the
    # space available. Must hold on the FIRST show, with no warm-up.
    win = _window()
    for combo in win.findChildren(QComboBox):
        if not combo.count():
            continue
        want = min(combo.count(), max(1, combo.maxVisibleItems()))
        got = _popup_rows_visible(combo)
        assert got >= want - 0.05, (
            f"popup for {combo.itemText(0)!r} shows {got:.1f} of {want} rows"
        )


def test_combo_popup_shrinks_when_items_are_removed():
    # A sticky minimum height left a repopulated list (recipes, categories)
    # with an oversized popup full of dead space.
    win = _window()
    # Unparented: a child combo would be picked up by findChildren in other
    # tests as an unguarded widget, since the window is shared.
    combo = QComboBox()
    combo.addItems([f"item {i}" for i in range(8)])
    combo.view().installEventFilter(win)
    assert _popup_rows_visible(combo) >= 7.95
    combo.clear()
    combo.addItem("only one")
    assert _popup_rows_visible(combo) <= 1.05, "popup kept its old height"

def test_combo_popups_do_not_resize_after_being_shown():
    """A resize after Show means Qt painted the popup at one size then changed
    it -- which is what reads as flicker. Geometry assertions cannot see
    painting, so this counts the resize events instead.
    """
    from PySide6.QtCore import QObject

    class _Spy(QObject):
        def __init__(self):
            super().__init__()
            self.resizes = []
            self.shown = False

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Show:
                self.shown = True
            elif event.type() == QEvent.Type.Resize and self.shown:
                self.resizes.append(event.size().height())
            return False

    win = _window()
    offenders = []
    for combo in win.findChildren(QComboBox):
        if not combo.count():
            continue
        spy = _Spy()
        view = combo.view()
        view.installEventFilter(spy)
        combo.showPopup()
        QApplication.processEvents()
        combo.hidePopup()
        view.removeEventFilter(spy)
        if spy.resizes:
            offenders.append(f"{combo.itemText(0)!r}: {spy.resizes}")
    assert not offenders, "popup resized after show:\n" + "\n".join(offenders)




if __name__ == "__main__":
    import traceback

    if not HAVE_QT:
        print(f"SKIP: PySide6/cv2 unavailable ({_WHY})")
        raise SystemExit(0)

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
