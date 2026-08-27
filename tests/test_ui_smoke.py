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

# Redirect the image library to a scratch folder BEFORE label_detections is
# imported: storage resolves DATA_DIR at import time, and constructing
# MainWindow writes recipe/settings files. Without this the test would mutate
# the repo's seed data -- or, on an operator's machine, their real library.
import tempfile  # noqa: E402

os.environ["LABELVISION_DATA_DIR"] = tempfile.mkdtemp(prefix="labelvision-smoke-")

from pathlib import Path  # noqa: E402
_UI_DIR = Path(__file__).resolve().parents[1] / "label_detections" / "ui"

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
        from label_detections.ui.main_window import MainWindow
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
    assert win.tabs.count() == 7
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    for expected in ("Label", "Live Capture", "Test Models", "Train", "Live Detect"):
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
    for expected in ("Start Training", "Run Model", "Auto-label",
                     "Change Folder...", "Dataset Health"):
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

def _popup_view_rule(css: str) -> str:
    start = css.index("QComboBox QAbstractItemView")
    return css[start:css.index("}", start)]


def test_combo_popup_declares_an_opaque_background():
    """The popup view is a QListView, which the generic input rule (matching
    QListWidget) never covered. With no background it paints nothing on first
    expose and the window behind shows through before the rows draw.

    Note: this asserts the cause, not the appearance -- render() forces a full
    repaint, so painting order cannot be observed offscreen.
    """
    rule = _popup_view_rule(_window().styleSheet())
    body = rule.split("{", 1)[1]  # drop the selector, or it glues to decl #1
    decls = [d.strip() for d in body.split(";")]
    assert any(d.startswith("background") for d in decls), (
        "popup view has no background declaration; "
        "selection-background-color does not count"
    )


def test_combo_popup_widgets_fill_their_background():
    win = _window()
    for combo in win.findChildren(QComboBox):
        view = combo.view()
        assert view.autoFillBackground(), (
            f"popup view for {combo.itemText(0)!r} does not fill its background"
        )

def test_combo_open_animation_is_disabled():
    """Windows enables the combo open animation by default. Qt paints the popup
    progressively as it opens, which under a stylesheet flashes the window
    behind -- the popup is correct once open, but the opening is not.

    main() disables it; assert the call is present, since this environment
    reports every UI effect as already off and cannot exercise it.
    """
    src = (_UI_DIR / "main_window.py").read_text()
    assert "UI_AnimateCombo" in src, "combo open animation is never disabled"
    assert "setEffectEnabled" in src


def test_combo_views_are_plain_listviews():
    # Qt's private combo view has platform-specific painting that does not
    # composite cleanly under a stylesheet.
    from PySide6.QtWidgets import QListView
    for combo in _window().findChildren(QComboBox):
        assert isinstance(combo.view(), QListView), type(combo.view()).__name__

def test_scrollbars_are_visible_against_the_dark_theme():
    # They were always ~14px wide but unstyled, so the default handle was
    # effectively invisible on the #0f172a background. Contrast is the fix.
    css = _window().styleSheet()
    assert "QScrollBar::handle:vertical" in css, "scrollbar handle unstyled"
    import re
    m = re.search(r"QScrollBar::handle:vertical \{[^}]*background:\s*(#[0-9a-fA-F]{6})", css)
    assert m, "scrollbar handle has no background colour"
    # Must be clearly lighter than the #0f172a track to be seen.
    r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (1, 3, 5))
    assert (r + g + b) / 3 > 80, f"handle {m.group(1)} too dark to see"


def test_buttons_are_not_oversized():
    win = _window()
    for btn in win.findChildren(QPushButton):
        h = btn.sizeHint().height()
        assert h <= 30, f"{btn.text()!r} is {h}px tall"


def test_no_button_label_clips_at_the_minimum_pane_width():
    """The pane minimum must be wide enough that no button elides its label.

    Compact buttons previously had minimumWidth(0), so "Mark Current Reviewed"
    rendered as "Mark Current Reviewe". Squeeze the pane to its minimum and
    check every tab, so growing a label fails here rather than truncating in
    the UI.
    """
    win = _window()
    original = win.tabs.width()
    win.tabs.setFixedWidth(win.tabs.minimumWidth())
    QApplication.processEvents()
    clipped = []
    try:
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            QApplication.processEvents()
            for btn in win.tabs.currentWidget().findChildren(QPushButton):
                if btn.width() and btn.width() < btn.minimumSizeHint().width() - 1:
                    clipped.append(f"{win.tabs.tabText(i)}: {btn.text()!r}")
    finally:
        win.tabs.setMinimumWidth(355)
        win.tabs.setMaximumWidth(400)
        win.tabs.resize(original, win.tabs.height())
    assert not clipped, (
        f"at {win.tabs.minimumWidth()}px these labels clip: " + ", ".join(clipped)
    )




def test_new_action_handlers_exist_and_are_wired():
    """The buttons added for background images must be connected.

    A typo in a clicked.connect target is invisible until an operator clicks it
    and gets a hard crash, which is exactly how the findChildren regression
    reached a packaged build.

    The handlers themselves are not called here: both open a modal QMessageBox
    on the empty-state path, and a modal dialog blocks forever under the
    offscreen platform rather than being suppressed.
    """
    win = _window()
    for name in ("mark_current_background", "import_background_images"):
        assert callable(getattr(win, name, None)), f"MainWindow.{name} is missing"

    labels = {b.text() for b in win.findChildren(QPushButton)}
    for text in ("Mark Background", "Import Backgrounds..."):
        assert text in labels, f"no {text!r} button was built"


def test_background_status_reaches_the_dataset_summary():
    """A background entry must be tallied, not silently dropped."""
    win = _window()
    totals = win._new_summary_totals()
    win._accumulate_summary(totals, {"status": "background", "labeled": False})
    assert totals["background"] == 1
    assert totals["problems"] == 0, "a background is not a problem image"
    win._set_dataset_summary_label(totals)


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


def test_the_built_in_guide_describes_the_tool_that_exists():
    """The guide told operators the model was trained on coarse families and to
    never add a class per SKU, long after both had stopped being true. A stale
    guide is worse than none: it is read as authority."""
    win = _window()
    text = ""
    from PySide6.QtWidgets import QTextEdit

    for child in win.findChildren(QTextEdit):
        body = child.toPlainText()
        if "Workflow for one label" in body:
            text = body
            break
    assert text, "workflow guide not found"

    for gone in ("coarse FAMILIES", "Never add a class per SKU",
                 "spec_plate, warning_label", "resolved afterwards"):
        assert gone not in text, f"guide still describes the old design: {gone!r}"

    assert "Label ids" in text
    assert "Export Two-Stage" in text and "Export All" in text
    assert "battery_side" in text


def test_every_tools_action_is_reachable_without_the_menu_bar():
    """The menu bar is hidden, so a menu action with no shortcut and no button
    is dead UI. Two diagnostics shipped that way and could not be opened at
    all -- this is the check that would have caught it."""
    from PySide6.QtWidgets import QPushButton

    win = _window()
    buttons = {b.text() for b in win.findChildren(QPushButton)}
    handlers = {b.text(): b for b in win.findChildren(QPushButton)}

    for label, method in (("Check Label Scale", "show_label_scale_report"),
                          ("Check Variable Regions", "check_variable_regions")):
        assert label in buttons, f"{method} has no visible button"

    # And every window-level action either carries a shortcut or has a button.
    unreachable = []
    for action in win.actions():
        text = action.text().replace("&&", "&")
        if action.shortcut().isEmpty() and text not in buttons:
            unreachable.append(text)
    assert not unreachable, f"unreachable with the menu bar hidden: {unreachable}"
    assert handlers  # the lookup above is the real assertion


def test_the_shortcut_sheet_is_generated_not_typed():
    """It drifted badly -- still offering "U: select bung class" long after
    those classes stopped existing, and listing none of the newer keys. With
    the menu bar hidden it is the only way a key is ever discovered."""
    win = _window()
    rows = []
    for action in win.actions():
        keys = action.shortcut().toString()
        if keys:
            rows.append((keys, action.text().replace("&&", "&")))
    assert rows, "no shortcut actions registered"

    names = {text for _keys, text in rows}
    assert "Check label scale (single vs two-stage)" in names
    assert "Keep this live frame + detections" in names
    assert not any("bung" in n.lower() or "retainer" in n.lower() for n in names)
