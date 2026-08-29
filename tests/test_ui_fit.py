"""Nothing in the left pane is clipped, elided, or off the edge.

The pane has a maximum width, so anything wider than it is not merely ugly --
it is unreachable without a horizontal scrollbar nobody expects to need in a
settings panel. And a button whose label does not fit does not wrap or shrink;
Qt draws "Mark Current Reviewe" and leaves it at that.

Both failures are invisible in a screenshot of a wide window and obvious on the
machine actually running the line, which is the wrong way round. This measures
them at the width the pane is actually pinned to.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-fit-"))

import pytest

try:
    from PySide6.QtWidgets import (QApplication, QCheckBox, QPushButton,
                                   QScrollArea)
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6 not available")

_win = None


def _window():
    """One window, shown and laid out, at the size the rig runs at."""
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
        _win.resize(1280, 720)
        _win.show()
        QApplication.instance().processEvents()
    return _win


def _laid_out(index: int):
    """Switch to a tab and let Qt lay it out. An inactive tab has no geometry."""
    win = _window()
    win.tabs.setCurrentIndex(index)
    app = QApplication.instance()
    app.processEvents()
    app.processEvents()
    return win.tabs.widget(index)


def _labelled(page):
    for kind in (QPushButton, QCheckBox):
        for widget in page.findChildren(kind):
            if widget.text() and widget.isVisible():
                yield widget


def _needs(widget) -> int:
    """Pixels the widget's own text wants, including its decoration."""
    # A checkbox spends room on its indicator before any text is drawn.
    padding = 30 if isinstance(widget, QCheckBox) else 24
    return widget.fontMetrics().horizontalAdvance(widget.text()) + padding


def test_no_control_in_the_left_pane_has_its_label_cut_off():
    """Qt elides rather than wrapping or shrinking, so a label one word too
    long is silently truncated -- "Capture Reference" rendered as "Capture
    Referenc". Every one of these was found this way rather than by looking."""
    win = _window()
    clipped = []
    for index in range(win.tabs.count()):
        page = _laid_out(index)
        for widget in _labelled(page):
            if widget.width() and _needs(widget) > widget.width() + 1:
                clipped.append(
                    f"{win.tabs.tabText(index)}: {widget.text()!r} "
                    f"wants {_needs(widget)}px, has {widget.width()}px")
    assert not clipped, "clipped in the left pane:\n  " + "\n  ".join(clipped)


def test_every_tab_fits_the_pane_it_lives_in():
    """The pane is capped, so a tab whose content cannot be squeezed below that
    cap scrolls sideways -- a settings panel with a horizontal scrollbar, which
    reads as broken rather than as scrollable.

    One long checkbox in the narrow half of a form layout is all it takes: it
    put Test Models at 397 against a 355 pane."""
    win = _window()
    cap = win.tabs.width()
    assert cap > 0
    too_wide = []
    for index in range(win.tabs.count()):
        page = _laid_out(index)
        areas = page.findChildren(QScrollArea)
        inner = areas[0].widget() if areas else page
        need = inner.minimumSizeHint().width()
        if need > cap:
            too_wide.append(f"{win.tabs.tabText(index)}: needs {need}px of {cap}px")
    assert not too_wide, "wider than the pane:\n  " + "\n  ".join(too_wide)


def test_the_annotation_pane_can_move_between_images_without_saving():
    """Save approves. An image you open, look at, and decide needs nothing had
    no button that would just move on -- only Save, Save + Next, or the image
    list."""
    win = _window()
    buttons = {b.text() for b in win.findChildren(QPushButton)}
    assert "Next ›" in buttons and "‹ Prev" in buttons
    assert "Save" in buttons and "Save + Next" in buttons


def test_deleting_a_sidecar_is_not_a_button_on_the_annotating_pane():
    """It is a recovery action, not part of labelling an image, and it sat one
    slip away from the buttons used a hundred and fifty times a day."""
    win = _window()
    buttons = {b.text() for b in win.findChildren(QPushButton)}
    assert "Delete Saved JSON" not in buttons


def test_a_training_run_needs_four_fields_and_the_rest_is_folded_away():
    """Both models and both datasets, up front. Everything else has a working
    default and was on screen at the same time as them."""
    from PySide6.QtWidgets import QGroupBox

    win = _window()
    page = _laid_out([win.tabs.tabText(i)
                      for i in range(win.tabs.count())].index("Train"))
    for widget in (win.train_model_edit, win.train_data_edit,
                   win.cls_model_edit, win.cls_data_edit):
        assert widget.isVisible(), "a model or dataset field is folded away"

    closed = [b for b in page.findChildren(QGroupBox)
              if b.isCheckable() and not b.isChecked()]
    assert len(closed) >= 2, "the advanced settings are not folded away"
    # And the knobs inside them are genuinely off screen, not merely greyed.
    assert not win.train_epochs_spin.isVisible()
    assert not win.cls_erasing_spin.isVisible()
