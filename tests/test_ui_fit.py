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
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                                   QPushButton, QScrollArea)
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


# What each kind of control spends on decoration before any text is drawn.
_PADDING = {QCheckBox: 30, QComboBox: 40, QPushButton: 24}


def _shown_text(widget) -> str:
    return (widget.currentText() if isinstance(widget, QComboBox)
            else widget.text())


def _labelled(root):
    for kind in _PADDING:
        for widget in root.findChildren(kind):
            if _shown_text(widget) and widget.isVisible():
                yield widget


def _needs(widget) -> int:
    """Pixels the widget's own text wants, including its decoration."""
    padding = next(p for k, p in _PADDING.items() if isinstance(widget, k))
    return widget.fontMetrics().horizontalAdvance(_shown_text(widget)) + padding


def _clipped(root) -> list[str]:
    return [f"{_shown_text(w)!r} wants {_needs(w)}px, has {w.width()}px"
            for w in _labelled(root)
            if w.width() and _needs(w) > w.width() + 1]


def test_no_control_has_its_label_cut_off():
    """Qt elides rather than wrapping or shrinking, so a label one word too
    long is silently truncated -- "Capture Reference" rendered as "Capture
    Referenc", and a combo showing the middle of a sentence because the model
    name and its note would not both fit. Every one of these was found by
    measuring rather than by looking."""
    win = _window()
    clipped = []
    for index in range(win.tabs.count()):
        page = _laid_out(index)
        clipped += [f"{win.tabs.tabText(index)}: {x}" for x in _clipped(page)]
    # The annotating pane too: it is on screen whichever tab is open, so a
    # clipped button there is clipped all day.
    clipped += [f"right pane: {x}" for x in _clipped(win)
                if not any(x in c for c in clipped)]
    assert not clipped, "clipped:\n  " + "\n  ".join(clipped)


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


def test_the_tab_bar_shows_every_tab_at_once():
    """A tab bar that does not fit grows arrows and elides -- "Contrast" drawn
    as "trast" between two scroll buttons, which is how a tab gets lost.

    Seven tabs never fit at any label length. Contrast folded into Capture,
    where those sliders belong: they adjust the camera preview."""
    win = _window()
    bar = win.tabs.tabBar()
    assert bar.sizeHint().width() <= win.tabs.width(), (
        f"the tab bar wants {bar.sizeHint().width()}px of "
        f"{win.tabs.width()}px -- it will scroll")
    assert bar.count() <= 6, "another tab means another thing that does not fit"


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


def test_the_label_library_is_listed_once():
    """Detector Classes listed the same label ids with the same export-ready
    counts as the Label tab, minus the target and the reference warning -- and
    under a two-stage export it named classes the detector does not have, since
    every label box exports as the generic class there.

    What it uniquely carried was the per-image breakdown, which moved beside
    the count it complements."""
    from PySide6.QtWidgets import QGroupBox

    win = _window()
    titles = {b.title() for b in win.findChildren(QGroupBox)}
    assert "Detector Classes" not in titles
    assert not hasattr(win, "class_list_widget")
    assert win.class_counts_label.isVisible()
