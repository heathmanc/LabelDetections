"""The generic wizard renderer: table rows, and how a region is presented."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-flow-"))

import pytest

try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6 not available")


def _table(rows=()):
    from label_detections.core.wizard import Question
    from label_detections.ui.flow_dialog import TableEditor

    QApplication.instance() or QApplication([])
    question = Question("codes", "Codes", "table", columns=[
        Question("role", "Role", "choice", choices=["serial", "lot"]),
        Question("region", "Region on the label", "region"),
        Question("policy", "Policy", "choice", choices=["must_decode", "ignore"]),
    ])
    editor = TableEditor(question, {})
    editor.setValue(list(rows))
    return editor


ROWS = [
    {"role": "serial", "region": [0.1, 0.2, 0.3, 0.4], "policy": "must_decode"},
    {"role": "lot", "region": [0.5, 0.6, 0.2, 0.2], "policy": "ignore"},
]


def test_each_row_carries_its_own_remove_button():
    """Every cell holds an editor, so the table's own selection is always -1.

    A single shared Remove button read currentRow() and was therefore a
    permanent no-op -- it never removed anything, on any row, ever.
    """
    editor = _table(ROWS)
    assert editor.table.rowCount() == 2
    assert editor.table.currentRow() == -1          # the reason the old one failed
    editor._remove_row_of(editor.table.cellWidget(0, 0))
    assert editor.table.rowCount() == 1
    assert editor.value()[0]["role"] == "lot"


def test_removing_the_second_row_leaves_the_first():
    editor = _table(ROWS)
    editor._remove_row_of(editor.table.cellWidget(1, 0))
    assert [r["role"] for r in editor.value()] == ["serial"]


def test_remove_buttons_stay_bound_to_their_own_row():
    """Indexes shift as rows above go; a stale index deletes the wrong row."""
    editor = _table(ROWS + [{"role": "serial", "region": [], "policy": "ignore"}])
    last = editor.table.cellWidget(2, 0)
    editor._remove_row_of(editor.table.cellWidget(0, 0))
    editor._remove_row_of(last)
    assert [r["role"] for r in editor.value()] == ["lot"]


def test_adding_a_row_starts_it_blank():
    editor = _table()
    editor.add_row()
    row = editor.value()[0]
    assert row["region"] == []
    assert row["role"] == "serial"          # first choice, not empty


def test_rows_round_trip_without_losing_the_drawn_geometry():
    editor = _table(ROWS)
    assert editor.value()[0]["region"] == [0.1, 0.2, 0.3, 0.4]


def test_a_region_column_is_read_only_because_regions_are_drawn():
    editor = _table(ROWS)
    field = editor.table.cellWidget(0, 2)   # column 0 is remove, 1 is role
    assert field._w.isReadOnly()
    assert field._w.text() == "0.100, 0.200, 0.300, 0.400"


def test_an_undrawn_region_reads_as_not_drawn():
    editor = _table([{"role": "serial", "region": [], "policy": "ignore"}])
    field = editor.table.cellWidget(0, 2)
    assert field._w.text() == ""
    assert field._w.placeholderText() == "not drawn"
    assert editor.value()[0]["region"] == []


def test_headers_are_short_enough_to_read_with_the_full_text_in_a_tooltip():
    editor = _table(ROWS)
    assert editor.table.horizontalHeaderItem(2).text() == "Region"
    assert "artwork" in editor.table.horizontalHeaderItem(2).toolTip().lower() \
        or "drawing" in editor.table.horizontalHeaderItem(2).toolTip().lower() \
        or editor.table.horizontalHeaderItem(2).toolTip() == "Region on the label"


def test_row_numbers_are_shown_so_a_row_2_error_can_be_found():
    """isVisible() is False until the widget is shown; isHidden() is the flag."""
    editor = _table(ROWS)
    assert editor.table.verticalHeader().isHidden() is False
