"""Generic Qt renderer for a ``core.wizard.Flow``.

The dialog knows about question *kinds*, never about labels or recipes. Both
wizards are the same widget pointed at different data, so a new question is a
one-line change in ``core/label_wizard.py`` or ``core/recipe_wizard.py`` and
nothing here has to move.

A plain ``QDialog`` with a stacked widget rather than ``QWizard``: pages appear
and disappear as answers change (a label with no barcode never sees a
symbology field), and re-deriving the visible page list on every step is far
simpler than fighting ``QWizard``'s static page ids.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..core.wizard import Flow, Page, Question

# Injected into answers so a question can offer the library as its choices
# without the framework importing the library.
LIBRARY_IDS_KEY = "__library_ids"


def _short_header(question: Question) -> str:
    """A column heading that fits. The full wording lives in the tooltip."""
    label = str(question.label)
    trimmed = label.split(" (")[0]
    for long, short in (
        ("Region on the label", "Region"),
        ("Printed width", "Width mm"),
        ("X-dimension / cell", "Module mm"),
        ("Quiet zone", "Quiet mm"),
        ("Content pattern", "Pattern"),
        ("Print-grade it", "Grade"),
        ("Max age", "Max age"),
        ("Field name", "Field"),
    ):
        if trimmed.startswith(long):
            return short
    return trimmed


# --- field editors ---------------------------------------------------------

class FieldEditor(QWidget):
    """Wraps one question's widget behind a uniform value/setValue pair."""

    def __init__(self, question: Question, answers: dict[str, Any], parent=None,
                 owner=None):
        super().__init__(parent)
        self.question = question
        self._answers = answers
        # The "regions" button is the one question that edits several answers at
        # once -- a drawn barcode fills in the codes table, the text table and
        # the anchor together. It needs the dialog, not just its own value.
        self._owner = owner
        self._build()

    def _build(self) -> None:
        q = self.question
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        kind = q.kind

        if kind in ("text", "count"):
            self._w = QLineEdit()
            if q.placeholder:
                self._w.setPlaceholderText(q.placeholder)
            layout.addWidget(self._w)
        elif kind == "textarea":
            self._w = QPlainTextEdit()
            self._w.setMaximumHeight(70)
            layout.addWidget(self._w)
        elif kind == "int":
            self._w = QSpinBox()
            self._w.setRange(0, 100000)
            layout.addWidget(self._w)
        elif kind == "float":
            self._w = QDoubleSpinBox()
            self._w.setDecimals(3)
            self._w.setRange(0.0, 100000.0)
            layout.addWidget(self._w)
        elif kind == "bool":
            self._w = QCheckBox()
            layout.addWidget(self._w)
            layout.addStretch(1)
        elif kind in ("choice", "label_picker"):
            self._w = QComboBox()
            self._w.addItems([str(c) for c in self._choices()])
            layout.addWidget(self._w)
        elif kind == "multichoice":
            self._w = QListWidget()
            self._w.setSelectionMode(QAbstractItemView.NoSelection)
            self._w.setMaximumHeight(90)
            for choice in self._choices():
                item = QListWidgetItem(str(choice))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self._w.addItem(item)
            layout.addWidget(self._w)
        elif kind == "path":
            self._w = QLineEdit()
            layout.addWidget(self._w)
            browse = QPushButton("Browse...")
            browse.clicked.connect(self._browse_one)
            layout.addWidget(browse)
        elif kind == "paths":
            self._w = QListWidget()
            self._w.setMaximumHeight(90)
            layout.addWidget(self._w)
            side = QVBoxLayout()
            add = QPushButton("Add...")
            add.clicked.connect(self._browse_many)
            remove = QPushButton("Remove")
            remove.clicked.connect(self._remove_path)
            side.addWidget(add)
            side.addWidget(remove)
            side.addStretch(1)
            layout.addLayout(side)
        elif kind in ("size_mm", "frame_size"):
            self._parts = [self._number(kind == "frame_size") for _ in range(2)]
            for caption, widget in zip(("W", "H"), self._parts):
                layout.addWidget(QLabel(caption))
                layout.addWidget(widget)
            layout.addStretch(1)
            self._w = None
        elif kind == "region":
            # Read-only on purpose. A region is dragged on the artwork, and four
            # editable spin boxes in a table row are both unusable at that width
            # and an invitation to type coordinates nobody can get right.
            self._w = QLineEdit()
            self._w.setReadOnly(True)
            self._w.setPlaceholderText("not drawn")
            self._w.setToolTip(
                "Fractions of the label, set by drawing on the artwork. Use "
                "Define Regions while labeling, or Draw Regions if this label "
                "already has artwork.")
            self._region_value: list[float] = []
            layout.addWidget(self._w)
        elif kind == "regions":
            self._w = QPushButton("Draw Regions on the Reference...")
            self._w.clicked.connect(self._open_region_editor)
            layout.addWidget(self._w)
            self._summary = QLabel()
            self._summary.setWordWrap(True)
            self._summary.setStyleSheet("color: #9aa4b2;")
            layout.addWidget(self._summary, 1)
            self._refresh_region_summary()
        else:
            self._w = QLineEdit()
            layout.addWidget(self._w)

        if q.help:
            self.setToolTip(q.help)
        # A floor per kind: without it the size-to-content columns collapse a
        # combo or a four-part region to a few unusable pixels.
        self.setMinimumWidth({
            "bool": 40, "int": 80, "float": 90, "count": 70,
            "choice": 130, "label_picker": 150, "region": 190,
            "size_mm": 160, "frame_size": 160, "text": 150, "textarea": 200,
        }.get(kind, 120))

    def _number(self, integer: bool, normalised: bool = False):
        if integer:
            widget = QSpinBox()
            widget.setRange(0, 100000)
            return widget
        widget = QDoubleSpinBox()
        if normalised:
            # ROIs are fractions of the frame. Constraining the spin box is the
            # cheapest possible way to stop someone typing pixels into it.
            widget.setDecimals(4)
            widget.setRange(0.0, 1.0)
            widget.setSingleStep(0.01)
        else:
            widget.setDecimals(2)
            widget.setRange(0.0, 100000.0)
        return widget

    def _choices(self) -> list[Any]:
        if self.question.kind == "label_picker" and not self.question.choices_from:
            return [""] + list(self._answers.get(LIBRARY_IDS_KEY, []))
        return self.question.resolve_choices(self._answers)

    def _browse_one(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self.question.label)
        if path:
            self._w.setText(path)

    def _browse_many(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, self.question.label)
        for path in paths:
            self._w.addItem(path)

    def _remove_path(self) -> None:
        for item in self._w.selectedItems():
            self._w.takeItem(self._w.row(item))

    # -- region drawing ---------------------------------------------------

    def _region_summary_text(self) -> str:
        answers = self._owner.answers if self._owner is not None else self._answers
        parts = []
        for row in answers.get("codes", []) or []:
            if row.get("region"):
                parts.append(f"code:{row.get('role', '?')}")
        for row in answers.get("text_fields", []) or []:
            if row.get("region"):
                parts.append(f"text:{row.get('name', '?')}")
        if answers.get("anchor_region"):
            parts.append("anchor")
        if not parts:
            return ("Nothing drawn yet. Regions are fractions of the label, so they "
                    "follow it onto any image at any angle -- no measuring, no "
                    "calibration.")
        return "Drawn: " + ", ".join(parts)

    def _refresh_region_summary(self) -> None:
        if hasattr(self, "_summary"):
            self._summary.setText(self._region_summary_text())

    def _open_region_editor(self) -> None:
        from .region_editor import RegionEditorDialog

        answers = self._owner.answers if self._owner is not None else self._answers
        references = answers.get("reference_images") or []
        if not references:
            QMessageBox.information(
                self, "Draw Regions",
                "Add a reference image on the Appearance page first -- it is what "
                "the regions are drawn on.")
            return

        dialog = RegionEditorDialog(
            str(references[0]),
            [dict(r) for r in answers.get("codes", []) or []],
            [dict(r) for r in answers.get("text_fields", []) or []],
            list(answers.get("anchor_region") or []),
            parent=self,
        )
        if not dialog.exec():
            return
        drawn = dialog.result_regions()
        answers["codes"] = drawn["codes"]
        answers["text_fields"] = drawn["text_fields"]
        answers["anchor_region"] = drawn["anchor_region"]
        if drawn["anchor_region"]:
            # An anchor only means anything on a variable-data label, so drawing
            # one says what the operator meant more clearly than the checkbox did.
            answers["variable_data"] = True
        self._refresh_region_summary()

    # -- value round-trip ---------------------------------------------------

    def value(self) -> Any:
        kind = self.question.kind
        if kind in ("text", "path"):
            return self._w.text().strip()
        if kind == "count":
            return self._w.text().strip() or 1
        if kind == "textarea":
            return self._w.toPlainText().strip()
        if kind in ("int", "float"):
            return self._w.value()
        if kind == "bool":
            return self._w.isChecked()
        if kind in ("choice", "label_picker"):
            return self._w.currentText()
        if kind == "multichoice":
            return [self._w.item(i).text() for i in range(self._w.count())
                    if self._w.item(i).checkState() == Qt.Checked]
        if kind == "paths":
            return [self._w.item(i).text() for i in range(self._w.count())]
        if kind == "regions":
            # The button owns no value of its own; it writes the tables directly.
            return self._answers.get(self.question.key, "")
        if kind == "region":
            return list(self._region_value)
        if kind in ("size_mm", "frame_size"):
            return [w.value() for w in self._parts]
        return self._w.text()

    def setValue(self, value: Any) -> None:
        kind = self.question.kind
        if value is None:
            return
        if kind in ("text", "path", "count"):
            self._w.setText(str(value))
        elif kind == "textarea":
            self._w.setPlainText(str(value))
        elif kind in ("int", "float"):
            self._w.setValue(type(self._w.value())(value or 0))
        elif kind == "bool":
            self._w.setChecked(bool(value))
        elif kind in ("choice", "label_picker"):
            index = self._w.findText(str(value))
            self._w.setCurrentIndex(index if index >= 0 else 0)
        elif kind == "multichoice":
            wanted = {str(v) for v in value or []}
            for i in range(self._w.count()):
                item = self._w.item(i)
                item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        elif kind == "paths":
            self._w.clear()
            self._w.addItems([str(v) for v in value or []])
        elif kind == "region":
            self._region_value = [float(v) for v in list(value or [])[:4]]
            self._w.setText(", ".join(f"{v:.3f}" for v in self._region_value)
                            if len(self._region_value) == 4 else "")
        elif kind in ("size_mm", "frame_size"):
            for widget, item in zip(self._parts, list(value or [])):
                widget.setValue(type(widget.value())(item or 0))
        elif kind == "regions":
            self._refresh_region_summary()


# --- table editor ----------------------------------------------------------

class TableEditor(QWidget):
    """Repeating rows -- the codes on a label, the text fields on it.

    Each row carries its own remove button rather than relying on the table's
    selection. Every cell holds an editor widget, so clicking one focuses that
    widget and the table's ``currentRow()`` stays at -1 forever -- which made a
    single shared Remove button a permanent no-op.
    """

    def __init__(self, question: Question, answers: dict[str, Any], parent=None):
        super().__init__(parent)
        self.question = question
        self._answers = answers

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.table = QTableWidget(0, len(question.columns) + 1)
        self.table.setHorizontalHeaderLabels(
            [""] + [_short_header(c) for c in question.columns])
        for index, column in enumerate(question.columns, start=1):
            item = self.table.horizontalHeaderItem(index)
            if item is not None:
                item.setToolTip(column.help or column.label)

        header = self.table.horizontalHeader()
        # Size to content and scroll, rather than stretching every column into
        # the same narrow slot: eight stretched columns elide their own headers
        # and crush their editors to unusable widths.
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 30)
        # Row numbers, so a "row 2" validation error can be found.
        self.table.verticalHeader().setVisible(True)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setMinimumHeight(170)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        add = QPushButton("Add row")
        add.clicked.connect(lambda: self.add_row())
        buttons.addWidget(add)
        self.hint = QLabel("Use the ✕ on a row to remove it.")
        self.hint.setStyleSheet("color: #9aa4b2; font-size: 8pt;")
        buttons.addWidget(self.hint)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def add_row(self, values: dict[str, Any] | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        remove = QPushButton("✕")
        remove.setToolTip("Remove this row")
        remove.setFixedWidth(26)
        remove.setProperty("rightPanelButton", True)
        # Bound to the button, not to a row index: indexes shift as rows above
        # are removed, and a stale index deletes the wrong row.
        remove.clicked.connect(lambda _checked=False, b=remove: self._remove_row_of(b))
        self.table.setCellWidget(row, 0, remove)

        for column_index, column in enumerate(self.question.columns, start=1):
            editor = FieldEditor(column, self._answers)
            editor.setValue((values or {}).get(column.key, column.blank()))
            self.table.setCellWidget(row, column_index, editor)
        # An explicit height rather than resizeRowsToContents: the cell widgets
        # have their own margins and the computed height came out a few pixels
        # short, so consecutive rows visibly touched.
        self.table.setRowHeight(row, 34)
        self._refresh_hint()

    def _remove_row_of(self, button: QPushButton) -> None:
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 0) is button:
                self.table.removeRow(row)
                self._refresh_hint()
                return

    def _refresh_hint(self) -> None:
        count = self.table.rowCount()
        self.hint.setText(
            "No rows yet." if not count else
            f"{count} row(s). Use the ✕ on a row to remove it.")

    def value(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            entry: dict[str, Any] = {}
            for column_index, column in enumerate(self.question.columns, start=1):
                editor = self.table.cellWidget(row, column_index)
                if editor is not None:
                    entry[column.key] = editor.value()
            rows.append(entry)
        return rows

    def setValue(self, value: Any) -> None:
        self.table.setRowCount(0)
        for row in value or []:
            self.add_row(row)
        self._refresh_hint()


# --- the dialog ------------------------------------------------------------

class FlowDialog(QDialog):
    """Walks a Flow's visible pages, then shows a summary before finishing."""

    def __init__(self, flow: Flow, answers: dict[str, Any] | None = None,
                 library=None, parent=None):
        super().__init__(parent)
        self.flow = flow
        self.answers = flow.defaults()
        self.answers.update(answers or {})
        if library is not None:
            self.answers[LIBRARY_IDS_KEY] = library.ids()
        self.result_object: Any = None

        self.setWindowTitle(flow.title)
        self.setMinimumSize(820, 560)

        root = QVBoxLayout(self)
        self._heading = QLabel()
        self._heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        self._blurb = QLabel()
        self._blurb.setWordWrap(True)
        self._blurb.setStyleSheet("color: #9aa4b2;")
        root.addWidget(self._heading)
        root.addWidget(self._blurb)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        root.addWidget(line)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self._errors = QLabel()
        self._errors.setWordWrap(True)
        self._errors.setStyleSheet("color: #f87171;")
        root.addWidget(self._errors)

        self.buttons = QDialogButtonBox()
        self._back = self.buttons.addButton("Back", QDialogButtonBox.ActionRole)
        self._next = self.buttons.addButton("Next", QDialogButtonBox.ActionRole)
        self._finish = self.buttons.addButton("Finish", QDialogButtonBox.AcceptRole)
        self.buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self._back.clicked.connect(self._go_back)
        self._next.clicked.connect(self._go_next)
        self._finish.clicked.connect(self._finish_clicked)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._index = 0
        self._editors: dict[str, Any] = {}
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._show_page()

    # -- page machinery -----------------------------------------------------

    def _pages(self) -> list[Page]:
        return self.flow.visible_pages(self.answers)

    def _is_summary(self) -> bool:
        return self._index >= len(self._pages())

    def _build_page(self, page: Page) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setLabelAlignment(Qt.AlignRight)
        self._editors = {}
        for question in page.visible_questions(self.answers):
            editor = (TableEditor(question, self.answers) if question.kind == "table"
                      else FieldEditor(question, self.answers, owner=self))
            editor.setValue(self.answers.get(question.key, question.blank()))
            caption = question.label + (" *" if question.required else "")
            form.addRow(caption, editor)
            if question.help:
                hint = QLabel(question.help)
                hint.setWordWrap(True)
                hint.setStyleSheet("color: #9aa4b2; font-size: 8pt;")
                form.addRow("", hint)
            self._editors[question.key] = editor
        return widget

    def _show_page(self) -> None:
        while self.stack.count():
            self.stack.removeWidget(self.stack.widget(0))
        pages = self._pages()

        if self._is_summary():
            self._heading.setText("Review")
            self._blurb.setText(
                "Warnings do not block finishing -- they describe what this will and "
                "will not catch."
            )
            self._summary.setPlainText(self._summary_text())
            self.stack.addWidget(self._summary)
            self._editors = {}
        else:
            page = pages[self._index]
            self._heading.setText(f"{page.title}   ({self._index + 1} of {len(pages) + 1})")
            self._blurb.setText(page.blurb)
            self.stack.addWidget(self._build_page(page))

        self._back.setEnabled(self._index > 0)
        self._next.setEnabled(not self._is_summary())
        self._finish.setEnabled(self._is_summary())
        self._errors.clear()

    def _harvest(self) -> None:
        for key, editor in self._editors.items():
            self.answers[key] = editor.value()

    def _summary_text(self) -> str:
        notes = self.flow.notes(self.answers)
        lines = ["Ready to save." if not notes else "Ready to save, with warnings:"]
        for note in notes:
            lines.append(f"  - {note}")
        lines.append("")
        lines.append(self.flow.to_text(self.answers))
        return "\n".join(lines)

    # -- navigation ---------------------------------------------------------

    def _go_back(self) -> None:
        if not self._is_summary():
            self._harvest()
        self._index = max(0, self._index - 1)
        self._show_page()

    def _go_next(self) -> None:
        self._harvest()
        page = self._pages()[self._index]
        errors = self.flow.validate_page(page, self.answers)
        if errors:
            self._errors.setText("\n".join(errors))
            return
        self._index += 1
        self._show_page()

    def _finish_clicked(self) -> None:
        errors = self.flow.validate(self.answers)
        if errors:
            # Answers can go stale when an earlier page reveals a later
            # requirement, so send the operator back to the offending page
            # rather than refusing with no way forward.
            self._errors.setText("\n".join(errors))
            self._index = 0
            self._show_page()
            self._errors.setText("\n".join(errors))
            return
        self.result_object = self.flow.result(self.answers)
        self.accept()
