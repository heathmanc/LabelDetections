"""One window that sets up a label's reference image, start to finish.

Before this, defining a label meant Capture Reference, then drawing a box on
the annotation canvas, then Define Regions, then Place Regions -- four controls
across two panes, in an order nothing enforced, each doing part of one job. The
parts were also separately reachable, so a label could sit half-defined
indefinitely and look finished.

It is one job: photograph the label, say where it is in the photograph, say
what to read on it. So it is one window, and the buttons that used to do the
pieces are gone.

The reference is written only when the whole thing is finished. Cancelling at
any point leaves the label exactly as it was, which is what makes the artwork
safe to treat as immutable -- there is no half-completed state that could have
moved it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ..core import reference as ref_logic
from .region_editor import RegionEditorBody

SOURCE_PAGE, DRAW_PAGE = 0, 1


class ReferenceSetupDialog(QDialog):
    """Photograph a label, outline it, and mark what to read on it."""

    def __init__(self, label, *, capture, images, parent=None):
        """``capture`` returns a freshly grabbed BGR frame or None.

        ``images`` are this label's existing captures, so a reference can be
        set up without the camera connected -- on a rig where the camera is on
        the line and the labelling is done at a desk, insisting on a live frame
        would mean the work can only happen in one room.
        """
        super().__init__(parent)
        self.label = label
        self._capture = capture
        self._images = [Path(p) for p in images or []]
        self.result: dict | None = None
        self._frame_path: Path | None = None

        label_id = str(getattr(label, "label_id", "") or "label")
        self.setWindowTitle(f"Reference image for {label_id}")
        self.resize(1150, 760)

        root = QVBoxLayout(self)
        self.heading = QLabel()
        self.heading.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.blurb = QLabel()
        self.blurb.setWordWrap(True)
        self.blurb.setStyleSheet("color: #9aa4b2;")
        root.addWidget(self.heading)
        root.addWidget(self.blurb)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_source_page())
        self.stack.addWidget(QWidget())          # replaced when a frame exists

        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #f87171;")
        root.addWidget(self.problem)

        self.buttons = QDialogButtonBox()
        self._back = self.buttons.addButton("Back", QDialogButtonBox.ActionRole)
        self._next = self.buttons.addButton("Next", QDialogButtonBox.ActionRole)
        self._finish = self.buttons.addButton("Save reference",
                                              QDialogButtonBox.AcceptRole)
        self.buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self._back.clicked.connect(self._go_back)
        self._next.clicked.connect(self._go_next)
        self._finish.clicked.connect(self._finish_clicked)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self._show_page()

    # -- page 1: where the photograph comes from --------------------------

    def _build_source_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)

        self.shoot_radio = QRadioButton("Capture a new frame")
        self.shoot_radio.setChecked(True)
        column.addWidget(self.shoot_radio)
        hint = QLabel(
            "    Frame the label square-on and filling as much of the view as "
            "the part allows. This one photograph is the label's artwork for "
            "good, and every region is measured on it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa4b2;")
        column.addWidget(hint)

        self.existing_radio = QRadioButton("Use one of this label's images")
        column.addWidget(self.existing_radio)
        row = QHBoxLayout()
        row.addSpacing(24)
        self.image_combo = QComboBox()
        for path in self._images:
            self.image_combo.addItem(path.name, str(path))
        if not self._images:
            self.image_combo.addItem("no captured images yet", "")
            self.existing_radio.setEnabled(False)
        self.image_combo.currentIndexChanged.connect(
            lambda _i: self.existing_radio.setChecked(True))
        row.addWidget(self.image_combo, 1)
        column.addLayout(row)

        column.addStretch(1)
        existing = ref_logic.reference_path(self.label)
        if existing:
            note = QLabel(
                f"This label already has artwork:\n{existing}\n\n"
                "Saving here replaces it, and every region measured against "
                "the old one has to be drawn again.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #fbbf24;")
            column.addWidget(note)
        return page

    # -- page 2: the outline and the regions ------------------------------

    def _build_draw_page(self, frame_path: Path) -> QWidget:
        codes = [c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__)
                 for c in getattr(self.label, "codes", None) or []]
        texts = [dict(t.__dict__)
                 for t in getattr(self.label, "text_fields", None) or []]
        self.body = RegionEditorBody(
            str(frame_path), codes, texts,
            list(getattr(self.label, "anchor_region", None) or []), self)
        return self.body

    # -- moving between them -----------------------------------------------

    def _frame(self) -> Path | None:
        """The photograph to work on, captured or chosen."""
        if self.existing_radio.isChecked():
            chosen = self.image_combo.currentData()
            return Path(chosen) if chosen else None
        return self._capture()

    def _go_next(self) -> None:
        self.problem.clear()
        if self.stack.currentIndex() != SOURCE_PAGE:
            return
        try:
            frame_path = self._frame()
        except Exception as exc:
            self.problem.setText(f"Could not get a frame: {exc}")
            return
        if frame_path is None or not Path(frame_path).is_file():
            self.problem.setText(
                "No frame. Open the live preview and try again, or pick one of "
                "this label's captured images.")
            return
        self._frame_path = Path(frame_path)
        page = self._build_draw_page(self._frame_path)
        old = self.stack.widget(DRAW_PAGE)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.stack.insertWidget(DRAW_PAGE, page)
        self.stack.setCurrentIndex(DRAW_PAGE)
        self._show_page()

    def _go_back(self) -> None:
        self.problem.clear()
        self.stack.setCurrentIndex(SOURCE_PAGE)
        self._show_page()

    def _show_page(self) -> None:
        drawing = self.stack.currentIndex() == DRAW_PAGE
        label_id = str(getattr(self.label, "label_id", "") or "label")
        if drawing:
            self.heading.setText(f"2 of 2 — Outline {label_id}, then its regions")
            self.blurb.setText(
                "Draw the label outline first: everything else is measured as a "
                "fraction of it, so it has to exist before a region means "
                "anything. Then draw the code and text areas inside it.")
        else:
            self.heading.setText(f"1 of 2 — A photograph of {label_id}")
            self.blurb.setText(
                "The reference image is this label's coordinate system. It is "
                "never edited afterwards, only replaced -- because every region "
                "on every image already reviewed is positioned against it.")
        self._back.setEnabled(drawing)
        self._next.setEnabled(not drawing)
        self._finish.setEnabled(drawing)

    # -- finishing ---------------------------------------------------------

    def _finish_clicked(self) -> None:
        self.problem.clear()
        body = getattr(self, "body", None)
        if body is None or not body.has_image():
            self.problem.setText("There is no photograph to work on yet.")
            return
        if not body.has_outline():
            self.problem.setText(
                "Draw the label outline first — pick 'Label outline' and drag "
                "around the whole label. Regions are fractions of it, so "
                "without one they have nothing to be a fraction of.")
            return
        drawn = body.result_regions()
        if not (drawn["codes"] or drawn["text_fields"]):
            answer = QMessageBox.question(
                self, "No regions",
                f"{getattr(self.label, 'label_id', 'This label')} will have a "
                f"reference image but nothing marked to read on it.\n\n"
                f"It can still be detected and classified. It cannot be "
                f"verified -- which is what stops a label nobody enrolled being "
                f"reported as this one.\n\nSave it anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        self.result = {
            "frame": str(self._frame_path),
            "outline": self._outline_rect(body),
            **drawn,
        }
        self.accept()

    @staticmethod
    def _outline_rect(body) -> list[float]:
        """The outline in image pixels: ``[x, y, w, h]``.

        The caller crops the artwork out of the photograph with it. Kept in
        pixels rather than fractions because the photograph is the only thing
        it has ever been relative to.
        """
        rect = body.canvas.outline
        return [float(rect.x()), float(rect.y()),
                float(rect.width()), float(rect.height())]
