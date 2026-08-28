"""One window that sets up a label's reference image, start to finish.

Before this, defining a label meant Capture Reference, then drawing a box on
the annotation canvas, then Define Regions, then Place Regions -- four controls
across two panes, in an order nothing enforced, each doing part of one job. The
parts were separately reachable, so a label could sit half-defined
indefinitely and look finished.

It is one job in three steps, and the window owns all three:

  1. **Frame.** Its own live preview and its own shutter. Reaching into the
     main window's preview meant the reference could only be shot from another
     tab that happened to be running, which is a dependency on where somebody
     had been rather than on what they are doing.
  2. **Outline.** Four corners, not a rectangle. A label at an angle inside an
     axis-aligned box brings a wedge of background in with it, and the outline
     is the coordinate system every region is a fraction of -- so that wedge
     shifts every region by its own size.
  3. **Regions.** Drawn on the artwork, which is the outline RECTIFIED: warped
     straight-on. That is what makes the regions simple. On a de-skewed
     artwork an axis-aligned region is the right shape, and it is exactly what
     the runtime maps back onto whatever oriented box the detector produces.

Nothing is written until the last step finishes. Cancelling at any point leaves
the label exactly as it was, which is what makes the artwork safe to treat as
immutable -- there is no half-completed state that could have moved it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QRadioButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ..core import reference as ref_logic
from .quad_canvas import QuadCanvas
from .region_editor import RegionEditorBody

FRAME_PAGE, OUTLINE_PAGE, REGION_PAGE = 0, 1, 2

# How often the built-in preview pulls a frame. The camera runs at its own rate
# and this only has to look live to a person holding a battery still.
PREVIEW_MS = 66


class ReferenceSetupDialog(QDialog):
    """Photograph a label, outline it, and mark what to read on it."""

    def __init__(self, label, *, frames, images, parent=None):
        """``frames`` returns the newest BGR frame from the camera, or None.

        ``images`` are this label's existing captures, so a reference can be
        set up without the camera -- on a rig where the camera is on the line
        and the labelling happens at a desk, insisting on a live frame would
        mean the work can only be done in one room.
        """
        super().__init__(parent)
        self.label = label
        self._frames = frames
        self._images = [Path(p) for p in images or []]
        self.result: dict | None = None
        self.frame = None                # the raw photograph, BGR
        self.artwork = None              # the rectified label, BGR
        # Set when the photograph came from an image already in the dataset.
        # Empty means it was just shot and the caller has to keep it: the
        # photograph the artwork was flattened from is the one to go back to
        # when a region looks wrong, and it is a perfectly good training image
        # of the label besides.
        self.source_path = ""

        label_id = str(getattr(label, "label_id", "") or "label")
        self.setWindowTitle(f"Reference image for {label_id}")
        self.resize(1180, 820)

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
        self.stack.addWidget(self._build_frame_page())
        self.stack.addWidget(self._build_outline_page())
        self.stack.addWidget(QWidget())        # built once artwork exists

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

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_preview)
        self._timer.start(PREVIEW_MS)
        self._show_page()

    # -- step 1: the photograph -------------------------------------------

    def _build_frame_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)

        self.preview = QuadCanvas()          # same widget, no outline drawn here
        column.addWidget(self.preview, 1)

        row = QHBoxLayout()
        self.shoot_btn = QPushButton("Capture this frame")
        self.shoot_btn.setStyleSheet("font-weight: 700;")
        self.shoot_btn.clicked.connect(self._shoot)
        row.addWidget(self.shoot_btn, 1)
        column.addLayout(row)

        pick = QHBoxLayout()
        self.existing_radio = QRadioButton("or use a captured image:")
        pick.addWidget(self.existing_radio)
        self.image_combo = QComboBox()
        for path in self._images:
            self.image_combo.addItem(path.name, str(path))
        if not self._images:
            self.image_combo.addItem("none yet", "")
            self.existing_radio.setEnabled(False)
        self.image_combo.currentIndexChanged.connect(
            lambda _i: self.existing_radio.setChecked(True))
        pick.addWidget(self.image_combo, 1)
        column.addLayout(pick)

        existing = ref_logic.reference_path(self.label)
        if existing:
            note = QLabel(
                f"Replacing the artwork at {existing} — every region measured "
                f"against the old one has to be drawn again.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #fbbf24;")
            column.addWidget(note)
        return page

    def _tick_preview(self) -> None:
        """Show the live camera until a frame has been taken."""
        if self.stack.currentIndex() != FRAME_PAGE or self.frame is not None:
            return
        try:
            live = self._frames()
        except Exception:
            live = None
        if live is not None:
            self.preview.set_frame(live)

    def _shoot(self) -> None:
        self.problem.clear()
        try:
            taken = self._frames()
        except Exception as exc:
            self.problem.setText(f"Could not read the camera: {exc}")
            return
        if taken is None:
            self.problem.setText(
                "No camera frame. Connect and start the camera, or pick one of "
                "this label's captured images below.")
            return
        self.frame = taken.copy()
        self.source_path = ""
        self.preview.set_frame(self.frame)
        self._go_next()

    # -- step 2: the outline ----------------------------------------------

    def _build_outline_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        self.outline_canvas = QuadCanvas()
        self.outline_canvas.changed.connect(self._show_page)
        column.addWidget(self.outline_canvas, 1)
        row = QHBoxLayout()
        clear = QPushButton("Clear outline")
        clear.clicked.connect(self.outline_canvas.clear)
        row.addWidget(clear)
        row.addStretch(1)
        column.addLayout(row)
        return page

    # -- step 3: the regions ----------------------------------------------

    def _build_region_page(self, artwork_path: Path) -> QWidget:
        codes = [c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__)
                 for c in getattr(self.label, "codes", None) or []]
        texts = [dict(t.__dict__)
                 for t in getattr(self.label, "text_fields", None) or []]
        self.body = RegionEditorBody(
            str(artwork_path), codes, texts,
            list(getattr(self.label, "anchor_region", None) or []), self)
        return self.body

    # -- moving between them -----------------------------------------------

    def _go_next(self) -> None:
        self.problem.clear()
        page = self.stack.currentIndex()
        if page == FRAME_PAGE:
            if self.frame is None and self.existing_radio.isChecked():
                import cv2

                chosen = self.image_combo.currentData()
                self.frame = cv2.imread(str(chosen)) if chosen else None
                self.source_path = str(chosen) if self.frame is not None else ""
            if self.frame is None:
                self.problem.setText(
                    "Capture a frame, or pick one of this label's images.")
                return
            self.outline_canvas.set_frame(self.frame)
            self.stack.setCurrentIndex(OUTLINE_PAGE)
        elif page == OUTLINE_PAGE:
            if not self.outline_canvas.has_quad():
                self.problem.setText(
                    "Draw the outline first: drag around the label, then drag "
                    "its corners onto the label's own corners.")
                return
            if not self._rectify():
                return
            self.stack.setCurrentIndex(REGION_PAGE)
        self._show_page()

    def _rectify(self) -> bool:
        """Warp the outlined quad straight-on and open the regions on it."""
        import cv2

        from ..core.imageio import rectify_quad
        from ..core.storage import DATA_DIR

        artwork = rectify_quad(self.frame, self.outline_canvas.quad,
                               out_width=900)
        if artwork is None or artwork.size == 0:
            self.problem.setText(
                "That outline could not be flattened -- its corners are in a "
                "line, or it is too small. Redraw it around the whole label.")
            return False
        self.artwork = artwork
        scratch = DATA_DIR / "library" / "references"
        scratch.mkdir(parents=True, exist_ok=True)
        # Written to a scratch name, not the label's: nothing claims to be this
        # label's artwork until the whole thing is finished.
        preview = scratch / "_pending_reference.png"
        cv2.imwrite(str(preview), artwork)
        old = self.stack.widget(REGION_PAGE)
        self.stack.removeWidget(old)
        old.deleteLater()
        self.stack.insertWidget(REGION_PAGE, self._build_region_page(preview))
        return True

    def _go_back(self) -> None:
        self.problem.clear()
        page = self.stack.currentIndex()
        if page == REGION_PAGE:
            self.stack.setCurrentIndex(OUTLINE_PAGE)
        elif page == OUTLINE_PAGE:
            # Back to the camera means taking another photograph, so the frame
            # goes: keeping it would show a still while the shutter is offered.
            self.frame = None
            self.source_path = ""
            self.stack.setCurrentIndex(FRAME_PAGE)
        self._show_page()

    def _show_page(self) -> None:
        page = self.stack.currentIndex()
        label_id = str(getattr(self.label, "label_id", "") or "label")
        headings = {
            FRAME_PAGE: (
                f"1 of 3 — Photograph {label_id}",
                "Hold the label square-on and filling as much of the view as "
                "the part allows. This photograph becomes the label's "
                "coordinate system, and it is never edited afterwards."),
            OUTLINE_PAGE: (
                f"2 of 3 — Outline {label_id}",
                "Drag around the label, then drag each corner onto the label's "
                "own corner. Four corners rather than a rectangle: a label at "
                "an angle inside a straight box brings a wedge of background "
                "with it, and every region is a fraction of this outline."),
            REGION_PAGE: (
                f"3 of 3 — What to read on {label_id}",
                "The artwork is your outline flattened straight-on. Draw the "
                "code and text areas on it -- they are stored as fractions of "
                "it, so they follow the label onto any frame at any angle."),
        }
        heading, blurb = headings.get(page, ("", ""))
        self.heading.setText(heading)
        self.blurb.setText(blurb)
        self._back.setEnabled(page != FRAME_PAGE)
        self._next.setEnabled(
            page == FRAME_PAGE
            or (page == OUTLINE_PAGE and self.outline_canvas.has_quad()))
        self._finish.setEnabled(page == REGION_PAGE)
        self.shoot_btn.setEnabled(page == FRAME_PAGE)

    # -- finishing ---------------------------------------------------------

    def _finish_clicked(self) -> None:
        self.problem.clear()
        body = getattr(self, "body", None)
        if body is None or self.artwork is None:
            self.problem.setText("There is no artwork to save yet.")
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
        self.result = {"artwork": self.artwork, "frame": self.frame,
                       "source_path": self.source_path,
                       "quad": list(self.outline_canvas.quad), **drawn}
        self.accept()

    def done(self, code: int) -> None:
        self._timer.stop()
        super().done(code)
