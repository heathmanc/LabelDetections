"""Draw read-regions on a label's reference artwork.

A region is an area *inside* a label that inspection has to read on its own:
a barcode, a serial, a date code. It is stored as **fractions of the label** --
``[x, y, w, h]``, each 0 to 1 -- which is what makes it survive everything that
happens to the label afterwards. Once an operator draws the label's four corners
on a real image, every region follows by homography
(``core.annotations.apply_reference_regions``), at whatever angle, distance or
resolution the camera happened to see it.

Fractions rather than millimetres, because the mapping is pure proportion. No
distance calibration, no measuring, nothing to look up: drag a box on the
artwork and it lands in the right place on every unit forever.

That is also why regions are the answer to changing text. The artwork around a
serial never moves; the serial does. Matching scores against the static part,
and the region says exactly where to go looking for the part that changes.

The label outline exists because a reference photo usually has margin around
the label. Everything is measured relative to that outline, not to the image.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..core.labels import CODE_ROLES, SYMBOLOGIES

# Roles a drawn region can carry. "outline" is the label itself within the
# reference photo -- everything else is measured relative to it, because a
# reference shot almost always has margin around the label.
OUTLINE = "outline"
ROLE_COLORS = {
    OUTLINE: QColor(96, 165, 250),
    "code": QColor(250, 204, 21),
    "text": QColor(34, 197, 94),
    "anchor": QColor(168, 85, 247),
}

_HANDLE = 7


class RegionCanvas(QWidget):
    """The reference image with draggable rectangles over it."""

    regions_changed = Signal()
    selection_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self._pixmap: QPixmap | None = None
        # Each region: {"role", "name", "rect": QRectF in image pixels}
        self.regions: list[dict] = []
        self.outline = QRectF()
        self.selected = -1
        self.draw_role = "code"
        self._drag_from: QPoint | None = None
        self._drag_rect: QRectF | None = None
        self._moving = False
        self._resizing = False
        self._move_origin = QRectF()

    # -- image ----------------------------------------------------------

    def load(self, path: str | Path) -> bool:
        image = QImage(str(path))
        if image.isNull():
            self._pixmap = None
            return False
        self._pixmap = QPixmap.fromImage(image)
        if self.outline.isNull():
            # Default the outline to the whole image: a cropped reference is the
            # common case, and an operator who did crop should not have to
            # re-draw what they already framed.
            self.outline = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
        self.update()
        return True

    def image_size(self) -> tuple[int, int]:
        if self._pixmap is None:
            return (0, 0)
        return (self._pixmap.width(), self._pixmap.height())

    # -- coordinate mapping ---------------------------------------------

    def _target(self) -> QRectF:
        """Where the image is drawn, letterboxed into the widget."""
        if self._pixmap is None or self._pixmap.width() == 0:
            return QRectF(self.rect())
        scale = min(self.width() / self._pixmap.width(),
                    self.height() / self._pixmap.height())
        w = self._pixmap.width() * scale
        h = self._pixmap.height() * scale
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def _to_image(self, point: QPoint) -> QRectF:
        target = self._target()
        if target.width() <= 0 or self._pixmap is None:
            return QRectF()
        sx = self._pixmap.width() / target.width()
        sy = self._pixmap.height() / target.height()
        return QRectF((point.x() - target.x()) * sx, (point.y() - target.y()) * sy, 0, 0)

    def _to_screen(self, rect: QRectF) -> QRectF:
        target = self._target()
        if self._pixmap is None or self._pixmap.width() == 0:
            return QRectF()
        sx = target.width() / self._pixmap.width()
        sy = target.height() / self._pixmap.height()
        return QRectF(target.x() + rect.x() * sx, target.y() + rect.y() * sy,
                      rect.width() * sx, rect.height() * sy)

    # -- mm conversion ---------------------------------------------------

    def to_fraction(self, rect: QRectF) -> list[float]:
        """A pixel rect as fractions of the label outline, clamped to it.

        Clamped because a region dragged past the label's edge is a slip, and
        storing it would place the crop off the label at runtime.
        """
        if self.outline.width() <= 0 or self.outline.height() <= 0:
            return []
        fx = (rect.x() - self.outline.x()) / self.outline.width()
        fy = (rect.y() - self.outline.y()) / self.outline.height()
        fw = rect.width() / self.outline.width()
        fh = rect.height() / self.outline.height()
        fx, fy = max(0.0, min(1.0, fx)), max(0.0, min(1.0, fy))
        fw = max(0.0, min(1.0 - fx, fw))
        fh = max(0.0, min(1.0 - fy, fh))
        return [round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)]

    def from_fraction(self, region: list[float]) -> QRectF:
        """The inverse, for showing regions that already exist."""
        if len(region) < 4 or self.outline.width() <= 0:
            return QRectF()
        return QRectF(
            self.outline.x() + float(region[0]) * self.outline.width(),
            self.outline.y() + float(region[1]) * self.outline.height(),
            float(region[2]) * self.outline.width(),
            float(region[3]) * self.outline.height(),
        )

    # -- interaction ------------------------------------------------------

    def _hit(self, point: QPoint) -> int:
        for index in range(len(self.regions) - 1, -1, -1):
            if self._to_screen(self.regions[index]["rect"]).contains(point):
                return index
        return -1

    def _on_handle(self, point: QPoint, index: int) -> bool:
        if index < 0:
            return False
        screen = self._to_screen(self.regions[index]["rect"])
        corner = QRectF(screen.right() - _HANDLE, screen.bottom() - _HANDLE,
                        _HANDLE * 2, _HANDLE * 2)
        return corner.contains(point)

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None:
            return
        point = event.position().toPoint()
        hit = self._hit(point)
        if event.button() == Qt.LeftButton and hit >= 0:
            self.selected = hit
            self.selection_changed.emit(hit)
            if self._on_handle(point, hit):
                self._resizing = True
            else:
                self._moving = True
                self._move_origin = QRectF(self.regions[hit]["rect"])
                self._drag_from = point
            self.update()
            return
        if event.button() == Qt.LeftButton:
            self._drag_from = point
            self._drag_rect = QRectF()
            self.selected = -1
            self.selection_changed.emit(-1)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._pixmap is None or self._drag_from is None:
            return
        point = event.position().toPoint()
        start = self._to_image(self._drag_from).topLeft()
        now = self._to_image(point).topLeft()

        if self._resizing and self.selected >= 0:
            rect = self.regions[self.selected]["rect"]
            rect.setBottomRight(now)
            self.update()
            return
        if self._moving and self.selected >= 0:
            dx = now.x() - start.x()
            dy = now.y() - start.y()
            origin = self._move_origin
            self.regions[self.selected]["rect"] = QRectF(
                origin.x() + dx, origin.y() + dy, origin.width(), origin.height())
            self.update()
            return
        self._drag_rect = QRectF(start, now).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._moving or self._resizing:
            self._moving = self._resizing = False
            self._drag_from = None
            # Normalise after a resize so a rectangle dragged up-left is still
            # a rectangle rather than a negative-width one.
            if self.selected >= 0:
                self.regions[self.selected]["rect"] = \
                    self.regions[self.selected]["rect"].normalized()
            self.regions_changed.emit()
            self.update()
            return
        if self._drag_rect is not None and self._drag_rect.width() > 3 \
                and self._drag_rect.height() > 3:
            if self.draw_role == OUTLINE:
                self.outline = QRectF(self._drag_rect)
            else:
                self.regions.append({
                    "role": self.draw_role,
                    "name": self._default_name(self.draw_role),
                    "rect": QRectF(self._drag_rect),
                })
                self.selected = len(self.regions) - 1
                self.selection_changed.emit(self.selected)
            self.regions_changed.emit()
        self._drag_from = None
        self._drag_rect = None
        self.update()

    def _default_name(self, role: str) -> str:
        if role == "code":
            used = {r["name"] for r in self.regions if r["role"] == "code"}
            for candidate in CODE_ROLES:
                if candidate not in used:
                    return candidate
            return "other"
        if role == "anchor":
            return "anchor"
        used = sum(1 for r in self.regions if r["role"] == "text")
        return f"field_{used + 1}"

    def delete_selected(self) -> None:
        if 0 <= self.selected < len(self.regions):
            del self.regions[self.selected]
            self.selected = -1
            self.selection_changed.emit(-1)
            self.regions_changed.emit()
            self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0f172a"))
        if self._pixmap is None:
            p.setPen(QColor("#94a3b8"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No reference image.\nAdd one on the Appearance page first.")
            return

        target = self._target()
        p.drawPixmap(target.toRect(), self._pixmap)

        outline_screen = self._to_screen(self.outline)
        pen = QPen(ROLE_COLORS[OUTLINE], 2, Qt.DashLine)
        p.setPen(pen)
        p.drawRect(outline_screen)
        p.drawText(outline_screen.topLeft() + QPoint(4, -4), "label outline")

        for index, region in enumerate(self.regions):
            colour = ROLE_COLORS.get(region["role"], QColor("#94a3b8"))
            selected = index == self.selected
            p.setPen(QPen(QColor(250, 204, 21) if selected else colour,
                          3 if selected else 2))
            screen = self._to_screen(region["rect"])
            p.drawRect(screen)
            p.drawText(screen.topLeft() + QPoint(4, -4),
                       f"{region['role']}: {region['name']}")
            if selected:
                p.fillRect(QRectF(screen.right() - _HANDLE, screen.bottom() - _HANDLE,
                                  _HANDLE * 2, _HANDLE * 2), QColor(250, 204, 21))

        if self._drag_rect is not None and not self._drag_rect.isNull():
            p.setPen(QPen(ROLE_COLORS.get(self.draw_role, QColor("#94a3b8")), 1, Qt.DashLine))
            p.drawRect(self._to_screen(self._drag_rect))


class RegionEditorDialog(QDialog):
    """Draw a label's read-regions on its reference artwork.

    Opened from the add-a-label wizard with whatever rows already exist, and it
    hands them back with their geometry filled in. The wizard's tables stay the
    one source of truth for policies and patterns; this only supplies the
    coordinates, which are the part nobody can type accurately.
    """

    def __init__(self, reference: str, codes: list[dict], text_fields: list[dict],
                 anchor: list[float] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw read-regions on the reference")
        self.resize(1080, 700)

        root = QVBoxLayout(self)
        blurb = QLabel(
            "Drag on the artwork to add a region. Regions are stored as fractions of "
            "the label, so they follow it onto any image at any angle and any "
            "distance -- which is how a serial that changes on every unit still gets "
            "read from the same place. Nothing here needs measuring or calibrating."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #9aa4b2;")
        root.addWidget(blurb)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        self.canvas = RegionCanvas()
        body.addWidget(self.canvas, 3)

        side = QVBoxLayout()
        body.addLayout(side, 2)

        side.addWidget(QLabel("Draw as"))
        self.role_combo = QComboBox()
        self.role_combo.addItem("Code (barcode / 2D)", "code")
        self.role_combo.addItem("Text field (OCR)", "text")
        self.role_combo.addItem("Static anchor (for matching)", "anchor")
        self.role_combo.addItem("Label outline", OUTLINE)
        self.role_combo.setToolTip(
            "Label outline marks where the label sits in this reference photo. "
            "Everything else is measured from it, so set it first if the photo "
            "has margin around the label."
        )
        self.role_combo.currentIndexChanged.connect(self._role_changed)
        side.addWidget(self.role_combo)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Role", "Name", "Detail", "x, y, w, h"])
        self.table.horizontalHeaderItem(3).setToolTip(
            "Position on the label as fractions of it, 0 to 1.")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._row_selected)
        side.addWidget(self.table, 1)

        edit_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("serial, date_code, lot...")
        self.name_edit.editingFinished.connect(self._rename_selected)
        edit_row.addWidget(QLabel("Name"))
        edit_row.addWidget(self.name_edit, 1)
        side.addLayout(edit_row)

        symb_row = QHBoxLayout()
        self.symbology_combo = QComboBox()
        self.symbology_combo.addItems(SYMBOLOGIES)
        self.symbology_combo.currentIndexChanged.connect(self._symbology_changed)
        symb_row.addWidget(QLabel("Symbology"))
        symb_row.addWidget(self.symbology_combo, 1)
        side.addLayout(symb_row)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.canvas.delete_selected)
        side.addWidget(delete_btn)

        self.feasibility = QLabel()
        self.feasibility.setWordWrap(True)
        self.feasibility.setStyleSheet("color: #fbbf24;")
        side.addWidget(self.feasibility)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.canvas.regions_changed.connect(self._refresh_table)
        self.canvas.selection_changed.connect(self._sync_side_panel)

        self._symbologies: dict[int, str] = {}
        self._widths: dict[int, float] = {}
        self._x_dims: dict[int, float] = {}
        self._extras: dict[str, dict] = {}
        self._original: dict = {"codes": [], "text_fields": [], "anchor_region": []}
        self._load(reference, codes, text_fields, anchor or [])

    # -- load / save ------------------------------------------------------

    def _load(self, reference: str, codes: list[dict], text_fields: list[dict],
              anchor: list[float]) -> None:
        # Capture what already exists BEFORE anything can fail. A reference image
        # that has moved must not cost the operator every policy and pattern
        # already entered -- the editor supplies geometry, it never owns the row.
        for row in codes or []:
            self._extras[f"code:{row.get('role', 'other')}"] = dict(row)
        for row in text_fields or []:
            self._extras[f"text:{row.get('name', '')}"] = dict(row)
        self._original = {
            "codes": [dict(r) for r in codes or []],
            "text_fields": [dict(r) for r in text_fields or []],
            "anchor_region": list(anchor),
        }

        if not self.canvas.load(reference):
            self.feasibility.setText(
                f"Could not open the reference image:\n{reference}\n\n"
                "Nothing already entered has been lost -- close this and point the "
                "label at a reference that exists.")
            return
        for row in codes or []:
            name = str(row.get("role", "other"))
            rect = self.canvas.from_fraction(list(row.get("region") or []))
            if rect.isNull() or rect.width() <= 0:
                continue
            self.canvas.regions.append({"role": "code", "name": name, "rect": rect})
            index = len(self.canvas.regions) - 1
            self._symbologies[index] = str(row.get("symbology", "datamatrix"))
            self._widths[index] = float(row.get("code_width_mm", 0.0) or 0.0)
            self._x_dims[index] = float(row.get("x_dim_mm", 0.0) or 0.0)
        for row in text_fields or []:
            name = str(row.get("name", ""))
            rect = self.canvas.from_fraction(list(row.get("region") or []))
            if rect.isNull() or rect.width() <= 0:
                continue
            self.canvas.regions.append({"role": "text", "name": name, "rect": rect})
        if len(anchor) >= 4:
            rect = self.canvas.from_fraction(list(anchor))
            if not rect.isNull() and rect.width() > 0:
                self.canvas.regions.append({"role": "anchor", "name": "anchor", "rect": rect})
        self._refresh_table()

    def result_regions(self) -> dict:
        """Wizard rows with their ``region`` filled in from what was drawn.

        Policies, patterns and print-spec numbers already entered are carried
        through untouched -- the editor supplies geometry, which is the part
        nobody can type accurately, and nothing else.

        With no usable artwork there is nothing to have drawn, so the rows come
        back exactly as they went in rather than empty.
        """
        if self.canvas.image_size() == (0, 0):
            return dict(self._original)

        codes: list[dict] = []
        text_fields: list[dict] = []
        anchor: list[float] = []

        for index, region in enumerate(self.canvas.regions):
            fraction = self.canvas.to_fraction(region["rect"])
            name = str(region["name"])
            if region["role"] == "code":
                row = dict(self._extras.get(f"code:{name}", {}))
                row.setdefault("policy", "must_decode")
                row["role"] = name
                row["symbology"] = self._symbologies.get(
                    index, row.get("symbology", "datamatrix"))
                row["region"] = fraction
                codes.append(row)
            elif region["role"] == "text":
                row = dict(self._extras.get(f"text:{name}", {}))
                row.setdefault("policy", "must_be_present")
                row["name"] = name
                row["region"] = fraction
                text_fields.append(row)
            elif region["role"] == "anchor":
                anchor = fraction
        return {"codes": codes, "text_fields": text_fields, "anchor_region": anchor}

    # -- side panel -------------------------------------------------------

    def _role_changed(self) -> None:
        self.canvas.draw_role = str(self.role_combo.currentData())

    def _row_selected(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if rows:
            self.canvas.selected = rows.pop()
            self.canvas.update()
            self._sync_side_panel(self.canvas.selected)

    def _sync_side_panel(self, index: int) -> None:
        enabled = 0 <= index < len(self.canvas.regions)
        self.name_edit.setEnabled(enabled)
        self.symbology_combo.setEnabled(enabled and
                                        self.canvas.regions[index]["role"] == "code")
        if not enabled:
            self.name_edit.clear()
            self.feasibility.clear()
            return
        self.name_edit.setText(str(self.canvas.regions[index]["name"]))
        if self.canvas.regions[index]["role"] == "code":
            symbology = self._symbologies.get(index, "datamatrix")
            position = self.symbology_combo.findText(symbology)
            self.symbology_combo.setCurrentIndex(max(0, position))
        self._update_feasibility(index)

    def _rename_selected(self) -> None:
        index = self.canvas.selected
        if 0 <= index < len(self.canvas.regions):
            self.canvas.regions[index]["name"] = self.name_edit.text().strip() or "unnamed"
            self._refresh_table()
            self.canvas.update()

    def _symbology_changed(self) -> None:
        index = self.canvas.selected
        if 0 <= index < len(self.canvas.regions):
            self._symbologies[index] = self.symbology_combo.currentText()
            self._update_feasibility(index)

    def _update_feasibility(self, index: int) -> None:
        """Say how many pixels this code needs, if the print spec is known.

        Optional and off the printed symbol, not the camera: a 10-mil DataMatrix
        wants roughly 3 px per cell wherever it is mounted, and no model recovers
        a code the optics never resolved. Silent when the numbers are not entered.
        """
        region = self.canvas.regions[index]
        if region["role"] != "code":
            self.feasibility.clear()
            return
        width_mm = self._widths.get(index, 0.0)
        x_dim = self._x_dims.get(index, 0.0)
        if not width_mm or not x_dim:
            self.feasibility.setText(
                "Enter this code's printed width and module size on the Code details "
                "page to see how many pixels the camera needs to decode it.")
            return
        symbology = self._symbologies.get(index, "datamatrix")
        per_module = 3.0 if symbology in ("qr", "datamatrix", "aztec") else 2.0
        needed = (width_mm / x_dim) * per_module
        self.feasibility.setText(
            f"{width_mm:g} mm wide at a {x_dim:g} mm module: the camera needs about "
            f"{needed:.0f} px across this code to decode it reliably."
        )

    def _refresh_table(self) -> None:
        self.table.setRowCount(0)
        for index, region in enumerate(self.canvas.regions):
            row = self.table.rowCount()
            self.table.insertRow(row)
            detail = self._symbologies.get(index, "") if region["role"] == "code" else ""
            fraction = self.canvas.to_fraction(region["rect"])
            values = [
                region["role"], region["name"], detail,
                ", ".join(f"{v:.3f}" for v in fraction) if fraction else "-",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        if 0 <= self.canvas.selected < self.table.rowCount():
            self.table.selectRow(self.canvas.selected)

    def accept(self) -> None:
        if self.canvas.image_size() == (0, 0):
            QMessageBox.information(
                self, "Regions",
                "There is no reference image to draw on. Add one on the Appearance "
                "page first.")
            return
        super().accept()
