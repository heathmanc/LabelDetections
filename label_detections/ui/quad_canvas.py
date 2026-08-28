"""Draw a four-corner outline on a photograph.

A label is rarely square-on to the camera, and an axis-aligned rectangle around
one that is not includes a wedge of whatever is behind it on two corners. That
matters more than it looks: the outline is the coordinate system every region
is a fraction of, so a wedge of background in it shifts every region by the
size of the wedge.

So the outline is a quad, and the artwork is the quad rectified -- warped
straight-on. Which is also what makes the regions simple: on a de-skewed
artwork an axis-aligned region is the right shape, and that is exactly what the
runtime maps back onto whatever oriented box the detector produces.

Drag on the image to lay one out, then drag its corners. Nothing here knows
about labels, cameras or files.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

HANDLE_PX = 9
OUTLINE = QColor(96, 165, 250)
HANDLE = QColor(250, 204, 21)


class QuadCanvas(QWidget):
    """An image with one draggable four-corner outline on it."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self._pixmap: QPixmap | None = None
        self.quad: list[list[float]] = []
        self._dragging = -1
        self._drag_from: QPointF | None = None
        self.setMouseTracking(True)

    # -- the image ---------------------------------------------------------

    def set_frame(self, image_bgr) -> None:
        """Show a BGR frame. Clears any outline: it belonged to the old one."""
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            self._pixmap = None
        else:
            import cv2

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            self._pixmap = QPixmap.fromImage(
                QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.quad = []
        self._dragging = -1
        self.changed.emit()
        self.update()

    def image_size(self) -> tuple[int, int]:
        if self._pixmap is None:
            return (0, 0)
        return (self._pixmap.width(), self._pixmap.height())

    def has_quad(self) -> bool:
        return len(self.quad) == 4

    # -- mapping between the widget and the image --------------------------

    def _target(self) -> QRectF:
        """Where the image is drawn, letterboxed into the widget."""
        if self._pixmap is None:
            return QRectF()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_image(self, point: QPoint) -> QPointF:
        box = self._target()
        if box.width() <= 0 or self._pixmap is None:
            return QPointF()
        scale = self._pixmap.width() / box.width()
        return QPointF((point.x() - box.x()) * scale,
                       (point.y() - box.y()) * scale)

    def _to_screen(self, x: float, y: float) -> QPointF:
        box = self._target()
        if self._pixmap is None or self._pixmap.width() <= 0:
            return QPointF()
        scale = box.width() / self._pixmap.width()
        return QPointF(box.x() + x * scale, box.y() + y * scale)

    # -- interaction -------------------------------------------------------

    def _corner_at(self, point: QPoint) -> int:
        for index, (x, y) in enumerate(self.quad):
            if (self._to_screen(x, y) - QPointF(point)).manhattanLength() <= HANDLE_PX * 2:
                return index
        return -1

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        hit = self._corner_at(event.position().toPoint())
        if hit >= 0:
            self._dragging = hit
            return
        self._drag_from = self._to_image(event.position().toPoint())
        self.quad = []
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._pixmap is None:
            return
        point = self._to_image(event.position().toPoint())
        if self._dragging >= 0:
            self.quad[self._dragging] = [point.x(), point.y()]
            self.changed.emit()
            self.update()
            return
        if self._drag_from is not None:
            # Lay out a rectangle while dragging; the corners come after.
            x0, y0 = self._drag_from.x(), self._drag_from.y()
            x1, y1 = point.x(), point.y()
            self.quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging >= 0:
            self._dragging = -1
            self._settle()
            return
        if self._drag_from is not None:
            self._drag_from = None
            if self.has_quad():
                self._settle()
            self.changed.emit()
            self.update()

    def _settle(self) -> None:
        """Canonical order and inside the image, after every change.

        Ordering here rather than at save time because the corner handles are
        positional: leaving a quad wound the other way would mean rectifying it
        into a mirror image, which is a failure that looks like nothing at all
        until a barcode will not decode.
        """
        from ..core.geometry import order_quad

        width, height = self.image_size()
        clamped = [[max(0.0, min(float(width), x)), max(0.0, min(float(height), y))]
                   for x, y in self.quad]
        self.quad = order_quad(clamped)
        self.changed.emit()
        self.update()

    def set_quad(self, quad) -> None:
        self.quad = [[float(x), float(y)] for x, y in (quad or [])[:4]]
        if self.has_quad():
            self._settle()
        else:
            self.changed.emit()
            self.update()

    def clear(self) -> None:
        self.quad = []
        self.changed.emit()
        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 18, 30))
        if self._pixmap is None:
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No frame yet.")
            return
        painter.drawPixmap(self._target(), self._pixmap,
                           QRectF(self._pixmap.rect()))
        if not self.has_quad():
            return
        painter.setPen(QPen(OUTLINE, 2))
        points = [self._to_screen(x, y) for x, y in self.quad]
        for index in range(4):
            painter.drawLine(points[index], points[(index + 1) % 4])
        painter.setPen(QPen(HANDLE, 2))
        for index, point in enumerate(points):
            painter.setBrush(HANDLE if index == 0 else Qt.NoBrush)
            painter.drawRect(QRectF(point.x() - HANDLE_PX / 2,
                                    point.y() - HANDLE_PX / 2,
                                    HANDLE_PX, HANDLE_PX))
        # The first corner filled, so which way up the artwork will come out is
        # visible before it is committed rather than after.
        painter.setPen(QPen(HANDLE))
        painter.drawText(points[0] + QPointF(8, -8), "top-left")
