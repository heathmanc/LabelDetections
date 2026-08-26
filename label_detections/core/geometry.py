"""Pure geometry for oriented boxes and battery-side rectification.

Stdlib only -- no numpy, no OpenCV -- so the inspection logic that depends on
it stays unit testable without a display or an image stack. The runtime may
well hand the same quads to ``cv2.warpPerspective``; these functions exist so
the *decisions* (is this label inside its zone, where does its barcode sit)
can be made and tested independently of that.

Coordinate conventions
----------------------
* An oriented box ("quad") is four ``[x, y]`` points in clockwise order
  starting top-left: TL, TR, BR, BL. This matches the sidecar format.
* "Label space" is one label's own flattened artwork as a **unit square**:
  ``[0, 0]`` is its top-left corner, ``[1, 1]`` its bottom-right. Barcode and
  text sub-regions live here, so they follow the label around.

  Deliberately unitless. A homography maps the label's rectangle onto whatever
  quad an operator drew, and it does not care whether that rectangle was
  measured in millimetres, pixels or fractions -- only its proportions matter.
  Fractions mean nothing has to be measured or calibrated to place a region.
"""
from __future__ import annotations

import math

Point = list[float]
Quad = list[Point]
Matrix = list[list[float]]


# --- containment / overlap -------------------------------------------------

def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon. Works for any simple polygon."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-12
            if x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned ``(x, y, w, h)`` rects."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --- quad helpers ----------------------------------------------------------

def rect_corners(x: float, y: float, w: float, h: float) -> Quad:
    """Axis-aligned rect as a clockwise TL/TR/BR/BL quad."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def quad_centroid(quad: Quad) -> tuple[float, float]:
    n = len(quad) or 1
    return (sum(p[0] for p in quad) / n, sum(p[1] for p in quad) / n)


def quad_bounds(quad: Quad) -> tuple[float, float, float, float]:
    """Enclosing axis-aligned ``(x, y, w, h)``."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def quad_edge_lengths(quad: Quad) -> list[float]:
    """The four side lengths, starting with TL->TR."""
    out = []
    for i in range(4):
        x0, y0 = quad[i]
        x1, y1 = quad[(i + 1) % 4]
        out.append(math.hypot(x1 - x0, y1 - y0))
    return out


def quad_size(quad: Quad) -> tuple[float, float]:
    """Mean (width, height) of the quad, averaging opposite edges.

    Averaging tolerates the slight keystone a camera puts on a label that sits
    on a curved battery side, where the two widths differ by a few percent.
    """
    e = quad_edge_lengths(quad)
    return (e[0] + e[2]) / 2.0, (e[1] + e[3]) / 2.0


def quad_angle_deg(quad: Quad) -> float:
    """Rotation of the quad's top edge, in degrees, in [-180, 180).

    Zero means the label reads left-to-right the same way its reference
    artwork does, which is what a rotation tolerance is measured against.
    """
    (x0, y0), (x1, y1) = quad[0], quad[1]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return ((angle + 180.0) % 360.0) - 180.0


def angle_delta_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


# --- homography ------------------------------------------------------------

def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None when singular."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        pval = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= pval
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def homography_from_points(src: Quad, dst: Quad) -> Matrix | None:
    """3x3 homography mapping four ``src`` points onto four ``dst`` points.

    Returns None for degenerate input (collinear or coincident corners), which
    is what a bad or half-drawn annotation looks like. Callers treat that as
    "cannot rectify" rather than crashing an inspection.
    """
    if len(src) < 4 or len(dst) < 4:
        return None
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (u, v), (x, y) in zip(src[:4], dst[:4]):
        u, v, x, y = float(u), float(v), float(x), float(y)
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        rhs.append(x)
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        rhs.append(y)
    h = _solve(rows, rhs)
    if h is None:
        return None
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_homography(h: Matrix, x: float, y: float) -> tuple[float, float]:
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-12:
        w = 1e-12
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def map_quad(h: Matrix, quad: Quad) -> Quad:
    return [list(apply_homography(h, p[0], p[1])) for p in quad]


UNIT_QUAD: Quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def homography_to_unit(quad: Quad) -> Matrix | None:
    """Image -> label space: flatten ``quad`` onto the unit square.

    Feed it a label's four drawn corners and every coordinate inside it becomes
    a fraction of that label, independent of angle, distance or resolution.
    """
    return homography_from_points(quad, UNIT_QUAD)


def homography_from_unit(quad: Quad) -> Matrix | None:
    """Label space -> image. The inverse direction of the above."""
    return homography_from_points(UNIT_QUAD, quad)


def place_unit_rect(quad: Quad, rect: list[float]) -> Quad | None:
    """Where a region defined on flat artwork lands on a drawn label.

    This is what puts a barcode box on screen without anyone drawing it. The
    library knows the code occupies ``rect`` -- ``[x, y, w, h]`` as fractions of
    the label -- the operator drew the label's four corners, and the code's
    image quad follows.

    Nothing is measured and nothing is calibrated: the mapping is pure
    proportion, so it holds at any distance and any angle.
    """
    h = homography_from_unit(quad)
    if h is None or len(rect) < 4:
        return None
    x, y, w, hh = (float(v) for v in rect[:4])
    if w <= 0 or hh <= 0:
        return None
    return map_quad(h, rect_corners(x, y, w, hh))
