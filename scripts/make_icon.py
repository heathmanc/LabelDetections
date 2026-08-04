#!/usr/bin/env python3
"""Generate the BungVision Label Studio application icon.

Renders procedurally with stdlib only (no Pillow) and writes a multi-resolution
Windows .ico. Entries are classic BMP/DIB rather than PNG-in-ICO, which is what
PyInstaller's resource embedder and every Windows version handle reliably.

    python scripts/make_icon.py                 # -> bung_labeler/ui/assets/app.ico
    python scripts/make_icon.py --preview       # also write PNG previews

Design: a dark viewfinder framing a battery plate of six bungs -- the app's
subject (batteries/bungs) inside its function (detection framing). Six matches
the default expected_bungs. Shapes are kept chunky so the 16px entry stays
legible in the taskbar.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

# Palette lifted from the app's own dark theme so the icon matches the UI.
BG_OUTER = (15, 23, 42)      # #0f172a  window background
BG_INNER = (30, 41, 59)      # #1e293b  panel background
BRACKET = (96, 165, 250)     # #60a5fa  detection-blue
PLATE = (71, 85, 105)        # #475569  battery plate
PLATE_EDGE = (148, 163, 184)  # #94a3b8  plate highlight
BUNG = (245, 158, 11)        # #f59e0b  amber bung
BUNG_EDGE = (253, 224, 71)   # #fde047  bung highlight

SIZES = (16, 24, 32, 48, 64, 128, 256)
SS = 4  # supersampling factor for anti-aliasing


def _rounded_rect(x: float, y: float, cx: float, cy: float, w: float, h: float, r: float) -> bool:
    """Point-in-rounded-rectangle, all args normalized to the 0..1 canvas."""
    dx = abs(x - cx) - (w / 2 - r)
    dy = abs(y - cy) - (h / 2 - r)
    if dx <= 0 and dy <= 0:
        return True
    if dx > 0 and dy > 0:
        return (dx * dx + dy * dy) <= r * r
    return (dx <= 0 and dy <= r) or (dy <= 0 and dx <= r)


def _circle(x: float, y: float, cx: float, cy: float, rad: float) -> bool:
    return (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad


def _bracket(x: float, y: float, small: bool = False) -> bool:
    """Four L-shaped viewfinder corners, reading as 'detection frame'."""
    if small:
        inset, arm, thick = 0.085, 0.26, 0.10
    else:
        inset, arm, thick = 0.115, 0.20, 0.062
    lo, hi = inset, 1.0 - inset
    for bx, sx in ((lo, 1), (hi, -1)):
        for by, sy in ((lo, 1), (hi, -1)):
            # Horizontal arm.
            x0, x1 = sorted((bx, bx + sx * arm))
            y0, y1 = sorted((by, by + sy * thick))
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
            # Vertical arm.
            x0, x1 = sorted((bx, bx + sx * thick))
            y0, y1 = sorted((by, by + sy * arm))
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
    return False


def _sample(x: float, y: float, small: bool = False) -> tuple[int, int, int, int]:
    """Painter's-algorithm scene sample at normalized (x, y). Returns RGBA.

    ``small`` switches to a simplified composition for 32px and below: fewer,
    larger bungs, a bigger plate, chunkier brackets, and no hairline highlights.
    At those sizes the detailed version collapses into an unreadable blob.
    """
    # Background: rounded square with a soft vertical gradient.
    if not _rounded_rect(x, y, 0.5, 0.5, 0.96, 0.96, 0.20):
        return (0, 0, 0, 0)
    t = y  # 0 at top -> 1 at bottom
    rgb = tuple(int(BG_INNER[i] + (BG_OUTER[i] - BG_INNER[i]) * t) for i in range(3))

    if _bracket(x, y, small):
        rgb = BRACKET

    if small:
        # Four fat bungs on a large plate: survives downsampling to 16px.
        if _rounded_rect(x, y, 0.5, 0.5, 0.56, 0.50, 0.09):
            rgb = PLATE
        for col in range(2):
            for row in range(2):
                cx = 0.5 + (col - 0.5) * 0.235
                cy = 0.5 + (row - 0.5) * 0.215
                if _circle(x, y, cx, cy, 0.088):
                    rgb = BUNG
        return (rgb[0], rgb[1], rgb[2], 255)

    # Battery plate.
    if _rounded_rect(x, y, 0.5, 0.5, 0.50, 0.38, 0.07):
        rgb = PLATE
        if not _rounded_rect(x, y, 0.5, 0.5, 0.465, 0.345, 0.055):
            rgb = PLATE_EDGE  # thin rim highlight

    # Six bungs: 3 columns x 2 rows, matching the default expected count.
    for col in range(3):
        for row in range(2):
            cx = 0.5 + (col - 1) * 0.148
            cy = 0.5 + (row - 0.5) * 0.155
            if _circle(x, y, cx, cy, 0.052):
                rgb = BUNG
                if not _circle(x, y, cx, cy, 0.038):
                    rgb = BUNG_EDGE

    return (rgb[0], rgb[1], rgb[2], 255)


# At or below this pixel size the simplified composition is used.
SMALL_MAX = 32


def render(size: int) -> list[list[tuple[int, int, int, int]]]:
    """Render one size as a row-major RGBA grid, supersampled for smooth edges."""
    small = size <= SMALL_MAX
    rows = []
    inv = 1.0 / (size * SS)
    for py in range(size):
        row = []
        for px in range(size):
            r = g = b = a = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = (px * SS + sx + 0.5) * inv
                    y = (py * SS + sy + 0.5) * inv
                    sr, sg, sb, sa = _sample(x, y, small)
                    # Weight colour by coverage so edges blend correctly.
                    r += sr * sa; g += sg * sa; b += sb * sa; a += sa
            if a == 0:
                row.append((0, 0, 0, 0))
            else:
                row.append((r // a, g // a, b // a, a // (SS * SS)))
        rows.append(row)
    return rows


def _dib(grid) -> bytes:
    """Pack an RGBA grid as an ICO-embedded BMP: header + BGRA + AND mask."""
    h = len(grid)
    w = len(grid[0])
    # biHeight is doubled: the DIB holds the colour bitmap plus the AND mask.
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, w * h * 4, 0, 0, 0, 0)

    pixels = bytearray()
    for row in reversed(grid):  # DIB rows run bottom-up
        for (r, g, b, a) in row:
            pixels += bytes((b, g, r, a))

    # AND mask: 1bpp, rows padded to 4-byte boundaries. Alpha carries the real
    # transparency, but the mask must still be present and correctly sized.
    row_bytes = ((w + 31) // 32) * 4
    mask = bytearray()
    for row in reversed(grid):
        bits = bytearray(row_bytes)
        for x, (_r, _g, _b, a) in enumerate(row):
            if a == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        mask += bits

    return header + bytes(pixels) + bytes(mask)


def write_ico(path: Path, sizes=SIZES) -> None:
    images = [_dib(render(s)) for s in sizes]
    out = bytearray(struct.pack("<HHH", 0, 1, len(images)))  # ICONDIR
    offset = 6 + 16 * len(images)
    for size, data in zip(sizes, images):
        dim = 0 if size >= 256 else size  # 0 encodes 256 in ICO
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    for data in images:
        out += data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def write_png(path: Path, size: int) -> None:
    """Minimal RGBA PNG, for previewing the icon outside Windows."""
    grid = render(size)
    raw = bytearray()
    for row in grid:
        raw.append(0)  # filter type 0
        for (r, g, b, a) in row:
            raw += bytes((r, g, b, a))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bung_labeler/ui/assets/app.ico")
    ap.add_argument("--preview", action="store_true", help="also write PNG previews")
    args = ap.parse_args()

    out = Path(args.out)
    write_ico(out)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes, sizes: {', '.join(map(str, SIZES))})")

    if args.preview:
        for s in (64, 256):
            p = out.with_name(f"app_preview_{s}.png")
            write_png(p, s)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
