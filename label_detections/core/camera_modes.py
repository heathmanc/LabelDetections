"""What sizes a camera will actually give you, instead of remembering them.

Typing a resolution from memory is how a rig ends up running at a size nobody
chose: 1920x1080 into a 2592x1944 sensor is not a smaller picture of the same
scene, it is a crop of the middle of it, and on a Basler that is exactly what
happens because Width and Height are AOI controls rather than display scaling.

A Basler does not have a LIST of supported resolutions. It has a range and a
step: any width between Width.Min and Width.Max that lands on Width.Inc is
valid, and so is any height. So there is nothing to enumerate -- what is worth
offering is the full sensor, and the fractions of it somebody actually wants,
each snapped to the step the camera will accept anyway.

Which is the useful part. The camera silently rounds a value that is off the
step, so a typed 1920 can become 1918 without saying so; picking from a list
that was built from the step means the number asked for is the number applied.
"""
from __future__ import annotations

# What to offer below the full sensor. Halves and thirds because those are the
# ones that keep a whole scene in view while costing less to move and decode --
# a smaller AOI on a Basler is a crop, so these are the sizes that stay useful.
FRACTIONS = (1.0, 0.75, 0.5, 1.0 / 3.0, 0.25)


def snap(value: int, low: int, high: int, step: int) -> int:
    """The nearest value at or below ``value`` the camera will accept.

    Down rather than to-nearest: a size the camera rounds UP is a size bigger
    than the one asked for, and the whole point of choosing from a list is that
    what is chosen is what is applied.
    """
    step = max(1, int(step))
    low, high = int(low), int(high)
    value = max(low, min(int(value), high))
    return low + ((value - low) // step) * step


def offered(width: tuple[int, int, int], height: tuple[int, int, int],
            fractions=FRACTIONS) -> list[tuple[int, int]]:
    """Sizes worth offering, largest first, from a camera's limits.

    ``width`` and ``height`` are each ``(min, max, step)`` as reported by the
    camera. Every size returned is on the step and inside the range, so nothing
    offered can be silently rounded to something else.
    """
    w_min, w_max, w_inc = (int(v) for v in width)
    h_min, h_max, h_inc = (int(v) for v in height)
    if w_max <= 0 or h_max <= 0:
        return []

    seen: list[tuple[int, int]] = []
    for fraction in fractions:
        if fraction <= 0:
            continue
        size = (snap(round(w_max * fraction), w_min, w_max, w_inc),
                snap(round(h_max * fraction), h_min, h_max, h_inc))
        if size[0] >= w_min and size[1] >= h_min and size not in seen:
            seen.append(size)
    return seen


def describe(size: tuple[int, int], full: tuple[int, int]) -> str:
    """One row of the list: the size, and what it is a fraction of.

    Naming the fraction matters more here than it looks. On a sensor whose AOI
    controls are being set, half the width is half the FIELD OF VIEW, not a
    half-scale picture of the same scene -- somebody choosing 1296x972 off a
    2592x1944 sensor is choosing to see a quarter of the battery, and the row
    should say so before they find out on the belt.
    """
    w, h = int(size[0]), int(size[1])
    fw, fh = int(full[0]), int(full[1])
    if not fw or not fh:
        return f"{w} x {h}"
    if (w, h) == (fw, fh):
        return f"{w} x {h}  (full sensor)"
    return f"{w} x {h}  ({w / fw:.0%} of the sensor width)"


def limits_note(width: tuple[int, int, int], height: tuple[int, int, int]) -> str:
    """What the camera said, for somebody who wants a size that is not listed."""
    w_min, w_max, w_inc = (int(v) for v in width)
    h_min, h_max, h_inc = (int(v) for v in height)
    if w_max <= 0 or h_max <= 0:
        return ""
    steps = ""
    if w_inc > 1 or h_inc > 1:
        steps = f", in steps of {w_inc} x {h_inc}"
    return (f"Camera accepts {w_min}-{w_max} wide by {h_min}-{h_max} "
            f"high{steps}. Anything on that grid works; the list is what is "
            f"worth choosing.")


# --- exposure ---------------------------------------------------------------
#
# Exposure has no grid and no natural set of modes, so the picker cannot be
# built the way the size one is. What it has instead is an anchor: the value
# auto exposure settles on for the light actually in the room.
#
# That is the whole workflow on a fixed rig. Let auto find it, read what it
# found, then freeze it -- because auto exposure on a production line is a
# variation nobody asked for. A battery that happens to follow a shiny one gets
# exposed differently from one that follows a matt one, and stage 2 is then
# being asked to learn a lighting difference that carries no information about
# which label it is.

# A bracket around whatever auto chose, for finding the edge of usable. Wide
# enough to see a difference, narrow enough that every step is still a picture.
BRACKET = (0.5, 0.71, 1.0, 1.41, 2.0)


def fps_ceiling(exposure_us: float) -> float:
    """The most frames a second an exposure this long allows. 0 when unbounded.

    The sensor cannot start the next exposure until this one ends, so a 50 ms
    exposure caps the camera at 20/s however it is configured -- and a rig
    asking for 30 fps then quietly runs at 20 with nothing saying why.
    """
    exposure_us = float(exposure_us or 0)
    return (1_000_000.0 / exposure_us) if exposure_us > 0 else 0.0


# Above this the exposure is not what limits the frame rate -- the interface,
# the model or the belt is -- so saying it would be noise on every row.
RATE_WORTH_SAYING = 200.0


def rate_note(exposure_us: float) -> str:
    """", max 60/s" when the exposure is what caps the rate, else "".

    Formatted rather than rounded to zero: a ten second exposure caps the
    camera at 0.1/s, and printing "max 0/s" reads as a fault instead of as the
    perfectly correct consequence of asking for a ten second exposure.
    """
    ceiling = fps_ceiling(exposure_us)
    if not ceiling or ceiling >= RATE_WORTH_SAYING:
        return ""
    return f", max {ceiling:.0f}/s" if ceiling >= 1 else f", max {ceiling:.2g}/s"


def exposure_choices(low: float, high: float, current: float,
                     auto: bool = True, bracket=BRACKET) -> list[tuple[str, int]]:
    """``(label, microseconds)`` rows for the picker, fastest first.

    Built around ``current`` rather than around round numbers, because the
    number that matters is the one that suits this light, and only the camera
    knows it. The ends of the range come along so a value can be typed knowing
    what it will be clamped to.

    ``auto`` only names the anchor row. It is worth naming truthfully: "what
    auto settled on" is a reading taken from the light in the room, while "what
    it is set to now" is a number somebody typed earlier, and only the first of
    those is evidence about anything.
    """
    low, high = float(low or 0), float(high or 0)
    if high <= 0:
        return []
    rows: list[tuple[str, int]] = []
    seen: set[int] = set()

    def add(label: str, value: float) -> None:
        clamped = int(round(max(low, min(float(value), high))))
        if clamped in seen:
            return
        seen.add(clamped)
        rows.append((f"{label}  --  {clamped:,} us{rate_note(clamped)}", clamped))

    # Order of adding decides which label a value keeps when two rows land on
    # the same number, and two collisions are worth getting right. A bracket
    # step past the end of the range clamps onto it, and calling the camera's
    # ceiling "2x that" loses the only row that says where the ceiling is. So
    # the reading goes in first -- it is the point of the whole picker -- then
    # the ends, then the steps around it, which are the rows that can be spared.
    if current > 0:
        add("what auto settled on" if auto else "what it is set to now", current)
    add("shortest the camera takes", low)
    add("longest the camera takes", high)
    if current > 0:
        for factor in sorted(bracket):
            if factor != 1.0:
                add(f"{factor:g}x that", current * factor)
    rows.sort(key=lambda row: row[1])
    return rows


def exposure_note(limits: dict, wanted_fps: float = 0.0) -> str:
    """What the camera said about exposure, and what it costs in frame rate."""
    if not limits:
        return ""
    low, high = float(limits.get("min", 0)), float(limits.get("max", 0))
    current = float(limits.get("current", 0))
    if high <= 0:
        return ""
    parts = [f"Camera takes {low:,.0f} to {high:,.0f} us"]
    if current > 0:
        parts.append(("auto is on and has settled at" if limits.get("auto")
                      else "currently") + f" {current:,.0f} us")
    ceiling = fps_ceiling(current)
    if current > 0 and wanted_fps and ceiling and wanted_fps > ceiling:
        parts.append(f"which caps the camera at {ceiling:.0f}/s -- below the "
                     f"{wanted_fps:.0f}/s asked for")
    return "; ".join(parts) + "."
