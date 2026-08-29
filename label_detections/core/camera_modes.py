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
