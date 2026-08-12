"""Class-filter precedence: names identify a class, IDs are only a fallback.

Exercised through the real MainWindow under the offscreen platform, with a
stub standing in for an Ultralytics result.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["BUNGVISION_DATA_DIR"] = tempfile.mkdtemp(prefix="bungvision-filters-")

try:
    import numpy as np
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception as exc:  # pragma: no cover
    HAVE_QT = False
    _WHY = exc

_SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]
_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from bung_labeler.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


class _Arr:
    def __init__(self, a): self.a = np.array(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _OBB:
    def __init__(self, polys, confs, clss):
        self.xyxyxyxy = _Arr(polys)
        self.conf = _Arr(confs)
        self.cls = _Arr(clss)
        self.xywhr = _Arr([])


class _Result:
    def __init__(self, names, obb):
        self.names, self.obb, self.boxes = names, obb, None


def _labels(names: dict, class_ids: list[int]) -> list[str]:
    """Run one fake detection set through the real auto-label pipeline."""
    win = _window()
    result = _Result(names, _OBB([_SQUARE] * len(class_ids),
                                 [0.9] * len(class_ids), class_ids))
    battery, count, _ = win._battery_obb_overlay_items([result])
    if count == 0:
        battery, count, _ = win._battery_box_overlay_items([result])
    bungs, _n = win._bung_overlay_items([result])
    other, _c = win._other_overlay_items([result])
    return [b["label"] for b in win._overlay_items_to_box_dicts(battery + bungs + other)]


def test_one_detection_yields_one_label():
    assert _labels({0: "battery_sealed"}, [0]) == ["battery_sealed"]
    assert _labels({0: "battery"}, [0]) == ["battery"]


def test_sealed_battery_at_class_id_one_is_not_also_a_bung():
    # Exporting battery + battery_sealed makes battery_sealed class 1, and the
    # default count filter is "bung,1". Matching that by ID counted the same
    # object as both a battery and a bung -- two labels per battery.
    assert _labels({0: "battery", 1: "battery_sealed"}, [1]) == ["battery_sealed"]


def test_each_detection_produces_exactly_one_label():
    for names, ids in (
        ({0: "battery", 1: "battery_sealed"}, [0, 1]),
        ({0: "battery", 1: "bung"}, [0, 1, 1]),
        ({0: "battery_sealed", 1: "bung"}, [0, 1]),
    ):
        assert len(_labels(names, ids)) == len(ids), (names, ids)


def test_normal_battery_and_bung_model_is_unaffected():
    assert _labels({0: "battery", 1: "bung"}, [0, 1, 1]) == ["battery", "bung", "bung"]


def test_unrecognised_class_is_labelled_with_its_real_name():
    # The ID fallback still routes an unrecognised class into the bung bucket
    # for counting, but the label must be the model's own class name -- writing
    # "bung" would silently mislabel the training data.
    assert _labels({0: "battery", 1: "rubber_plug"}, [1]) == ["rubber_plug"]


if __name__ == "__main__":
    import traceback

    if not HAVE_QT:
        print(f"SKIP: PySide6 unavailable ({_WHY})")
        raise SystemExit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
