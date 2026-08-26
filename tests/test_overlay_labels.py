"""Tests for mapping model detections to editable labels (headless, no Qt).

The method is extracted from main_window.py's AST and bound to a stub, so this
exercises the real shipped code without importing PySide6.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MW = Path(__file__).resolve().parents[1] / "label_detections" / "ui" / "main_window.py"


def _stub(class_names=("battery_side", "spec_plate", "warning_label")):
    fn = next(n for n in ast.walk(ast.parse(_MW.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "_label_for_overlay_item")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "m", "exec"), ns)

    class Stub:
        _label_for_overlay_item = ns["_label_for_overlay_item"]

    s = Stub()
    s.class_names = list(class_names)
    return s


def test_configured_families_use_configured_ids():
    s = _stub()
    assert s._label_for_overlay_item(
        {"type": "other_obb", "name": "battery_side"}) == ("battery_side", 0)
    assert s._label_for_overlay_item(
        {"type": "other_obb", "name": "spec_plate"}) == ("spec_plate", 1)
    assert s._label_for_overlay_item(
        {"type": "other_obb", "name": "warning_label"}) == ("warning_label", 2)


def test_unknown_family_keeps_its_own_name():
    # A newly trained family must be labelable before it is added to the class
    # config, otherwise it cannot be auto-labelled or validated at all.
    s = _stub()
    assert s._label_for_overlay_item(
        {"type": "other_obb", "name": "hologram_seal", "cls_id": 3}
    ) == ("hologram_seal", 3)


def test_a_similar_name_is_not_absorbed_into_a_configured_family():
    # "spec_plate_backup" is not "spec_plate". The label comes from the model's
    # own class name, never from a prefix match.
    s = _stub()
    assert s._label_for_overlay_item(
        {"type": "other_obb", "name": "spec_plate_backup", "cls_id": 3}
    ) == ("spec_plate_backup", 3)


def test_legacy_item_with_only_a_display_label():
    s = _stub()
    assert s._label_for_overlay_item(
        {"type": "other_obb", "label": "spec_plate 0.94"}) == ("spec_plate", 1)


def test_type_is_the_last_resort():
    s = _stub()
    assert s._label_for_overlay_item({"type": "spec_plate_box"}) == ("spec_plate", 1)


def test_unusable_item_is_skipped_not_guessed():
    s = _stub()
    assert s._label_for_overlay_item({"type": "other_obb"}) == ("", -1)
    assert s._label_for_overlay_item({}) == ("", -1)


def test_name_match_is_case_insensitive():
    s = _stub()
    assert s._label_for_overlay_item({"name": "Spec_Plate"}) == ("spec_plate", 1)


def test_missing_cls_id_falls_back_to_minus_one():
    s = _stub()
    assert s._label_for_overlay_item({"name": "brand_new_class"}) == ("brand_new_class", -1)


if __name__ == "__main__":
    import traceback

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
