#!/usr/bin/env python3
"""Diagnose "the saved capture looks darker than the live preview".

The live view renders the *adjusted* frame; a raw capture writes the
*unadjusted* one. If a recipe's adjustment sliders are not neutral, the two
genuinely differ and the raw file is legitimately darker than what was on
screen. This prints the recipe's actual slider values and measures how much
they would brighten a real capture, turning "looks darker" into a number.

    python scripts/check_capture_brightness.py
    python scripts/check_capture_brightness.py --recipe "Default__Battery_Model"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEUTRAL = {
    "brightness": 0,
    "contrast": 0,
    "gamma": 1.0,
    "clahe_enabled": False,
    "sharpen": 0,
}


def _data_dir() -> Path:
    """User data folder: beside the exe when installed, else the repo."""
    local = Path.home() / "AppData" / "Local" / "LabelVisionStudio" / "data"
    if local.is_dir():
        return local
    return ROOT / "data"


def recipe_settings(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ! could not read {path.name}: {exc}")
        return {}


def report_recipe(path: Path) -> bool:
    """Print a recipe's adjustment values. Returns True if any are non-neutral."""
    data = recipe_settings(path)
    if not data:
        return False
    drift = {k: data.get(k) for k, v in NEUTRAL.items() if data.get(k, v) != v}
    print(f"\n  {path.stem}")
    for key, neutral in NEUTRAL.items():
        value = data.get(key, neutral)
        mark = "  <-- NOT NEUTRAL" if value != neutral else ""
        print(f"    {key:15} = {value!r}{mark}")
    for extra in ("clahe_clip", "clahe_grid"):
        if extra in data:
            print(f"    {extra:15} = {data[extra]!r}")
    return bool(drift)


def measure(image_path: Path, settings: dict) -> None:
    """Compare a capture's brightness before and after the recipe adjustments."""
    try:
        import cv2
        import numpy as np
        from label_detections.core.image_adjust import apply_adjustments
    except Exception as exc:
        print(f"\n  (skipping pixel measurement: {exc})")
        return

    raw = cv2.imread(str(image_path))
    if raw is None:
        print(f"  ! could not decode {image_path.name}")
        return

    adjusted = apply_adjustments(
        raw,
        brightness=int(settings.get("brightness", 0)),
        contrast=int(settings.get("contrast", 0)),
        gamma=float(settings.get("gamma", 1.0)),
        clahe_enabled=bool(settings.get("clahe_enabled", False)),
        clahe_clip=float(settings.get("clahe_clip", 2.0)),
        clahe_grid=int(settings.get("clahe_grid", 8)),
        sharpen=int(settings.get("sharpen", 0)),
    )
    raw_mean = float(np.mean(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)))
    adj_mean = float(np.mean(cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)))
    delta = adj_mean - raw_mean

    print(f"\n  {image_path.name}")
    print(f"    saved file (raw)      mean luminance {raw_mean:6.2f} / 255")
    print(f"    what the preview shows              {adj_mean:6.2f} / 255")
    if abs(delta) < 0.5:
        print("    -> identical. The preview and a raw capture are the same pixels,")
        print("       so any difference you see is in how the file is being viewed,")
        print("       not in the file. Check Windows HDR and your image viewer.")
    else:
        pct = (delta / max(raw_mean, 1.0)) * 100.0
        print(f"    -> preview is {delta:+.2f} ({pct:+.1f}%) vs the saved raw file.")
        print("       This is the difference you are seeing. Capture adjusted, or")
        print("       reset the sliders on the Contrast tab.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", help="recipe stem to inspect (default: all)")
    ap.add_argument("--image", help="specific capture to measure")
    args = ap.parse_args()

    data_dir = _data_dir()
    print(f"Data folder: {data_dir}")

    recipe_dir = data_dir / "recipes"
    recipes = sorted(recipe_dir.glob("*.json")) if recipe_dir.is_dir() else []
    if args.recipe:
        recipes = [p for p in recipes if p.stem == args.recipe]
    if not recipes:
        print(f"No recipes found in {recipe_dir}")
        return 1

    print("\n=== Recipe adjustment settings ===")
    any_drift = False
    for path in recipes:
        any_drift |= report_recipe(path)

    print("\n=== Verdict ===")
    if any_drift:
        print("  At least one recipe has NON-NEUTRAL adjustments. The live view shows")
        print("  the adjusted frame, so a RAW capture will look darker than the screen.")
    else:
        print("  All recipes are neutral. Preview and raw capture are the same pixels,")
        print("  so a visible difference is in the viewer, not the saved file.")

    print("\n=== Pixel measurement ===")
    settings = recipe_settings(recipes[0])
    if args.image:
        measure(Path(args.image), settings)
    else:
        caps = data_dir / "captures"
        images = sorted(caps.rglob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True) if caps.is_dir() else []
        if not images:
            print(f"  No captures found under {caps}")
        for img in images[:3]:
            measure(img, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
