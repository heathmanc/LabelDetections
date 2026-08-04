"""Unit tests for the GPU wheel-selection logic (headless-safe, no torch needed)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import setup_gpu as g


def test_blackwell_gets_cu128():
    # RTX 50-series is sm_120 and REQUIRES cu128; this is the deployment target.
    url, _ = g.wheel_index_for_capability((12, 0))
    assert url == g.CU128


def test_ada_and_hopper_get_cu124():
    assert g.wheel_index_for_capability((8, 9))[0] == g.CU124   # RTX 40-series
    assert g.wheel_index_for_capability((9, 0))[0] == g.CU124   # H100


def test_ampere_gets_cu124():
    assert g.wheel_index_for_capability((8, 6))[0] == g.CU124   # RTX 30-series
    assert g.wheel_index_for_capability((8, 0))[0] == g.CU124   # A100


def test_turing_gets_cu121():
    assert g.wheel_index_for_capability((7, 5))[0] == g.CU121   # RTX 20-series


def test_pascal_and_older_rejected():
    for cap in ((6, 1), (5, 0), (3, 5)):
        try:
            g.wheel_index_for_capability(cap)
        except ValueError as exc:
            assert "too old" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {cap}")


def test_future_arch_still_maps_to_newest():
    # An arch newer than Blackwell should not fall through to an old wheel.
    assert g.wheel_index_for_capability((13, 0))[0] == g.CU128


def test_parse_compute_cap():
    assert g.parse_compute_cap("12.0") == (12, 0)
    assert g.parse_compute_cap("  8.9 \n") == (8, 9)
    assert g.parse_compute_cap("") is None
    assert g.parse_compute_cap("N/A") is None


def test_capability_from_name_fallback():
    assert g.capability_from_name("NVIDIA GeForce RTX 5090") == (12, 0)
    assert g.capability_from_name("NVIDIA GeForce RTX 5070 Ti") == (12, 0)
    assert g.capability_from_name("NVIDIA GeForce RTX 4090") == (8, 9)
    assert g.capability_from_name("NVIDIA GeForce RTX 3080") == (8, 6)
    assert g.capability_from_name("Quadro P2000") is None


def test_name_fallback_agrees_with_capability_mapping():
    # The 50-series name hint must route to the same wheel as a real sm_120 read.
    cap = g.capability_from_name("NVIDIA GeForce RTX 5080")
    assert g.wheel_index_for_capability(cap)[0] == g.CU128


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
