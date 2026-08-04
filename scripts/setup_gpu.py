#!/usr/bin/env python3
"""Detect the installed NVIDIA GPU and install the matching PyTorch build.

Motivation
----------
``pip install torch`` gives a CPU-only wheel on Windows, and even the CUDA
wheels only contain compiled kernels for a fixed set of GPU architectures.
Installing a cu124 wheel on a Blackwell (RTX 50-series, sm_120) card produces:

    CUDA error: no kernel image is available for execution on the device

...at the first real kernel launch, *after* ``torch.cuda.is_available()`` has
already returned True. This script picks the right wheel index up front.

Usage
-----
    python scripts/setup_gpu.py            # detect, install, verify
    python scripts/setup_gpu.py --dry-run  # show what it would do
    python scripts/setup_gpu.py --check    # verify an existing install only
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Minimum torch version that ships kernels for each architecture family.
# Blackwell (sm_120) support landed in torch 2.7 / CUDA 12.8.
CU128 = "https://download.pytorch.org/whl/cu128"
CU124 = "https://download.pytorch.org/whl/cu124"
CU121 = "https://download.pytorch.org/whl/cu121"


def wheel_index_for_capability(cap: tuple[int, int]) -> tuple[str, str]:
    """Map a CUDA compute capability to (wheel_index_url, human_reason).

    Raises ValueError for architectures no current PyTorch build supports.
    """
    major, minor = cap
    sm = major * 10 + minor

    if sm >= 120:
        # Blackwell: RTX 50-series, B100/B200. Needs CUDA 12.8 + torch >= 2.7.
        return CU128, f"sm_{sm} (Blackwell) requires CUDA 12.8 wheels"
    if sm >= 89:
        # Ada (sm_89) / Hopper (sm_90).
        return CU124, f"sm_{sm} (Ada/Hopper) works with CUDA 12.4 wheels"
    if sm >= 80:
        # Ampere: RTX 30-series, A100.
        return CU124, f"sm_{sm} (Ampere) works with CUDA 12.4 wheels"
    if sm >= 75:
        # Turing: RTX 20-series, GTX 16-series.
        return CU121, f"sm_{sm} (Turing) works with CUDA 12.1 wheels"
    if sm >= 70:
        # Volta.
        return CU121, f"sm_{sm} (Volta) works with CUDA 12.1 wheels"

    raise ValueError(
        f"sm_{sm} is too old for current PyTorch builds (Pascal and earlier were "
        f"dropped). Pin an older torch, or run on CPU by setting the device "
        f"field to 'cpu' in the Model Test and Train tabs."
    )


def parse_compute_cap(text: str) -> tuple[int, int] | None:
    """Parse the compute capability from `nvidia-smi --query-gpu=compute_cap`."""
    m = re.search(r"(\d+)\.(\d+)", (text or "").strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# GPU-name fallback for drivers too old to support --query-gpu=compute_cap.
_NAME_HINTS = (
    (re.compile(r"RTX\s*50\d\d", re.I), (12, 0)),
    (re.compile(r"RTX\s*40\d\d", re.I), (8, 9)),
    (re.compile(r"RTX\s*30\d\d", re.I), (8, 6)),
    (re.compile(r"RTX\s*20\d\d", re.I), (7, 5)),
    (re.compile(r"GTX\s*16\d\d", re.I), (7, 5)),
)


def capability_from_name(name: str) -> tuple[int, int] | None:
    for pattern, cap in _NAME_HINTS:
        if pattern.search(name or ""):
            return cap
    return None


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else None
    except Exception:
        return None


def detect() -> tuple[str, tuple[int, int]]:
    """Return (gpu_name, compute_capability). Exits with guidance on failure."""
    name = _nvidia_smi("name")
    if name is None:
        sys.exit(
            "ERROR: nvidia-smi not found or returned no GPU.\n"
            "  - Confirm an NVIDIA GPU is present and the driver is installed.\n"
            "  - Run 'nvidia-smi' manually to see the underlying error.\n"
            "  - Without a working driver, no PyTorch CUDA build will help."
        )

    cap = parse_compute_cap(_nvidia_smi("compute_cap") or "")
    source = "nvidia-smi"
    if cap is None:
        cap = capability_from_name(name)
        source = "GPU name lookup (driver too old for --query-gpu=compute_cap)"
    if cap is None:
        sys.exit(
            f"ERROR: could not determine compute capability for {name!r}.\n"
            "Update the NVIDIA driver, or install torch manually from\n"
            "https://pytorch.org/get-started/locally/"
        )

    driver = _nvidia_smi("driver_version") or "unknown"
    print(f"GPU        : {name}")
    print(f"Driver     : {driver}")
    print(f"Capability : sm_{cap[0] * 10 + cap[1]}  (via {source})")
    return name, cap


def verify() -> bool:
    """Launch a real CUDA kernel. is_available() alone does NOT catch sm mismatch."""
    code = (
        "import torch;"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda);"
        "print('archs', torch.cuda.get_arch_list());"
        "assert torch.cuda.is_available(), 'torch.cuda.is_available() is False';"
        "a = torch.randn(64, 64, device='cuda');"
        "b = (a @ a).sum().item();"
        "print('kernel launch OK, result', b)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr.strip(), file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="show the pip command without running it")
    ap.add_argument("--check", action="store_true", help="only verify the current install")
    args = ap.parse_args()

    if args.check:
        print("=== Verifying existing install ===")
        ok = verify()
        print("\nRESULT:", "CUDA is working." if ok else "CUDA is NOT working.")
        return 0 if ok else 1

    print("=== Detecting GPU ===")
    _name, cap = detect()

    try:
        index_url, reason = wheel_index_for_capability(cap)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Wheel index: {index_url}\n  reason: {reason}\n")

    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall",
           "torch", "torchvision", "--index-url", index_url]

    if args.dry_run:
        print("DRY RUN, would execute:\n  " + " ".join(cmd))
        return 0

    print("=== Installing PyTorch (this downloads ~2.5 GB) ===")
    # --force-reinstall so an existing CPU-only or wrong-arch wheel is replaced.
    if subprocess.run(cmd).returncode != 0:
        sys.exit("ERROR: pip install failed. See the output above.")

    print("\n=== Verifying ===")
    ok = verify()
    if not ok:
        print(
            "\nCUDA still not working. Next steps:\n"
            "  - Update the NVIDIA driver (a 50-series card needs a recent one).\n"
            "  - Re-run with --check after updating.\n"
            "  - As a stopgap, set the device field to 'cpu' in the app.",
            file=sys.stderr,
        )
        return 1
    print("\nRESULT: CUDA is working. Set the device field to '0' in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
