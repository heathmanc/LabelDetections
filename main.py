#!/usr/bin/env python3
"""LabelVision Studio launcher.

Use this file directly from the extracted folder:
    python main.py

This launcher forces the extracted application folder onto sys.path so the
bundled label_detections package is found even when launched from a shortcut or
with a different working directory.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

# MUST be the first thing that runs, before any other import or argv handling.
#
# Ultralytics' DataLoader spawns worker processes (the `workers` training
# parameter). On Windows multiprocessing uses the "spawn" start method, which
# re-executes this executable to create each child. In a frozen build that
# means every DataLoader worker would re-run this script and open another copy
# of the GUI, while training stalls waiting for children that never report in.
#
# freeze_support() detects a spawned child, hands control to multiprocessing,
# and never returns -- so the GUI below is only ever reached by a real launch.
multiprocessing.freeze_support()

# Ultralytics pulls in matplotlib. Force the headless Agg backend before any
# import can pick a GUI one: packaged builds exclude tkinter, so a TkAgg
# default would fail at import time. Harmless when running from source -- this
# app never renders matplotlib figures itself.
os.environ.setdefault("MPLBACKEND", "Agg")

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# A frozen build re-invokes itself to run training/evaluation with its own
# bundled Ultralytics (there is no `yolo` CLI or system Python inside a package).
# Dispatch before importing Qt so a worker process never starts a GUI.
from label_detections.worker import maybe_run_worker

_worker_exit = maybe_run_worker(sys.argv)
if _worker_exit is not None:
    raise SystemExit(_worker_exit)

try:
    from label_detections.ui.main_window import main
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "")
    if missing == "label_detections":
        raise SystemExit(
            "LabelVision Studio could not find its bundled 'label_detections' folder.\n"
            "Make sure the zip is fully extracted before running, and run from the "
            "extracted folder."
        ) from exc
    raise

if __name__ == "__main__":
    main()
