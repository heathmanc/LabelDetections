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

# A native crash -- a segfault in torch, CUDA or Qt -- ends the process with no
# Python traceback at all. From outside that is indistinguishable from a clean
# exit, and "it just closes" is the whole of what anyone can report. faulthandler
# prints the C-level stack for every thread when it happens, which turns that
# into something diagnosable. Written to a file as well as stderr, because a
# console window launched from a shortcut disappears with the process.
try:
    import faulthandler

    faulthandler.enable()
    _crash_log = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "labelvision_crash.log"
    try:
        _crash_fp = open(_crash_log, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_crash_fp, all_threads=True)
    except Exception:
        pass
except Exception:
    pass

# Ultralytics posts anonymised usage events to its own endpoint on a background
# thread. A py-spy dump of this application showed that thread sitting in an SSL
# handshake, which is what it looks like on a line with no route out: a thread
# that exists to phone home, blocked, in a tool running production hardware.
# It is not why anything hung, and it is not something to find out about from a
# stack dump either. YOLO_OFFLINE is Ultralytics' own switch (utils.is_online),
# read at import, so it must be set before the first ultralytics import.
os.environ.setdefault("YOLO_OFFLINE", "True")

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
