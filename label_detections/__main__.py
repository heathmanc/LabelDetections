"""``python -m label_detections`` entry point.

Mirrors ``main.py``'s startup contract: freeze_support first, worker dispatch
before Qt. Running the package as a module has to be as safe as the launcher,
or a frozen build started that way spawns GUIs from DataLoader workers.
"""
from __future__ import annotations

import multiprocessing
import os
import sys

multiprocessing.freeze_support()
os.environ.setdefault("MPLBACKEND", "Agg")

from label_detections.worker import maybe_run_worker

_worker_exit = maybe_run_worker(sys.argv)
if _worker_exit is not None:
    raise SystemExit(_worker_exit)

from label_detections.ui.main_window import main

if __name__ == "__main__":
    main()
