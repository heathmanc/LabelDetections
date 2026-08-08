"""Worker entry points used when the app is frozen.

A packaged build bundles Ultralytics but has no ``yolo`` CLI and no system
Python, so Training and Evaluate cannot shell out the way they do from source.
Instead the frozen executable re-invokes *itself* with a worker flag and runs
the job with its own bundled Ultralytics.

main.py dispatches here before any Qt import, so a worker process never starts
a GUI.
"""
from __future__ import annotations

import os
import sys


def _ensure_std_streams() -> None:
    """Guarantee usable sys.stdout/sys.stderr in a windowed build.

    PyInstaller's windowed bootloader leaves sys.stdout as None because there is
    no console. When QProcess spawns us the OS handles are real pipes, so we can
    reopen them by file descriptor -- without this the parent's log pane stays
    empty. Line buffering keeps training progress streaming instead of arriving
    in one block at exit.
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name, None) is not None:
            continue
        try:
            stream = os.fdopen(fd, "w", buffering=1, errors="replace")
        except Exception:
            stream = open(os.devnull, "w")
        setattr(sys, name, stream)


def run_train_worker(argv: list[str]) -> int:
    """Run an Ultralytics training job from ``key=value`` arguments."""
    _ensure_std_streams()
    from bung_labeler.core import training as training_logic

    params = training_logic.parse_worker_args(argv)
    task = str(params.pop("task", "obb"))
    model_path = str(params.pop("model", "")).strip()
    if not model_path:
        print("[error] no model specified for training", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[error] bundled Ultralytics failed to load: {exc}", file=sys.stderr)
        return 3

    print(f"[worker] training {task} from {model_path}", flush=True)
    try:
        model = YOLO(model_path, task=task) if task else YOLO(model_path)
        model.train(**params)
    except Exception as exc:
        print(f"[error] training failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


def run_eval_worker(argv: list[str]) -> int:
    """Run the metrics runner in-process, mirroring `python -m ...eval_runner`."""
    _ensure_std_streams()
    from bung_labeler import eval_runner

    try:
        return int(eval_runner.main(argv) or 0)
    except SystemExit as exc:  # argparse/exit inside the runner
        return int(exc.code or 0)
    except Exception as exc:
        print(f"[error] evaluation failed: {exc}", file=sys.stderr, flush=True)
        return 1


def maybe_run_worker(argv: list[str]) -> int | None:
    """Return an exit code if argv selects a worker, else None to start the GUI."""
    if len(argv) < 2:
        return None
    from bung_labeler.core import training as training_logic

    flag, rest = argv[1], argv[2:]
    if flag == training_logic.TRAIN_WORKER_FLAG:
        return run_train_worker(rest)
    if flag == training_logic.EVAL_WORKER_FLAG:
        return run_eval_worker(rest)
    return None
