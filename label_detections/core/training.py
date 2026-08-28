"""Pure helpers for launching Ultralytics YOLO training as a subprocess.

This module builds and validates the training command but does not run it (the
UI runs it via QProcess so it stays cancelable and non-blocking). Keeping the
command construction here makes it unit testable without Qt, OpenCV, or
Ultralytics installed.

The generated command targets the Ultralytics ``yolo`` CLI, e.g.:
    yolo obb train model=yolo11s-obb.pt data=data.yaml imgsz=736 batch=16 ...
"""
from __future__ import annotations

from pathlib import Path

VALID_TASKS = ("obb", "detect", "segment", "pose", "classify")


def default_params() -> dict:
    return {
        "task": "obb",
        "model": "yolo11s-obb.pt",
        "data": "",
        "imgsz": 736,
        "batch": 16,
        "epochs": 100,
        "device": "0",
        "project": "data/training",
        "name": "bungvision",
        "patience": 50,
        "workers": 8,
        "resume": False,
    }


def validate_train_params(params: dict) -> list[str]:
    """Return a list of human-readable problems. Empty == ready to run."""
    errors: list[str] = []

    task = str(params.get("task", "")).strip().lower()
    if task not in VALID_TASKS:
        errors.append(f"Task must be one of: {', '.join(VALID_TASKS)}.")

    model = str(params.get("model", "")).strip()
    if not model:
        errors.append("Base model is required (e.g. yolo11s-obb.pt or a .pt checkpoint).")

    data = str(params.get("data", "")).strip()
    if not data:
        errors.append("Data YAML is required. Export a dataset, then point to its data.yaml.")
    elif not Path(data).exists():
        errors.append(f"Data YAML not found: {data}")

    for key, lo, hi in (("imgsz", 32, 8192), ("epochs", 1, 100000), ("patience", 0, 100000), ("workers", 0, 256)):
        try:
            v = int(params.get(key))
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer.")
            continue
        if not (lo <= v <= hi):
            errors.append(f"{key} must be between {lo} and {hi}.")

    # batch may be -1 (Ultralytics auto-batch) or a positive integer.
    try:
        batch = int(params.get("batch"))
        if batch == 0 or batch < -1:
            errors.append("batch must be a positive integer, or -1 for auto.")
    except (TypeError, ValueError):
        errors.append("batch must be an integer (or -1 for auto).")

    if not str(params.get("name", "")).strip():
        errors.append("Run name is required.")

    return errors


# Augmentation, set explicitly rather than left to Ultralytics.
#
# Ultralytics ships mosaic=1.0, scale=0.5, fliplr=0.5, and for classification
# erasing=0.4. Those are tuned for COCO-like data: many object scales, objects
# that can face either way, and far more images than a line ever produces.
#
# A bolted-down camera at a fixed working distance sees exactly one scene
# geometry. Mosaic tiles four images into one, roughly halving each, and scale
# jitters size on top of that -- so a detector whose image size was computed
# from measured label pixels ("a label must arrive at 256 px or better") spends
# most of training seeing labels at a fraction of that. The arithmetic and the
# augmentation end up arguing. Mosaic also pushes labels onto tile boundaries,
# teaching the model to fire on partial labels, which on this line is a reject.
#
# A mirrored label is a label that does not exist, so fliplr goes to zero. And
# a classifier crop is identified by its printed text: erasing a patch of it
# 40% of the time and still asking for the part number is teaching noise.
#
# Values, not policy: every one of these is a field on the Train tab.
AUGMENT_DEFAULTS = {
    "mosaic": 0.0,
    "scale": 0.2,
    "fliplr": 0.0,
    "erasing": 0.0,
}

# Ultralytics turns mosaic off for the last N epochs so a run finishes on clean
# images. The trainer fires that on the epoch count alone -- `if epoch ==
# (self.epochs - self.args.close_mosaic)` -- and never asks whether mosaic was
# ever on, so it announces "Closing dataloader mosaic" even at mosaic=0, and
# rebuilds the transform pipeline and resets the loader to set four values that
# already hold. Derived rather than exposed: with mosaic off there is nothing
# for it to mean, and with mosaic on it must not be zero.
DEFAULT_CLOSE_MOSAIC = 10

# Which of them the task actually honours. Classification builds a torchvision
# pipeline (see ultralytics.data.augment.classify_augmentations) that has no
# mosaic at all, and only classification has erasing -- passing a key a task
# ignores is silent, which is exactly how a setting comes to look applied when
# it is not.
_DETECTION_AUGMENT = ("mosaic", "scale", "fliplr")
_CLASSIFY_AUGMENT = ("fliplr", "erasing")


def augment_kwargs(params: dict) -> dict:
    """The augmentation settings that apply to this run's task."""
    task = str(params.get("task", "obb")).strip().lower()
    keys = _CLASSIFY_AUGMENT if task == "classify" else _DETECTION_AUGMENT
    out: dict = {}
    for key in keys:
        value = params.get(key, AUGMENT_DEFAULTS[key])
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = AUGMENT_DEFAULTS[key]
    if "mosaic" in out:
        try:
            requested = int(params.get("close_mosaic", DEFAULT_CLOSE_MOSAIC))
        except (TypeError, ValueError):
            requested = DEFAULT_CLOSE_MOSAIC
        out["close_mosaic"] = requested if out["mosaic"] > 0 else 0
    return out


def train_kwargs(params: dict) -> dict:
    """Params as keyword arguments for ``YOLO.train()``.

    Mirrors build_train_command, but for driving Ultralytics in-process. A
    packaged build bundles Ultralytics yet has no ``yolo`` CLI on PATH, so the
    frozen app re-invokes itself as a worker and calls train() with these.
    Empty device/project/name are omitted so Ultralytics keeps its own defaults.
    """
    kwargs: dict = {
        "data": str(params.get("data", "")).strip(),
        "imgsz": int(params.get("imgsz", 736)),
        "batch": int(params.get("batch", 16)),
        "epochs": int(params.get("epochs", 100)),
        "patience": int(params.get("patience", 50)),
        "workers": int(params.get("workers", 8)),
    }
    for key in ("device", "project", "name"):
        value = str(params.get(key, "")).strip()
        if value:
            kwargs[key] = value
    if params.get("resume"):
        kwargs["resume"] = True
    kwargs.update(augment_kwargs(params))
    return kwargs


# argv flags the frozen executable understands instead of launching the GUI.
TRAIN_WORKER_FLAG = "--train-worker"
EVAL_WORKER_FLAG = "--eval-worker"


def build_worker_train_command(exe: str, params: dict) -> list[str]:
    """argv for re-invoking the frozen app as a training worker.

    Parameters travel as the same ``key=value`` strings the CLI uses, so the
    worker and the CLI path stay in sync.
    """
    cmd = [exe, TRAIN_WORKER_FLAG,
           f"task={str(params.get('task', 'obb')).strip().lower()}",
           f"model={str(params.get('model', '')).strip()}"]
    for key, value in train_kwargs(params).items():
        cmd.append(f"{key}={value}")
    return cmd


# Keys whose values are integers; everything else stays a string. Typing these
# explicitly rather than guessing from the text matters: a run name or project
# path of "2024" would otherwise be coerced to an int and break path building,
# and device="0" would stop being the string Ultralytics documents.
_WORKER_INT_KEYS = frozenset({"imgsz", "batch", "epochs", "patience", "workers",
                              "close_mosaic"})
_WORKER_BOOL_KEYS = frozenset({"resume"})
# Augmentation values are floats. Without this they would reach train() as the
# strings they travelled as, and "0.0" is not 0.0 to Ultralytics.
_WORKER_FLOAT_KEYS = frozenset(AUGMENT_DEFAULTS)  # close_mosaic is an int


def parse_worker_args(argv: list[str]) -> dict:
    """Parse ``key=value`` worker arguments back into a params dict."""
    out: dict = {}
    for item in argv:
        if "=" not in item:
            continue
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not key:
            continue
        if key in _WORKER_BOOL_KEYS:
            out[key] = raw.strip().lower() in ("1", "true", "yes")
        elif key in _WORKER_INT_KEYS:
            try:
                out[key] = int(raw)
            except ValueError:
                continue  # drop rather than hand train() a bad type
        elif key in _WORKER_FLOAT_KEYS:
            try:
                out[key] = float(raw)
            except ValueError:
                continue
        else:
            out[key] = raw
    return out


def build_train_command(yolo_exe: str, params: dict) -> list[str]:
    """Build the argv list for the Ultralytics CLI from validated params.

    yolo_exe is the executable/entrypoint to invoke (default "yolo"); it is left
    overridable so a full path or wrapper can be supplied in environments where
    ``yolo`` is not on PATH.
    """
    task = str(params.get("task", "obb")).strip().lower()
    cmd = [yolo_exe or "yolo", task, "train"]

    def add(key: str, value) -> None:
        cmd.append(f"{key}={value}")

    add("model", str(params.get("model", "")).strip())
    add("data", str(params.get("data", "")).strip())
    add("imgsz", int(params.get("imgsz", 736)))
    add("batch", int(params.get("batch", 16)))
    add("epochs", int(params.get("epochs", 100)))
    add("patience", int(params.get("patience", 50)))
    add("workers", int(params.get("workers", 8)))

    device = str(params.get("device", "")).strip()
    if device:
        add("device", device)

    project = str(params.get("project", "")).strip()
    if project:
        add("project", project)
    name = str(params.get("name", "")).strip()
    if name:
        add("name", name)

    if params.get("resume"):
        add("resume", "True")

    # Emitted always, not only when they differ from Ultralytics' defaults:
    # these are the settings most likely to be questioned later, and a command
    # line that states them is the record of what was actually trained.
    for key, value in augment_kwargs(params).items():
        add(key, value)

    return cmd


# --- Live training-metrics parsing -------------------------------------------
# Ultralytics writes <project>/<name>/results.csv, one row per finished epoch.
# Parsing that file (rather than the tqdm stdout, which uses carriage returns)
# gives a clean, pollable source for the live loss/mAP chart.

def parse_results_csv(text: str) -> list[dict]:
    """Parse an Ultralytics results.csv into a list of per-epoch row dicts.

    Numeric cells are converted to floats; non-numeric cells stay as strings.
    Malformed rows (wrong column count) are skipped so a half-written file
    being polled mid-train does not raise.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: list[dict] = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            continue
        row: dict = {}
        for key, raw in zip(header, parts):
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        rows.append(row)
    return rows


def _first_matching_column(columns: list[str], needle: str) -> str | None:
    needle = needle.lower()
    for col in columns:
        if needle in col.lower():
            return col
    return None


def metric_series(rows: list[dict], needle: str) -> list[float]:
    """Numeric series for the first column whose name contains ``needle``."""
    if not rows:
        return []
    col = _first_matching_column(list(rows[0].keys()), needle)
    if col is None:
        return []
    out: list[float] = []
    for r in rows:
        v = r.get(col)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


# Series shown on the live chart: a short label and the substring used to find
# the matching results.csv column across detect/obb/segment/pose runs.
CHART_SERIES = (
    ("box_loss", "train/box_loss"),
    ("cls_loss", "train/cls_loss"),
    ("mAP50", "metrics/mAP50("),
    ("mAP50-95", "metrics/mAP50-95("),
    # A classification run writes none of the above. Its loss column is
    # "train/loss" -- which is not a substring of "train/box_loss", so the two
    # cannot cross-match -- and it reports top-1/top-5 accuracy instead of mAP.
    # Listed together rather than split by task because a series that is not in
    # the file is skipped, so one table serves both stages.
    ("loss", "train/loss"),
    ("top1", "metrics/accuracy_top1"),
    ("top5", "metrics/accuracy_top5"),
)


def chart_series(rows: list[dict]) -> dict[str, list[float]]:
    """Return {label: series} for the standard chart metrics that are present."""
    out: dict[str, list[float]] = {}
    for label, needle in CHART_SERIES:
        series = metric_series(rows, needle)
        if series:
            out[label] = series
    return out


# Validation metrics shown in the training-finished summary.
SUMMARY_METRICS = (
    ("precision", "metrics/precision"),
    ("recall", "metrics/recall"),
    ("mAP50", "metrics/mAP50("),
    ("mAP50-95", "metrics/mAP50-95("),
    ("accuracy_top1", "metrics/accuracy_top1"),
    ("accuracy_top5", "metrics/accuracy_top5"),
)


def summarize_results(rows: list[dict]) -> dict:
    """Summarize a parsed results.csv into final + best validation metrics.

    Returns a dict with:
      epochs       - epoch number of the last row (or row count if no column)
      rows         - number of recorded epochs
      final        - {metric: value} from the last epoch
      best         - {metric: value} from the best epoch (ranked by mAP50-95,
                     falling back to mAP50)
      best_epoch   - epoch number of that best row
    Empty rows yield {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}.
    """
    if not rows:
        return {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}

    epoch_series = metric_series(rows, "epoch")
    epochs = int(epoch_series[-1]) if epoch_series else len(rows)

    final: dict[str, float] = {}
    for label, needle in SUMMARY_METRICS:
        series = metric_series(rows, needle)
        if series:
            final[label] = series[-1]

    # Rank epochs by mAP50-95, then mAP50, then top-1 accuracy, to find the
    # best checkpoint. Without the last one a classification run ranked on an
    # empty series and "best" silently meant "last".
    rank = (metric_series(rows, "mAP50-95(")
            or metric_series(rows, "mAP50(")
            or metric_series(rows, "metrics/accuracy_top1"))
    best_idx = max(range(len(rank)), key=lambda i: rank[i]) if rank else len(rows) - 1

    best: dict[str, float] = {}
    for label, needle in SUMMARY_METRICS:
        series = metric_series(rows, needle)
        if series and best_idx < len(series):
            best[label] = series[best_idx]

    if epoch_series and best_idx < len(epoch_series):
        best_epoch = int(epoch_series[best_idx])
    else:
        best_epoch = best_idx + 1

    return {"epochs": epochs, "rows": len(rows), "final": final, "best": best, "best_epoch": best_epoch}


def format_duration(seconds: float) -> str:
    """Human duration like '1h 23m 4s' / '5m 12s' / '47s'."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
