"""Unit tests for the pure training command builder/validator (headless-safe)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import training as t


def _params(tmp_path, **over):
    data = tmp_path / "data.yaml"
    data.write_text("names: [battery, bung]\n", encoding="utf-8")
    p = t.default_params()
    p["data"] = str(data)
    p.update(over)
    return p


def test_defaults_are_obb():
    assert t.default_params()["task"] == "obb"


def test_validate_clean(tmp_path):
    assert t.validate_train_params(_params(tmp_path)) == []


def test_validate_missing_data():
    errors = t.validate_train_params(t.default_params())
    assert any("Data YAML is required" in e for e in errors)


def test_validate_missing_data_file(tmp_path):
    p = _params(tmp_path, data=str(tmp_path / "nope.yaml"))
    assert any("not found" in e for e in t.validate_train_params(p))


def test_validate_bad_task(tmp_path):
    p = _params(tmp_path, task="banana")
    assert any("Task must be one of" in e for e in t.validate_train_params(p))


def test_validate_bad_batch(tmp_path):
    assert any("batch" in e for e in t.validate_train_params(_params(tmp_path, batch=0)))
    # -1 auto-batch is allowed.
    assert t.validate_train_params(_params(tmp_path, batch=-1)) == []


def test_validate_imgsz_range(tmp_path):
    assert any("imgsz" in e for e in t.validate_train_params(_params(tmp_path, imgsz=16)))


def test_build_command_basic(tmp_path):
    p = _params(tmp_path, imgsz=640, batch=8, epochs=50, device="0", name="run1")
    cmd = t.build_train_command("yolo", p)
    assert cmd[:3] == ["yolo", "obb", "train"]
    assert f"data={p['data']}" in cmd
    assert "imgsz=640" in cmd and "batch=8" in cmd and "epochs=50" in cmd
    assert "device=0" in cmd and "name=run1" in cmd
    assert "model=yolo11s-obb.pt" in cmd


def test_build_command_omits_empty_device_and_resume(tmp_path):
    p = _params(tmp_path, device="", resume=False)
    cmd = t.build_train_command("yolo", p)
    assert not any(c.startswith("device=") for c in cmd)
    assert not any(c.startswith("resume=") for c in cmd)


def test_build_command_custom_exe_and_resume(tmp_path):
    p = _params(tmp_path, resume=True)
    cmd = t.build_train_command("/opt/venv/bin/yolo", p)
    assert cmd[0] == "/opt/venv/bin/yolo"
    assert "resume=True" in cmd


_RESULTS_CSV = (
    "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.5,2.0,0.10,0.05\n"
    "2,1.2,1.7,0.30,0.15\n"
    "3,1.0,1.5,0.55,0.32\n"
)


def test_parse_results_csv():
    rows = t.parse_results_csv(_RESULTS_CSV)
    assert len(rows) == 3
    assert rows[0]["epoch"] == 1.0
    assert rows[2]["train/box_loss"] == 1.0


def test_parse_results_csv_skips_malformed():
    bad = _RESULTS_CSV + "4,oops\n"
    rows = t.parse_results_csv(bad)
    assert len(rows) == 3


def test_parse_results_csv_empty():
    assert t.parse_results_csv("") == []
    assert t.parse_results_csv("epoch,train/box_loss\n") == []


def test_metric_series_by_substring():
    rows = t.parse_results_csv(_RESULTS_CSV)
    assert t.metric_series(rows, "box_loss") == [1.5, 1.2, 1.0]
    assert t.metric_series(rows, "mAP50-95") == [0.05, 0.15, 0.32]
    assert t.metric_series(rows, "nonexistent") == []


def test_chart_series_present_only():
    rows = t.parse_results_csv(_RESULTS_CSV)
    series = t.chart_series(rows)
    assert set(series) == {"box_loss", "cls_loss", "mAP50", "mAP50-95"}
    assert series["mAP50"] == [0.10, 0.30, 0.55]


_RESULTS_CSV_VAL = (
    "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.5,0.40,0.30,0.20,0.10\n"
    "2,1.2,0.70,0.60,0.65,0.40\n"
    "3,1.0,0.60,0.55,0.50,0.35\n"
)


def test_summarize_results_final_and_best():
    rows = t.parse_results_csv(_RESULTS_CSV_VAL)
    s = t.summarize_results(rows)
    assert s["epochs"] == 3 and s["rows"] == 3
    # Final row is epoch 3.
    assert s["final"]["mAP50-95"] == 0.35
    assert s["final"]["precision"] == 0.60
    # Best by mAP50-95 is epoch 2.
    assert s["best_epoch"] == 2
    assert s["best"]["mAP50-95"] == 0.40
    assert s["best"]["mAP50"] == 0.65


def test_summarize_results_empty():
    s = t.summarize_results([])
    assert s == {"epochs": 0, "rows": 0, "final": {}, "best": {}, "best_epoch": 0}


def test_format_duration():
    assert t.format_duration(47) == "47s"
    assert t.format_duration(312) == "5m 12s"
    assert t.format_duration(3784) == "1h 3m 4s"


if __name__ == "__main__":
    import tempfile
    import traceback
    import inspect
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)


# --- Frozen-build worker round-trip -------------------------------------------
# A packaged build has no `yolo` CLI, so it re-invokes itself with these args.

def _worker_params():
    return dict(task="obb", model="yolo11s-obb.pt", data="d.yaml", imgsz=960,
                batch=-1, epochs=50, patience=10, workers=8, device="0",
                project="C:/out", name="run1", resume=False)


def test_build_worker_train_command_shape():
    cmd = t.build_worker_train_command("app.exe", _worker_params())
    assert cmd[0] == "app.exe"
    assert cmd[1] == t.TRAIN_WORKER_FLAG
    assert "task=obb" in cmd and "model=yolo11s-obb.pt" in cmd


def test_worker_args_round_trip_types():
    cmd = t.build_worker_train_command("x", _worker_params())
    back = t.parse_worker_args(cmd[2:])
    assert back["imgsz"] == 960 and isinstance(back["imgsz"], int)
    assert back["batch"] == -1 and back["epochs"] == 50 and back["workers"] == 8
    assert back["device"] == "0" and isinstance(back["device"], str)


def test_worker_args_keep_numeric_strings_as_strings():
    # A run name or project path of "2024" must not become an int, or path
    # building downstream breaks.
    cmd = t.build_worker_train_command("x", dict(_worker_params(), name="2024", project="C:/1"))
    back = t.parse_worker_args(cmd[2:])
    assert back["name"] == "2024" and isinstance(back["name"], str)
    assert back["project"] == "C:/1" and isinstance(back["project"], str)


def test_worker_args_resume_is_bool():
    cmd = t.build_worker_train_command("x", dict(_worker_params(), resume=True))
    assert t.parse_worker_args(cmd[2:])["resume"] is True


def test_worker_args_ignores_junk():
    assert t.parse_worker_args(["novalue", "", "=x", "imgsz=notanint"]) == {}


def test_train_kwargs_omits_empty_optionals():
    k = t.train_kwargs(dict(_worker_params(), device="", project="", name=""))
    assert "device" not in k and "project" not in k and "name" not in k
    assert k["imgsz"] == 960 and k["epochs"] == 50


# --- following the run's real output directory ------------------------------

try:
    from PySide6.QtWidgets import QApplication
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _HAVE_QT = True
except Exception:
    _HAVE_QT = False

import pytest

qt_only = pytest.mark.skipif(not _HAVE_QT, reason="PySide6 not available")

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


@qt_only
def test_the_save_directory_is_taken_from_the_start_of_training_line():
    """Ultralytics announces its output directory twice: as training starts,
    and again when it ends. Only the second was read -- so for the whole of
    training there was nothing to follow, and the chart fell back to globbing
    <project>/<name>* for a directory Ultralytics does not necessarily use. It
    resolves the project against its own runs root, so project 'data/training'
    lands in runs/obb/data/training/<name> and the glob finds nothing."""
    win = _window()
    win._results_csv_path = None
    win._train_save_dir = None

    win._scan_for_save_dir(
        "\x1b[1mLogging results to \x1b[1mruns/obb/data/training/lv-15\x1b[0m\n")
    assert win._results_csv_path is not None
    assert str(win._results_csv_path).endswith("runs/obb/data/training/lv-15/results.csv")


@qt_only
def test_the_end_of_training_line_is_still_read():
    win = _window()
    win._results_csv_path = None
    win._train_save_dir = None
    win._scan_for_save_dir("Results saved to \x1b[1mruns/obb/train7\x1b[0m")
    assert str(win._results_csv_path).endswith("runs/obb/train7/results.csv")


@qt_only
def test_an_announced_directory_is_not_second_guessed_by_the_glob():
    """Once Ultralytics has said where it is writing, an absent results.csv
    means 'not yet', not 'look elsewhere'. Falling through would lock onto some
    other run's file and chart the wrong training."""
    win = _window()
    win._results_csv_path = None
    win._train_save_dir = None
    win._scan_for_save_dir("Logging results to runs/obb/not-written-yet")
    assert win._resolve_results_csv() is None
