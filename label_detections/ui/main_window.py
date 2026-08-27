from __future__ import annotations

import os
import sys
import traceback
import time
import math
import json
from pathlib import Path

# Allow this file to be launched directly during troubleshooting, e.g.
# python label_detections/ui/main_window.py, without losing access to the
# bundled label_detections package.
# Skipped when frozen: PyInstaller already resolves the bundled package, and
# parents[2] would point inside the bundle rather than at a real source tree.
if not getattr(sys, "frozen", False):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

import cv2
from PySide6.QtCore import QTimer, Qt, QProcess, QRectF, QPointF, QEvent
from PySide6.QtGui import QAction, QKeySequence, QIntValidator, QTextCursor, QColor, QPainter, QPen, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QGroupBox,
    QInputDialog,
    QHBoxLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListView,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QAbstractSpinBox,
    QAbstractItemView,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSizePolicy,
)

from label_detections.core.camera import CameraSource, quick_test_source
from label_detections.core.image_adjust import apply_adjustments
from label_detections.core.imageio import (
    IMPORT_IMAGE_EXTS,
    import_images,
    save_capture,
)
from label_detections.core.storage import (
    DATA_DIR,
    EXPORT_DIR,
    dataset_folder,
    image_label_json_path,
    label_folder,
    list_datasets,
    list_images,
    load_annotations,
    load_camera_settings,
    load_class_config,
    load_test_settings,
    load_training_settings,
    save_annotations,
    save_camera_settings,
    save_test_settings,
    save_training_settings,
)
from label_detections.core.yolo_export import export_label_yolo, export_all_labels_yolo
from label_detections.core import review as review_logic
from label_detections.core import geometry as geom
from label_detections.core import export_report
from label_detections.core import relabel as relabel_logic
from label_detections.core import active_learning
from label_detections.core import training as training_logic
from label_detections.core import evaluation as evaluation_logic
from label_detections.core import dataset_health
from label_detections.core import storage as storage_mod
from label_detections.core import class_stats
from label_detections.core import persistence
from label_detections.core import imageio
from label_detections.core import labels as labels_mod
from label_detections.core import annotations as ann_logic
from label_detections.core import augment as augment_logic
from label_detections.core import dataset as dataset_logic
from label_detections.core import live_detect as live_logic
from label_detections.version import APP_TITLE
from label_detections.ui import wizards
from label_detections.ui.canvas import ImageCanvas


class TrainingMetricsChart(QWidget):
    """Multi-series line chart for live training metrics with dual Y axes.

    Losses are plotted against an autoscaled left Y axis; mAP metrics against a
    0-1 right Y axis.  Both axes are numbered and the X axis is the epoch number.
    The legend shows each series' latest value.
    """

    _COLORS = {
        "box_loss": "#f87171",
        "cls_loss": "#fb923c",
        "mAP50": "#34d399",
        "mAP50-95": "#60a5fa",
    }
    _DEFAULT = "#cbd5e1"
    _AXIS = "#475569"
    _GRID = "#1e293b"
    _TEXT = "#94a3b8"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._epochs: list[float] = []
        self._series: dict[str, list[float]] = {}
        self.setMinimumHeight(200)
        self.setToolTip(
            "Live training curves from results.csv. Losses should trend down; "
            "mAP should trend up. X axis is the epoch number."
        )

    def set_data(self, epochs: list[float], series: dict[str, list[float]]) -> None:
        self._series = {k: list(v) for k, v in (series or {}).items() if v}
        # Fall back to 1..N if the epoch column was missing.
        n = max((len(v) for v in self._series.values()), default=0)
        if epochs and len(epochs) >= n:
            self._epochs = [float(e) for e in epochs[:n]]
        else:
            self._epochs = [float(i + 1) for i in range(n)]
        self.update()

    def clear(self) -> None:
        self._epochs = []
        self._series = {}
        self.update()

    @staticmethod
    def _fmt(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        full = self.rect()
        p.fillRect(full, QColor("#0b1220"))

        if not self._series:
            p.setPen(QColor(self._DEFAULT))
            p.drawText(full, Qt.AlignCenter, "Training curves appear here once epochs complete.")
            p.end()
            return

        fm = p.fontMetrics()

        # Split series across two Y axes: losses on the left, mAP/metrics on the
        # right (0-1 band).  Either axis is omitted if it has no series yet.
        loss_series = {k: v for k, v in self._series.items() if "loss" in k.lower()}
        metric_series = {k: v for k, v in self._series.items() if "loss" not in k.lower()}

        # Margins: leave room for a right Y axis only when there are metrics.
        # The bottom band holds the epoch tick row plus a dedicated legend row so
        # the legend never overlaps the axis labels.
        left = 52
        right = 52 if metric_series else 14
        top = 16
        bottom = 44
        plot = QRectF(
            full.left() + left, full.top() + top,
            max(1, full.width() - left - right),
            max(1, full.height() - top - bottom),
        )

        def axis_range(vals: list[float], include_zero: bool, floor_max: float | None) -> tuple[float, float]:
            lo = min(vals)
            hi = max(vals)
            if include_zero:
                lo = min(0.0, lo)
            if floor_max is not None:
                hi = max(hi, floor_max)
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi + (hi - lo) * 0.05

        lmin, lmax = axis_range([v for s in loss_series.values() for v in s] or [0.0, 1.0], True, None)
        # mAP lives in 0-1; keep that fixed range unless a value somehow exceeds 1.
        rmin, rmax = axis_range([v for s in metric_series.values() for v in s] or [0.0, 1.0], True, 1.0)

        # X range from epoch numbers.
        emin = self._epochs[0] if self._epochs else 1.0
        emax = self._epochs[-1] if self._epochs else 1.0
        if emax <= emin:
            emax = emin + 1.0

        def px(epoch: float) -> float:
            return plot.left() + (epoch - emin) / (emax - emin) * plot.width()

        def py(val: float, lo: float, hi: float) -> float:
            return plot.bottom() - (val - lo) / (hi - lo) * plot.height()

        def py_left(val: float) -> float:
            return py(val, lmin, lmax)

        def py_right(val: float) -> float:
            return py(val, rmin, rmax)

        # --- Y ticks + gridlines (shared y positions; left + right labels) ---
        y_ticks = 5
        for i in range(y_ticks + 1):
            frac = i / y_ticks
            y = plot.bottom() - frac * plot.height()
            p.setPen(QPen(QColor(self._GRID), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            if loss_series:
                p.setPen(QColor(self._TEXT))
                p.drawText(QRectF(full.left(), y - 8, left - 6, 16),
                           Qt.AlignRight | Qt.AlignVCenter, self._fmt(lmin + (lmax - lmin) * frac))
            if metric_series:
                p.setPen(QColor(self._COLORS.get("mAP50", self._TEXT)))
                p.drawText(QRectF(plot.right() + 4, y - 8, right - 6, 16),
                           Qt.AlignLeft | Qt.AlignVCenter, self._fmt(rmin + (rmax - rmin) * frac))

        # --- X axis ticks (epoch numbers) ---
        n_epochs = len(self._epochs)
        x_ticks = min(6, n_epochs) if n_epochs > 1 else 1
        for i in range(x_ticks):
            frac = i / (x_ticks - 1) if x_ticks > 1 else 0.0
            epoch = emin + (emax - emin) * frac
            p.setPen(QColor(self._TEXT))
            p.drawText(QRectF(px(epoch) - 24, plot.bottom() + 4, 48, 16),
                       Qt.AlignCenter, str(int(round(epoch))))

        # Axis lines.
        p.setPen(QPen(QColor(self._AXIS), 1))
        p.drawLine(QPointF(plot.left(), plot.top()), QPointF(plot.left(), plot.bottom()))
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        if metric_series:
            p.drawLine(QPointF(plot.right(), plot.top()), QPointF(plot.right(), plot.bottom()))

        # --- Series lines ---
        for name, values in self._series.items():
            color = QColor(self._COLORS.get(name, self._DEFAULT))
            ymap = py_right if name in metric_series else py_left
            p.setPen(QPen(color, 2))
            prev = None
            for i, val in enumerate(values):
                epoch = self._epochs[i] if i < len(self._epochs) else float(i + 1)
                pt = QPointF(px(epoch), ymap(val))
                if prev is not None:
                    p.drawLine(prev, pt)
                else:
                    p.drawEllipse(pt, 2, 2)
                prev = pt

        # --- Legend on its own row along the bottom (below the epoch ticks) ---
        # Right-axis (mAP) series get an (R) marker so the dual axes are clear.
        entries = []
        for name, values in self._series.items():
            suffix = " (R)" if name in metric_series else ""
            entries.append((name, f"{name} {self._fmt(values[-1])}{suffix}"))
        spacing = 16
        total_w = sum(fm.horizontalAdvance(text) for _n, text in entries) + spacing * max(0, len(entries) - 1)
        legend_x = max(full.left() + 6, full.left() + (full.width() - total_w) / 2)
        legend_y = full.bottom() - 5
        for name, text in entries:
            color = QColor(self._COLORS.get(name, self._DEFAULT))
            p.setPen(color)
            p.drawText(QPointF(legend_x, legend_y), text)
            legend_x += fm.horizontalAdvance(text) + spacing
        p.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        _icon = Path(__file__).resolve().parent / "assets" / "app.ico"
        if _icon.exists():
            self.setWindowIcon(QIcon(str(_icon)))
        self.resize(1450, 850)
        self.setMinimumSize(1000, 650)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        self.camera = CameraSource()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self._last_preview_status_t = 0.0
        self._preview_frame_counter = 0
        self._preview_fps_t0 = time.perf_counter()
        self._preview_fps = 0.0
        # Last camera frame sequence processed by the display timer. Used to skip
        # re-decoding/re-painting an unchanged frame (see _on_timer).
        self._last_frame_seq = None
        self.last_raw = None
        self.last_adjusted = None
        self.current_image_path: Path | None = None
        # Active-learning review queue (model-prioritized unreviewed images).
        self._review_queue: list[Path] = []
        self._review_queue_pos = -1
        self.camera_settings = load_camera_settings()
        self.class_config = load_class_config()
        self._test_model = None
        self._test_model_path = ""
        self._model_test_overlay_active = False

        # Performance cache, ported from v0.9.35: index the active dataset once
        # and only re-read a sidecar when that specific file changes. Avoids
        # parsing hundreds of JSON files after every box drag, save or review.
        self._dataset_index_dirty = True
        self._image_paths_cache: list[Path] = []
        self._image_status_cache: dict[str, dict] = {}
        # Set by Capture Reference: the next label box drawn opens the region
        # editor on it, so the whole flow is capture, draw, draw regions.
        self._awaiting_reference_box = False
        # Live detection: model on its own thread, overlays scaled from the
        # full-resolution frame it ran on down to the preview being displayed.
        self._live_thread = None
        self._live_worker = None
        self._live_busy = False
        self._live_frame = None
        self._live_overlay_scale = (1.0, 1.0)
        self._live_rolling = live_logic.Rolling()
        self._live_gate = live_logic.CaptureGate()
        self._live_tracks = live_logic.TrackBook()
        self._live_tracking = True
        self._live_last_started = 0.0
        # The last frame of a capture burst, opened when the preview stops.
        self._last_capture_path: Path | None = None
        self._session_captures = 0

        # One label at a time. `label_id` is the dataset being worked on -- the
        # whole app is scoped to it, because a label is trained on its own
        # images and nothing else's. There is no recipe here: which labels a
        # battery must carry, and where, is the front end's business.
        self.library = persistence.load_library()
        # Derived, so it cannot drift from the labels it is meant to describe.
        # Must follow the library load for that reason.
        self.class_names = self.library.detector_classes()
        self.label_id: str = self._initial_label_id()

        self.canvas = ImageCanvas()
        self.canvas.boxes_changed.connect(self._update_box_count)

        self.label_list = QListWidget()
        self.label_list.itemDoubleClicked.connect(self._load_selected_label)

        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self._load_selected_image)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._build_ui()
        self.polish_buttons()
        # Now that button minimums exist, size the pane so nothing can elide.
        # 355px: binary-searched as the narrowest width at which no button
        # label elides, measured on a 1280x720 window where the vertical
        # scrollbar is present and steals ~14px (351 exactly). Measuring on a
        # larger window gives 337, which then clips at 720p. Qt's own layout
        # minimum cannot produce either -- it reports a value that still clips.
        # test_ui_smoke re-checks this, so a longer label fails the suite
        # instead of truncating in the UI.
        self.tabs.setMinimumWidth(355)
        self._build_menu()
        self._refresh_labels()
        self._refresh_images()
        self._apply_theme()
        # After _apply_theme: popup heights are computed from styled row
        # metrics, and the stylesheet changes row height. Sizing them before
        # theming left the height stale, so the popup resized on first open --
        # visible as a flicker.
        self._install_wheel_guards()
        self._class_changed(0)
        self._update_box_count()

    def _initial_label_id(self) -> str:
        """The label to open on launch: the first in the library, else none.

        An empty library is the normal first-run state, not an error. The Label
        tab says so and offers the wizard rather than inventing a placeholder
        label that would then need deleting.
        """
        ids = self.library.ids()
        if ids:
            return ids[0]
        existing = list_datasets()
        return existing[0] if existing else ""
    def _build_menu(self) -> None:
        """Build normal drop-down menus instead of dumping every action on the menu bar.

        The previous first pass added actions directly to menuBar(), which made Qt show
        them as a long row of random labels across the top of the window. Keeping the
        actions inside File/Edit/View/Class/Navigate menus preserves shortcuts without
        cluttering the UI.
        """
        menubar = self.menuBar()
        menubar.clear()

        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")
        class_menu = menubar.addMenu("Class")
        nav_menu = menubar.addMenu("Navigate")
        capture_menu = menubar.addMenu("Capture")
        tools_menu = menubar.addMenu("Tools")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo_canvas)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo_canvas)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()

        open_action = QAction("Open image", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save labels", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_labels)
        file_menu.addAction(save_action)

        delete_action = QAction("Delete selected annotation", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self._guarded(self.canvas.delete_selected))
        edit_menu.addAction(delete_action)

        delete_image_action = QAction("Delete captured image", self)
        delete_image_action.setShortcut("Shift+Delete")
        delete_image_action.triggered.connect(self.delete_selected_image)
        edit_menu.addAction(delete_image_action)

        zoom_in_action = QAction("Zoom in", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("Fit image", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)

        refresh_index_action = QAction("Refresh dataset index", self)
        refresh_index_action.setShortcut("Ctrl+F5")
        refresh_index_action.triggered.connect(lambda: self._refresh_images(force=True))
        view_menu.addAction(refresh_index_action)

        # Two shortcuts, not one per class. Class names are label ids now, so
        # there is no fixed set to assign letters to -- and there is no need:
        # the tool is scoped to one label, so the only two things ever drawn on
        # an image are that label and the battery face.
        class_actions = []
        for _name, _key, _text in (
            ("battery_side", "B", "Class: battery face"),
            ("", "L", "Class: the label being trained"),
        ):
            _action = QAction(_text, self)
            _action.setShortcut(_key)
            _action.triggered.connect(self._guarded(
                lambda _checked=False, n=_name: self.set_class_by_name(n or self.label_id)))
            class_menu.addAction(_action)
            class_actions.append(_action)

        next_action = QAction("Next image", self)
        next_action.setShortcut("N")
        next_action.triggered.connect(self._guarded(self.next_image))
        nav_menu.addAction(next_action)

        prev_action = QAction("Previous image", self)
        prev_action.setShortcut("P")
        prev_action.triggered.connect(self._guarded(self.previous_image))
        nav_menu.addAction(prev_action)

        unreviewed_action = QAction("Find next unreviewed", self)
        unreviewed_action.setShortcut("Ctrl+U")
        unreviewed_action.triggered.connect(self.find_next_unreviewed_image)
        nav_menu.addAction(unreviewed_action)

        mark_reviewed_action = QAction("Mark current reviewed", self)
        mark_reviewed_action.setShortcut("Ctrl+Shift+R")
        mark_reviewed_action.triggered.connect(self.mark_current_reviewed)
        nav_menu.addAction(mark_reviewed_action)

        force_review_action = QAction("Force review current", self)
        force_review_action.setShortcut("Ctrl+Shift+F")
        force_review_action.triggered.connect(self.force_mark_current_reviewed)
        nav_menu.addAction(force_review_action)

        capture_action = QAction("Capture adjusted", self)
        capture_action.setShortcut("C")
        capture_action.triggered.connect(self._guarded(lambda: self.capture_frame(save_adjusted=True)))
        capture_menu.addAction(capture_action)

        auto_label_action = QAction("Auto-label current (model)", self)
        auto_label_action.setShortcut("Ctrl+L")
        auto_label_action.triggered.connect(self.auto_label_current)
        tools_menu.addAction(auto_label_action)

        validate_action = QAction("Validate current image", self)
        validate_action.setShortcut("Ctrl+Shift+V")
        validate_action.triggered.connect(self.validate_current_image)
        tools_menu.addAction(validate_action)

        relabel_action = QAction("Bulk relabel class...", self)
        relabel_action.triggered.connect(self.bulk_relabel_dialog)
        tools_menu.addAction(relabel_action)

        tools_menu.addSeparator()

        define_regions_action = QAction("Define read-regions from this image", self)
        define_regions_action.setShortcut("Ctrl+Shift+D")
        define_regions_action.triggered.connect(self._guarded(self.define_read_regions))
        tools_menu.addAction(define_regions_action)

        edit_regions_action = QAction("Edit read-regions", self)
        edit_regions_action.triggered.connect(self._guarded(self.edit_read_regions))
        tools_menu.addAction(edit_regions_action)

        replace_artwork_action = QAction("Replace label artwork...", self)
        replace_artwork_action.setToolTip(
            "Re-flatten this label's artwork from the box on this image. Every "
            "region is positioned against it, so this is behind a confirmation.")
        replace_artwork_action.triggered.connect(self._guarded(self.replace_label_artwork))
        tools_menu.addAction(replace_artwork_action)

        regions_action = QAction("Place read-regions", self)
        regions_action.setShortcut("Ctrl+R")
        regions_action.setToolTip(
            "Fill in the active label's read-regions from its artwork.")
        regions_action.triggered.connect(self._guarded(self.place_regions_on_canvas))
        tools_menu.addAction(regions_action)

        prelabel_action = QAction("Pre-label unlabeled && review (model)", self)
        prelabel_action.setShortcut("Ctrl+Shift+P")
        prelabel_action.triggered.connect(self.prelabel_and_review)
        tools_menu.addAction(prelabel_action)

        build_queue_action = QAction("Build review queue (model)", self)
        build_queue_action.triggered.connect(self.build_review_queue)
        tools_menu.addAction(build_queue_action)

        next_queue_action = QAction("Next in review queue", self)
        next_queue_action.setShortcut("Ctrl+Shift+N")
        next_queue_action.triggered.connect(self.next_in_review_queue)
        tools_menu.addAction(next_queue_action)

        tools_menu.addSeparator()

        live_detect_action = QAction("Start/stop live detect", self)
        live_detect_action.setShortcut("Ctrl+D")
        live_detect_action.triggered.connect(self._guarded(self.toggle_live_detect))
        tools_menu.addAction(live_detect_action)

        keep_frame_action = QAction("Keep this live frame (image only)", self)
        keep_frame_action.setShortcut("Ctrl+K")
        keep_frame_action.triggered.connect(self._guarded(self.keep_live_frame))
        tools_menu.addAction(keep_frame_action)

        keep_json_action = QAction("Keep this live frame + detections", self)
        keep_json_action.setShortcut("Ctrl+Shift+K")
        keep_json_action.triggered.connect(
            self._guarded(self.keep_live_frame_with_detections))
        tools_menu.addAction(keep_json_action)

        scale_action = QAction("Check label scale (single vs two-stage)", self)
        scale_action.setToolTip(
            "Measure how many pixels wide your labels actually are, and say "
            "whether cropping would help or hurt them.")
        scale_action.triggered.connect(self._guarded(self.show_label_scale_report))
        tools_menu.addAction(scale_action)

        variance_action = QAction("Check variable regions", self)
        variance_action.setToolTip(
            "Measure how much each label's date codes and serials actually differ "
            "across its images.")
        variance_action.triggered.connect(self._guarded(self.check_variable_regions))
        tools_menu.addAction(variance_action)

        health_action = QAction("Dataset health dashboard", self)
        health_action.triggered.connect(self.show_dataset_health)
        tools_menu.addAction(health_action)

        data_folder_action = QAction("Data folder (image library)...", self)
        data_folder_action.triggered.connect(self.change_data_folder)
        tools_menu.addAction(data_folder_action)

        shortcuts_action = QAction("Keyboard shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_shortcuts_reference)
        tools_menu.addAction(shortcuts_action)

        # The menu bar is hidden (see below), and Qt does NOT dispatch the
        # shortcuts of actions that live only inside a hidden menu bar. Register
        # every shortcut action on the window itself so the keys keep working.
        for action in (
            undo_action, redo_action, open_action, save_action,
            delete_action, delete_image_action,
            zoom_in_action, zoom_out_action, fit_action, refresh_index_action,
            *class_actions,
            next_action, prev_action,
            unreviewed_action, mark_reviewed_action, force_review_action,
            capture_action, auto_label_action, validate_action,
            prelabel_action, next_queue_action, shortcuts_action, regions_action,
            define_regions_action, edit_regions_action, replace_artwork_action,
            variance_action, scale_action, live_detect_action, keep_frame_action,
            keep_json_action,
        ):
            self.addAction(action)

        self.menuBar().setVisible(False)

    def _typing_in_text_field(self) -> bool:
        """True when a text-entry widget has focus.

        Single-letter shortcuts (the class keys, N/P/C) use the window-wide
        shortcut context, so without this guard they would steal keystrokes
        while the user is typing into a run-name / filter / device field.
        """
        from PySide6.QtWidgets import QAbstractSpinBox, QComboBox
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QTextEdit, QSpinBox, QAbstractSpinBox)):
            return True
        if isinstance(w, QComboBox) and w.isEditable():
            return True
        return False

    def _guarded(self, fn):
        """Wrap a single-key shortcut slot so it no-ops while typing in a field."""
        def runner(*_args):
            if self._typing_in_text_field():
                return
            fn()
        return runner

    def eventFilter(self, obj, event):
        """Stop the mouse wheel from changing input values.

        Qt lets a wheel event over a spinbox, combo or slider edit it even
        without focus. Inside a scrolling panel that means scrolling past a
        field silently alters a setting -- changing image size or confidence
        while the operator only meant to scroll the page. Wheel events on these
        widgets are redirected to the scroll area instead.
        """
        # Qt sizes a combo popup from an unstyled row metric, but the themed
        # rows render taller, so the last row was always clipped -- every
        # dropdown showed n-0.5 items no matter how much space was free.
        # Resize the view to fit its rows as it is shown.
        if event.type() == QEvent.Type.Show and isinstance(obj, QAbstractItemView):
            self._fit_combo_popup(obj)
            return False
        if event.type() == QEvent.Type.Wheel and isinstance(
            obj, (QAbstractSpinBox, QComboBox, QSlider)
        ):
            area = self._parent_scroll_area(obj)
            if area is not None:
                # Hand the scroll to the page so the wheel still does the thing
                # the operator expects, just without editing the field.
                QApplication.sendEvent(area.viewport(), event)
            return True
        return super().eventFilter(obj, event)

    @staticmethod
    def _parent_scroll_area(widget) -> QScrollArea | None:
        node = widget.parentWidget()
        while node is not None:
            if isinstance(node, QScrollArea):
                return node
            node = node.parentWidget()
        return None

    @staticmethod
    def _fit_combo_popup(view) -> None:
        """Make a combo popup exactly tall enough for the rows it will show."""
        combo = view.parent()
        while combo is not None and not isinstance(combo, QComboBox):
            combo = combo.parent()
        if combo is None or combo.count() <= 0:
            return
        rows = min(combo.count(), max(1, combo.maxVisibleItems()))
        row_h = view.sizeHintForRow(0)
        if row_h <= 0:
            row_h = view.fontMetrics().height() + 8
        # Fixed, not minimum: a minimum is sticky, so a combo that shrinks (a
        # repopulated recipe or category list) would keep the taller popup.
        view.setFixedHeight(rows * row_h)

    def _install_wheel_guards(self) -> None:
        """Apply the wheel guard to every value widget currently in the UI."""
        # One call per type: PySide6's findChildren takes a single type, not a
        # tuple (unlike isinstance). Passing a tuple raises at startup.
        # QAbstractSpinBox covers QSpinBox and QDoubleSpinBox via subclassing.
        for cls in (QAbstractSpinBox, QComboBox, QSlider):
            for widget in self.findChildren(cls):
                # NoFocus would break keyboard use; StrongFocus keeps click/tab
                # focus while the event filter suppresses wheel edits.
                widget.setFocusPolicy(Qt.StrongFocus)
                widget.installEventFilter(self)
                if isinstance(widget, QComboBox):
                    # Replace Qt's private combo view with a plain QListView.
                    # The default view has platform-specific painting that does
                    # not composite cleanly under a stylesheet, which shows as a
                    # flash of the window behind while the popup opens.
                    # Must happen before the filters below: setView() discards
                    # the old view and anything installed on it.
                    widget.setView(QListView(widget))
                    # Size the popup up front and whenever its items change.
                    # Doing it in the Show handler meant Qt painted the popup at
                    # the wrong size and then resized it, which is visible as a
                    # flicker. The Show handler stays as a safety net; when the
                    # height is already correct it is a no-op and emits nothing.
                    widget.view().installEventFilter(self)
                    # Fill opaquely: without this the popup frame paints nothing
                    # on its first expose and the window behind shows through.
                    view = widget.view()
                    view.setAutoFillBackground(True)
                    container = view.parentWidget()
                    if container is not None:
                        container.setAutoFillBackground(True)
                    self._fit_combo_popup(view)
                    model = widget.model()
                    for signal in (model.rowsInserted, model.rowsRemoved,
                                   model.modelReset, model.layoutChanged):
                        signal.connect(
                            lambda *_a, v=widget.view(): self._fit_combo_popup(v)
                        )

    @staticmethod
    def _scrollable_tab() -> tuple[QWidget, QWidget, QVBoxLayout]:
        """Build a vertically scrolling tab body.

        Returns (outer, inner, layout): add content to ``layout`` and return
        ``outer`` from the tab factory. Tabs that skipped this simply clipped
        their lower content on shorter windows with no way to reach it.
        """
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        return outer, inner, layout

    def _scroll_panel(self, widget: QWidget, min_width: int, preferred_width: int) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # AsNeeded, not AlwaysOff: the pane is capped at a maximum width, so
        # any row wider than the cap was silently clipped with no way to reach
        # it. A bar now appears only when something genuinely does not fit.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(min_width)
        scroll.resize(preferred_width, scroll.height())
        widget.setMinimumWidth(min_width - 22)
        scroll.setWidget(widget)
        return scroll

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = self._left_panel()
        # Keep the right rail scroll-safe so Linux/Qt themes do not visually
        # stack or clip buttons on shorter windows.
        right = self._scroll_panel(self._right_panel(), min_width=360, preferred_width=380)
        # v0.9.18: v0.9.17 overcorrected and made the entire left rail too wide.
        # Keep enough room for the capture tab, but let the image canvas stay dominant.
        # Minimum is derived from the widest tab's own content, not guessed.
        # Guessing it low let compact buttons elide their labels -- "Mark
        # Current Reviewed" rendered as "Mark Current Reviewe". Computed after
        # the tabs exist so it tracks content changes instead of going stale.
        # Minimum is set after polish_buttons in __init__: the button widths it
        # assigns are what the content minimum depends on, so computing it here
        # would read pre-polish sizes and come out too small.
        left.setMaximumWidth(400)
        right.setMinimumWidth(360)
        right.setMaximumWidth(450)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([410, 800, 380])
        self.setCentralWidget(splitter)

    def _left_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._label_tab(), "Label")
        tabs.addTab(self._capture_tab(), "Live Capture")
        tabs.addTab(self._adjust_tab(), "Contrast")
        self._test_tab_widget = self._model_test_tab()
        tabs.addTab(self._test_tab_widget, "Test Models")
        self._train_tab_widget = self._train_tab()
        tabs.addTab(self._train_tab_widget, "Train")
        self._live_tab_widget = self._live_detect_tab()
        tabs.addTab(self._live_tab_widget, "Live Detect")
        tabs.addTab(self._help_tab(), "Instructions")
        self.tabs = tabs
        return tabs



    def _model_test_tab(self) -> QWidget:
        """Model sandbox: run one trained OBB model on one image.

        This tab intentionally does not change saved labels or live inspection state. It is
        just a verification tool so the user can confirm battery and bung detections
        from the same model before using rotation-aware count testing.
        """
        outer, w, layout = self._scrollable_tab()

        title = QLabel("Test trained model / Count Test only")
        title.setStyleSheet("font-size: 10pt; font-weight: 700; color: #bfdbfe;")
        title.setWordWrap(True)
        layout.addWidget(title)

        help_text = QLabel(
            "Load your trained LabelVision OBB model, select a test image, then run the model or run a count test."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        form_box = QGroupBox("Model files")
        form = QVBoxLayout(form_box)

        _saved_test = load_test_settings()
        self.test_model_edit = QLineEdit(str(_saved_test.get("model", "")))
        self.test_model_edit.setPlaceholderText("LabelVision OBB best.pt or .engine")
        self.test_model_edit.setMinimumWidth(60)
        self.test_model_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse_model = QPushButton("Model...")
        browse_model.clicked.connect(self.browse_test_model)
        row = QHBoxLayout(); row.addWidget(self.test_model_edit, 1); row.addWidget(browse_model)
        form.addWidget(QLabel("OBB model")); form.addLayout(row)

        self.test_image_edit = QLineEdit(str(_saved_test.get("image", "")))
        self.test_image_edit.setPlaceholderText("test image path")
        self.test_image_edit.setMinimumWidth(60)
        self.test_image_edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse_img = QPushButton("Image...")
        browse_img.clicked.connect(self.browse_test_image)
        use_current = QPushButton("Use Current")
        use_current.clicked.connect(self.use_current_test_image)
        row = QHBoxLayout(); row.addWidget(self.test_image_edit, 1); row.addWidget(browse_img); row.addWidget(use_current)
        form.addWidget(QLabel("Test image")); form.addLayout(row)


        layout.addWidget(form_box)

        settings_box = QGroupBox("Inference settings")
        settings = QFormLayout(settings_box)
        _ts = _saved_test
        self.test_imgsz_spin = QSpinBox(); self.test_imgsz_spin.setRange(320, 2048); self.test_imgsz_spin.setSingleStep(32)
        self.test_imgsz_spin.setValue(int(_ts.get("imgsz", 736)))
        self.test_conf_spin = QDoubleSpinBox(); self.test_conf_spin.setRange(0.01, 0.99); self.test_conf_spin.setSingleStep(0.05)
        self.test_conf_spin.setValue(float(_ts.get("conf", 0.45)))
        self.test_device_edit = QLineEdit(str(_ts.get("device", "0")))
        self.test_device_edit.setPlaceholderText("0, cpu, cuda:0")
        settings.addRow("Image size", self.test_imgsz_spin)
        settings.addRow("Confidence", self.test_conf_spin)
        settings.addRow("Device", self.test_device_edit)
        self.test_hide_saved_labels_check = QCheckBox("Hide saved labels while testing")
        self.test_hide_saved_labels_check.setChecked(bool(_ts.get("hide_saved_labels", True)))
        self.test_hide_saved_labels_check.setToolTip("Hides existing/manual labels on the canvas during model testing without deleting them.")
        settings.addRow("Display", self.test_hide_saved_labels_check)

        layout.addWidget(settings_box)

        # Persist as the operator edits, so nothing has to be re-entered next
        # launch. editingFinished rather than textChanged keeps this to one
        # write per edit instead of one per keystroke.
        for _edit in (self.test_model_edit, self.test_image_edit,
                      self.test_device_edit):
            _edit.editingFinished.connect(self._save_test_settings)
        self.test_imgsz_spin.valueChanged.connect(lambda _v: self._save_test_settings())
        self.test_conf_spin.valueChanged.connect(lambda _v: self._save_test_settings())
        self.test_hide_saved_labels_check.stateChanged.connect(lambda _s: self._save_test_settings())

        run_btn = QPushButton("Run Model")
        run_btn.clicked.connect(self.run_model_test)
        clear_btn = QPushButton("Clear Overlay")
        clear_btn.setToolTip("Clears the model-test overlay and hides saved labels on the canvas. It does not delete or save labels.")
        clear_btn.clicked.connect(self.clear_model_test_overlay)
        show_labels_btn = QPushButton("Show Labels")
        show_labels_btn.setToolTip("Shows saved/manual labels again. This does not affect model-test overlays.")
        show_labels_btn.clicked.connect(self.show_saved_annotations)
        auto_label_btn = QPushButton("Auto-label")
        auto_label_btn.setToolTip("Pre-label the current image with this model. Predictions become editable labels you correct and save. Undo with Ctrl+Z.")
        auto_label_btn.clicked.connect(self.auto_label_current)
        prelabel_btn = QPushButton("Pre-label All")
        prelabel_btn.setToolTip("Run the model on every unlabeled image, save the predictions as un-reviewed labels, and open them in the review queue lowest-confidence first. Existing labels are untouched. (Ctrl+Shift+P)")
        prelabel_btn.clicked.connect(self.prelabel_and_review)

        run_grid = QGridLayout()
        run_grid.setHorizontalSpacing(8)
        run_grid.setVerticalSpacing(8)
        test_buttons = (run_btn, clear_btn, show_labels_btn, auto_label_btn, prelabel_btn)
        for i, btn in enumerate(test_buttons):
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            run_grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(run_grid)

        self.test_results_text = QTextEdit()
        self.test_results_text.setReadOnly(True)
        self.test_results_text.setMinimumHeight(180)
        self.test_results_text.setPlainText(
            "Step-by-step:\n"
            "1. Load the LabelVision OBB best.pt or .engine model.\n"
            "2. Choose a saved/captured test image.\n"
            "3. Click Run Model or Run Count.\n\n"
            "Expected result: blue battery polygons, green bung polygons/centers, and a text summary."
        )
        layout.addWidget(self.test_results_text)
        layout.addStretch(1)
        return outer

    def _train_tab(self) -> QWidget:
        """Launch Ultralytics YOLO training on an exported dataset as a subprocess.

        Training runs via the `yolo` CLI in a QProcess so the UI stays responsive
        and the run is cancelable; stdout streams into the log below.
        """
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll)

        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Train a YOLO model")
        title.setStyleSheet("font-size: 10pt; font-weight: 700; color: #bfdbfe;")
        layout.addWidget(title)

        help_text = QLabel(
            "Export a reviewed dataset first, then point Data YAML at its data.yaml. "
            "Training runs the Ultralytics 'yolo' command in the background."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        saved = load_training_settings()
        params = training_logic.default_params()
        params.update({k: saved[k] for k in params if k in saved})

        files_box = QGroupBox("Dataset and model")
        files = QVBoxLayout(files_box)

        self.train_model_edit = QLineEdit(str(params["model"]))
        self.train_model_edit.setPlaceholderText("yolo11s-obb.pt or path to a .pt checkpoint")
        model_browse = QPushButton("Model...")
        model_browse.clicked.connect(self.browse_train_model)
        # The edit takes the slack; a long checkpoint path must not push the
        # browse button off the edge of the pane.
        self.train_model_edit.setMinimumWidth(120)
        r = QHBoxLayout(); r.addWidget(self.train_model_edit, 1); r.addWidget(model_browse)
        files.addWidget(QLabel("Base model")); files.addLayout(r)

        self.train_data_edit = QLineEdit(str(params["data"]))
        self.train_data_edit.setPlaceholderText("data/exports/<name>/data.yaml")
        data_browse = QPushButton("YAML...")
        data_browse.clicked.connect(self.browse_train_data)
        data_latest = QPushButton("Latest export")
        data_latest.setToolTip("Fill in the most recently created export's data.yaml.")
        data_latest.clicked.connect(self.use_latest_export_for_training)
        # Buttons on their own row: a long path plus two buttons on one line
        # forced the group wider than the left pane's maximum width.
        files.addWidget(QLabel("Data YAML"))
        files.addWidget(self.train_data_edit)
        r = QHBoxLayout()
        for b in (data_browse, data_latest):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            r.addWidget(b)
        files.addLayout(r)
        layout.addWidget(files_box)

        params_box = QGroupBox("Training parameters")
        grid = QGridLayout(params_box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        def _lbl(text):
            l = QLabel(text); l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return l

        self.train_task_combo = QComboBox()
        self.train_task_combo.addItems(list(training_logic.VALID_TASKS))
        if str(params["task"]) in training_logic.VALID_TASKS:
            self.train_task_combo.setCurrentText(str(params["task"]))

        self.train_device_edit = QLineEdit(str(params["device"]))
        self.train_device_edit.setPlaceholderText("0, cpu, cuda:0")

        self.train_imgsz_spin = QSpinBox(); self.train_imgsz_spin.setRange(32, 8192)
        self.train_imgsz_spin.setSingleStep(32); self.train_imgsz_spin.setValue(int(params["imgsz"]))

        self.train_batch_spin = QSpinBox(); self.train_batch_spin.setRange(-1, 1024)
        self.train_batch_spin.setValue(int(params["batch"]))
        self.train_batch_spin.setToolTip("-1 lets Ultralytics auto-pick the batch size for your GPU.")

        self.train_epochs_spin = QSpinBox(); self.train_epochs_spin.setRange(1, 100000)
        self.train_epochs_spin.setValue(int(params["epochs"]))

        self.train_patience_spin = QSpinBox(); self.train_patience_spin.setRange(0, 100000)
        self.train_patience_spin.setValue(int(params["patience"]))

        self.train_workers_spin = QSpinBox(); self.train_workers_spin.setRange(0, 256)
        self.train_workers_spin.setValue(int(params["workers"]))

        self.train_yolo_exe_edit = QLineEdit(str(saved.get("yolo_exe", "yolo")))
        self.train_yolo_exe_edit.setToolTip("Ultralytics CLI entrypoint. Use a full path if 'yolo' is not on PATH.")

        # Row 0: Task | Device
        grid.addWidget(_lbl("Task"), 0, 0); grid.addWidget(self.train_task_combo, 0, 1)
        grid.addWidget(_lbl("Device"), 0, 2); grid.addWidget(self.train_device_edit, 0, 3)
        # Row 1: Image size | Batch
        grid.addWidget(_lbl("Image size"), 1, 0); grid.addWidget(self.train_imgsz_spin, 1, 1)
        _batch_lbl = _lbl("Batch")
        _batch_lbl.setToolTip("-1 lets Ultralytics auto-pick the batch size for your GPU.")
        grid.addWidget(_batch_lbl, 1, 2); grid.addWidget(self.train_batch_spin, 1, 3)
        # Row 2: Epochs | Patience
        grid.addWidget(_lbl("Epochs"), 2, 0); grid.addWidget(self.train_epochs_spin, 2, 1)
        grid.addWidget(_lbl("Patience"), 2, 2); grid.addWidget(self.train_patience_spin, 2, 3)
        # Row 3: Workers | yolo executable
        grid.addWidget(_lbl("Workers"), 3, 0); grid.addWidget(self.train_workers_spin, 3, 1)
        grid.addWidget(_lbl("yolo exe"), 3, 2); grid.addWidget(self.train_yolo_exe_edit, 3, 3)
        # Row 4: Output folder (edit spans cols 1-2, browse button in col 3)
        self.train_project_edit = QLineEdit(str(params["project"]))
        project_browse = QPushButton("Folder...")
        project_browse.clicked.connect(self.browse_train_project)
        grid.addWidget(_lbl("Output folder"), 4, 0)
        grid.addWidget(self.train_project_edit, 4, 1, 1, 2)
        grid.addWidget(project_browse, 4, 3)
        # Row 5: Run name | Resume
        self.train_name_edit = QLineEdit(str(params["name"]))
        # Text lives in the row label; repeating it here only widened the grid.
        self.train_resume_check = QCheckBox()
        self.train_resume_check.setToolTip("Resume an interrupted run from its last checkpoint.")
        self.train_resume_check.setChecked(bool(params.get("resume", False)))
        grid.addWidget(_lbl("Run name"), 5, 0); grid.addWidget(self.train_name_edit, 5, 1)
        grid.addWidget(_lbl("Resume"), 5, 2); grid.addWidget(self.train_resume_check, 5, 3)

        # The four-column grid is the widest thing in the left pane. Without an
        # explicit small minimum, each field demands its content width and the
        # row overflows the pane's maximum width -- which is what clipped
        # Device / Batch / Patience / yolo exe off the right edge.
        for _w in (self.train_task_combo, self.train_device_edit,
                   self.train_imgsz_spin, self.train_batch_spin,
                   self.train_epochs_spin, self.train_patience_spin,
                   self.train_workers_spin, self.train_yolo_exe_edit,
                   self.train_project_edit, self.train_name_edit):
            _w.setMinimumWidth(72)
            _w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # The checkbox label is long; let it elide rather than widen the grid.
        self.train_resume_check.setMinimumWidth(0)
        self.train_resume_check.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        grid.setColumnStretch(1, 1); grid.setColumnStretch(3, 1)
        layout.addWidget(params_box)

        btn_row = QHBoxLayout()
        self.train_start_btn = QPushButton("Start Training")
        self.train_start_btn.clicked.connect(self.start_training)
        self.train_stop_btn = QPushButton("Stop")
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.clicked.connect(self.stop_training)
        # Start takes the extra room; Stop only needs its own label width.
        self.train_start_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_row.addWidget(self.train_start_btn, 1); btn_row.addWidget(self.train_stop_btn)
        layout.addLayout(btn_row)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(140)
        self.train_log.setPlaceholderText("Training output appears here.")
        layout.addWidget(self.train_log)

        chart_box = QGroupBox("Live training curves")
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        self.train_metrics_chart = TrainingMetricsChart()
        chart_layout.addWidget(self.train_metrics_chart)
        layout.addWidget(chart_box)

        # Polls the run's results.csv while training so the chart updates per epoch.
        self._results_csv_path: Path | None = None
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(3000)
        self._metrics_timer.timeout.connect(self._poll_training_metrics)

        # --- Evaluate / promote -------------------------------------------
        eval_box = QGroupBox("Evaluate and promote")
        eval_layout = QVBoxLayout(eval_box)
        eval_help = QLabel(
            "Score a trained model against a labeled split (uses the Data YAML and "
            "Task above), then promote it so Test/Auto-label/Count use it."
        )
        eval_help.setWordWrap(True)
        eval_layout.addWidget(eval_help)

        self.eval_model_edit = QLineEdit()
        self.eval_model_edit.setPlaceholderText("trained best.pt to evaluate")
        eval_model_browse = QPushButton("Model...")
        eval_model_browse.clicked.connect(self.browse_eval_model)
        eval_use_trained = QPushButton("Use trained")
        eval_use_trained.setToolTip("Fill in <output folder>/<run name>/weights/best.pt from the training settings above.")
        eval_use_trained.clicked.connect(self.use_trained_weights_for_eval)
        # The line edit takes the slack so the two buttons keep their full width.
        r = QHBoxLayout()
        r.addWidget(self.eval_model_edit, 1)
        r.addWidget(eval_model_browse)
        r.addWidget(eval_use_trained)
        eval_layout.addWidget(QLabel("Model to evaluate")); eval_layout.addLayout(r)

        # Split selector and the action buttons get their own rows: packing a
        # label, a combo and two buttons onto one line clipped the wider labels
        # in the narrow left pane.
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Split"))
        self.eval_split_combo = QComboBox()
        self.eval_split_combo.addItems(list(evaluation_logic.VALID_SPLITS))
        split_row.addWidget(self.eval_split_combo, 1)
        eval_layout.addLayout(split_row)

        eval_btn_row = QHBoxLayout()
        self.eval_start_btn = QPushButton("Evaluate")
        self.eval_start_btn.clicked.connect(self.start_evaluation)
        self.promote_btn = QPushButton("Promote model")
        self.promote_btn.setEnabled(False)
        self.promote_btn.setToolTip("Copy this model into data/models and set it as the active model for Test / Auto-label / Count / review queue.")
        self.promote_btn.clicked.connect(self.promote_model)
        for b in (self.eval_start_btn, self.promote_btn):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            eval_btn_row.addWidget(b)
        eval_layout.addLayout(eval_btn_row)

        self.eval_metrics_text = QTextEdit()
        self.eval_metrics_text.setReadOnly(True)
        self.eval_metrics_text.setMinimumHeight(140)
        self.eval_metrics_text.setPlaceholderText("mAP / precision / recall appear here after evaluation.")
        eval_layout.addWidget(self.eval_metrics_text)
        layout.addWidget(eval_box)
        layout.addStretch(1)

        self._train_process = None
        self._eval_process = None
        self._eval_buffer = ""
        self._eval_last_model = ""
        return outer

    def _gather_eval_params(self) -> dict:
        return {
            "task": self.train_task_combo.currentText(),
            "model": self.eval_model_edit.text().strip(),
            "data": self.train_data_edit.text().strip(),
            "imgsz": int(self.train_imgsz_spin.value()),
            "device": self.train_device_edit.text().strip(),
            "split": self.eval_split_combo.currentText(),
        }

    def browse_eval_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model to evaluate", "", "Model (*.pt *.engine);;All files (*)")
        if path:
            self.eval_model_edit.setText(path)

    def use_trained_weights_for_eval(self) -> None:
        project = self.train_project_edit.text().strip() or "data/training"
        name = self.train_name_edit.text().strip() or "bungvision"
        best = Path(project) / name / "weights" / "best.pt"
        self.eval_model_edit.setText(str(best))
        if not best.exists():
            self.status.showMessage("Trained best.pt not found yet; train first or browse to a checkpoint.", 6000)

    def _python_for_subprocess(self) -> str:
        """Interpreter to use for `python -m ...` child processes.

        sys.executable is the right answer when running from source, but in a
        PyInstaller build it is the .exe itself, which ignores -m and would just
        launch a second copy of the GUI. Frozen builds therefore fall back to
        the `python` on PATH -- the same interpreter that provides the `yolo`
        CLI used for training.
        """
        if getattr(sys, "frozen", False):
            return "python"
        return sys.executable

    def start_evaluation(self) -> None:
        if self._eval_process is not None:
            QMessageBox.information(self, "Evaluate", "An evaluation is already in progress.")
            return
        params = self._gather_eval_params()
        errors = evaluation_logic.validate_eval_params(params)
        if errors:
            QMessageBox.warning(self, "Evaluate", "Cannot evaluate:\n\n" + "\n".join(f"• {e}" for e in errors))
            return
        if getattr(sys, "frozen", False):
            # Same reasoning as training: run the metrics runner inside our own
            # process image instead of a `python -m` that does not exist here.
            cmd = [sys.executable, training_logic.EVAL_WORKER_FLAG] + \
                evaluation_logic.build_eval_args(params)
        else:
            cmd = evaluation_logic.build_eval_command(self._python_for_subprocess(), params)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(str(DATA_DIR.parent))
        proc.readyReadStandardOutput.connect(self._on_eval_stdout)
        proc.finished.connect(self._on_eval_finished)
        proc.errorOccurred.connect(self._on_eval_error)
        self._eval_process = proc
        self._eval_buffer = ""
        self._eval_last_model = params["model"]

        self.eval_metrics_text.setPlainText("Running evaluation...\n$ " + " ".join(cmd) + "\n")
        self.eval_start_btn.setEnabled(False)
        self.promote_btn.setEnabled(False)
        self.status.showMessage("Evaluating model...", 5000)
        proc.start(cmd[0], cmd[1:])

    def _on_eval_stdout(self) -> None:
        if self._eval_process is None:
            return
        data = bytes(self._eval_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._eval_buffer += data

    def _on_eval_error(self, _error) -> None:
        self.eval_metrics_text.append("\n[error] Could not run evaluation. Check that Ultralytics is installed.")

    def _on_eval_finished(self, exit_code: int, _status) -> None:
        metrics = evaluation_logic.parse_metrics_output(self._eval_buffer)
        self._eval_process = None
        self.eval_start_btn.setEnabled(True)
        if exit_code == 0 and metrics:
            self.eval_metrics_text.setPlainText(evaluation_logic.format_metrics(metrics))
            self.promote_btn.setEnabled(bool(self._eval_last_model))
            self.status.showMessage("Evaluation complete.", 6000)
        else:
            tail = "\n".join(self._eval_buffer.strip().splitlines()[-15:])
            self.eval_metrics_text.setPlainText(
                f"Evaluation exited with code {exit_code} and no metrics were parsed.\n\n{tail}"
            )
            self.status.showMessage("Evaluation failed; see the metrics panel.", 8000)

    def promote_model(self) -> None:
        model = self._eval_last_model or self.eval_model_edit.text().strip()
        if not model or not Path(model).exists():
            QMessageBox.information(self, "Promote", "Evaluate a model first; its file must exist to promote.")
            return
        import shutil
        models_dir = DATA_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        name = (self.train_name_edit.text().strip() or "model")
        dest = models_dir / f"{name}{Path(model).suffix or '.pt'}"
        try:
            shutil.copy2(model, dest)
        except Exception as e:
            QMessageBox.warning(self, "Promote", f"Could not copy model:\n{e}")
            return
        # Make the promoted model the active one for test/auto-label/count/queue.
        if hasattr(self, "test_model_edit"):
            self.test_model_edit.setText(str(dest))
        QMessageBox.information(
            self, "Promote",
            f"Promoted model to:\n{dest}\n\nIt is now the active model for Test, Auto-label, Count, and the review queue.",
        )
        self.status.showMessage(f"Promoted model: {dest.name}", 8000)

    def _gather_train_params(self) -> dict:
        return {
            "task": self.train_task_combo.currentText(),
            "model": self.train_model_edit.text().strip(),
            "data": self.train_data_edit.text().strip(),
            "imgsz": int(self.train_imgsz_spin.value()),
            "batch": int(self.train_batch_spin.value()),
            "epochs": int(self.train_epochs_spin.value()),
            "patience": int(self.train_patience_spin.value()),
            "workers": int(self.train_workers_spin.value()),
            "device": self.train_device_edit.text().strip(),
            "project": self.train_project_edit.text().strip(),
            "name": self.train_name_edit.text().strip(),
            "resume": bool(self.train_resume_check.isChecked()),
        }

    def browse_train_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select base model", "", "Model (*.pt *.yaml);;All files (*)")
        if path:
            self.train_model_edit.setText(path)

    def browse_train_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", str(EXPORT_DIR), "YAML (*.yaml *.yml);;All files (*)")
        if path:
            self.train_data_edit.setText(path)

    def browse_train_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", self.train_project_edit.text().strip() or str(DATA_DIR))
        if path:
            self.train_project_edit.setText(path)

    def use_latest_export_for_training(self) -> None:
        candidates = [p / "data.yaml" for p in EXPORT_DIR.glob("*") if (p / "data.yaml").exists()]
        if not candidates:
            QMessageBox.information(self, "Train", "No exports found. Export a reviewed dataset first.")
            return
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        self.train_data_edit.setText(str(latest))
        # If the export recorded its task, match the training task to it.
        task_file = latest.parent / "task.txt"
        if task_file.exists():
            try:
                task = task_file.read_text(encoding="utf-8").strip().lower()
                if task in training_logic.VALID_TASKS:
                    self.train_task_combo.setCurrentText(task)
            except Exception:
                pass
        self.status.showMessage(f"Using dataset: {latest}", 6000)

    def start_training(self) -> None:
        if self._train_process is not None:
            QMessageBox.information(self, "Train", "A training run is already in progress.")
            return
        params = self._gather_train_params()
        errors = training_logic.validate_train_params(params)
        if errors:
            QMessageBox.warning(self, "Train", "Cannot start training:\n\n" + "\n".join(f"• {e}" for e in errors))
            return

        yolo_exe = self.train_yolo_exe_edit.text().strip() or "yolo"
        if getattr(sys, "frozen", False):
            # A packaged build has no `yolo` CLI and no system Python, but it
            # does bundle Ultralytics -- the same copy Auto-label already uses.
            # Re-invoke ourselves as a worker rather than requiring a separate
            # install just to train.
            cmd = training_logic.build_worker_train_command(sys.executable, params)
        else:
            cmd = training_logic.build_train_command(yolo_exe, params)

        # Persist for next session.
        settings = dict(params)
        settings["yolo_exe"] = yolo_exe
        try:
            save_training_settings(settings)
        except Exception:
            pass

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(str(DATA_DIR.parent))
        proc.readyReadStandardOutput.connect(self._on_train_stdout)
        proc.finished.connect(self._on_train_finished)
        proc.errorOccurred.connect(self._on_train_error)
        self._train_process = proc

        self.train_log.clear()
        self.train_log.append("$ " + " ".join(cmd) + "\n")
        self.train_start_btn.setEnabled(False)
        self.train_stop_btn.setEnabled(True)
        self.status.showMessage("Training started...", 5000)

        # Clear the chart immediately so the previous run's curves don't persist.
        if hasattr(self, "train_metrics_chart"):
            self.train_metrics_chart.clear()

        # Determine which results.csv to follow.  YOLO creates a NEW directory
        # when <project>/<name> already exists (bungvision -> bungvision2 or
        # bungvision-2 depending on version), so we cannot trust the bare path:
        # the base dir's results.csv is stale.  Instead we record the start time
        # and, while polling, lock onto whichever <name>* results.csv was written
        # AFTER training started.
        self._train_project = params.get("project") or "data/training"
        self._train_name = params.get("name") or "bungvision"
        self._train_start_time = time.time()
        self._train_stopped = False
        self._results_csv_path = None  # resolved on first successful poll

        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.stop()
            self._metrics_timer.start()

        proc.start(cmd[0], cmd[1:])

    def _resolve_results_csv(self) -> Path | None:
        """Find the results.csv for the active run.

        Once locked, keep using it.  Otherwise scan <project>/<name>* for a
        results.csv modified at/after training start — that excludes the stale
        base-directory file from a previous run and picks the freshly created
        numbered directory (bungvision2 / bungvision-2 / ...).
        """
        path = getattr(self, "_results_csv_path", None)
        if path is not None and Path(path).exists():
            return Path(path)
        project = getattr(self, "_train_project", None)
        name = getattr(self, "_train_name", None)
        if not project or not name:
            return None
        import glob as _glob
        start = getattr(self, "_train_start_time", 0.0)
        pattern = str(Path(project) / f"{name}*" / "results.csv")
        fresh = []
        for p in _glob.glob(pattern):
            pp = Path(p)
            try:
                if pp.exists() and pp.stat().st_mtime >= start - 1.0:
                    fresh.append(pp)
            except OSError:
                continue
        if not fresh:
            return None
        chosen = max(fresh, key=lambda p: p.stat().st_mtime)
        self._results_csv_path = chosen
        return chosen

    def _poll_training_metrics(self) -> None:
        """Re-read the active run's results.csv and refresh the live chart."""
        if not hasattr(self, "train_metrics_chart"):
            return
        path = self._resolve_results_csv()
        if path is None:
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        rows = training_logic.parse_results_csv(text)
        if rows:
            epochs = training_logic.metric_series(rows, "epoch")
            series = training_logic.chart_series(rows)
            self.train_metrics_chart.set_data(epochs, series)

    def _on_train_stdout(self) -> None:
        if self._train_process is None:
            return
        data = bytes(self._train_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self.train_log.moveCursor(QTextCursor.End)
        self.train_log.insertPlainText(data)
        self.train_log.moveCursor(QTextCursor.End)
        # Detect YOLO's "Results saved to <dir>" line so we follow the actual
        # output directory even when YOLO appended a numeric suffix (bungvision2,
        # bungvision3 ...) because the run name already existed on disk.
        for line in data.splitlines():
            line = line.strip()
            # Ultralytics prints: "Results saved to runs/obb/train2" or similar.
            # The CWD for the subprocess is the project root so the path may be
            # relative, and ANSI escape codes may surround it.
            if "results saved to" in line.lower():
                import re
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                m = re.search(r"results saved to\s+(.+)", clean, re.IGNORECASE)
                if m:
                    save_dir = Path(m.group(1).strip())
                    if not save_dir.is_absolute():
                        save_dir = Path(DATA_DIR.parent) / save_dir
                    candidate = save_dir / "results.csv"
                    self._results_csv_path = candidate

    def _on_train_error(self, _error) -> None:
        if self._train_process is None:
            return
        self.train_log.append(
            f"\n[error] Could not run '{self.train_yolo_exe_edit.text().strip() or 'yolo'}'. "
            "Check that Ultralytics is installed and the yolo executable is correct."
        )

    def _on_train_finished(self, exit_code: int, _status) -> None:
        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.stop()
        self._poll_training_metrics()  # final refresh to catch the last epoch row
        self.train_log.append(f"\n[done] Training process exited with code {exit_code}.")

        elapsed = time.time() - getattr(self, "_train_start_time", time.time())
        stopped = getattr(self, "_train_stopped", False)
        csv_path = self._resolve_results_csv()
        run_dir = csv_path.parent if csv_path else None
        weights = (run_dir / "weights" / "best.pt") if run_dir else None

        if weights:
            self.train_log.append(f"[done] Best weights (if produced): {weights}")
        if exit_code == 0 and not stopped:
            self.status.showMessage("Training finished.", 8000)
        else:
            self.status.showMessage(f"Training exited with code {exit_code}.", 8000)

        self.train_start_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self._train_process = None
        self._show_training_summary(exit_code, stopped, elapsed, csv_path, run_dir, weights)

    def _show_training_summary(self, exit_code, stopped, elapsed, csv_path, run_dir, weights) -> None:
        """Popup summarizing the finished run: validation metrics, time, paths."""
        params = self._gather_train_params()
        dur = training_logic.format_duration(elapsed)

        summary = {}
        if csv_path and Path(csv_path).exists():
            try:
                rows = training_logic.parse_results_csv(Path(csv_path).read_text(encoding="utf-8", errors="replace"))
                summary = training_logic.summarize_results(rows)
            except Exception:
                summary = {}

        lines: list[str] = []
        if stopped:
            headline = "Training stopped by user."
        elif exit_code == 0:
            headline = "Training completed successfully."
        else:
            headline = f"Training exited with code {exit_code} (it may not have finished)."
        lines.append(headline)
        lines.append("")
        lines.append(f"Task / model: {params.get('task')} · {params.get('model')}")
        lines.append(f"Dataset: {params.get('data') or '(none)'}")
        lines.append(f"Time spent training: {dur}")

        epochs_done = summary.get("rows", 0)
        if epochs_done:
            lines.append(f"Epochs recorded: {epochs_done} (requested {params.get('epochs')})")

        def _metric_block(title: str, metrics: dict, epoch: int | None = None) -> None:
            if not metrics:
                return
            suffix = f" (epoch {epoch})" if epoch is not None else ""
            lines.append("")
            lines.append(f"{title}{suffix}:")
            order = ["precision", "recall", "mAP50", "mAP50-95"]
            for key in order:
                if key in metrics:
                    lines.append(f"  • {key}: {metrics[key]:.4f}")

        if summary.get("final") or summary.get("best"):
            _metric_block("Final validation metrics", summary.get("final", {}))
            _metric_block("Best validation metrics", summary.get("best", {}), summary.get("best_epoch"))
        else:
            lines.append("")
            lines.append("No validation metrics were found in results.csv for this run.")

        lines.append("")
        if weights and Path(weights).exists():
            lines.append(f"Best weights: {weights}")
            lines.append("Use the buttons below to make this the active model or continue training from it.")
        elif weights:
            lines.append(f"Best weights (expected): {weights}")
        if run_dir:
            lines.append(f"Run folder: {run_dir}")

        box = QMessageBox(self)
        box.setWindowTitle("Training Summary")
        box.setIcon(QMessageBox.Information if (exit_code == 0 and not stopped) else QMessageBox.Warning)
        box.setText(headline)
        box.setInformativeText("\n".join(lines[2:]))  # body after the headline + blank

        # One-click follow-ups when best.pt actually exists: skip the manual
        # copy-the-path dance between the Train, Test, and Evaluate tabs.
        have_weights = bool(weights and Path(weights).exists())
        last_weights = (Path(run_dir) / "weights" / "last.pt") if run_dir else None
        can_resume = bool(stopped and last_weights and last_weights.exists())
        use_btn = train_more_btn = resume_btn = None
        if have_weights:
            use_btn = box.addButton("Use as active model", QMessageBox.AcceptRole)
            if can_resume:
                # Interrupted run: Ultralytics can resume last.pt to its original
                # epoch target. (A completed run cannot be resumed.)
                resume_btn = box.addButton("Resume training", QMessageBox.ActionRole)
            else:
                # Completed run: continue from best.pt as a fresh fine-tune run.
                train_more_btn = box.addButton("Train more from best.pt", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()

        clicked = box.clickedButton()
        if have_weights and clicked is use_btn:
            self._use_trained_as_active(Path(weights))
        elif can_resume and clicked is resume_btn:
            self._resume_training_from(last_weights)
        elif have_weights and clicked is train_more_btn:
            self._finetune_training_from(Path(weights))

    def _use_trained_as_active(self, weights: Path) -> None:
        """Make a finished run's best.pt the active model for Test/Auto-label/etc."""
        if hasattr(self, "test_model_edit"):
            self.test_model_edit.setText(str(weights))
        if hasattr(self, "eval_model_edit"):
            self.eval_model_edit.setText(str(weights))
        if hasattr(self, "_test_tab_widget") and hasattr(self, "tabs"):
            self.tabs.setCurrentWidget(self._test_tab_widget)
        self.status.showMessage(
            f"Active model set to {weights.name}. Used by Test, Auto-label, Count, Pre-label, and the review queue.",
            8000,
        )

    def _resume_training_from(self, weights: Path) -> None:
        """Pre-fill the Train tab to resume an interrupted run from its last.pt.

        resume=True tells Ultralytics to continue the same run to its original
        epoch target; the model field carries the last.pt checkpoint.
        """
        if hasattr(self, "train_model_edit"):
            self.train_model_edit.setText(str(weights))
        if hasattr(self, "train_resume_check"):
            self.train_resume_check.setChecked(True)
        if hasattr(self, "_train_tab_widget") and hasattr(self, "tabs"):
            self.tabs.setCurrentWidget(self._train_tab_widget)
        self.status.showMessage(
            f"Train tab ready to resume from {weights.name}. Review settings, then Start Training.",
            8000,
        )

    def _finetune_training_from(self, weights: Path) -> None:
        """Pre-fill the Train tab to fine-tune from a completed run's best.pt.

        This starts a fresh run initialized from best.pt (resume stays off, since
        a finished run cannot be resumed). The run name is bumped so the new run
        does not collide with the original output folder.
        """
        if hasattr(self, "train_model_edit"):
            self.train_model_edit.setText(str(weights))
        if hasattr(self, "train_resume_check"):
            self.train_resume_check.setChecked(False)
        if hasattr(self, "train_name_edit"):
            self.train_name_edit.setText(self._next_run_name(self.train_name_edit.text().strip()))
        if hasattr(self, "_train_tab_widget") and hasattr(self, "tabs"):
            self.tabs.setCurrentWidget(self._train_tab_widget)
        self.status.showMessage(
            f"Train tab ready to fine-tune from {weights.name} as a new run. Adjust epochs, then Start Training.",
            8000,
        )

    @staticmethod
    def _next_run_name(name: str) -> str:
        """Bump a trailing -N suffix so a fine-tune run gets a fresh folder."""
        name = name or "bungvision"
        import re
        m = re.search(r"^(.*?)(?:[-_]ft(\d+))?$", name)
        base = m.group(1) if m else name
        n = int(m.group(2)) + 1 if (m and m.group(2)) else 2
        return f"{base}-ft{n}"

    def stop_training(self) -> None:
        if self._train_process is None:
            return
        self._train_stopped = True
        self.train_log.append("\n[stop] Stopping training...")
        self._train_process.kill()

    def _help_tab(self) -> QWidget:
        outer, w, layout = self._scrollable_tab()

        title = QLabel(f"{APP_TITLE} — built-in workflow guide")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #bfdbfe;")
        layout.addWidget(title)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setMinimumHeight(560)
        body.setPlainText(
            "Purpose\n"
            "This tool produces one trained label at a time. It does not decide which\n"
            "labels a battery must carry, or where each one belongs -- that is the\n"
            "recipe, and it is authored in the vision front end.\n"
            "\n"
            "What the model reports\n"
            "Label ids. The front end's recipe is a list of label ids and quantities,\n"
            "so the detector is trained on exactly that -- what it returns is already\n"
            "countable against a recipe, with nothing in between to resolve.\n"
            "battery_side is the one exception: it is the whole face, not a label.\n"
            "\n"
            "Two ways to train, and when each is right\n"
            "SINGLE-STAGE (Export All): one detector, one class per label id, one pass.\n"
            "Simplest to run and deploy. Right when your labels look clearly different\n"
            "from each other.\n"
            "\n"
            "TWO-STAGE (Export Two-Stage): a detector that finds WHERE a label is (one\n"
            "generic `label` class), then a classifier over the cropped label that\n"
            "decides WHICH. Right when labels differ in fine detail -- a revision\n"
            "letter, a language, a small block of text.\n"
            "The reason is resolution. Detection sees the whole frame at 640-1024 px, so\n"
            "a label 100 px wide gives it 100 px to read a revision letter off. Cropped\n"
            "and resized, the classifier gets 224 px of label and that same letter lands\n"
            "at 25-40 px. Training cannot recover detail the pixels never held.\n"
            "Two-stage also makes a new label cheaper: the detector already finds labels\n"
            "it has never seen, so only the classifier is retrained.\n"
            "Both halves export from the SAME annotations and share one split and seed,\n"
            "so they hold out the same batteries. Nothing is labeled twice.\n"
            "\n"
            "Not sure? Start single-stage. Move to two-stage when two labels get\n"
            "confused for each other in Live Detect.\n"
            "\n"
            "Workflow for one label\n"
            "1. Label tab -> Add Label... Answer the wizard: artwork, surface, rotation,\n"
            "   any barcodes and where they sit on the artwork.\n"
            "2. Capture or import images of that label into its dataset.\n"
            "3. Pick the label in the Class box and draw its oriented box. The class\n"
            "   IS the label id, so the box carries its identity as drawn.\n"
            "4. Save. An image carrying the label is approved by saving; one that does\n"
            "   not is saved un-reviewed, because editing is not approving.\n"
            "5. Mark Background for negatives -- a bare fixture, a battery without\n"
            "   this label. They teach the model where NOT to fire.\n"
            "6. Force Review for deliberate defect examples. It asks what is wrong\n"
            "   (torn, smeared code, wrong revision...) and records the answer, so the\n"
            "   defect library stays queryable.\n"
            "7. Dataset Health shows how close each label is to its training target.\n"
            "8. Export (All, or Two-Stage), then Train.\n"
            "9. Live Detect: point the camera at a real battery and watch what the\n"
            "   model says. Keep the frames it gets wrong -- straight back to step 3.\n"
            "\n"
            "Export trains every label at once\n"
            "Labels are gathered and reviewed one at a time, but they are TRAINED\n"
            "together: one detector over every label id. A model trained on a single\n"
            "label has nothing to tell it apart from, and will report the one class it\n"
            "knows on anything label-shaped.\n"
            "The split never separates a capture group -- burst frames of the same\n"
            "physical label stay on one side, or validation just measures memorisation.\n"
            "split_report.txt ships inside every export so you can check that.\n"
            "\n"
            "Reviewed-only export\n"
            "Only images approved inside this tool can train a model. A generic\n"
            "reviewed:true field from another program does not count, on purpose.\n"
            "\n"
            "OBB controls\n"
            "- Drag to draw, then move the four corner handles.\n"
            "- Right-click or Ctrl-click selects an annotation.\n"
            "- Arrow keys nudge; Shift+Arrow nudges 10 pixels.\n"
            "- Mouse wheel zooms. Middle-drag or Alt-drag pans.\n"
            "- Ctrl+Z / Ctrl+Y undo and redo.\n"
            "Labels sit on curved, tilted faces, so oriented boxes are the norm; an\n"
            "axis-aligned box around a rotated label swallows its neighbours.\n"
            "\n"
            "Detector classes\n"
            "One class per label id, plus battery_side for the whole face. The class\n"
            "list IS the label library -- there is nothing separate to maintain, and\n"
            "nothing that can disagree with it.\n"
            "So the model reports the id the recipe is written in, with no second step\n"
            "to resolve what it found. The price is that a new label is not detected\n"
            "until it has images and a training run: define it, gather its dataset,\n"
            "Export All, retrain.\n"
            "battery_side holds class 0 and the rest sort by id, so the numbering only\n"
            "shifts when labels are added -- retrain after that, since a model maps\n"
            "class id to name by position.\n"
            "\n"
            "What this tool does NOT do\n"
            "No recipes: which labels a battery must carry, how many, and where, is\n"
            "authored in the front end.\n"
            "No pass/fail: Live Detect validates the model, it does not inspect. A\n"
            "second half-built HMI in the labeling tool would be the worst of both.\n"
            "No code reading: read-regions record WHERE a barcode or a date code sits\n"
            "on the label, as fractions of it, for the front end to read. Nothing here\n"
            "decodes them."
        )
        layout.addWidget(body)
        return outer
    def _live_detect_tab(self) -> QWidget:
        """Watch the model work through the real camera, and keep what it fails on.

        Deliberately not an inspection view: no verdict, no latching, no reject
        output. Whether a battery passes is the front end's decision, and a
        second half-built HMI in the labeling tool would be the worst of both.
        What this is for is the question a saved-image test cannot answer --
        does the model work through *this* lens, at *this* standoff, under
        *this* light -- and the loop back into labeling when it does not.
        """
        outer, w, layout = self._scrollable_tab()

        info = QLabel(
            "Runs the Test Models tab's model on the live camera. This is model "
            "validation, not inspection: nothing here passes or fails a battery."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #9aa4b2;")
        layout.addWidget(info)

        control_box = QGroupBox("Live detection")
        cv_ = QVBoxLayout(control_box)
        cv_.setContentsMargins(8, 8, 8, 8)
        cv_.setSpacing(4)

        self.live_start_btn = QPushButton("Start Live Detect")
        self.live_start_btn.setToolTip(
            "Opens the camera if it is not already running, loads the model off "
            "the GUI thread, and overlays detections on the live view.")
        self.live_start_btn.clicked.connect(self.start_live_detect)
        self.live_stop_btn = QPushButton("Stop")
        self.live_stop_btn.setEnabled(False)
        self.live_stop_btn.clicked.connect(self.stop_live_detect)
        for _b in (self.live_start_btn, self.live_stop_btn):
            _b.setProperty("compactCaptureButton", True)
        row = QHBoxLayout()
        row.addWidget(self.live_start_btn)
        row.addWidget(self.live_stop_btn)
        cv_.addLayout(row)

        cls_row = QHBoxLayout()
        self.live_classifier_edit = QLineEdit()
        self.live_classifier_edit.setPlaceholderText(
            "optional: classifier .pt for stage 2 (leave blank for stage 1 only)")
        self.live_classifier_edit.setToolTip(
            "The second stage. Each detection is cropped out of the FULL-RESOLUTION "
            "frame and this model names it.\n\n"
            "Leave blank and boxes show whatever class the detector reports -- "
            "which under a two-stage export is just 'label', with no identity.")
        cls_browse = QPushButton("...")
        cls_browse.setMaximumWidth(34)
        cls_browse.clicked.connect(self._browse_live_classifier)
        cls_row.addWidget(self.live_classifier_edit, 1)
        cls_row.addWidget(cls_browse)
        cv_.addWidget(QLabel("Stage 2 classifier"))
        cv_.addLayout(cls_row)

        self.live_track_check = QCheckBox("Track objects across frames")
        self.live_track_check.setChecked(True)
        self.live_track_check.setToolTip(
            "Gives each object a stable id, so confidence can be read as a held "
            "average instead of a per-frame flicker.\n\n"
            "An object that keeps being lost and re-acquired under a new id is "
            "the failure a single confidence number hides completely.\n\n"
            "Takes effect on the next Start.")
        cv_.addWidget(self.live_track_check)

        self.live_status_label = QLabel("Stopped.")
        self.live_status_label.setWordWrap(True)
        cv_.addWidget(self.live_status_label)

        self.live_readout = QTextEdit()
        self.live_readout.setReadOnly(True)
        self.live_readout.setMinimumHeight(120)
        self.live_readout.setPlaceholderText(
            "Detections, latency and rate appear here once it is running.")
        cv_.addWidget(self.live_readout)

        boundary = QLabel(
            "The detector is trained on label ids, so what it reports here is "
            "the same identity the recipe is written in -- no resolution step "
            "in between. A label absent from the trained model cannot appear, "
            "however clearly it is on the battery."
        )
        boundary.setWordWrap(True)
        boundary.setStyleSheet("color: #9aa4b2;")
        cv_.addWidget(boundary)
        layout.addWidget(control_box)

        keep_box = QGroupBox("Keep what it fails on")
        kv = QVBoxLayout(keep_box)
        kv.setContentsMargins(8, 8, 8, 8)
        kv.setSpacing(4)
        note = QLabel(
            "A frame the model handles badly is the most valuable training image "
            "available. Keeping it puts it straight into this label's dataset, "
            "un-reviewed, ready to label.\n\n"
            "Keep the image alone when the model got it wrong: a bad box gets "
            "nudged rather than redrawn, so wrong proposals end up in the "
            "dataset as slightly-wrong truth. Keep the detections too when it "
            "was close, and correct them instead of drawing from nothing."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa4b2;")
        kv.addWidget(note)

        self.live_keep_btn = QPushButton("Keep Image Only")
        self.live_keep_btn.setToolTip(
            "Save the frame currently on screen into this label's dataset, with "
            "no annotation at all. Label it from scratch. (Ctrl+K)")
        self.live_keep_btn.clicked.connect(self.keep_live_frame)

        self.live_keep_json_btn = QPushButton("Keep Image + Detections")
        self.live_keep_json_btn.setToolTip(
            "Save the frame together with a sidecar holding the boxes the model "
            "just found, so review is correction rather than drawing.\n\n"
            "Saves the frame the boxes were computed on, which can be a frame or "
            "two behind the preview -- an image and boxes that disagree would be "
            "worse than either alone.\n\n"
            "Written un-reviewed, and marked as machine-proposed. (Ctrl+Shift+K)")
        self.live_keep_json_btn.clicked.connect(self.keep_live_frame_with_detections)

        for _b in (self.live_keep_btn, self.live_keep_json_btn):
            _b.setProperty("compactCaptureButton", True)
        keep_row = QHBoxLayout()
        keep_row.addWidget(self.live_keep_btn)
        keep_row.addWidget(self.live_keep_json_btn)
        kv.addLayout(keep_row)

        self.live_auto_check = QCheckBox("Keep frames the model struggles with")
        self.live_auto_check.setToolTip(
            "Walk batteries past the camera and it collects the hard ones by "
            "itself. Rate-limited and capped per session: one battery it cannot "
            "handle would otherwise produce hundreds of near-identical frames, "
            "which is worse than none -- they all say the same thing and each "
            "costs a review.\n\n"
            "Always image-only. It fires on frames the model disagreed with "
            "itself about, which is exactly where its boxes are worth least.")
        self.live_auto_check.toggled.connect(self._on_live_auto_toggled)
        kv.addWidget(self.live_auto_check)

        self.live_capture_label = QLabel("")
        self.live_capture_label.setWordWrap(True)
        self.live_capture_label.setStyleSheet("color: #93c5fd;")
        kv.addWidget(self.live_capture_label)
        layout.addWidget(keep_box)

        return outer
    # --- live detection ---------------------------------------------------

    def start_live_detect(self) -> None:
        """Load the model on its own thread and start overlaying the live view."""
        if getattr(self, "_live_thread", None) is not None:
            return
        model_path = self.test_model_edit.text().strip()
        if not model_path:
            QMessageBox.information(
                self, "Live Detect",
                "Pick a model on the Test Models tab first -- live detection uses "
                "the same one, so there is only ever one model in play.")
            return
        if not self._camera_is_live():
            self.open_camera()
            if not self._camera_is_live():
                return

        from PySide6.QtCore import QThread
        from .live_detect import InferenceWorker

        self._live_rolling = live_logic.Rolling()
        self._live_gate = live_logic.CaptureGate()
        self._live_tracks = live_logic.TrackBook()
        self._live_last_started = 0.0
        self._live_busy = False
        self._live_frame = None
        self._live_counts = {}
        self._live_result_frame = None
        self._live_result_items = []
        self._live_session = live_logic.proposal_session(time.time())

        self._live_thread = QThread(self)
        self._live_tracking = self.live_track_check.isChecked()
        classifier_path = (self.live_classifier_edit.text().strip()
                           if hasattr(self, "live_classifier_edit") else "")
        self._live_worker = InferenceWorker(
            model_path, int(self.test_imgsz_spin.value()),
            float(self.test_conf_spin.value()), self._model_test_device_arg(),
            track=self._live_tracking, classifier_path=classifier_path,
            crop_px=self._live_crop_px(classifier_path))
        self._live_worker.moveToThread(self._live_thread)
        self._live_worker.loaded.connect(self._on_live_loaded)
        self._live_worker.failed.connect(self._on_live_failed)
        self._live_worker.result.connect(self._on_live_result)
        self._live_thread.started.connect(self._live_worker.load)
        self._live_thread.start()

        self.live_start_btn.setEnabled(False)
        self.live_stop_btn.setEnabled(True)
        self.live_status_label.setText("Loading the model...")
        self.status.showMessage("Live detect: loading model", 4000)

    def stop_live_detect(self) -> None:
        thread = getattr(self, "_live_thread", None)
        worker = getattr(self, "_live_worker", None)
        if worker is not None:
            worker.stop()
        if thread is not None:
            thread.quit()
            # Bounded: a stuck inference must not hang the window on close.
            thread.wait(3000)
        self._live_thread = None
        self._live_worker = None
        self._live_busy = False
        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if hasattr(self, "live_start_btn"):
            self.live_start_btn.setEnabled(True)
            self.live_stop_btn.setEnabled(False)
            self.live_status_label.setText("Stopped.")
        self.status.showMessage("Live detect stopped", 4000)

    def toggle_live_detect(self) -> None:
        if self._live_running():
            self.stop_live_detect()
        else:
            self.start_live_detect()
    def _live_running(self) -> bool:
        return getattr(self, "_live_thread", None) is not None

    def _on_live_loaded(self, message: str) -> None:
        self.live_status_label.setText(f"{message}\nWatching the live view.")

    def _on_live_failed(self, message: str) -> None:
        self._live_busy = False
        self.live_status_label.setText(f"Inference problem: {message}")

    def _pump_live_detect(self, frame) -> None:
        """Hand the model a frame, if it is idle and enough time has passed.

        Called from the camera tick. Skipping while busy is what keeps the view
        live: the preview runs at the camera's rate and overlays refresh
        whenever the model finishes, instead of the preview stuttering along at
        the model's pace.
        """
        if not self._live_running() or frame is None:
            return
        now = time.monotonic()
        if not live_logic.should_infer(self._live_busy, now - self._live_last_started):
            return
        self._live_busy = True
        self._live_last_started = now
        self._live_frame = frame.copy()
        # Queued across the thread boundary by Qt, so this returns immediately.
        self._live_worker.infer(self._live_frame)

    def _on_live_result(self, results, latency: float, identities=None) -> None:
        self._live_busy = False
        self._live_rolling.record(latency)
        items, _counts = self._detection_overlay_items(results)
        items = live_logic.apply_identities(items, identities)
        # Counted after stage 2, so the readout reports label ids rather than
        # a screenful of "label" -- which is what the recipe is written in.
        counts: dict[str, int] = {}
        for item in items:
            counts[item["name"]] = counts.get(item["name"], 0) + 1
        self._live_counts = counts
        # Hold the frame *these* boxes came from. By the time an operator
        # clicks, _live_frame has usually moved on, and saving a newer image
        # against older boxes writes a sidecar that is wrong everywhere.
        self._live_result_frame = self._live_frame
        self._live_result_items = items
        if hasattr(self.canvas, "set_model_test_overlays"):
            self.canvas.set_model_test_overlays(
                self._scaled_overlay_items(items, self._live_overlay_scale))

        if getattr(self, "_live_tracking", False):
            self._live_tracks.update(
                [(i.get("track_id"), i.get("name"), i.get("conf", 0.0)) for i in items])
            self.live_readout.setPlainText(
                live_logic.track_summary(self._live_tracks, self.label_id,
                                         self._live_rolling))
        else:
            self.live_readout.setPlainText(live_logic.frame_summary(
                counts, self.label_id, self._live_rolling))

        if not self.live_auto_check.isChecked():
            return
        found, total, avg_conf = self._detection_disagreement(results)
        score = active_learning.disagreement_score(found, 1, total, avg_conf)
        keep, reason = self._live_gate.consider(score)
        if keep:
            self._keep_frame(self._live_frame, reason)
        self.live_capture_label.setText(live_logic.capture_note(
            self._live_gate.captured, self._live_gate.limit, reason))

    @staticmethod
    def _scaled_overlay_items(items: list[dict], scale) -> list[dict]:
        """Move overlays from full-frame coordinates into the preview's.

        Skipped entirely at 1:1, which is the common case once preview scaling
        is off -- copying every point to multiply it by one is pure waste on a
        path that runs several times a second.
        """
        sx, sy = float(scale[0]), float(scale[1])
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return items
        out: list[dict] = []
        for item in items:
            scaled = dict(item)
            if item.get("points"):
                scaled["points"] = [[x * sx, y * sy] for x, y in item["points"]]
            if item.get("xyxy"):
                x1, y1, x2, y2 = item["xyxy"]
                scaled["xyxy"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            if "cx" in item:
                scaled["cx"] = item["cx"] * sx
            if "cy" in item:
                scaled["cy"] = item["cy"] * sy
            out.append(scaled)
        return out

    def _on_live_auto_toggled(self, armed: bool) -> None:
        if armed:
            self._live_gate = live_logic.CaptureGate()
            self.live_capture_label.setText(live_logic.capture_note(
                0, live_logic.CAPTURE_SESSION_LIMIT, ""))
        else:
            self.live_capture_label.setText("")

    def keep_live_frame(self) -> None:
        """Save what is on screen into this label's dataset, by hand."""
        frame = getattr(self, "_live_frame", None)
        if frame is None:
            frame = self.last_raw
        if frame is None:
            QMessageBox.information(
                self, "Keep Frame",
                "No frame yet. Open the live preview and try again.")
            return
        self._keep_frame(frame, "kept by hand")

    def keep_live_frame_with_detections(self) -> None:
        """Save the last inferred frame plus the boxes the model found on it.

        Deliberately the frame inference ran on, not the freshest one on
        screen: those boxes describe that image and no other.
        """
        frame = getattr(self, "_live_result_frame", None)
        items = getattr(self, "_live_result_items", None) or []
        if frame is None:
            QMessageBox.information(
                self, "Keep Frame",
                "Nothing has been through the model yet. Start live detect and "
                "give it a frame first.")
            return
        if not items:
            # Degrade rather than refuse: the operator wanted this frame kept,
            # and an empty sidecar would read as "labeled, nothing in it".
            self._keep_frame(frame, "kept by hand -- no detections to propose")
            return
        self._keep_frame(frame, f"kept with {len(items)} proposed box(es)",
                         items=items)

    def closeEvent(self, event) -> None:
        """Bring the inference thread down before the window goes.

        A QThread outliving its window is how a Qt app exits with a crash
        instead of a return code, and the operator sees a dialog rather than a
        clean shutdown.
        """
        try:
            if self._live_running():
                self.stop_live_detect()
            if self._camera_is_live():
                self.close_camera()
        except Exception:
            pass
        super().closeEvent(event)
    def _keep_frame(self, frame, reason: str, items=None) -> None:
        """Save a live frame into the open label's dataset.

        With ``items`` it also writes an un-reviewed sidecar holding those
        detections as proposals. Without, no sidecar is written at all and the
        image lands unlabeled.
        """
        if not self.label_id:
            QMessageBox.information(
                self, "Keep Frame",
                "Open a label first -- the frame goes into that label's dataset.")
            return
        raw_path, _adjusted = save_capture(self.label_id, frame, None, save_raw=True)
        if raw_path is None:
            return
        if items:
            height, width = (int(frame.shape[0]), int(frame.shape[1])) \
                if getattr(frame, "shape", None) else (0, 0)
            data = live_logic.proposed_annotation(
                raw_path.name, self.label_id, items,
                width=width, height=height,
                session=getattr(self, "_live_session", ""))
            persistence.save_annotation(self.label_id, raw_path.name, data)
        if hasattr(self, "_live_gate"):
            self._live_gate.mark()
        self._last_capture_path = raw_path
        self._dataset_index_dirty = True
        self._refresh_images(force=True)
        self._update_dataset_summary()
        self.status.showMessage(
            f"Kept {raw_path.name} into {self.label_id} — {reason}", 6000)
    def _label_tab(self) -> QWidget:
        """The label library and the dataset for the label being trained.

        This is where the recipe tab used to be, and it is deliberately the same
        shape: pick a thing on the left, its images fill the list, everything
        else in the app follows. What changed is what you pick. A recipe said
        "this battery model needs six bungs"; a label says "this is the 31-AGM
        spec plate, here is what it looks like, here are its images". Which
        labels a battery must carry, and where each one belongs, is the front
        end's business and is authored there.
        """
        outer, w, layout = self._scrollable_tab()

        info_box = QGroupBox("Label being trained")
        info = QFormLayout(info_box)
        self.label_id_label = QLabel("—")
        self.label_id_label.setWordWrap(True)
        self.label_id_label.setStyleSheet("color: #93c5fd; font-weight: 700;")
        self.label_meta_label = QLabel("—")
        self.label_meta_label.setWordWrap(True)
        self.label_meta_label.setStyleSheet("color: #94a3b8;")
        self.label_progress_label = QLabel("—")
        self.label_progress_label.setWordWrap(True)
        info.addRow("Label", self.label_id_label)
        info.addRow("Details", self.label_meta_label)
        info.addRow("Dataset", self.label_progress_label)

        add_btn = QPushButton("Add Label...")
        add_btn.setToolTip(
            "Define a new label: its artwork, size, codes and how it should be "
            "inspected. Adding one never retrains anything else."
        )
        add_btn.clicked.connect(self.add_label_via_wizard)
        edit_btn = QPushButton("Edit Selected")
        edit_btn.setToolTip("Re-open the selected label's definition in the wizard.")
        edit_btn.clicked.connect(self.edit_selected_label)
        open_btn = QPushButton("Open Selected")
        open_btn.setToolTip("Switch the whole app to the selected label's dataset.")
        open_btn.clicked.connect(self._load_selected_label)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip(
            "Remove the label from the library. Its captured images and sidecars "
            "are NOT deleted."
        )
        remove_btn.clicked.connect(self.remove_selected_label)
        for _b in (add_btn, edit_btn, open_btn, remove_btn):
            _b.setProperty("compactCaptureButton", True)

        label_btn_row = QHBoxLayout()
        label_btn_row.setSpacing(6)
        label_btn_row.addWidget(open_btn)
        label_btn_row.addWidget(edit_btn)

        self.label_search_edit = QLineEdit()
        self.label_search_edit.setPlaceholderText("Filter labels...")
        self.label_search_edit.setClearButtonEnabled(True)
        self.label_search_edit.setToolTip(
            "Every word has to appear somewhere -- id, description, revision, "
            "part number or vendor -- in any order.\n\n"
            "So \"g31 warn\" finds the G31 warning label without having to "
            "remember whether it was named warning_g31 or g31_warning.")
        # Filtering on each keystroke: the library is in memory and the list is
        # a few hundred rows at worst, so there is nothing to debounce.
        self.label_search_edit.textChanged.connect(lambda _t: self._refresh_labels())

        self.label_count_label = QLabel()
        self.label_count_label.setStyleSheet("color: #94a3b8;")

        # Image library location. On a visible tab rather than only in the Tools
        # menu, because the menu bar is hidden (see _build_menu).
        library_box = QGroupBox("Image library")
        library_layout = QVBoxLayout(library_box)
        library_layout.setContentsMargins(8, 8, 8, 8)
        library_layout.setSpacing(4)
        self.library_path_label = QLabel()
        self.library_path_label.setWordWrap(True)
        self.library_path_label.setStyleSheet("color: #94a3b8;")
        self.library_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.library_combo = QComboBox()
        self.library_combo.setToolTip("Switch to a previously used image library.")
        # Paths are long and arbitrary; let the combo elide rather than force
        # the whole left pane wider than every other control needs.
        self.library_combo.setMinimumContentsLength(12)
        self.library_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.library_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.library_combo.activated.connect(self._on_library_combo_activated)

        change_library_btn = QPushButton("Change Folder...")
        change_library_btn.setToolTip(
            "Point captures, labels and exports at another folder, such as a "
            "shared drive. Takes effect after restarting."
        )
        change_library_btn.setProperty("compactCaptureButton", True)
        change_library_btn.clicked.connect(self.change_data_folder)

        library_layout.addWidget(self.library_path_label)
        library_layout.addWidget(self.library_combo)
        library_layout.addWidget(change_library_btn)
        self._refresh_library_label()

        layout.addWidget(info_box)
        layout.addWidget(add_btn)
        layout.addLayout(label_btn_row)
        layout.addWidget(remove_btn)
        layout.addWidget(self.label_search_edit)
        layout.addWidget(self.label_count_label)
        layout.addWidget(self.label_list)
        layout.addWidget(library_box)
        return outer

    def _refresh_labels(self) -> None:
        """Repopulate the library list and the active-label summary.

        Each row carries its export-ready count against the label's own target,
        so the list doubles as the answer to "what is left to do" without
        opening anything.
        """
        query = (self.label_search_edit.text()
                 if hasattr(self, "label_search_edit") else "")
        matched = self.library.search(query)
        self.label_list.clear()
        for label in matched:
            statuses = list(persistence.dataset_statuses(label.label_id).values())
            ready = sum(1 for s in statuses if review_logic.export_ready(s))
            target = max(1, int(getattr(label, "train_target", 150) or 150))
            mark = "✓" if ready >= target else " "
            item = QListWidgetItem(
                f"{mark} {label.label_id}  {ready}/{target}")
            item.setData(Qt.ItemDataRole.UserRole, label.label_id)
            self.label_list.addItem(item)

        if hasattr(self, "label_count_label"):
            total = len(self.library)
            if len(matched) == total:
                self.label_count_label.setText(f"{total} label(s)")
            else:
                self.label_count_label.setText(f"{len(matched)} of {total} label(s)")

        self._refresh_active_label_panel()

    def _refresh_regions_button(self) -> None:
        """Say which of the two things the button will do, before it is pressed."""
        if not hasattr(self, "define_regions_btn"):
            return
        label = self.library.get(self.label_id) if self.label_id else None
        has_artwork = label is not None and self._existing_artwork(label) is not None
        self.define_regions_btn.setText(
            "Edit Regions..." if has_artwork else "Define Regions...")
        self.define_regions_btn.setToolTip(
            "Open this label's artwork and adjust its read-regions. The artwork "
            "is kept -- to re-flatten it from this image, use "
            "Tools > Replace label artwork."
            if has_artwork else
            "Flatten the label box on this image into straight-on artwork and draw "
            "the areas to read inside it -- a barcode, a serial, a date code. "
            "Stored as fractions of the label, so they then apply to every image "
            "of it. (Ctrl+Shift+R)")

    def _refresh_active_label_panel(self) -> None:
        if not hasattr(self, "label_id_label"):
            return
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            self.label_id_label.setText(self.label_id or "No label selected")
            self.label_meta_label.setText(
                "Add a label to define what to train, or open one from the list below."
                if not self.library else
                "This dataset has images but no definition in the library."
            )
            self.label_progress_label.setText("—")
            return

        self.label_id_label.setText(label.label_id)
        bits = []
        if label.revision:
            bits.append(f"rev {label.revision}")
        size = list(label.size_mm or [])
        if len(size) >= 2 and size[0] and size[1]:
            bits.append(f"{size[0]:g} x {size[1]:g} mm")
        if label.codes:
            bits.append(", ".join(f"{c.role}/{c.symbology}" for c in label.codes))
        if label.surface and label.surface != "matte":
            bits.append(label.surface)
        self.label_meta_label.setText(" · ".join(bits) or str(label.name or "—"))

        self._refresh_regions_button()
        statuses = list(persistence.dataset_statuses(label.label_id).values())
        self.label_progress_label.setText(
            review_logic.dataset_summary(
                label.label_id, statuses,
                want=max(1, int(getattr(label, "train_target", 150) or 150)))
        )

    def _selected_label_id(self) -> str:
        item = self.label_list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _load_selected_label(self, *_args) -> None:
        """Switch the whole app to another label's dataset."""
        label_id = self._selected_label_id()
        if not label_id:
            QMessageBox.information(self, "Label", "Select a label in the list first.")
            return
        self.set_active_label(label_id)

    def set_active_label(self, label_id: str) -> None:
        """Point the whole app at one label's dataset.

        The Class combo follows to that label, because the next thing the
        operator does is draw one -- and a box drawn under the wrong class
        is a mislabel that survives all the way into training.
        """
        self.label_id = str(label_id)
        self._disarm_reference_capture()
        # Re-derive first: a label added since construction is not in the combo
        # yet, and set_class_by_name would silently leave the class on
        # battery_side -- every box then drawn would carry the wrong identity.
        self._refresh_class_combo()
        label = self.library.get(self.label_id)
        if label is not None:
            self.set_class_by_name(str(label.label_id))
        self.current_image_path = None
        self.canvas.clear_boxes()
        self._dataset_index_dirty = True
        self._image_status_cache.clear()
        self._review_queue = []
        self._review_queue_pos = -1
        self._refresh_images()
        self._refresh_active_label_panel()
        self._update_box_count()
        self.status.showMessage(f"Now labeling: {self.label_id}", 5000)

    def add_label_via_wizard(self) -> None:
        label = wizards.add_label(self)
        if label is None:
            return
        self.library = persistence.load_library()
        self._refresh_class_combo()
        self._refresh_class_list_widget()
        self._refresh_labels()
        self.set_active_label(label.label_id)

    def edit_selected_label(self) -> None:
        label_id = self._selected_label_id() or self.label_id
        existing = self.library.get(label_id)
        if existing is None:
            QMessageBox.information(self, "Label", "Select a label in the list first.")
            return
        if wizards.edit_label(self, existing) is None:
            return
        self.library = persistence.load_library()
        self._refresh_class_combo()
        self._refresh_class_list_widget()
        self._refresh_labels()

    def remove_selected_label(self) -> None:
        label_id = self._selected_label_id()
        if not label_id:
            QMessageBox.information(self, "Label", "Select a label in the list first.")
            return
        reply = QMessageBox.question(
            self, "Remove Label",
            f"Remove '{label_id}' from the library?\n\n"
            "Its captured images and saved labels are NOT deleted -- only the "
            "definition. Re-adding the same id picks the dataset back up.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.library.remove(label_id)
        persistence.save_library(self.library)
        if self.label_id == label_id:
            self.label_id = self._initial_label_id()
        self._refresh_labels()
        self._refresh_images()
        self.status.showMessage(f"Removed {label_id} from the library", 5000)

    def _assign_active_label(self, boxes: list[dict]) -> list[dict]:
        """Carry each prediction's identity across, and fill in its regions.

        The detector is trained on label ids, so a prediction already names
        which label it is -- there is nothing to guess. Every box whose class
        is a library label gets that label's id, not just the one being
        trained: a battery carries several labels and the recipe counts them
        all, so throwing away the others would be discarding real answers.
        """
        for box in boxes:
            name = str(box.get("label", ""))
            label = self.library.get(name) if name else None
            if label is not None:
                box["label_id"] = name
                # The read-regions follow from the four corners just drawn, so
                # the operator confirms a barcode box rather than drawing one.
                # Hand-adjusted regions are preserved.
                ann_logic.apply_reference_regions(box, label)
        return boxes
    def _refresh_library_label(self) -> None:
        """Show the active library path, how it was chosen, and any pending change."""
        if not hasattr(self, "library_path_label"):
            return
        env = os.environ.get(storage_mod.DATA_DIR_ENV, "").strip()
        configured = storage_mod.read_configured_data_dir()
        if env:
            source = f"set by {storage_mod.DATA_DIR_ENV}"
        elif configured is not None:
            source = "custom location"
        else:
            source = "default location"

        text = f"{DATA_DIR}\n({source})"
        # Covers both a change made this session and a configured location that
        # could not be opened at startup -- in each case the folder in use is
        # not the one configured, and saying so avoids confusion.
        if configured is not None and not env and Path(configured) != DATA_DIR:
            text += f"\n\nPending restart:\n{configured}"
        self.library_path_label.setText(text)
        self._reload_library_combo()

    def _reload_library_combo(self) -> None:
        """Populate the quick-switch list with known library locations."""
        if not hasattr(self, "library_combo"):
            return
        active = str(storage_mod.read_configured_data_dir() or DATA_DIR)
        entries = [str(p) for p in storage_mod.read_recent_data_dirs()]
        default = str(storage_mod._app_root() / "data")
        for extra in (active, default):
            if extra not in entries:
                entries.append(extra)

        self.library_combo.blockSignals(True)
        self.library_combo.clear()
        for path in entries:
            label = path + ("   [default]" if path == default else "")
            self.library_combo.addItem(label, path)
        idx = self.library_combo.findData(active)
        self.library_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.library_combo.blockSignals(False)

    def _on_library_combo_activated(self, index: int) -> None:
        """Switch to a remembered library picked from the list."""
        target = self.library_combo.itemData(index)
        if not target:
            return
        target = Path(target)
        if target == (storage_mod.read_configured_data_dir() or DATA_DIR):
            return

        if not storage_mod._ensure_data_dirs(target):
            QMessageBox.warning(
                self, "Image library",
                f"Cannot open that library:\n{target}\n\n"
                "If it is on a network drive, check that it is connected. "
                "The library was not changed.",
            )
            self._reload_library_combo()
            return

        # Selecting the built-in default clears the override rather than
        # pinning that path, so a portable copy still follows its own folder.
        if target == storage_mod._app_root() / "data":
            storage_mod.write_configured_data_dir(None)
        else:
            storage_mod.write_configured_data_dir(target)
        self._refresh_library_label()
        QMessageBox.information(
            self, "Image library",
            f"Library set to:\n{target}\n\nRestart the application to load it.",
        )
        self.status.showMessage(f"Library set to {target} (restart to apply)", 10000)

    def _capture_tab(self) -> QWidget:
        """Live capture tab.

        v0.9.19 keeps the capture controls compact and readable.
        Resolution fields are plain manual-entry boxes with no spinner arrows,
        defaulting to the Basler 5MP resolution used for LabelVision testing.
        """
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll)

        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        cam_box = QGroupBox("Camera / Stream")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setContentsMargins(10, 10, 10, 10)
        cam_layout.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Auto", "V4L2", "GStreamer", "GStreamer (native)", "FFmpeg", "Basler/Pylon"])
        self.backend_combo.setCurrentText(str(self.camera_settings.get("camera_backend", "V4L2")))
        self.backend_combo.currentTextChanged.connect(self._on_camera_backend_changed)
        self.backend_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Backend", self.backend_combo)

        self.source_edit = QLineEdit(str(self.camera_settings.get("camera_source", "0")))
        self.source_edit.setPlaceholderText("0, /dev/video0, video.mp4, rtsp://, or Basler serial")
        self.source_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Source", self.source_edit)

        self.width_spin = QLineEdit(str(int(self.camera_settings.get("width", 2592) or 2592)))
        self.width_spin.setValidator(QIntValidator(0, 8192, self))
        self.width_spin.setPlaceholderText("2592")
        self.height_spin = QLineEdit(str(int(self.camera_settings.get("height", 1944) or 1944)))
        self.height_spin.setValidator(QIntValidator(0, 8192, self))
        self.height_spin.setPlaceholderText("1944")
        self.fps_spin = QLineEdit(str(int(self.camera_settings.get("fps", 0) or 0)))
        self.fps_spin.setValidator(QIntValidator(0, 240, self))
        self.fps_spin.setPlaceholderText("Default")
        self.preview_scale_combo = QComboBox()
        self.preview_scale_combo.addItems(["Full", "1/2", "1/3", "1/4"])
        self.preview_scale_combo.setCurrentText(str(self.camera_settings.get("preview_scale", "1/2") or "1/2"))
        self.exposure_auto_check = QCheckBox("Auto exposure")
        self.exposure_auto_check.setChecked(bool(self.camera_settings.get("exposure_auto", True)))
        self.exposure_auto_check.stateChanged.connect(self._on_exposure_auto_changed)
        self.exposure_us_edit = QLineEdit(str(int(self.camera_settings.get("exposure_us", 0) or 0)))
        self.exposure_us_edit.setValidator(QIntValidator(0, 10000000, self))
        self.exposure_us_edit.setPlaceholderText("Manual us")
        self.exposure_us_edit.setAlignment(Qt.AlignCenter)
        self.exposure_us_edit.setFixedWidth(92)
        self.apply_exposure_btn = QPushButton("Apply Exposure")
        self.apply_exposure_btn.clicked.connect(self.apply_exposure_to_camera)

        # Keep manual camera-format fields compact. These are QLineEdit boxes
        # on purpose: the operator types exact resolutions instead of clicking
        # spinner arrows.
        for edit in (self.width_spin, self.height_spin):
            edit.setFixedWidth(72)
            edit.setAlignment(Qt.AlignCenter)
        self.fps_spin.setFixedWidth(58)
        self.fps_spin.setAlignment(Qt.AlignCenter)
        self.preview_scale_combo.setFixedWidth(82)

        format_grid = QGridLayout()
        format_grid.setHorizontalSpacing(8)
        format_grid.setVerticalSpacing(6)
        format_grid.addWidget(QLabel("Width"), 0, 0)
        format_grid.addWidget(self.width_spin, 0, 1)
        format_grid.addWidget(QLabel("Height"), 0, 2)
        format_grid.addWidget(self.height_spin, 0, 3)
        format_grid.addWidget(QLabel("FPS"), 1, 0)
        format_grid.addWidget(self.fps_spin, 1, 1)
        format_grid.addWidget(QLabel("Preview"), 1, 2)
        format_grid.addWidget(self.preview_scale_combo, 1, 3)
        format_grid.setColumnStretch(4, 1)
        form.addRow("Format", format_grid)

        self.apply_exposure_btn.setProperty("compactCaptureButton", True)
        self.apply_exposure_btn.setMinimumHeight(24)
        self.apply_exposure_btn.setMaximumHeight(26)
        self.apply_exposure_btn.setMaximumWidth(150)
        self.apply_exposure_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        exposure_box = QGroupBox("Exposure")
        exposure_layout = QGridLayout(exposure_box)
        exposure_layout.setHorizontalSpacing(8)
        exposure_layout.setVerticalSpacing(8)
        exposure_layout.addWidget(self.exposure_auto_check, 0, 0, 1, 2)
        exposure_layout.addWidget(QLabel("Manual us"), 1, 0)
        exposure_layout.addWidget(self.exposure_us_edit, 1, 1)
        exposure_layout.addWidget(self.apply_exposure_btn, 2, 0, 1, 2, Qt.AlignLeft)
        exposure_layout.setColumnStretch(1, 1)

        self._on_exposure_auto_changed()
        # The detailed camera controls are edited in a popup to keep this tab compact.

        self.basler_hint_label = QLabel("Basler: Source may be blank or a serial/model filter.")
        self.basler_hint_label.setWordWrap(True)
        self.basler_hint_label.setStyleSheet("color: #94a3b8; font-size: 8pt;")

        opts_box = QGroupBox("Camera Options")
        opts_layout = QVBoxLayout(opts_box)
        opts_layout.setContentsMargins(10, 10, 10, 10)
        opts_layout.setSpacing(4)
        self.force_v4l2_check = QCheckBox("Force V4L2 — use Linux USB-camera backend")
        self.force_v4l2_check.setChecked(bool(self.camera_settings.get("force_v4l2", True)))
        self.low_latency_check = QCheckBox("Low latency — reduce buffering/delay")
        self.low_latency_check.setChecked(bool(self.camera_settings.get("low_latency", True)))
        self.threaded_camera_check = QCheckBox("Threaded reader — smoother preview capture")
        self.threaded_camera_check.setChecked(bool(self.camera_settings.get("threaded_camera", True)))
        self.mjpg_check = QCheckBox("MJPG — request compressed camera stream")
        self.mjpg_check.setChecked(bool(self.camera_settings.get("mjpg", True)))
        self.skip_heavy_live_check = QCheckBox("Skip heavy filters — keep live view faster")
        self.skip_heavy_live_check.setChecked(bool(self.camera_settings.get("skip_heavy_live", True)))
        for widget in [
            self.force_v4l2_check,
            self.low_latency_check,
            self.threaded_camera_check,
            self.mjpg_check,
            self.skip_heavy_live_check,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            opts_layout.addWidget(widget)

        self._camera_settings_hidden = QWidget()
        self._camera_settings_hidden.setVisible(False)
        hidden_layout = QVBoxLayout(self._camera_settings_hidden)
        hidden_layout.setContentsMargins(0, 0, 0, 0)
        hidden_layout.setSpacing(0)
        hidden_layout.addLayout(form)
        hidden_layout.addWidget(exposure_box)
        hidden_layout.addWidget(self.basler_hint_label)
        hidden_layout.addWidget(opts_box)
        cam_layout.addWidget(self._camera_settings_hidden)
        self._on_camera_backend_changed(self.backend_combo.currentText())

        camera_settings_btn = QPushButton("Camera...")
        camera_settings_btn.setToolTip("Open the camera settings dialog.")
        camera_settings_btn.clicked.connect(self.open_camera_settings_dialog)
        camera_settings_btn.setProperty("compactCaptureButton", True)
        camera_settings_btn.setMinimumHeight(24)
        camera_settings_btn.setMaximumHeight(26)
        self.camera_settings_summary = QLabel()
        self.camera_settings_summary.setWordWrap(True)
        self.camera_settings_summary.setStyleSheet("color: #cbd5e1;")
        cam_layout.addWidget(camera_settings_btn, 0, Qt.AlignLeft)
        cam_layout.addWidget(self.camera_settings_summary)
        self._update_camera_settings_summary()

        self.test_cam_btn = QPushButton("Test")
        self.test_cam_btn.clicked.connect(self.test_camera)
        self.open_cam_btn = QPushButton("Open Preview")
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.close_cam_btn = QPushButton("Stop")
        self.close_cam_btn.clicked.connect(self.close_camera)
        cap_raw = QPushButton("Capture Raw")
        cap_raw.clicked.connect(lambda: self.capture_frame(save_adjusted=False))
        cap_adj = QPushButton("Capture Adjusted")
        cap_adj.clicked.connect(lambda: self.capture_frame(save_adjusted=True))
        cap_ref = QPushButton("Capture Reference")
        cap_ref.setToolTip(
            "Capture a frame to define this label's read-regions from. It is saved "
            "into the dataset like any other capture -- then draw the label's box "
            "and it opens flattened, straight-on, to draw the regions on.")
        cap_ref.clicked.connect(self.capture_reference)

        control_box = QGroupBox("Actions")
        control_layout = QGridLayout(control_box)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setHorizontalSpacing(6)
        control_layout.setVerticalSpacing(4)
        control_buttons = [self.test_cam_btn, self.open_cam_btn, self.close_cam_btn,
                           cap_raw, cap_adj, cap_ref]
        for i, btn in enumerate(control_buttons):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            control_layout.addWidget(btn, i // 2, i % 2)
        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)

        list_header = QLabel("Captured Images")
        list_header.setStyleSheet("font-weight: 700;")

        review_box = QGroupBox("Review Filter")
        review_layout = QVBoxLayout(review_box)
        review_layout.setContentsMargins(8, 8, 8, 8)
        review_layout.setSpacing(4)
        self.show_unreviewed_only_check = QCheckBox("Show only needs review")
        self.show_unreviewed_only_check.setToolTip("Show only images with imported/saved JSON labels that have not been marked reviewed.")
        self.show_unreviewed_only_check.stateChanged.connect(self._refresh_images)
        find_unreviewed = QPushButton("Find Unreviewed")
        find_unreviewed.clicked.connect(self.find_next_unreviewed_image)
        mark_reviewed = QPushButton("Mark Reviewed")
        mark_reviewed.setToolTip("Mark the current image reviewed.")
        mark_reviewed.clicked.connect(self.mark_current_reviewed)
        force_reviewed = QPushButton("Force Review")
        force_reviewed.setToolTip("Use this only when you intentionally want a mismatch image exported, such as a missing-bung/fail example.")
        force_reviewed.clicked.connect(self.force_mark_current_reviewed)
        mark_background = QPushButton("Mark Background")
        mark_background.setToolTip(
            "Mark the current image as containing no objects at all -- an empty\n"
            "conveyor or fixture. It exports as an empty label file, which is how\n"
            "YOLO learns what a negative looks like."
        )
        mark_background.clicked.connect(self.mark_current_background)
        for btn in (find_unreviewed, mark_reviewed, force_reviewed, mark_background):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        review_layout.addWidget(self.show_unreviewed_only_check)
        review_grid = QGridLayout()
        review_grid.setHorizontalSpacing(6)
        review_grid.setVerticalSpacing(4)
        for i, btn in enumerate((find_unreviewed, mark_reviewed, force_reviewed, mark_background)):
            review_grid.addWidget(btn, i // 2, i % 2)
        review_grid.setColumnStretch(0, 1)
        review_grid.setColumnStretch(1, 1)
        review_layout.addLayout(review_grid)

        load_selected = QPushButton("Load Selected")
        load_selected.clicked.connect(self._load_selected_image)
        delete_selected = QPushButton("Delete Image")
        delete_selected.clicked.connect(self.delete_selected_image)
        import_images_btn = QPushButton("Import Images...")
        import_images_btn.setToolTip("Copy existing image files into this recipe. You can optionally specify a separate folder containing matching LabelVision label JSON files.")
        import_images_btn.clicked.connect(self.import_images_to_recipe)
        import_bg_btn = QPushButton("Import Backgrounds...")
        import_bg_btn.setToolTip(
            "Copy in images that contain no objects at all -- empty conveyor, bare\n"
            "fixture. Each one is marked background on import and exports as an\n"
            "empty label file, so no hand-labeling is needed."
        )
        import_bg_btn.clicked.connect(self.import_background_images)
        for btn in (load_selected, delete_selected, import_images_btn, import_bg_btn):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        image_button_row = QHBoxLayout()
        image_button_row.setContentsMargins(0, 0, 0, 0)
        image_button_row.setSpacing(6)
        image_button_row.addWidget(load_selected)
        image_button_row.addWidget(delete_selected)
        image_button_row2 = QHBoxLayout()
        image_button_row2.setContentsMargins(0, 0, 0, 0)
        image_button_row2.setSpacing(6)
        image_button_row2.addWidget(import_images_btn)
        image_button_row2.addWidget(import_bg_btn)

        layout.addWidget(cam_box)
        layout.addWidget(control_box)
        layout.addWidget(review_box)
        layout.addWidget(list_header)
        layout.addWidget(self.image_list, 1)
        layout.addLayout(image_button_row)
        layout.addLayout(image_button_row2)
        return outer

    def _slider(self, minv, maxv, val, cb) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(val)
        s.valueChanged.connect(cb)
        return s

    def _adjust_tab(self) -> QWidget:
        outer, w, layout = self._scrollable_tab()
        box = QGroupBox("Non-destructive Preview")
        form = QFormLayout(box)
        self.brightness_slider = self._slider(-100, 100, 0, self._adjustment_changed)
        self.contrast_slider = self._slider(-100, 100, 0, self._adjustment_changed)
        self.gamma_slider = self._slider(20, 300, 100, self._adjustment_changed)
        self.sharpen_slider = self._slider(0, 100, 0, self._adjustment_changed)
        self.clahe_check = QCheckBox("Enable CLAHE")
        self.clahe_check.setChecked(False)
        self.clahe_check.stateChanged.connect(self._adjustment_changed)
        self.clahe_clip_slider = self._slider(5, 100, 20, self._adjustment_changed)
        self.clahe_grid_slider = self._slider(2, 16, 8, self._adjustment_changed)
        form.addRow("Brightness", self.brightness_slider)
        form.addRow("Contrast", self.contrast_slider)
        form.addRow("Gamma", self.gamma_slider)
        form.addRow("Sharpen", self.sharpen_slider)
        form.addRow("CLAHE", self.clahe_check)
        form.addRow("CLAHE clip", self.clahe_clip_slider)
        form.addRow("CLAHE grid", self.clahe_grid_slider)

        reset = QPushButton("Reset")
        reset.setToolTip("Reset all contrast/brightness adjustments to their defaults.")
        reset.clicked.connect(self.reset_adjustments)
        layout.addWidget(box)
        layout.addWidget(reset)
        layout.addStretch()
        return outer


    def _right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        label_box = QGroupBox("Annotation")
        v = QVBoxLayout(label_box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)
        self.mode_label = QLabel("Mode: Labeling")
        self.mode_label.setWordWrap(True)
        self.guidance_label = QLabel("OBB labels: drag to draw, then adjust the four corner handles.")
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setStyleSheet("color: #bfdbfe; font-weight: 600;")

        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.class_combo.currentIndexChanged.connect(self._class_changed)
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("OBB / 4-corner", "obb")
        self.tool_combo.addItem("Box fallback", "box")
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        self.count_label = QLabel("Battery: 0 / 1   Bungs: 0 / expected 6")
        self.dataset_label = QLabel("Dataset: 0 images, 0 labeled, 0 ready")
        # Both grow with the dataset and were clipped at the rail's edge.
        for _lbl in (self.count_label, self.dataset_label):
            _lbl.setWordWrap(True)

        save = QPushButton("Save")
        save.clicked.connect(self.save_labels)
        save_next = QPushButton("Save + Next")
        save_next.clicked.connect(self.save_and_next)
        copy_prev = QPushButton("Copy Prev")
        copy_prev.clicked.connect(self.copy_previous_labels)
        qa_btn = QPushButton("Find Problem")
        qa_btn.clicked.connect(self.find_next_problem_image)
        self.define_regions_btn = QPushButton("Define Regions...")
        self.define_regions_btn.clicked.connect(self.define_read_regions)
        define_regions_btn = self.define_regions_btn
        self._refresh_regions_button()
        regions_btn = QPushButton("Place Regions")
        regions_btn.setToolTip(
            "Fill in this label's read-regions -- barcodes, text fields, the match "
            "anchor -- from its artwork. They are stored as fractions of the label, "
            "so drawing its four corners is all the positioning they need. (Ctrl+R)")
        regions_btn.clicked.connect(self.place_regions_on_canvas)
        delete = QPushButton("Delete Box")
        delete.setToolTip("Delete only the selected on-screen box. Click Save when you want to write the change.")
        delete.clicked.connect(self.canvas.delete_selected)
        clear = QPushButton("Clear Boxes")
        clear.setToolTip("Clear the on-screen boxes for this image without deleting or overwriting the saved JSON label file.")
        clear.clicked.connect(self.clear_boxes_unsaved)
        clear_saved = QPushButton("Delete Saved JSON")
        clear_saved.setToolTip("Delete the saved .json labels for this image after confirmation.")
        clear_saved.clicked.connect(self.delete_saved_labels_confirmed)

        zminus = QPushButton("−")
        zminus.clicked.connect(self.canvas.zoom_out)
        zfit = QPushButton("Fit")
        zfit.clicked.connect(self.canvas.fit_to_window)
        zplus = QPushButton("+")
        zplus.clicked.connect(self.canvas.zoom_in)

        right_panel_buttons = (save, save_next, copy_prev, qa_btn,
                               define_regions_btn, regions_btn,
                               delete, clear, clear_saved, zminus, zfit, zplus)
        for btn in right_panel_buttons:
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.addRow("Class", self.class_combo)
        form.addRow("Tool", self.tool_combo)

        def button_row(*buttons: QPushButton) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(0, 0, 0, 0)
            for b in buttons:
                row.addWidget(b)
            return row

        health_btn = QPushButton("Dataset Health")
        health_btn.setToolTip("Per-recipe / per-category readiness dashboard: labeled, reviewed, and export-ready counts.")
        health_btn.clicked.connect(self.show_dataset_health)
        shortcuts_btn = QPushButton("⌨ Shortcuts")
        shortcuts_btn.setToolTip("Show the keyboard shortcut reference (F1).")
        shortcuts_btn.clicked.connect(self.show_shortcuts_reference)
        for btn in (health_btn, shortcuts_btn):
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        v.addWidget(self.mode_label)
        v.addWidget(self.guidance_label)
        v.addLayout(form)
        v.addWidget(self.count_label)
        v.addWidget(self.dataset_label)
        v.addLayout(button_row(health_btn, shortcuts_btn))
        v.addLayout(button_row(zminus, zfit, zplus))
        v.addLayout(button_row(save, save_next))
        v.addLayout(button_row(copy_prev, qa_btn))
        v.addLayout(button_row(define_regions_btn, regions_btn))
        v.addLayout(button_row(delete, clear))
        v.addWidget(clear_saved)

        class_box = QGroupBox("Detector Classes")
        cv = QVBoxLayout(class_box)
        cv.setContentsMargins(8, 8, 8, 8)
        cv.setSpacing(4)
        class_note = QLabel(
            "The classes the model is trained on: one per label, plus the "
            "battery face. This list is the label library -- add a label to add "
            "a class. Every new one needs a retrain before it is detected."
        )
        class_note.setWordWrap(True)
        class_note.setStyleSheet("color: #9aa4b2;")
        cv.addWidget(class_note)
        self.class_list_widget = QListWidget()
        self.class_list_widget.setMaximumHeight(120)
        self.class_list_widget.setToolTip(
            "Derived from the label library, so it cannot disagree with what the\n"
            "detector is actually trained on. Read-only on purpose: a class list\n"
            "edited separately from the labels is two answers to one question."
        )
        cv.addWidget(self.class_list_widget)
        self.class_counts_label = QLabel("Current image: no labels")
        self.class_counts_label.setWordWrap(True)
        self.class_counts_label.setStyleSheet("color: #94a3b8;")
        self.class_counts_label.setToolTip("Per-class box counts for the image currently on the canvas.")
        cv.addWidget(self.class_counts_label)
        self._refresh_class_list_widget()

        export_box = QGroupBox("Export")
        ev = QVBoxLayout(export_box)
        ev.setContentsMargins(8, 8, 8, 8)
        ev.setSpacing(4)
        self.export_task_combo = QComboBox()
        self.export_task_combo.addItem("OBB dataset - all labeled classes", "obb")
        self.export_task_combo.addItem("Detect boxes dataset - compatibility", "detect")
        self.export_augment_spin = QSpinBox()
        self.export_augment_spin.setRange(0, 10)
        self.export_augment_spin.setValue(0)
        self.export_augment_spin.setToolTip(
            "Extra training copies per image with variable regions -- date codes, "
            "serials -- grafted from other images of the same label.\n\n"
            "Only written for labels whose regions turn out to be near-identical "
            "across the dataset; ones that already vary get none, because "
            "recombining them teaches nothing and dilutes the real images.\n\n"
            "Tools > Check variable regions says which is which.")
        exp = QPushButton("Export Dataset")
        exp.clicked.connect(self.export_yolo)
        exp_all = QPushButton("Export All")
        exp_all.clicked.connect(self.export_all_yolo)
        exp_all.setToolTip(
            "Combine every label's dataset into one training set. This is the "
            "normal export: the detector learns every label at once."
        )
        for btn in (exp, exp_all):
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        exp_two = QPushButton("Export Two-Stage")
        exp_two.clicked.connect(self.export_two_stage)
        exp_two.setToolTip(
            "Write both halves of a detect-then-classify pipeline from the same "
            "annotations: a detector that finds WHERE a label is (one generic "
            "class), and crops foldered by label id for a classifier that "
            "decides WHICH.\n\n"
            "Worth it when the deciding detail is small in the frame. Crop size "
            "is measured from your own boxes rather than fixed -- a 224 px crop "
            "is a large gain for a small label and an outright loss for one the "
            "detector already resolves to 500 px. Tools > Check label scale "
            "shows the working.\n\n"
            "Both halves share one split and seed, so they hold out the same "
            "batteries."
        )
        for btn in (exp_two,):
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        export_btn_row = QHBoxLayout()
        export_btn_row.setSpacing(6)
        export_btn_row.addWidget(exp)
        export_btn_row.addWidget(exp_all)
        ev.addWidget(QLabel("Export task"))
        ev.addWidget(self.export_task_combo)
        augment_row = QHBoxLayout()
        augment_row.addWidget(QLabel("Variable-region copies"))
        augment_row.addWidget(self.export_augment_spin, 1)
        ev.addLayout(augment_row)
        ev.addLayout(export_btn_row)
        ev.addWidget(exp_two)
        exp_regions = QPushButton("Export Region Crops")
        exp_regions.clicked.connect(self.export_region_crops)
        exp_regions.setToolTip(
            "Crop each read-region out of the FULL-RESOLUTION frame, foldered by "
            "label id.\n\n"
            "This is what separates two labels that differ only by a revision "
            "letter or a language line. Nothing at detector resolution reaches "
            "that detail, and a whole-label crop reaches it less -- the region "
            "cropped from the original keeps every pixel it had.\n\n"
            "Ground truth comes free: the label id already says which revision "
            "it is.")
        exp_regions.setProperty("rightPanelButton", True)
        exp_regions.setMinimumHeight(24)
        exp_regions.setMaximumHeight(26)
        exp_regions.setMinimumWidth(0)
        exp_regions.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ev.addWidget(exp_regions)
        export_note = QLabel("Exports annotation class names as-is. Reviewed and force-reviewed images only.")
        export_note.setWordWrap(True)
        export_note.setStyleSheet("color: #94a3b8;")
        ev.addWidget(export_note)

        layout.addWidget(label_box)
        layout.addWidget(class_box)
        layout.addWidget(export_box)
        layout.addStretch(1)
        return w


    def _class_changed(self, idx: int) -> None:
        if idx < 0:
            idx = 0
        if idx >= len(self.class_names):
            return
        self.canvas.class_id = idx
        self.canvas.class_name = self.class_names[idx]
        default_tool = self._default_tool_for_class(idx)
        if hasattr(self, "tool_combo"):
            target = "obb" if default_tool == "OBB" else "box"
            tool_idx = self.tool_combo.findData(target)
            if tool_idx >= 0:
                self.tool_combo.blockSignals(True)
                self.tool_combo.setCurrentIndex(tool_idx)
                self.tool_combo.blockSignals(False)
                self.canvas.set_annotation_kind(target)
        if hasattr(self, "guidance_label"):
            if self.canvas.annotation_kind == "obb":
                self.guidance_label.setText("OBB Tool — drag to draw, then adjust the four corner handles.")
            else:
                self.guidance_label.setText("Box fallback — draw a normal axis-aligned YOLO box.")

    def _tool_changed(self, idx: int) -> None:
        kind = self.tool_combo.currentData() if hasattr(self, "tool_combo") else "box"
        self.canvas.set_annotation_kind(kind)
        if hasattr(self, "guidance_label"):
            if kind == "obb":
                self.guidance_label.setText("OBB Tool — drag to draw, then adjust the four corner handles.")
            else:
                self.guidance_label.setText("Box fallback — draw a normal axis-aligned YOLO box.")

    def _default_tool_for_class(self, class_id: int) -> str:
        for c in self.class_config:
            if int(c.get("id", -1)) == int(class_id):
                tool = str(c.get("default_tool", "OBB")).upper()
                return "BOX" if tool == "BOX" else "OBB"
        return "OBB"

    def _refresh_class_combo(self) -> None:
        self.class_names = self.library.detector_classes()
        if not hasattr(self, "class_combo"):
            return
        current = self.class_combo.currentIndex()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        self.class_combo.setCurrentIndex(max(0, min(current, len(self.class_names) - 1)))
        self.class_combo.blockSignals(False)
        self._class_changed(self.class_combo.currentIndex())

    def _refresh_class_list_widget(self) -> None:
        """The detector's classes, with how many export-ready images back each.

        The count is the useful number: a class with no images behind it is one
        the model is asked to learn and given nothing to learn it from, which
        is how a class ends up detected nowhere or everywhere.
        """
        if not hasattr(self, "class_list_widget"):
            return
        self.class_list_widget.clear()
        for name in self.library.detector_classes():
            if name in labels_mod.STRUCTURAL_CLASSES:
                suffix = "the battery face"
            else:
                statuses = list(persistence.dataset_statuses(name).values())
                ready = sum(1 for st in statuses if review_logic.export_ready(st))
                # Deliberately no class number here. The export numbers classes
                # from what the annotations actually contain, so a label with
                # nothing reviewed is absent from the model and every class
                # after it sits at a different index than this list would
                # imply. Names are what inference returns; a number shown here
                # would be a number that is right only by luck.
                suffix = f"{ready} ready" if ready else "not in the model yet"
            item = QListWidgetItem(f"{name}  ({suffix})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.class_list_widget.addItem(item)


    def set_class_by_name(self, name: str) -> None:
        if not hasattr(self, "class_combo"):
            return
        for i, n in enumerate(self.class_names):
            if n == name:
                self.class_combo.setCurrentIndex(i)
                return

    def _apply_theme(self) -> None:
        _assets = Path(__file__).resolve().parent / "assets"
        checkbox_check = (_assets / "checkbox_check.svg").as_posix()
        spin_up = (_assets / "spin_up.svg").as_posix()
        spin_down = (_assets / "spin_down.svg").as_posix()
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0f172a;
                color: #e5e7eb;
                font-size: 9pt;
            }
            QMenu {
                background: #111827;
                color: #e5e7eb;
                border: 1px solid #334155;
                padding: 6px;
            }
            QMenu::item { padding: 7px 28px 7px 18px; border-radius: 5px; }
            QMenu::item:selected { background: #1d4ed8; }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #93c5fd;
            }
            QPushButton {
                min-height: 22px;
                padding: 3px 8px;
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 7px;
                font-weight: 700;
                text-align: center;
            }
            QPushButton[compactCaptureButton="true"], QPushButton[rightPanelButton="true"] {
                min-height: 22px;
                max-height: 26px;
                padding: 2px 7px;
                border-radius: 5px;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:pressed { background: #1e40af; }
            QCheckBox {
                spacing: 8px;
                color: #e5e7eb;
                min-height: 24px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #93c5fd;
                border-radius: 4px;
                background: #020617;
            }
            QCheckBox::indicator:hover { border: 2px solid #bfdbfe; }
            QCheckBox::indicator:checked {
                background: #020617;
                border: 2px solid #bfdbfe;
                image: url("__CHECKBOX_CHECK__");
            }
            QCheckBox::indicator:disabled {
                border: 2px solid #475569;
                background: #0f172a;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 6px;
                color: #e5e7eb;
                min-height: 24px;
            }
            /* Styling a QSpinBox at all replaces the native step buttons with
               unstyled subcontrols that are tiny and nearly unclickable. Once
               the widget is themed, the buttons and arrows must be sized and
               positioned explicitly or they do not usably work. */
            QSpinBox, QDoubleSpinBox {
                padding-right: 22px;   /* reserve the button column */
                min-height: 28px;      /* two 14px halves stay clickable */
            }
            /* height is required, not optional: without it both buttons claim
               the full widget height and overlap, so the lower one swallows the
               upper one's clicks and stepping up appears dead. */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                height: 13px;
                border-left: 1px solid #334155;
                background: #1e293b;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                height: 13px;
                border-left: 1px solid #334155;
                background: #1e293b;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #334155;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background: #1d4ed8;
            }
            /* SVG chevrons rather than CSS border-triangles: the image: form is
               the reliably supported way to set a subcontrol glyph in Qt, and
               it is already proven by the checkbox above. */
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url("__SPIN_UP__");
                width: 10px;
                height: 7px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url("__SPIN_DOWN__");
                width: 10px;
                height: 7px;
            }
            QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
            QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
                opacity: 80;
            }
            /* Styling a QComboBox replaces its native drop-down button with an
               unstyled subcontrol, the same trap as the spinbox step buttons.
               Size and position it explicitly, and reuse the spinner chevron. */
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: center right;
                width: 22px;
                border-left: 1px solid #334155;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                background: #1e293b;
            }
            QComboBox::drop-down:hover { background: #334155; }
            QComboBox::down-arrow {
                image: url("__SPIN_DOWN__");
                width: 10px;
                height: 7px;
            }
            QComboBox::down-arrow:disabled { opacity: 80; }
            /* No min-height: forcing 120px made a 2-item popup render as a tall
               box of dead space that Qt then had to reposition, which is what
               made short dropdowns feel glitchy. maxVisibleItems caps the tall
               ones instead. */
            /* The popup view is a QListView, which the generic input rule above
               (QLineEdit, ..., QListWidget) never matched -- so it had no
               background and the first paint showed straight through to the
               window behind before the rows drew. That is the flicker.
               QComboBox QListView is spelled out too: the popup is a QListView
               and matching it directly avoids relying on the abstract base. */
            /* The scrollbars were already ~14px wide, but unstyled: the
               default handle against the #0f172a theme is effectively
               invisible. The fix is contrast, not width -- keep the platform
               width and give the handle a light, obvious colour. */
            QScrollBar:vertical {
                background: #0f172a;
                width: 14px;
                margin: 0;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #64748b;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover { background: #94a3b8; }
            QScrollBar:horizontal {
                background: #0f172a;
                height: 14px;
                margin: 0;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: #64748b;
                min-width: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover { background: #94a3b8; }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0; width: 0; border: none; background: none;
            }
            QScrollBar::add-page, QScrollBar::sub-page { background: none; }
            QComboBox QAbstractItemView, QComboBox QListView {
                background: #111827;
                color: #e5e7eb;
                border: 1px solid #334155;
                selection-background-color: #1d4ed8;
                selection-color: #ffffff;
                outline: none;
            }
            QComboBox QAbstractItemView::item { min-height: 22px; }
            QTextEdit { min-height: 56px; }
            QLabel { padding: 2px 0; }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
            }
            QTabBar::tab {
                background: #111827;
                color: #cbd5e1;
                padding: 6px 8px;
                margin-right: 3px;
                border: 1px solid #334155;
                border-bottom: 0;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 70px;
            }
            QTabBar::tab:selected { background: #1e293b; color: white; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::handle:horizontal { width: 18px; background: #60a5fa; margin: -6px 0; border-radius: 9px; }
            QScrollArea { border: 0; }
        """.replace("__CHECKBOX_CHECK__", checkbox_check)
           .replace("__SPIN_UP__", spin_up)
           .replace("__SPIN_DOWN__", spin_down))

        for widget in self.findChildren(QWidget):
            layout = widget.layout()
            if layout is not None:
                layout.setContentsMargins(6, 6, 6, 6)
                layout.setSpacing(6)

    def _on_exposure_auto_changed(self, *args) -> None:
        manual = not (self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True)
        if hasattr(self, "exposure_us_edit"):
            self.exposure_us_edit.setEnabled(manual)
        if hasattr(self, "apply_exposure_btn"):
            self.apply_exposure_btn.setEnabled(True)

    def apply_exposure_to_camera(self) -> None:
        auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0
        msg = self.camera.set_exposure(auto, exposure_us)
        self.status.showMessage(msg, 8000)
        if not self.camera.is_open():
            QMessageBox.information(self, "Exposure", msg + "\n\nOpen Preview first to apply exposure to the active camera.")

    def _update_camera_settings_summary(self) -> None:
        if not hasattr(self, "camera_settings_summary"):
            return
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        source = self.source_edit.text().strip() if hasattr(self, "source_edit") else ""
        if not source and backend == "Basler/Pylon":
            source = "Any Basler"
        elif not source:
            source = "0"
        width = self._int_line_value(self.width_spin, 0) if hasattr(self, "width_spin") else 0
        height = self._int_line_value(self.height_spin, 0) if hasattr(self, "height_spin") else 0
        fps = self._int_line_value(self.fps_spin, 0) if hasattr(self, "fps_spin") else 0
        preview = self.preview_scale_combo.currentText() if hasattr(self, "preview_scale_combo") else "1/2"
        exposure = "auto" if (self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True) else f"{self._int_line_value(self.exposure_us_edit, 0)} us"
        size = f"{width}x{height}" if width and height else "default size"
        fps_text = f"{fps} FPS" if fps else "default FPS"
        self.camera_settings_summary.setText(f"{backend} | Source {source} | {size}, {fps_text} | Preview {preview} | Exposure {exposure}")

    def _camera_stream_signature(self) -> tuple:
        """Return settings that require a camera reopen to take effect.

        Preview scale and exposure can be applied without renegotiating the
        stream, but backend/source/format/backend options cannot. Keeping this
        separate prevents the UI from looking like a resolution change applied
        while the live reader is still showing the old camera mode.
        """
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        is_basler = backend == "Basler/Pylon"
        return (
            backend,
            self.source_edit.text().strip() if hasattr(self, "source_edit") else "0",
            self._int_line_value(self.width_spin, 0) if hasattr(self, "width_spin") else 0,
            self._int_line_value(self.height_spin, 0) if hasattr(self, "height_spin") else 0,
            self._int_line_value(self.fps_spin, 0) if hasattr(self, "fps_spin") else 0,
            False if is_basler else (self.force_v4l2_check.isChecked() if hasattr(self, "force_v4l2_check") else False),
            False if is_basler else (self.mjpg_check.isChecked() if hasattr(self, "mjpg_check") else False),
            self.low_latency_check.isChecked() if hasattr(self, "low_latency_check") else True,
            self.threaded_camera_check.isChecked() if hasattr(self, "threaded_camera_check") else True,
        )

    def open_camera_settings_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Camera Settings")
        dlg.setMinimumWidth(460)
        dlg_font = self.font()
        if dlg_font.pointSize() <= 0:
            dlg_font.setPointSize(9)
        dlg.setFont(dlg_font)
        layout = QVBoxLayout(dlg)

        form_box = QGroupBox("Camera / Stream")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        backend_combo = QComboBox()
        backend_combo.addItems(["Auto", "V4L2", "GStreamer", "GStreamer (native)", "FFmpeg", "Basler/Pylon"])
        backend_combo.setCurrentText(self.backend_combo.currentText())
        source_edit = QLineEdit(self.source_edit.text())
        source_edit.setPlaceholderText("0, /dev/video0, video.mp4, rtsp://, or Basler serial")

        width_edit = QLineEdit(self.width_spin.text())
        width_edit.setValidator(QIntValidator(0, 8192, dlg))
        height_edit = QLineEdit(self.height_spin.text())
        height_edit.setValidator(QIntValidator(0, 8192, dlg))
        fps_edit = QLineEdit(self.fps_spin.text())
        fps_edit.setValidator(QIntValidator(0, 240, dlg))
        preview_combo = QComboBox()
        preview_combo.addItems(["Full", "1/2", "1/3", "1/4"])
        preview_combo.setCurrentText(self.preview_scale_combo.currentText())

        format_grid = QGridLayout()
        format_grid.setHorizontalSpacing(8)
        format_grid.setVerticalSpacing(6)
        for edit in (width_edit, height_edit):
            edit.setFixedWidth(72)
            edit.setAlignment(Qt.AlignCenter)
        fps_edit.setFixedWidth(58)
        fps_edit.setAlignment(Qt.AlignCenter)
        preview_combo.setFixedWidth(82)
        format_grid.addWidget(QLabel("Width"), 0, 0)
        format_grid.addWidget(width_edit, 0, 1)
        format_grid.addWidget(QLabel("Height"), 0, 2)
        format_grid.addWidget(height_edit, 0, 3)
        format_grid.addWidget(QLabel("FPS"), 1, 0)
        format_grid.addWidget(fps_edit, 1, 1)
        format_grid.addWidget(QLabel("Preview"), 1, 2)
        format_grid.addWidget(preview_combo, 1, 3)
        format_grid.setColumnStretch(4, 1)

        form.addRow("Backend", backend_combo)
        form.addRow("Source", source_edit)
        form.addRow("Format", format_grid)
        layout.addWidget(form_box)

        exposure_box = QGroupBox("Exposure")
        exposure_layout = QGridLayout(exposure_box)
        exposure_layout.setHorizontalSpacing(8)
        exposure_layout.setVerticalSpacing(6)
        exposure_auto_check = QCheckBox("Auto exposure")
        exposure_auto_check.setChecked(self.exposure_auto_check.isChecked())
        exposure_us_edit = QLineEdit(self.exposure_us_edit.text())
        exposure_us_edit.setValidator(QIntValidator(0, 10000000, dlg))
        exposure_us_edit.setFixedWidth(92)
        exposure_us_edit.setAlignment(Qt.AlignCenter)
        apply_exposure_btn = QPushButton("Apply Exposure")
        apply_exposure_btn.setProperty("compactCaptureButton", True)
        exposure_layout.addWidget(exposure_auto_check, 0, 0, 1, 2)
        exposure_layout.addWidget(QLabel("Manual us"), 1, 0)
        exposure_layout.addWidget(exposure_us_edit, 1, 1)
        exposure_layout.addWidget(apply_exposure_btn, 2, 0, 1, 2, Qt.AlignLeft)
        exposure_layout.setColumnStretch(1, 1)
        layout.addWidget(exposure_box)

        opts_box = QGroupBox("Camera Options")
        opts_layout = QVBoxLayout(opts_box)
        force_v4l2_check = QCheckBox("Force V4L2 - use Linux USB-camera backend")
        force_v4l2_check.setChecked(self.force_v4l2_check.isChecked())
        low_latency_check = QCheckBox("Low latency - reduce buffering/delay")
        low_latency_check.setChecked(self.low_latency_check.isChecked())
        threaded_camera_check = QCheckBox("Threaded reader - smoother preview capture")
        threaded_camera_check.setChecked(self.threaded_camera_check.isChecked())
        mjpg_check = QCheckBox("MJPG - request compressed camera stream")
        mjpg_check.setChecked(self.mjpg_check.isChecked())
        skip_heavy_live_check = QCheckBox("Skip heavy filters - keep live view faster")
        skip_heavy_live_check.setChecked(self.skip_heavy_live_check.isChecked())
        for widget in (force_v4l2_check, low_latency_check, threaded_camera_check, mjpg_check, skip_heavy_live_check):
            opts_layout.addWidget(widget)
        layout.addWidget(opts_box)

        def sync_enabled(*args) -> None:
            manual = not exposure_auto_check.isChecked()
            exposure_us_edit.setEnabled(manual)
            is_basler = backend_combo.currentText() == "Basler/Pylon"
            force_v4l2_check.setEnabled(not is_basler)
            mjpg_check.setEnabled(not is_basler)

        def apply_to_current(*, save: bool = False, close: bool = True) -> None:
            was_open = self.camera.is_open() if hasattr(self, "camera") else False
            old_stream_sig = self._camera_stream_signature() if was_open else None

            self.backend_combo.setCurrentText(backend_combo.currentText())
            self.source_edit.setText(source_edit.text().strip())
            self.width_spin.setText(width_edit.text().strip())
            self.height_spin.setText(height_edit.text().strip())
            self.fps_spin.setText(fps_edit.text().strip())
            self.preview_scale_combo.setCurrentText(preview_combo.currentText())
            self.exposure_auto_check.setChecked(exposure_auto_check.isChecked())
            self.exposure_us_edit.setText(exposure_us_edit.text().strip())
            self.force_v4l2_check.setChecked(force_v4l2_check.isChecked())
            self.low_latency_check.setChecked(low_latency_check.isChecked())
            self.threaded_camera_check.setChecked(threaded_camera_check.isChecked())
            self.mjpg_check.setChecked(mjpg_check.isChecked())
            self.skip_heavy_live_check.setChecked(skip_heavy_live_check.isChecked())
            self._on_exposure_auto_changed()
            self._on_camera_backend_changed(self.backend_combo.currentText())
            self._update_camera_settings_summary()

            new_stream_sig = self._camera_stream_signature() if was_open else None
            stream_changed = bool(was_open and old_stream_sig != new_stream_sig)

            if save:
                self.camera_settings = {
                    "camera_source": self.source_edit.text().strip(),
                    "camera_backend": self.backend_combo.currentText(),
                    "width": self._int_line_value(self.width_spin, 2592),
                    "height": self._int_line_value(self.height_spin, 1944),
                    "fps": self._int_line_value(self.fps_spin, 0),
                    "preview_scale": self.preview_scale_combo.currentText(),
                    "exposure_auto": self.exposure_auto_check.isChecked(),
                    "exposure_us": self._int_line_value(self.exposure_us_edit, 0),
                    "force_v4l2": self.force_v4l2_check.isChecked(),
                    "low_latency": self.low_latency_check.isChecked(),
                    "threaded_camera": self.threaded_camera_check.isChecked(),
                    "mjpg": self.mjpg_check.isChecked(),
                    "skip_heavy_live": self.skip_heavy_live_check.isChecked(),
                }
                path = save_camera_settings(self.camera_settings)
                self.status.showMessage(f"Saved camera settings: {path}", 5000)
            else:
                self.status.showMessage("Camera settings applied", 4000)

            if stream_changed:
                # Width/height/FPS/backend changes do not affect an already-open
                # capture device until it is reopened. Reopen immediately so the
                # screen reflects the requested resolution instead of continuing
                # to show the stale negotiated mode.
                self.timer.stop()
                self.camera.close()
                self.blank_frame_count = 0
                self.open_camera()

            if close:
                dlg.accept()

        exposure_auto_check.stateChanged.connect(sync_enabled)
        backend_combo.currentTextChanged.connect(sync_enabled)
        apply_exposure_btn.clicked.connect(lambda: (apply_to_current(save=False, close=False), self.apply_exposure_to_camera()))
        sync_enabled()

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        apply_btn = QPushButton("Apply")
        save_btn = QPushButton("Save Settings")
        cancel_btn = QPushButton("Cancel")
        apply_btn.clicked.connect(lambda: apply_to_current(save=False))
        save_btn.clicked.connect(lambda: apply_to_current(save=True))
        cancel_btn.clicked.connect(dlg.reject)
        for btn in (apply_btn, save_btn, cancel_btn):
            btn.setProperty("compactCaptureButton", True)
            button_row.addWidget(btn)
        layout.addLayout(button_row)
        dlg.exec()

    def _int_line_value(self, edit: QLineEdit, default: int = 0) -> int:
        try:
            text = edit.text().strip()
            return int(text) if text else default
        except Exception:
            return default

    def _set_int_line_value(self, edit: QLineEdit, value: int) -> None:
        edit.setText(str(max(1, min(99, int(value)))))

    def _review_record(self, reason: str = "operator_review", *,
                       force: bool = False, defect_reason: str = "") -> dict:
        return review_logic.make_review_record(
            reason, force=force, defect_reason=defect_reason)
    def _annotation_reviewed(self, data: dict | None) -> bool:
        return review_logic.annotation_reviewed(data)

    def _annotation_force_reviewed(self, data: dict | None) -> bool:
        return review_logic.annotation_force_reviewed(data)

    def _needs_review_for_image(self, path: Path, data: dict | None = None) -> bool:
        if data is None:
            data = load_annotations(path)
        if not data or not data.get("boxes"):
            return False
        return not review_logic.annotation_reviewed(data)

    def _reset_dataset_image_index(self) -> None:
        """Force a full recipe-folder reindex on the next image-list refresh."""
        self._dataset_index_dirty = True
        self._image_paths_cache = []
        self._image_status_cache.clear()

    def _invalidate_image_status(self, path: Path | None) -> None:
        """Drop only one image from the cached review/status table."""
        if not path:
            return
        try:
            self._image_status_cache.pop(str(Path(path).resolve()), None)
        except Exception:
            self._image_status_cache.pop(str(path), None)

    def _json_mtime_ns(self, path: Path) -> tuple[bool, int]:
        json_path = image_label_json_path(path)
        try:
            exists = json_path.exists()
            return exists, json_path.stat().st_mtime_ns if exists else 0
        except Exception:
            return False, 0

    def _get_dataset_image_paths(self, *, force: bool = False) -> list[Path]:
        """Cached images for the label being trained, newest first."""
        if force or getattr(self, "_dataset_index_dirty", True):
            if self.label_id:
                self._image_paths_cache = sorted(list_images(self.label_id), reverse=True)
            else:
                self._image_paths_cache = []
            valid = {str(p.resolve()) for p in self._image_paths_cache}
            for key in list(self._image_status_cache.keys()):
                if key not in valid:
                    self._image_status_cache.pop(key, None)
            self._dataset_index_dirty = False
        return list(self._image_paths_cache)
    def _cached_image_status(self, path: Path, *, force: bool = False) -> dict:
        """Fast per-image status lookup for the active dataset.

        A sidecar is parsed only when it is new or its mtime changed. The active
        label id is part of the cache key material, because the same image can
        be ready for one label and merely context for another.
        """
        key = str(Path(path).resolve())
        try:
            image_mtime = Path(path).stat().st_mtime_ns if Path(path).exists() else 0
        except Exception:
            image_mtime = 0
        json_exists, json_mtime = self._json_mtime_ns(path)
        label = self.library.get(self.label_id) if self.label_id else None
        reference_source = str(getattr(label, "reference_source", "") or "")
        is_reference = bool(reference_source) and key == str(Path(reference_source).resolve())

        cached = self._image_status_cache.get(key)
        if (
            cached
            and not force
            and cached.get("image_mtime") == image_mtime
            and cached.get("json_exists") == json_exists
            and cached.get("json_mtime") == json_mtime
            and cached.get("label_id") == self.label_id
            # Part of the key, or redefining regions would leave the old marker
            # on the old image and none on the new one.
            and cached.get("is_reference") == is_reference
        ):
            return cached

        data = None
        if json_exists:
            try:
                data = load_annotations(path)
            except Exception:
                data = None

        status = review_logic.annotation_status(data, self.label_id)
        boxes = (data or {}).get("boxes") or []
        own = sum(1 for b in boxes if str(b.get("label_id", "")) == self.label_id)

        prefix = {
            "ready": "✓ REVIEWED OK  ",
            "forced": "⚠ FORCE REVIEW  ",
            "problem": "⚠ REVIEWED CHECK  ",
            "needs_review": f"🟡 REVIEW {own}x  ",
            "background": "▨ BACKGROUND  ",
            "empty": "◇ JSON EMPTY  ",
            "unlabeled": "□ NO JSON  ",
        }.get(status, "□ NO JSON  ")
        # Leads the line: the artwork every read-region on this label is
        # positioned against came from this shot, and redefining regions from a
        # different one silently moves every region.
        if is_reference:
            prefix = "◆ REFERENCE  " + prefix

        entry = {
            "path": path,
            "image_mtime": image_mtime,
            "json_exists": bool(json_exists),
            "json_mtime": json_mtime,
            "label_id": self.label_id,
            "is_reference": is_reference,
            "status": status,
            "prefix": prefix,
            "own_count": int(own),
            "box_count": len(boxes),
            "labeled": bool(boxes),
            "reviewed": review_logic.annotation_reviewed(data),
            "needs_review": status == "needs_review",
            "forced": status == "forced",
        }
        self._image_status_cache[key] = entry
        return entry
    def _refresh_images(self, *args, force: bool = False) -> None:
        # Qt checkbox signals pass an int state; do not treat that as a force refresh.
        if args and isinstance(args[0], bool):
            force = force or bool(args[0])

        review_only = bool(
            hasattr(self, "show_unreviewed_only_check")
            and self.show_unreviewed_only_check.isChecked()
        )
        # Build the visible list and tally the dataset summary in one pass. The
        # summary counts every image in the recipe regardless of the review-only
        # view filter, so it stays correct without a second walk of the cache.
        totals = self._new_summary_totals()
        self.image_list.setUpdatesEnabled(False)
        try:
            self.image_list.clear()
            for p in self._get_dataset_image_paths(force=force):
                entry = self._cached_image_status(p)
                self._accumulate_summary(totals, entry)
                if review_only and not entry.get("needs_review", False):
                    continue
                item = QListWidgetItem(entry.get("prefix", "") + p.name)
                # The name, not the decorated text: prefixes stack and change.
                item.setData(Qt.ItemDataRole.UserRole, p.name)
                self.image_list.addItem(item)
        finally:
            self.image_list.setUpdatesEnabled(True)
        self._set_dataset_summary_label(totals)
        # Keep the row for the image being edited highlighted across list rebuilds.
        self._select_image_in_list()

    def _on_camera_backend_changed(self, backend: str) -> None:
        is_basler = backend == "Basler/Pylon"
        is_gst_native = backend == "GStreamer (native)"
        if hasattr(self, "source_edit"):
            self.source_edit.setEnabled(True)
            if is_basler:
                self.source_edit.setPlaceholderText("Optional Basler serial/model")
            elif is_gst_native:
                self.source_edit.setPlaceholderText("0  or  /dev/video0  (required for GStreamer native)")
            else:
                self.source_edit.setPlaceholderText("0, /dev/video0, video.mp4, or rtsp://")
        if hasattr(self, "force_v4l2_check"):
            self.force_v4l2_check.setEnabled(not is_basler and not is_gst_native)
        if hasattr(self, "mjpg_check"):
            self.mjpg_check.setEnabled(not is_basler and not is_gst_native)
        if hasattr(self, "basler_hint_label"):
            self.basler_hint_label.setVisible(is_basler)
        if is_basler and hasattr(self, "status"):
            self.status.showMessage("Basler/Pylon selected. Source may be left blank or set to a serial/model filter.", 5000)
        if is_gst_native and hasattr(self, "status"):
            self.status.showMessage(
                "GStreamer (native) selected — bypasses OpenCV GStreamer build flags. "
                "Requires python3-gi + JetPack GStreamer plugins for hardware MJPG decode.", 7000
            )

    def _parse_source(self):
        """Return the camera source in the form expected by the selected backend.

        For normal OpenCV/V4L2 sources, blank means camera index 0. For
        Basler/Pylon, blank is valid and means "use the first Pylon camera";
        keeping it blank avoids accidentally treating the optional Basler source
        field like a USB /dev/video index.
        """
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        text = self.source_edit.text().strip()
        if backend == "Basler/Pylon" and not text:
            return ""
        src = text or "0"
        if src.isdigit():
            return int(src)
        return src

    def test_camera(self) -> None:
        """Open the camera briefly and report whether frames are readable."""
        src = self._parse_source()
        width = self._int_line_value(self.width_spin, 0) or None
        height = self._int_line_value(self.height_spin, 0) or None
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        exposure_auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0
        result = quick_test_source(src, backend=backend, width=width, height=height, exposure_auto=exposure_auto, exposure_us=exposure_us)
        title = "Camera Test Passed" if result.ok else "Camera Test Failed"
        QMessageBox.information(self, title, result.message)
        self.status.showMessage(result.message, 8000)

    def open_camera(self) -> None:
        # Keep the recipe object in sync with the capture fields, but do not
        # persist anything here. Adjustments are live camera controls and
        # the status bar, which hid camera-open failures and made the Open
        # Preview button look like it was wired to Save Recipe.
        src = self._parse_source()
        width = self._int_line_value(self.width_spin, 0) or None
        height = self._int_line_value(self.height_spin, 0) or None
        fps = self._int_line_value(self.fps_spin, 0) or None if hasattr(self, "fps_spin") else None
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        is_basler = backend == "Basler/Pylon"
        exposure_auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0

        self.status.showMessage(f"Opening {backend} camera preview...", 3000)
        self.blank_frame_count = 0
        if not self.camera.open(
            src,
            width,
            height,
            fps=fps,
            backend=backend,
            low_latency=self.low_latency_check.isChecked() if hasattr(self, "low_latency_check") else True,
            mjpg=self.mjpg_check.isChecked() if hasattr(self, "mjpg_check") else True,
            threaded=self.threaded_camera_check.isChecked() if hasattr(self, "threaded_camera_check") else True,
            force_v4l2=(False if is_basler else (self.force_v4l2_check.isChecked() if hasattr(self, "force_v4l2_check") else False)),
            exposure_auto=exposure_auto,
            exposure_us=exposure_us,
        ):
            QMessageBox.warning(
                self,
                "Camera",
                self.camera.last_result.message
                + "\n\nTry these quick checks:\n"
                + "• Source 0, then 1, then /dev/video0 for OpenCV cameras\n"
                + "• Backend V4L2 for normal USB webcams\n"
                + "• Backend GStreamer (native) for Jetson — bypasses OpenCV GStreamer build flags\n"
                + "  Requires: sudo apt install python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools\n"
                + "• Backend Basler/Pylon for Basler industrial cameras\n"
                + "• Width/Height set to Default\n"
                + "• Basler test: python -c \"from pypylon import pylon; print(pylon.TlFactory.GetInstance().EnumerateDevices())\"",
            )
            return
        # Force the first tick after (re)opening to process a frame.
        self._last_frame_seq = None
        self.timer.start(16)
        # Streaming now: drawing is blocked until a frame is captured, because a
        # box drawn on a frame that is replaced 30 times a second belongs to no
        # image and cannot be saved against one.
        self._refresh_live_mode()
        self.status.showMessage(self.camera.last_result.message, 8000)

    def close_camera(self) -> None:
        # Nothing to infer on once the stream stops, and a worker left running
        # would hold a model and a thread for no reason.
        if self._live_running():
            self.stop_live_detect()
        self.timer.stop()
        self.camera.close()
        self._refresh_live_mode()
        # A burst ends ready to label: open the last frame captured in it,
        # unless the operator has already opened something else.
        last = getattr(self, "_last_capture_path", None)
        if last is not None and self.current_image_path is None and Path(last).exists():
            self._load_image_path(Path(last))
        count = getattr(self, "_session_captures", 0)
        self._session_captures = 0
        self.status.showMessage(
            f"Live view stopped — {count} captured this session" if count
            else "Live view stopped", 6000)

    def _adjustment_changed(self, *args) -> None:
        if self.last_raw is not None and not self.camera.is_open():
            self.last_adjusted = self._adjust_frame(self.last_raw)
            self.canvas.set_frame(self.last_adjusted)
    def _adjust_frame(self, frame):
        return apply_adjustments(
            frame,
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            gamma=self.gamma_slider.value() / 100.0,
            clahe_enabled=self.clahe_check.isChecked(),
            clahe_clip=self.clahe_clip_slider.value() / 10.0,
            clahe_grid=self.clahe_grid_slider.value(),
            sharpen=self.sharpen_slider.value(),
        )

    def _adjust_live_frame(self, frame):
        """Fast preview adjustment path. Keeps live view responsive while preserving full-quality capture."""
        skip_heavy = getattr(self, "skip_heavy_live_check", None)
        if skip_heavy is not None and skip_heavy.isChecked():
            return apply_adjustments(
                frame,
                brightness=self.brightness_slider.value(),
                contrast=self.contrast_slider.value(),
                gamma=self.gamma_slider.value() / 100.0,
                clahe_enabled=False,
                clahe_clip=self.clahe_clip_slider.value() / 10.0,
                clahe_grid=self.clahe_grid_slider.value(),
                sharpen=0,
            )
        return self._adjust_frame(frame)

    def _scale_preview_frame(self, frame):
        combo = getattr(self, "preview_scale_combo", None)
        if combo is None:
            return frame
        value = combo.currentText()
        if value == "Full":
            return frame
        scale = {"1/2": 0.5, "1/3": 1.0 / 3.0, "1/4": 0.25}.get(value, 1.0)
        if scale >= 0.999:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _on_timer(self) -> None:
        # The display timer runs faster than most cameras deliver frames, so the
        # threaded reader often still holds the same frame as the previous tick.
        # Skip the decode/adjust/scale/repaint pipeline until a new frame arrives.
        if getattr(self.camera, "threaded", False):
            seq = self.camera.frame_seq()
            if seq == self._last_frame_seq and self.last_raw is not None:
                return
        else:
            seq = None

        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.blank_frame_count = getattr(self, "blank_frame_count", 0) + 1
            if self.blank_frame_count in (1, 30, 120):
                self.status.showMessage(
                    f"Camera is open but no frame was read ({self.blank_frame_count} misses). Try Stop/Open, V4L2 backend, source 1, or default resolution.",
                    8000,
                )
            return

        self._last_frame_seq = seq
        self.blank_frame_count = 0
        # Drop stale buffered frames when requested. This makes the display feel current,
        # even if it means skipping intermediate frames.
        if hasattr(self, "low_latency_check") and self.low_latency_check.isChecked():
            self.camera.drain(1)

        self.last_raw = frame
        preview_frame = self._scale_preview_frame(frame)
        self.last_adjusted = self._adjust_live_frame(preview_frame)
        self.canvas.set_frame(self.last_adjusted)

        # Inference runs on the full-resolution frame, not the preview: the
        # point of this view is to see what the model does at the resolution
        # production will hand it. The overlays come back in full-frame
        # coordinates and are scaled to the preview before being drawn.
        if self._live_running():
            try:
                self._live_overlay_scale = (
                    preview_frame.shape[1] / float(frame.shape[1]),
                    preview_frame.shape[0] / float(frame.shape[0]),
                )
            except Exception:
                self._live_overlay_scale = (1.0, 1.0)
            self._pump_live_detect(frame)

        self._preview_frame_counter += 1
        now_t = time.perf_counter()
        elapsed = now_t - self._preview_fps_t0
        if elapsed >= 1.0:
            self._preview_fps = self._preview_frame_counter / elapsed
            self._preview_frame_counter = 0
            self._preview_fps_t0 = now_t
            cam_fps = self.camera.read_fps() if hasattr(self.camera, "read_fps") else 0.0
            self.status.showMessage(f"Live view: display {self._preview_fps:.1f} FPS, camera read {cam_fps:.1f} FPS", 1200)


    def _refresh_live_mode(self) -> None:
        """One source of truth: is a camera streaming right now?

        Derived rather than tracked, because every path that could get it wrong
        (capture, open an image, stop the camera, a failed open) would otherwise
        need its own flag update.
        """
        try:
            live = bool(self.camera.is_open())
        except Exception:
            live = False
        self._set_live_mode(live)

    def _set_live_mode(self, live: bool) -> None:
        """Block annotation while a camera is streaming, and say why.

        Not a tab check: what matters is whether the frame under the cursor is a
        still. Capturing or opening a saved image ends live mode even though the
        Live Capture tab is still in front.
        """
        if hasattr(self.canvas, "set_drawing_enabled"):
            self.canvas.set_drawing_enabled(not live)
        if hasattr(self, "mode_label"):
            self.mode_label.setText("Mode: Live preview" if live else "Mode: Labeling")
        if hasattr(self, "guidance_label") and not getattr(self, "_awaiting_reference_box", False):
            self.guidance_label.setText(
                "Live preview — capture a frame before drawing. A box drawn on a "
                "live frame belongs to no image."
                if live else
                "OBB labels: drag to draw, then adjust the four corner handles.")

    def capture_frame(self, save_adjusted: bool) -> None:
        """Save the current frame into the active label's dataset.

        The camera keeps streaming. Capturing is a burst activity -- frame,
        shoot, reposition, shoot again -- and stopping the preview after every
        press would make a session of twenty captures twenty reopenings.

        Nothing is loaded onto the canvas either, for the same reason: the live
        frame would overwrite it on the next tick, and pointing
        ``current_image_path`` at a still nobody is looking at is how labels get
        saved against the wrong image. The last capture is remembered and
        opened when the preview stops, so a burst ends ready to label.
        """
        if self.last_raw is None:
            QMessageBox.information(self, "Capture", "No frame available yet. Open live view first.")
            return
        # Full-resolution adjustment, matching what the live view shows -- the
        # preview is downscaled and may skip CLAHE/sharpen, but the saved frame
        # always gets the complete pipeline at full size.
        adjusted = self._adjust_frame(self.last_raw) if save_adjusted else None
        # Capturing adjusted saves only the adjusted frame; the raw twin was
        # doubling the dataset with images nobody wanted to label.
        raw_path, adj_path = save_capture(
            self.label_id, self.last_raw, adjusted, save_raw=not save_adjusted
        )
        path = adj_path if adj_path else raw_path
        self._last_capture_path = path
        self._session_captures = getattr(self, "_session_captures", 0) + 1
        self._dataset_index_dirty = True
        self._refresh_images(force=True)
        self._update_dataset_summary()

        if not self._camera_is_live():
            # Captured from a still (no preview running): show it straight away.
            self.current_image_path = path
            self.canvas.load_image(path)
            self.canvas.clear_boxes()

        # Name which variant landed on disk: the live view always renders the
        # adjusted frame, so a raw capture can look darker than what was on
        # screen and the difference is otherwise invisible.
        kind = "adjusted" if save_adjusted else "raw (unadjusted)"
        total = len(self._get_dataset_image_paths())
        self.status.showMessage(
            f"Captured {kind}: {path.name} — {self._session_captures} this session, "
            f"{total} in {self.label_id}", 6000
        )

    def _camera_is_live(self) -> bool:
        try:
            return bool(self.camera.is_open())
        except Exception:
            return False
    def capture_reference(self) -> None:
        """Capture a frame in order to define this label's read-regions.

        The frame is saved into the dataset like any other capture -- a picture
        of the label is training data whatever else it is also used for. What
        this adds is the follow-through: draw the label's box and the region
        editor opens on it, flattened, without anyone hunting for a menu.
        """
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            QMessageBox.information(
                self, "Capture Reference",
                "Open a label first -- read-regions belong to a label.")
            return
        if self._existing_artwork(label) is not None:
            # A label's artwork is defined once. Shooting another reference for
            # it is the replace path, and that is behind a confirmation because
            # every region is positioned against the artwork it replaces.
            QMessageBox.information(
                self, "Capture Reference",
                f"'{self.label_id}' already has artwork, and its read-regions are "
                "positioned on it.\n\n"
                "Use Edit Regions to adjust them, or Tools > Replace label artwork "
                "if the printed label itself has changed.")
            return
        if self.last_raw is None:
            QMessageBox.information(
                self, "Capture Reference",
                "No frame yet. Open the live preview, frame the label, and try again.")
            return

        adjusted = bool(getattr(self, "capture_adjusted_default", False))
        self.capture_frame(save_adjusted=adjusted)
        # Unlike a burst capture, this one exists to be drawn on, so the preview
        # stops and the frame is opened.
        self.close_camera()
        last = getattr(self, "_last_capture_path", None)
        if last is not None and Path(last).exists():
            self._load_image_path(Path(last))
        # Armed rather than modal: the operator still has to say where the label
        # is, and a dialog sitting over the canvas is the worst possible place
        # to ask for that.
        self._awaiting_reference_box = True
        if hasattr(self, "guidance_label"):
            self.guidance_label.setText(
                f"Reference capture — draw the {self.label_id} box and it opens "
                "flattened to draw regions on.")
        self.status.showMessage(
            "Reference captured. Draw the label's box to define its read-regions.",
            10000)

    def _reference_box_drawn(self) -> bool:
        """True when an armed reference capture now has its label box."""
        if not getattr(self, "_awaiting_reference_box", False):
            return False
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            self._awaiting_reference_box = False
            return False
        return any(str(getattr(b, "label", "")) == self.label_id
                   for b in self.canvas.boxes)

    def _disarm_reference_capture(self) -> None:
        self._awaiting_reference_box = False
        if hasattr(self, "guidance_label"):
            self.guidance_label.setText(
                "OBB labels: drag to draw, then adjust the four corner handles.")
    def _save_test_settings(self) -> None:
        """Persist the Model Test tab so nothing is re-entered next launch."""
        if not hasattr(self, "test_model_edit"):
            return
        try:
            save_test_settings({
                "model": self.test_model_edit.text().strip(),
                "image": self.test_image_edit.text().strip(),
                "imgsz": int(self.test_imgsz_spin.value()),
                "conf": float(self.test_conf_spin.value()),
                "device": self.test_device_edit.text().strip(),
                "hide_saved_labels": bool(self.test_hide_saved_labels_check.isChecked()),
            })
        except Exception:
            # Settings are a convenience; never let a write failure break testing.
            pass

    def browse_test_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select LabelVision OBB model", str(EXPORT_DIR), "YOLO Model (*.pt *.onnx *.engine);;All files (*.*)")
        if path:
            self.test_model_edit.setText(path)
            self._save_test_settings()

    def browse_test_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select test image", str(dataset_folder(self.label_id)), "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.test_image_edit.setText(path)
            self._save_test_settings()
            self._load_image_path(Path(path))

    def use_current_test_image(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Test Models", "Open or capture an image first, then click Use Current.")
            return
        self.test_image_edit.setText(str(self.current_image_path))
        self._save_test_settings()

    def clear_model_test_overlay(self) -> None:
        """Remove visual test layers without deleting saved label data."""
        if hasattr(self.canvas, "clear_all_visual_overlays"):
            self.canvas.clear_all_visual_overlays()
        elif hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        self._model_test_overlay_active = False
        self.status.showMessage("Visual overlays cleared; saved labels were not deleted", 4000)

    def show_saved_annotations(self) -> None:
        """Show saved/manual labels again after testing."""
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(True)
        self.status.showMessage("Saved labels are visible again", 3000)

    def _model_test_device_arg(self):
        text = self.test_device_edit.text().strip() if hasattr(self, "test_device_edit") else "0"
        if not text:
            return None
        if text.lower() == "cpu":
            return "cpu"
        if text.isdigit():
            return int(text)
        return text

    def _load_test_model(self, model_path: str, model_name: str):
        if not model_path:
            raise RuntimeError(f"Select a {model_name} model first.")
        p = Path(model_path)
        if not p.exists():
            raise RuntimeError(f"{model_name} model not found:\n{p}")

        # Cache the test model so repeated Run Test clicks do not reload .pt files.
        if self._test_model is not None and self._test_model_path == str(p):
            return self._test_model

        try:
            from ultralytics import YOLO
        except Exception as e:
            if getattr(sys, "frozen", False):
                # pip cannot help here: a packaged build imports only from its
                # own bundle. This means the build is missing a dependency
                # (Ultralytics itself, or something it imports such as
                # matplotlib), so say that instead of sending the user to pip.
                raise RuntimeError(
                    "This build cannot load the model backend.\n\n"
                    "Ultralytics or one of its dependencies is missing from the "
                    "packaged application. Installing it with pip will not help, "
                    "because a packaged build only loads modules bundled inside "
                    "it.\n\n"
                    "This is a packaging defect -- please report it with the "
                    "error below.\n\n"
                    f"Original error: {e}"
                )
            raise RuntimeError(
                "Ultralytics is not installed in this Python environment.\n\n"
                "Install it with:\n"
                "pip install ultralytics\n\n"
                f"Original error: {e}"
            )
        model = YOLO(str(p))
        self._test_model = model
        self._test_model_path = str(p)
        return model

    def run_model_test(self) -> None:
        if not hasattr(self, "test_results_text"):
            return
        model_path = self.test_model_edit.text().strip()
        image_text = self.test_image_edit.text().strip()
        if not image_text and self.current_image_path:
            image_text = str(self.current_image_path)
            self.test_image_edit.setText(image_text)
        if not image_text:
            QMessageBox.information(self, "Test Models", "Select a test image first.")
            return
        image_path = Path(image_text)
        if not image_path.exists():
            QMessageBox.warning(self, "Test Models", f"Test image not found:\n{image_path}")
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            QMessageBox.warning(self, "Test Models", f"Could not read image:\n{image_path}")
            return

        try:
            self.status.showMessage("Loading/running model...", 2000)
            QApplication.processEvents()
            model = self._load_test_model(model_path, "LabelVision OBB")

            imgsz = int(self.test_imgsz_spin.value())
            device = self._model_test_device_arg()
            conf = float(self.test_conf_spin.value())

            common_args = {"imgsz": imgsz, "verbose": False}
            if device is not None:
                common_args["device"] = device

            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, **common_args)
            t1 = time.perf_counter()
        except Exception as e:
            tb = traceback.format_exc()
            self.test_results_text.setPlainText(f"Model test failed:\n{e}\n\n{tb}")
            QMessageBox.warning(self, "Test Models", str(e))
            return

        items, counts = self._detection_overlay_items(results)

        # Keep model-test graphics in a separate canvas overlay layer. Do not bake
        # them into the image pixmap, and do not convert them into saved labels.
        # Always start from a clean visual overlay state so repeated tests never stack.
        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if self.current_image_path != image_path:
            self._load_image_path(image_path)
        elif self.last_raw is None:
            self.last_raw = frame
            self.last_adjusted = self._adjust_frame(frame)
            self.canvas.set_frame(self.last_adjusted)
            self.canvas.image_path = image_path
        # For model testing, hide saved/manual labels by default. They are still loaded
        # and saved normally; this only prevents visual stacking over model results.
        hide_saved = True
        if hasattr(self, "test_hide_saved_labels_check"):
            hide_saved = self.test_hide_saved_labels_check.isChecked()
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(not hide_saved)
        if hasattr(self.canvas, "set_model_test_overlays"):
            self.canvas.set_model_test_overlays(items)
        self.current_image_path = image_path
        self._model_test_overlay_active = True

        total = sum(counts.values())
        summary = []
        summary.append(f"Image: {image_path.name}")
        summary.append(f"Image size: {frame.shape[1]} x {frame.shape[0]}")
        summary.append(f"Detections: {total}")
        for _name, _n in sorted(counts.items()):
            summary.append(f"  {_name}: {_n}")
        summary.append(f"Model time: {(t1 - t0) * 1000:.1f} ms")

        # The number that matters when training one label: did the model find
        # this label, and how surely.
        if self.label_id:
            summary.append("")
            summary.append(f"Active label: {self.label_id}")
            summary.append(f"  {counts.get(self.label_id, 0)} detection(s) of it")

        summary.append("")
        summary.append("Overlay legend:")
        summary.append("Amber polygon/box = a detection, captioned with its class and confidence.")
        summary.append("")
        summary.append("This is preview-only. It does not save labels or affect live inspection.")
        self.test_results_text.setPlainText("\n".join(summary))
        self.status.showMessage(f"Model test complete: {total} detections", 7000)


    def _run_test_model_on_image(self, image_path: Path):
        model_path = self.test_model_edit.text().strip()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image:\n{image_path}")
        model = self._load_test_model(model_path, "LabelVision OBB")
        imgsz = int(self.test_imgsz_spin.value())
        device = self._model_test_device_arg()
        conf = float(self.test_conf_spin.value())
        common_args = {"imgsz": imgsz, "verbose": False}
        if device is not None:
            common_args["device"] = device
        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, **common_args)
        t1 = time.perf_counter()
        return frame, results, t0, t1

    def _label_for_overlay_item(self, item: dict) -> tuple[str, int]:
        """Map a detection to an editable label and class id.

        Driven by the model's own class name rather than the overlay type, so
        every class the model predicts can be labelled -- not just battery and
        bung. When the name matches a configured class its id is used; an
        unknown class keeps its name and the model's id, so a newly trained
        class is still usable before it is added to the class config.
        """
        name = str(item.get("name", "")).strip()
        if not name:
            # Older overlay items carry only a display label like "battery 0.94".
            name = str(item.get("label", "")).strip().split()[0] if item.get("label") else ""
        if not name:
            typ = str(item.get("type", "")).lower()
            name = typ.rsplit("_", 1)[0] if "_" in typ else typ
        if not name or name == "other":
            return "", -1

        for idx, configured in enumerate(self.class_names or []):
            if str(configured).strip().lower() == name.lower():
                return str(configured), idx

        try:
            model_id = int(item.get("cls_id", -1))
        except (TypeError, ValueError):
            model_id = -1
        return name, model_id

    def _overlay_items_to_box_dicts(self, items: list[dict]) -> list[dict]:
        """Convert model-test overlay items into editable canvas box dicts.

        Every predicted class becomes an editable label (OBB, or a plain box for
        detect models), so the operator corrects predictions instead of drawing
        from scratch.
        """
        boxes: list[dict] = []
        for it in items:
            label, class_id = self._label_for_overlay_item(it)
            if not label:
                continue
            pts = it.get("points") or []
            if len(pts) >= 4:
                boxes.append({
                    "kind": "obb",
                    "points": [[float(x), float(y)] for x, y in pts[:4]],
                    "label": label,
                    "class_id": class_id,
                })
            elif "xyxy" in it:
                x1, y1, x2, y2 = [float(v) for v in it.get("xyxy", [0, 0, 0, 0])]
                boxes.append({
                    "kind": "box",
                    "x": x1, "y": y1,
                    "w": max(1.0, x2 - x1), "h": max(1.0, y2 - y1),
                    "label": label,
                    "class_id": class_id,
                })
        return boxes

    def auto_label_current(self) -> None:
        """Pre-label the current image with the trained model, leaving the result
        as editable labels for the operator to correct and save."""
        image_path = self.current_image_path or self._current_test_image_path()
        if image_path is None:
            QMessageBox.information(self, "Auto-label", "Open or capture an image first.")
            return
        image_path = Path(image_path)
        if not image_path.exists():
            QMessageBox.warning(self, "Auto-label", f"Image not found:\n{image_path}")
            return
        model_path = self.test_model_edit.text().strip() if hasattr(self, "test_model_edit") else ""
        if not model_path:
            QMessageBox.information(
                self, "Auto-label",
                "Set a trained OBB model in the Model Test tab first, then try Auto-label again.",
            )
            return

        existing = len(self.canvas.boxes)
        if existing:
            reply = QMessageBox.question(
                self, "Auto-label",
                f"Replace the {existing} existing label(s) on this image with model predictions?\n\n"
                "You can Undo (Ctrl+Z) afterwards.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            self.status.showMessage("Auto-labeling with model...", 2000)
            QApplication.processEvents()
            frame, results, _t0, _t1 = self._run_test_model_on_image(image_path)
        except Exception as e:
            QMessageBox.warning(self, "Auto-label", str(e))
            return

        items, _counts = self._detection_overlay_items(results)
        box_dicts = self._overlay_items_to_box_dicts(items)
        # Predictions arrive already carrying their identity. Stamp it on, so
        # the operator confirms an identity
        # rather than typing it on every box -- and leave the rest unnamed,
        # because guessing an identity is exactly the mistake to avoid.
        box_dicts = self._assign_active_label(box_dicts)
        if not box_dicts:
            QMessageBox.information(
                self, "Auto-label",
                "The model produced no detections at the current confidence.\n"
                "Lower Confidence in the Model Test tab and try again.",
            )
            return

        # Make sure we are editing this image, then replace boxes as one undo step.
        if self.current_image_path != image_path:
            self._load_image_path(image_path)
        self.canvas.clear_model_test_overlays()
        self.canvas.set_annotation_visibility(True)
        self._model_test_overlay_active = False
        self.canvas.push_undo_snapshot()
        self.canvas.set_boxes_from_dicts(box_dicts)
        extra = "".join(f", {n} {nm}" for nm, n in sorted(other_counts.items()))
        self.status.showMessage(
            f"Auto-labeled {battery_count} batteries, {bung_count} bungs{extra}. "
            "Correct as needed, then Save Labels.",
            8000,
        )

    def _detection_disagreement(self, results) -> tuple[int, int, float]:
        """Summarise one model pass for the review queue.

        Returns (detections of the active label, total detections, mean
        confidence). The queue ranks on those: an image where the model found
        nothing, or found the label with low confidence, teaches more than one
        it already handles cleanly.
        """
        items, _counts = self._detection_overlay_items(results)
        if not items:
            return 0, 0, 0.0
        own = sum(1 for i in items
                  if self.label_id and i.get("name") == self.label_id)
        avg_conf = sum(float(i.get("conf", 0.0)) for i in items) / len(items)
        return own, len(items), avg_conf
    def build_review_queue(self) -> None:
        """Run the model across unreviewed images and order them by how much the
        detections disagree with the recipe, so the most informative images are
        labeled first."""
        model_path = self.test_model_edit.text().strip() if hasattr(self, "test_model_edit") else ""
        if not model_path:
            QMessageBox.information(
                self, "Review queue",
                "Set a trained OBB model in the Model Test tab first, then build the queue.",
            )
            return

        todo = []
        for p in self._get_dataset_image_paths():
            entry = self._cached_image_status(p)
            if entry.get("status") not in ("ready", "forced"):
                todo.append(p)
        if not todo:
            QMessageBox.information(self, "Review queue", "No unreviewed images to prioritize in this recipe.")
            return
        if len(todo) > 200:
            reply = QMessageBox.question(
                self, "Review queue",
                f"Run the model on {len(todo)} unreviewed images? This may take a while.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        scored = []
        for i, p in enumerate(todo):
            self.status.showMessage(f"Scoring {i + 1}/{len(todo)}: {p.name}", 1000)
            QApplication.processEvents()
            try:
                _frame, results, _t0, _t1 = self._run_test_model_on_image(p)
            except Exception:
                # A model failure on an image is itself a reason to look at it.
                scored.append(active_learning.QueueItem(str(p), active_learning.MISS_PENALTY))
                continue
            found, total, avg_conf = self._detection_disagreement(results)
            score = active_learning.disagreement_score(found, 1, total, avg_conf)
            scored.append(active_learning.QueueItem(str(p), score))

        ranked = active_learning.rank_items(scored)
        self._review_queue = [Path(it.key) for it in ranked]
        self._review_queue_pos = -1

        top = ranked[: min(5, len(ranked))]
        lines = [f"{Path(it.key).name}: score {it.score:.1f}" for it in top]
        QMessageBox.information(
            self, "Review queue",
            f"Prioritized {len(ranked)} unreviewed image(s), highest disagreement first:\n\n"
            + "\n".join(lines)
            + "\n\nUse Tools > Next in review queue (Ctrl+Shift+N) to step through them.",
        )
        self.next_in_review_queue()

    def prelabel_and_review(self) -> None:
        """Smart pre-labeling loop.

        Runs the trained model across every *unlabeled* image in the recipe,
        writes the predictions to disk as un-reviewed labels (so they appear as
        "needs review" and are excluded from training/export until confirmed),
        then drops the operator into the review queue ordered lowest-confidence
        first. Each queued image opens with the model's boxes already loaded, so
        labeling becomes correcting rather than drawing from scratch.
        """
        model_path = self.test_model_edit.text().strip() if hasattr(self, "test_model_edit") else ""
        if not model_path:
            QMessageBox.information(
                self, "Pre-label & review",
                "Set a trained OBB model in the Model Test tab first, then pre-label.",
            )
            return

        todo = [
            p for p in self._get_dataset_image_paths()
            if self._cached_image_status(p).get("status") == "unlabeled"
        ]
        if not todo:
            QMessageBox.information(
                self, "Pre-label & review",
                "No unlabeled images in this recipe. Pre-labeling only writes to images "
                "that have no saved labels yet, so it never overwrites your work.",
            )
            return

        reply = QMessageBox.question(
            self, "Pre-label & review",
            f"Run the model on {len(todo)} unlabeled image(s), save the predictions as "
            "un-reviewed labels, and open them in the review queue (lowest confidence "
            "first)?\n\nExisting labeled images are left untouched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        scored = []
        written = 0
        empty = 0
        errors = 0
        for i, p in enumerate(todo):
            self.status.showMessage(f"Pre-labeling {i + 1}/{len(todo)}: {p.name}", 1000)
            QApplication.processEvents()
            try:
                frame, results, _t0, _t1 = self._run_test_model_on_image(p)
            except Exception:
                errors += 1
                # A model failure on an image is itself a reason to look at it.
                scored.append(active_learning.QueueItem(str(p), active_learning.MISS_PENALTY))
                continue

            items, _counts = self._detection_overlay_items(results)
            box_dicts = self._assign_active_label(self._overlay_items_to_box_dicts(items))

            if box_dicts:
                h, w = frame.shape[:2]
                # review=None + clear_review=False => saved but not reviewed,
                # so the image shows as "needs review" until the operator confirms.
                save_annotations(p, int(w), int(h), box_dicts, self.class_names, review=None)
                self._invalidate_image_status(p)
                written += 1
            else:
                empty += 1

            found, total, avg_conf = self._detection_disagreement(results)
            score = active_learning.disagreement_score(found, 1, total, avg_conf)
            scored.append(active_learning.QueueItem(str(p), score))

        ranked = active_learning.rank_items(scored)
        self._review_queue = [Path(it.key) for it in ranked]
        self._review_queue_pos = -1

        self._update_dataset_summary()
        self._refresh_images()

        msg = (
            f"Pre-labeled {written} image(s) with model predictions"
            + (f", {empty} had no detections" if empty else "")
            + (f", {errors} failed" if errors else "")
            + ".\n\nQueued "
            f"{len(ranked)} image(s), lowest confidence first. Correct each, then Save "
            "(or Mark reviewed). Use Next in review queue (Ctrl+Shift+N) to advance."
        )
        QMessageBox.information(self, "Pre-label & review", msg)
        self.next_in_review_queue()

    def next_in_review_queue(self) -> None:
        if not self._review_queue:
            QMessageBox.information(
                self, "Review queue",
                "Build the review queue first (Tools > Build review queue).",
            )
            return
        # Advance past any images that have since been deleted.
        while self._review_queue_pos + 1 < len(self._review_queue):
            self._review_queue_pos += 1
            path = self._review_queue[self._review_queue_pos]
            if path.exists():
                self._load_image_path(path)
                self.status.showMessage(
                    f"Review queue {self._review_queue_pos + 1}/{len(self._review_queue)}: {path.name}",
                    6000,
                )
                return
        self.status.showMessage("End of review queue.", 5000)

    def validate_current_image(self) -> None:
        """Run label-quality linting on the on-canvas boxes and report issues."""
        boxes = [b.to_dict() for b in self.canvas.boxes]
        if not boxes:
            QMessageBox.information(self, "Validate", "This image has no labels to validate.")
            return
        issues = review_logic.validate_boxes(self._current_annotation_for_validation(), self.label_id)
        if not issues:
            QMessageBox.information(self, "Validate", "No label-quality issues found.")
        else:
            QMessageBox.warning(
                self, "Validate",
                f"Found {len(issues)} issue(s):\n\n" + "\n".join(f"• {s}" for s in issues),
            )
        self.status.showMessage(f"Validation: {len(issues)} issue(s)", 6000)

    def _current_annotation_for_validation(self) -> dict:
        """The on-canvas state shaped like a sidecar, for the linter.

        Validation runs against what is drawn right now, not what was last
        saved -- catching a degenerate box before it is written is the whole
        point of an advisory check.
        """
        return {
            "image": str(self.current_image_path or ""),
            "label_id": self.label_id,
            "width": int(self.canvas.image_w),
            "height": int(self.canvas.image_h),
            "boxes": self._assign_active_label([b.to_dict() for b in self.canvas.boxes]),
        }
    def bulk_relabel_dialog(self) -> None:
        """Rename/renumber a class across every saved label in the current recipe.

        Operates on the on-disk sidecars, previews the impact first, and clears
        the review marker on changed images so they re-enter the review queue.
        """
        names = [str(n) for n in (self.class_names or [])]
        if len(names) < 2:
            QMessageBox.information(self, "Bulk relabel", "Define at least two classes before relabeling.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Bulk relabel class (current recipe)")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            f"Label: {self.label_id}\n"
            "Reassign every box of one class to another across this recipe's saved labels.\n"
            "Changed images are returned to the review queue."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        source_combo = QComboBox(); source_combo.addItems(names)
        target_combo = QComboBox(); target_combo.addItems(names)
        if len(names) > 1:
            target_combo.setCurrentIndex(1)
        form.addRow("From class", source_combo)
        form.addRow("To class", target_combo)
        layout.addLayout(form)

        preview_label = QLabel("Click Preview to count affected labels.")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.setEnabled(False)
        btn_row.addWidget(preview_btn); btn_row.addWidget(apply_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        ldir = label_folder(self.label_id)

        def do_preview() -> None:
            src = source_combo.currentText()
            tgt = target_combo.currentText()
            if src == tgt:
                preview_label.setText("Source and target classes are the same; nothing to do.")
                apply_btn.setEnabled(False)
                return
            report = relabel_logic.scan_relabel(
                ldir, match_label=src, new_label=tgt, new_class_id=names.index(tgt)
            )
            if report["boxes"] == 0:
                preview_label.setText(f"No '{src}' labels found in this recipe.")
                apply_btn.setEnabled(False)
            else:
                preview_label.setText(
                    f"Will change {report['boxes']} box(es) across {report['images']} image(s) "
                    f"from '{src}' to '{tgt}'.\nThose images will be marked needs-review."
                )
                apply_btn.setEnabled(True)

        def do_apply() -> None:
            src = source_combo.currentText()
            tgt = target_combo.currentText()
            if src == tgt:
                return
            reply = QMessageBox.question(
                dlg, "Bulk relabel",
                f"Relabel all '{src}' to '{tgt}' across the {self.label_id} dataset?\n\n"
                "This edits saved label files and cannot be undone from the canvas.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            report = relabel_logic.apply_relabel(
                ldir, match_label=src, new_label=tgt, new_class_id=names.index(tgt)
            )
            dlg.accept()
            self._reset_dataset_image_index()
            self._refresh_images(force=True)
            self._update_dataset_summary()
            if self.current_image_path and Path(self.current_image_path).exists():
                self._load_image_path(Path(self.current_image_path))
            self.status.showMessage(
                f"Relabeled {report['boxes']} box(es) across {report['images']} image(s); marked needs-review.",
                8000,
            )

        preview_btn.clicked.connect(do_preview)
        apply_btn.clicked.connect(do_apply)
        cancel_btn.clicked.connect(dlg.reject)
        # Re-preview whenever the selection changes so Apply reflects current choice.
        source_combo.currentIndexChanged.connect(lambda _i: apply_btn.setEnabled(False))
        target_combo.currentIndexChanged.connect(lambda _i: apply_btn.setEnabled(False))
        dlg.exec()

    def undo_canvas(self) -> None:
        if not self.canvas.undo():
            self.status.showMessage("Nothing to undo", 3000)

    def redo_canvas(self) -> None:
        if not self.canvas.redo():
            self.status.showMessage("Nothing to redo", 3000)

    def _current_test_image_path(self) -> Path | None:
        image_text = self.test_image_edit.text().strip() if hasattr(self, "test_image_edit") else ""
        if not image_text and self.current_image_path:
            image_text = str(self.current_image_path)
            self.test_image_edit.setText(image_text)
        if not image_text:
            return None
        return Path(image_text)


    def _model_class_name(self, names, cls_id: int) -> str:
        """Return a stable class name from Ultralytics result/model names.

        Ultralytics can expose names as a dict ({0: 'bung'}) or a list
        (['bung']). Older code only handled dicts, which made Run Count filter
        out valid bungs when names were list-like.
        """
        try:
            if isinstance(names, dict):
                return str(names.get(cls_id, f"class_{cls_id}"))
            if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                return str(names[cls_id])
        except Exception:
            pass
        return f"class_{cls_id}"

    def _normalize_class_token(self, value: str) -> str:
        """Normalize class/filter text for forgiving matching."""
        import re
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _class_filter_match(self, name: str, cls_id: int, names: set[str], ids: set[int]) -> bool:
        """Return True when a detection belongs to a requested model class.

        Single-model OBB testing must not treat every polygon as both a
        battery and a bung.  Matching is forgiving for names but exact for
        numeric IDs:
        - numeric class IDs match exactly
        - exact lowercase names match
        - normalized names match (rubber_bung == rubber bung)
        - partial tokens match (bung matches bungs/rubber_bung/bung_cap)
        """
        if cls_id in ids:
            return True
        lname = str(name).strip().lower()
        nname = self._normalize_class_token(lname)
        for token in names:
            token_l = str(token).strip().lower()
            ntok = self._normalize_class_token(token_l)
            if not ntok:
                continue
            if lname == token_l or nname == ntok:
                return True
            if ntok in nname:
                return True
        return False

    def _name_matches(self, name: str, tokens: set[str]) -> bool:
        """Name-only half of _class_filter_match, ignoring class IDs."""
        lname = str(name).strip().lower()
        if not lname:
            return False
        nname = self._normalize_class_token(lname)
        for token in tokens:
            token_l = str(token).strip().lower()
            ntok = self._normalize_class_token(token_l)
            if not ntok:
                continue
            if lname == token_l or nname == ntok or ntok in nname:
                return True
        return False

    def _filter_names_from_edit(self, edit_attr: str, default_text: str) -> set[str]:
        widget = getattr(self, edit_attr, None)
        text = widget.text().strip() if widget is not None else default_text
        names = {part.strip().lower() for part in text.split(",") if part.strip() and not part.strip().isdigit()}
        default_names = {part.strip().lower() for part in default_text.split(",") if part.strip() and not part.strip().isdigit()}
        return names or default_names

    def _filter_ids_from_edit(self, edit_attr: str, default_text: str) -> set[int]:
        widget = getattr(self, edit_attr, None)
        text = widget.text().strip() if widget is not None else default_text
        ids: set[int] = set()
        for part in text.split(","):
            value = part.strip()
            if value.isdigit():
                ids.add(int(value))
        return ids

    def _point_inside_polygon(self, x: float, y: float, poly: list[list[float]]) -> bool:
        try:
            return geom.point_in_polygon(x, y, poly)
        except Exception:
            return False

    def _normalize_angle_deg(self, angle: float) -> float:
        return geom.normalize_angle_deg(angle)

    def _polygon_long_edge_angle(self, pts) -> tuple[float | None, float]:
        return geom.polygon_long_edge_angle(pts)

    def _detection_overlay_items(self, results) -> tuple[list[dict], dict[str, int]]:
        """Overlay items for every detection the model returned.

        No class filter: every class the detector reports is interesting, and the
        old battery/bung split silently dropped anything outside those two --
        so a class could be trained and never seen or validated here.

        Returns (items, {class_name: count}).
        """
        items: list[dict] = []
        counts: dict[str, int] = {}

        for r in results or []:
            names = getattr(r, "names", {}) or {}

            obb = getattr(r, "obb", None)
            if obb is not None:
                try:
                    polys = obb.xyxyxyxy.cpu().numpy()
                except Exception:
                    polys = []
                confs = self._safe_np(obb, "conf")
                clss = self._safe_np(obb, "cls")
                ids = self._safe_np(obb, "id")
                if len(polys):
                    for i, poly in enumerate(polys):
                        pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                        if len(pts) < 4:
                            continue
                        cls_id = int(clss[i]) if i < len(clss) else 0
                        name = self._model_class_name(names, cls_id)
                        conf = float(confs[i]) if i < len(confs) else 0.0
                        track_id = int(ids[i]) if i < len(ids) else None
                        items.append({
                            "type": "other_obb",
                            "track_id": track_id,
                            "points": [[float(x), float(y)] for x, y in pts],
                            "cx": float(np.mean(pts[:, 0])),
                            "cy": float(np.mean(pts[:, 1])),
                            "conf": conf,
                            "cls_id": cls_id,
                            "name": name,
                            "label": (f"{name} #{track_id} {conf:.2f}"
                                      if track_id is not None else f"{name} {conf:.2f}"),
                        })
                        counts[name] = counts.get(name, 0) + 1
                    continue

            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = self._safe_np(boxes, "xyxy")
            confs = self._safe_np(boxes, "conf")
            clss = self._safe_np(boxes, "cls")
            ids = self._safe_np(boxes, "id")
            for i, box in enumerate(xyxy):
                cls_id = int(clss[i]) if i < len(clss) else 0
                name = self._model_class_name(names, cls_id)
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                conf = float(confs[i]) if i < len(confs) else 0.0
                track_id = int(ids[i]) if i < len(ids) else None
                items.append({
                    "type": "other_box",
                    "track_id": track_id,
                    "xyxy": [x1, y1, x2, y2],
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "conf": conf,
                    "cls_id": cls_id,
                    "name": name,
                    "label": (f"{name} #{track_id} {conf:.2f}" if track_id is not None
                              else f"{name} {conf:.2f}"),
                })
                counts[name] = counts.get(name, 0) + 1
        return items, counts

    @staticmethod
    def _safe_np(obj, attr):
        """An Ultralytics tensor attribute as numpy, or [] if it is not there.

        Torch tensors need .cpu() first; anything already array-like is taken as
        it comes. Requiring .cpu() unconditionally meant a plain array returned
        [] and the detections vanished with no error anywhere -- the worst shape
        for a bug on a display path.
        """
        value = getattr(obj, attr, None)
        if value is None:
            return []
        try:
            return value.cpu().numpy()
        except AttributeError:
            pass
        except Exception:
            return []
        try:
            return np.asarray(value)
        except Exception:
            return []
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", str(dataset_folder(self.label_id)), "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self._load_image_path(Path(path))

    def _load_selected_image(self) -> None:
        item = self.image_list.currentItem()
        if not item:
            return
        name = self._image_name_from_list_item(item)
        self._load_image_path(dataset_folder(self.label_id) / name)

    def _load_image_path(self, path: Path) -> None:
        # A still is on screen from here on, so drawing comes back. Guarded
        # rather than unconditional: close_camera opens the last capture when a
        # burst ends, and closing an already-closed camera from in here made
        # that mutually recursive.
        if self._camera_is_live():
            self.close_camera()
        else:
            self.timer.stop()
            self._refresh_live_mode()
        if not self.canvas.load_image(path):
            QMessageBox.warning(self, "Image", "Could not load image.")
            return
        self.current_image_path = path
        self.last_raw = cv2.imread(str(path))
        self.last_adjusted = self._adjust_frame(self.last_raw)
        self.canvas.set_frame(self.last_adjusted)
        # Keep labels tied to the selected image dimensions/path, even after preview adjustments.
        self.canvas.image_path = path
        data = load_annotations(path)
        if data:
            self.canvas.set_boxes_from_dicts(data.get("boxes", []))
        else:
            self.canvas.clear_boxes()
        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(True)
        self._model_test_overlay_active = False
        self._select_image_in_list(path)
        self.status.showMessage(f"Loaded image: {path.name}", 5000)

    def _select_image_in_list(self, path: Path | None = None) -> None:
        """Highlight the row for the given (or current) image in the captured-images
        list so the operator can always see which file is being edited."""
        if path is None:
            path = self.current_image_path
        if not path or not hasattr(self, "image_list"):
            return
        target = Path(path).name
        list_widget = self.image_list
        blocked = list_widget.blockSignals(True)
        try:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if self._image_name_from_list_item(item) == target:
                    list_widget.setCurrentRow(i)
                    list_widget.scrollToItem(item)
                    return
            # The current image is filtered out of the view (e.g. review-only
            # filter); drop any stale highlight rather than point at a different file.
            list_widget.setCurrentRow(-1)
        finally:
            list_widget.blockSignals(blocked)


    def _image_name_from_list_item(self, item_or_text) -> str:
        """The file name behind a list row.

        Rows carry their file name in a data role, because parsing it back out
        of the display text is what broke: prefixes stack ("REFERENCE" in front
        of "REVIEWED OK"), and splitting on the first double space handed the
        rest of the prefix back as part of the name -- so opening that row tried
        to read a file called "REVIEWED OK  capture.jpg".

        The text parsing survives only as a fallback for rows built elsewhere,
        and now takes the LAST prefix separator rather than the first.
        """
        if isinstance(item_or_text, QListWidgetItem):
            stored = item_or_text.data(Qt.ItemDataRole.UserRole)
            if stored:
                return str(stored)
            text = item_or_text.text()
        else:
            text = str(item_or_text)

        if "  " in text:
            return text.rsplit("  ", 1)[1]
        if text[:2] in ("✓ ", "⚠ ", "□ ", "◇ ", "▨ ", "◆ ", "🟡"):
            return text[2:].lstrip()
        return text
    def delete_selected_image(self) -> None:
        item = self.image_list.currentItem() if hasattr(self, "image_list") else None
        path = None

        if item:
            name = self._image_name_from_list_item(item)
            path = dataset_folder(self.label_id) / name
        elif self.current_image_path:
            path = self.current_image_path

        if path is None:
            QMessageBox.information(self, "Delete Image", "Select a captured image first.")
            return

        if not path.exists():
            QMessageBox.information(self, "Delete Image", f"Image does not exist:\n{path}")
            self._refresh_images()
            return

        related = [path]
        label_path = image_label_json_path(path)
        if label_path.exists():
            related.append(label_path)

        # If deleting a raw image, include the matching adjusted image and label.
        if not path.stem.endswith("_adjusted"):
            adjusted = path.with_name(path.stem + "_adjusted" + path.suffix)
            if adjusted.exists():
                related.append(adjusted)
                adj_label = image_label_json_path(adjusted)
                if adj_label.exists():
                    related.append(adj_label)

        # If deleting an adjusted image, keep the raw unless explicitly selected separately.
        unique = []
        seen = set()
        for p in related:
            if p not in seen:
                unique.append(p)
                seen.add(p)

        msg = "Delete selected captured image?"
        msg += "\n\nFiles to delete:\n" + "\n".join(p.name for p in unique)

        reply = QMessageBox.question(
            self,
            "Delete Captured Image",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        deleted = []
        for p in unique:
            try:
                if p.exists():
                    p.unlink()
                    deleted.append(p.name)
            except Exception as e:
                QMessageBox.warning(self, "Delete Image", f"Could not delete:\n{p}\n\n{e}")
                return

        if self.current_image_path in unique:
            self.current_image_path = None
            self.last_raw = None
            self.last_adjusted = None
            self.canvas.clear_boxes()
            self.canvas.pixmap = None
            self.canvas.update()

        self._dataset_index_dirty = True
        for p in unique:
            self._invalidate_image_status(p)
        self._refresh_images(force=True)
        self.status.showMessage("Deleted: " + ", ".join(deleted), 6000)

    def copy_previous_labels(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Labels", "Open or capture an image before copying labels.")
            return
        current_json = image_label_json_path(self.current_image_path)
        label_dir = current_json.parent
        candidates = [p for p in sorted(label_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if p != current_json]
        for p in candidates:
            try:
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                boxes = [self._normalize_import_box(b) for b in data.get("boxes", [])]
                if boxes:
                    self.canvas.push_undo_snapshot()
                    self.canvas.set_boxes_from_dicts(boxes)
                    self.status.showMessage(f"Copied labels from: {p.name}", 5000)
                    self._update_box_count()
                    return
            except Exception:
                continue
        QMessageBox.information(self, "Labels", "No previous saved label file was found for this recipe.")

    def mark_current_reviewed(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Review", "Open or capture an image before marking it reviewed.")
            return
        boxes = [b.to_dict() for b in self.canvas.boxes]
        boxes = self._assign_active_label(boxes)
        if not boxes:
            QMessageBox.information(self, "Review", "This image has no labels to review yet.")
            return
        if not any(str(b.get("label_id", "")) == self.label_id for b in boxes):
            QMessageBox.information(
                self, "Review",
                f"Nothing on this image is labeled '{self.label_id}', but it is in "
                f"that label's dataset.\n\n"
                "Draw the label and set its identity, mark the image a background, "
                "or use Force Review if this is a deliberate defect example.",
            )
            return
        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=self._review_record("manual_mark_reviewed"),
        )
        self._invalidate_image_status(self.current_image_path)
        self.status.showMessage(f"Marked reviewed: {path.name}", 5000)
        self._refresh_images()
        self._update_dataset_summary()
    def force_mark_current_reviewed(self) -> None:
        """Keep an image on purpose, and record *why*.

        The reason is not optional. A defect library where every forced image
        just says "mismatch" cannot answer "do I have enough torn-label
        examples yet?", which is the only question it exists to answer.
        """
        if not self.current_image_path:
            QMessageBox.information(self, "Force Review", "Open or capture an image before force-reviewing it.")
            return
        boxes = self._assign_active_label([b.to_dict() for b in self.canvas.boxes])
        if not boxes:
            QMessageBox.information(self, "Force Review", "This image has no labels to review yet.")
            return

        reasons = review_logic.DEFECT_REASONS
        choice, ok = QInputDialog.getItem(
            self, "Force Review",
            "Keep this image as a deliberate defect example.\n\n"
            "What is wrong with the label?",
            [r.replace("_", " ") for r in reasons], 0, False,
        )
        if not ok:
            return
        defect = reasons[[r.replace("_", " ") for r in reasons].index(choice)]

        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=self._review_record("force_review", force=True, defect_reason=defect),
        )
        self._invalidate_image_status(self.current_image_path)
        self.status.showMessage(f"Force-reviewed as {defect}: {path.name}", 7000)
        self._refresh_images()
        self._update_dataset_summary()
    def mark_current_background(self) -> None:
        """Record the current image as a deliberate negative (no objects).

        Background samples are how the model learns that a bare conveyor is not
        a battery. They are stored as a reviewed annotation with zero boxes and
        an explicit background flag, and export as an empty label file.
        """
        if not self.current_image_path:
            QMessageBox.information(
                self, "Background", "Open or capture an image before marking it background."
            )
            return
        existing = len(self.canvas.boxes)
        if existing:
            reply = QMessageBox.question(
                self, "Background",
                f"This image has {existing} label(s).\n\n"
                "Marking it background will discard them and record the image as "
                "containing no objects. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self.canvas.push_undo_snapshot()
            self.canvas.clear_boxes()

        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            [],
            self.class_names,
            review=review_logic.make_background_record(),
            background=True,
        )
        self._invalidate_image_status(self.current_image_path)
        self.status.showMessage(f"Marked background (no objects): {path.name}", 6000)
        self._update_box_count()
        self._refresh_images()
        self._update_dataset_summary()

    def find_next_unreviewed_image(self) -> None:
        images = self._get_dataset_image_paths()
        if not images:
            QMessageBox.information(self, "Review", "No captured/imported images found for this recipe.")
            return
        start = self._current_image_index()
        order = list(range(max(0, start + 1), len(images))) + list(range(0, max(0, start + 1)))
        for idx in order:
            entry = self._cached_image_status(images[idx])
            if entry.get("needs_review", False):
                batt = int(entry.get("battery_count", 0))
                bung = int(entry.get("bung_count", 0))
                self._load_image_path(images[idx])
                self.status.showMessage(f"Review: loaded unreviewed image ({batt} battery, {bung} bungs)", 6000)
                return
        QMessageBox.information(self, "Review", "No unreviewed labeled images found for this recipe.")

    def save_labels(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Labels", "Open or capture an image before saving labels.")
            return
        boxes = self._assign_active_label([b.to_dict() for b in self.canvas.boxes])
        has_own = any(str(b.get("label_id", "")) == self.label_id for b in boxes)
        # Saving an image that carries the label it was collected for approves
        # it. Saving one that does not, saves the geometry and clears any old
        # approval: editing is not approving, and a stale reviewed marker on an
        # image that no longer shows the label is the bug worth preventing.
        review = self._review_record("save_labels") if has_own else None
        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=review,
            clear_review=(review is None),
        )
        self._invalidate_image_status(self.current_image_path)
        if review is None:
            self.status.showMessage(
                f"Saved labels only; not reviewed -- nothing on this image is "
                f"'{self.label_id}'. Mark it a background, or Force Review if "
                "it is a deliberate defect example.",
                8000,
            )
        else:
            self.status.showMessage(f"Saved labels and marked reviewed: {path}", 5000)
        self._update_dataset_summary()
        self._refresh_images()
    def polish_buttons(self) -> None:
        """Prevent clipped button text on Linux/Qt themes without breaking compact panels."""
        # Recomputed on demand so a button whose text changes later still fits.
        for btn in self.findChildren(QPushButton):
            if btn.property("compactCaptureButton"):
                btn.setMinimumHeight(24)
                btn.setMaximumHeight(26)
                # A zero minimum let the label elide -- "Mark Current Reviewed"
                # rendered as "Mark Current Reviewe". Compact buttons keep their
                # text width; the pane minimum below is sized to fit them.
                btn.setMinimumWidth(self._button_text_width(btn))
                if btn.maximumWidth() > 16777214:
                    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                else:
                    btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                continue
            if btn.property("rightPanelButton"):
                btn.setMinimumHeight(24)
                btn.setMaximumHeight(26)
                btn.setMinimumWidth(self._button_text_width(btn))
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                continue
            # Measure the actual rendered text rather than guessing 6px/char.
            # The old estimate under-measured wide glyphs and Windows DPI
            # scaling, and its 118px ceiling then capped the minimum *below*
            # what the label needed -- so "Promote model" / "Start Training"
            # were clipped. No ceiling: a button must never be narrower than
            # its own text.
            btn.setMinimumHeight(24)
            btn.setMinimumWidth(self._button_text_width(btn))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        for combo in self.findChildren(QComboBox):
            # Let the popup size itself to its contents; maxVisibleItems caps
            # long lists. A forced minimum height left short lists with dead
            # space and made them feel glitchy.
            combo.setMaxVisibleItems(12)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
            combo.setMinimumHeight(28)

    @staticmethod
    def _button_text_width(btn: QPushButton) -> int:
        """Minimum width that fits a button's text, from real font metrics.

        Uses QFontMetrics rather than a character count so it stays correct
        across fonts, locales, and Windows DPI scaling. The padding allowance
        covers the stylesheet's horizontal padding plus the border.
        """
        text = btn.text().replace("&&", "&")
        advance = btn.fontMetrics().horizontalAdvance(text)
        # Qt's own minimumSizeHint accounts for style padding the font advance
        # does not, and it is what actually governs elision -- take whichever
        # is larger or the label still clips.
        return max(44, advance + 18, btn.minimumSizeHint().width())

    def _normalize_import_box(self, box: dict) -> dict:
        """Normalize LabelVision runtime JSON boxes to the editor's simple labels."""
        return review_logic.normalize_box(box)

    def _box_kind(self, box) -> str:
        label = getattr(box, "label", "") or ""
        class_id = int(getattr(box, "class_id", -1))
        if str(label).startswith("battery") or class_id == 0:
            return "battery"
        if str(label).startswith("bung") or class_id == 1:
            return "bung"
        if str(label).startswith("retainer") or class_id == 2:
            return "retainer"
        return str(label)

    def place_regions_on_canvas(self) -> None:
        """Fill in the active label's read-regions on every matching box.

        Runs after a draw, so the barcode and text areas appear the moment the
        label's corners exist -- the whole point of storing them as fractions of
        the label. Purely visual until Save, and a single Undo step.
        """
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None or not label.regions():
            return
        placed = 0
        if hasattr(self.canvas, "push_undo_snapshot"):
            self.canvas.push_undo_snapshot()
        for box in self.canvas.boxes:
            if str(getattr(box, "label", "")) != self.label_id:
                continue
            payload = box.to_dict()
            ann_logic.apply_reference_regions(payload, label)
            box.label_id = self.label_id
            box.regions = payload.get("regions", [])
            placed += len(box.regions)
        self.canvas.update()
        self.status.showMessage(
            f"Placed {placed} read-region(s) from {self.label_id}'s artwork", 5000)

    def define_read_regions(self, *, replace: bool = False) -> None:
        """Draw this label's read-regions, on artwork it keeps for good.

        A label gets its artwork once. The first time, the box on screen is
        flattened straight-on and kept; from then on this opens that same
        artwork, and drawing on a different capture is refused.

        The refusal matters because regions are fractions of the label. Flatten
        a second shot whose outline is drawn even slightly differently and every
        region on the label moves, quietly, against images that were already
        reviewed. Replacing artwork is a deliberate act -- ``replace=True``,
        behind a confirmation -- not a side effect of pressing the same button
        again on a different image.
        """
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            QMessageBox.information(
                self, "Read-Regions",
                "Open a label first -- read-regions belong to a label, not to an image.")
            return

        existing = self._existing_artwork(label)
        if existing is not None and not replace:
            # Already has artwork: edit the regions on it rather than making
            # new artwork out of whatever happens to be on screen.
            self._open_region_editor(label, existing)
            return

        box = self._label_box_for_regions(label)
        if box is None:
            return

        frame = self.last_raw
        if frame is None and self.current_image_path:
            frame = cv2.imread(str(self.current_image_path))
        if frame is None:
            QMessageBox.information(
                self, "Read-Regions",
                "Open one of this label's images and draw its box first. The box is "
                "what gets flattened into the artwork you draw regions on.")
            return

        flattened = imageio.rectify_quad(frame, box.to_dict().get("points") or [],
                                         out_width=900)
        if flattened is None:
            QMessageBox.warning(
                self, "Read-Regions",
                "That box could not be flattened -- its corners are collinear or "
                "too small. Redraw it around the whole label and try again.")
            return

        reference = imageio.save_reference(self.label_id, flattened)
        self._open_region_editor(label, str(reference),
                                 source=str(self.current_image_path or ""))

    def _existing_artwork(self, label) -> str | None:
        """This label's artwork, if it has any that is still on disk.

        A reference whose file has gone is treated as none: recovering from a
        deleted file is not the same act as replacing artwork that exists, and
        should not need a confirmation.
        """
        for reference in label.reference_images:
            if reference and Path(reference).is_file():
                return str(reference)
        return None

    def replace_label_artwork(self) -> None:
        """Deliberately re-flatten this label's artwork from the current box.

        Needed when the artwork itself changes -- a new print revision, a
        materially better shot. The existing regions are carried over and shown
        on the new artwork so they can be checked, because the one thing that
        silently breaks here is an outline drawn differently from last time.
        """
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            QMessageBox.information(self, "Replace Artwork", "Open a label first.")
            return
        if self._existing_artwork(label) is None:
            # Nothing to replace: this is just the first definition.
            self.define_read_regions()
            return

        drawn = len(label.regions())
        reply = QMessageBox.question(
            self, "Replace Artwork",
            f"'{self.label_id}' already has artwork, and {drawn} region(s) are "
            "positioned on it.\n\n"
            "Replacing it re-flattens the artwork from the box on this image. "
            "Regions are fractions of the label, so if you outline it even "
            "slightly differently than last time, every one of them moves -- "
            "against images that were already reviewed.\n\n"
            "The existing regions are carried over and shown on the new artwork "
            "so you can check them.\n\nReplace it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.define_read_regions(replace=True)
    def _label_box_for_regions(self, label):
        """The on-canvas box to flatten: the selection, else the only candidate."""
        candidates = [b for b in self.canvas.boxes
                      if str(getattr(b, "label", "")) == self.label_id]
        index = getattr(self.canvas, "selected_idx", None)
        if index is not None and 0 <= index < len(self.canvas.boxes):
            chosen = self.canvas.boxes[index]
            if str(getattr(chosen, "label", "")) == self.label_id:
                return chosen
            QMessageBox.information(
                self, "Read-Regions",
                f"The selected box is a {getattr(chosen, 'label', '?')}. "
                f"Select the {self.label_id} box instead.")
            return None
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            QMessageBox.information(
                self, "Read-Regions",
                f"Draw the {self.label_id} box on this image first. Regions are "
                "positioned inside it, so it has to exist before they can.")
            return None
        QMessageBox.information(
            self, "Read-Regions",
            f"There are {len(candidates)} {self.label_id} boxes on this image. Click the "
            "one to define regions on, then try again.")
        return None

    def _open_region_editor(self, label, reference: str, source: str = "") -> None:
        """Run the region editor against ``reference`` and save what was drawn."""
        from .region_editor import RegionEditorDialog

        dialog = RegionEditorDialog(
            reference,
            [c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__) for c in label.codes],
            [dict(t.__dict__) for t in label.text_fields],
            list(label.anchor_region),
            parent=self,
        )
        if not dialog.exec():
            return

        drawn = dialog.result_regions()
        updated = labels_mod.LabelDef.from_dict({
            **label.to_dict(),
            "codes": drawn["codes"],
            "text_fields": drawn["text_fields"],
            "anchor_region": drawn["anchor_region"],
            # A flattened capture is a better reference than vendor artwork: it
            # is this label, under this line's lighting, through this lens.
            "reference_images": [reference] + [
                r for r in label.reference_images if r != reference],
            "reference_source": source or label.reference_source,
            "variable_data": bool(label.variable_data or drawn["anchor_region"]),
        })
        self.library.add(updated, replace=True)
        persistence.save_library(self.library)
        self.library = persistence.load_library()

        count = len(drawn["codes"]) + len(drawn["text_fields"]) + \
            (1 if drawn["anchor_region"] else 0)
        self._refresh_active_label_panel()
        self._refresh_regions_button()
        self.place_regions_on_canvas()
        self.status.showMessage(
            f"Saved {count} read-region(s) on {self.label_id}. They now apply to "
            "every image of it.", 8000)

    def edit_read_regions(self) -> None:
        """Open this label's regions on the artwork it already has."""
        label = self.library.get(self.label_id) if self.label_id else None
        if label is None:
            QMessageBox.information(self, "Read-Regions", "Open a label first.")
            return
        existing = self._existing_artwork(label)
        if existing is None:
            QMessageBox.information(
                self, "Read-Regions",
                f"'{self.label_id}' has no artwork yet.\n\n"
                "Open one of its images, draw the label box, then use "
                "Define Regions -- the box gets flattened into the artwork.")
            return
        self._open_region_editor(label, existing)

    def _update_box_count(self) -> None:
        """Refresh the on-canvas tallies, and follow through on a reference capture.

        Two numbers, because they answer different questions: how many of *this*
        label are drawn (is this image usable for its dataset at all), and what
        else is on the image (context labels, which are welcome but are not what
        this dataset is for).
        """
        boxes = [b.to_dict() for b in self.canvas.boxes]
        mine = sum(1 for b in boxes if str(b.get("label_id", "")) == self.label_id)
        unnamed = sum(1 for b in boxes
                      if not str(b.get("label_id", "")).strip()
                      and str(b.get("label", "")) != ann_logic.BATTERY_SIDE)
        if hasattr(self, "count_label"):
            target = self.label_id or "no label selected"
            state = "OK" if mine else "NONE"
            extra = f"   {unnamed} unassigned" if unnamed else ""
            self.count_label.setText(
                f"{target}: {mine} drawn  [{state}]{extra}"
            )
        if hasattr(self, "class_counts_label"):
            counts = class_stats.count_labels(boxes)
            total = sum(counts.values())
            self.class_counts_label.setText(
                f"Current image ({total} boxes): {class_stats.format_counts(counts)}"
            )
        # Editing on-screen boxes does not change the on-disk dataset, so the
        # summary is refreshed by save/review/delete/capture and label changes
        # instead of walking every sidecar on each box draw or nudge.

        if self._reference_box_drawn():
            self._disarm_reference_capture()
            # Deferred: this runs inside the canvas's boxes_changed signal, and
            # opening a modal from a signal handler while the mouse is still
            # down leaves the canvas mid-drag.
            QTimer.singleShot(0, self.define_read_regions)
    def clear_boxes_unsaved(self) -> None:
        """Clear the editable canvas only; never overwrite or delete saved JSON."""
        if not self.canvas.boxes:
            self.status.showMessage("No on-screen boxes to clear", 3000)
            return
        self.canvas.push_undo_snapshot()
        self.canvas.clear_boxes()
        self._update_box_count()
        self.status.showMessage("On-screen boxes cleared. Saved JSON was not changed; click Save to overwrite it.", 6000)

    def delete_saved_labels_confirmed(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Delete Saved JSON", "Open or capture an image first.")
            return
        label_path = image_label_json_path(self.current_image_path)
        if not label_path.exists():
            self.status.showMessage("No saved JSON exists for this image", 3000)
            return
        reply = QMessageBox.question(
            self,
            "Delete Saved JSON",
            f"Delete the saved label JSON for this image?\n\n{label_path.name}\n\n"
            "This does not delete the image. It will remove the file-list JSON indicator until you save labels again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            label_path.unlink()
        except Exception as exc:
            QMessageBox.warning(self, "Delete Saved JSON", f"Could not delete saved JSON:\n{exc}")
            return
        self.canvas.clear_boxes()
        self._update_box_count()
        self._invalidate_image_status(self.current_image_path)
        self._refresh_images()
        self.status.showMessage("Saved JSON deleted for current image", 5000)

    # Backward-compatible slot name used by older builds/actions.
    def clear_labels_confirmed(self) -> None:
        self.delete_saved_labels_confirmed()

    def _image_status(self, path: Path) -> tuple[str, int, int]:
        entry = self._cached_image_status(path)
        status = entry.get("status", "unlabeled")
        # For QA/problem search, unreviewed labeled images are still problems.
        if status == "needs_review":
            status = "problem"
        return status, int(entry.get("own_count", 0)), int(entry.get("box_count", 0))
    @staticmethod
    def _new_summary_totals() -> dict:
        return {"total": 0, "labeled": 0, "ready": 0, "forced": 0, "problems": 0,
                "needs_review": 0, "background": 0}

    def _accumulate_summary(self, totals: dict, entry: dict) -> None:
        """Fold one cached image-status entry into the running dataset totals."""
        totals["total"] += 1
        status = entry.get("status", "unlabeled")
        if entry.get("labeled", False):
            totals["labeled"] += 1
        if status == "ready":
            totals["ready"] += 1
        elif status == "forced":
            totals["forced"] += 1
        elif status == "problem":
            totals["problems"] += 1
        elif status == "needs_review":
            totals["needs_review"] += 1
            totals["problems"] += 1
        elif status == "background":
            totals["background"] += 1

    def _set_dataset_summary_label(self, totals: dict) -> None:
        if not hasattr(self, "dataset_label"):
            return
        self.dataset_label.setText(
            f"Dataset: {totals['total']} images, {totals['labeled']} labeled, "
            f"{totals['ready']} ready, {totals['forced']} forced, "
            f"{totals['problems']} problems, {totals['needs_review']} needs review, "
            f"{totals['background']} background"
        )

    def _update_dataset_summary(self) -> None:
        if not hasattr(self, "dataset_label"):
            return
        totals = self._new_summary_totals()
        for p in self._get_dataset_image_paths():
            self._accumulate_summary(totals, self._cached_image_status(p))
        self._set_dataset_summary_label(totals)

    def import_images_to_recipe(self) -> None:
        """Copy external image files (plus any sidecar label JSON) into the recipe."""
        exts = " ".join(f"*{e}" for e in IMPORT_IMAGE_EXTS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images into this recipe", "",
            f"Images ({exts});;All files (*)",
        )
        if not paths:
            return

        # Ask whether label JSON files are in a separate directory.
        json_dir: Path | None = None
        ask = QMessageBox(self)
        ask.setWindowTitle("Import Labels")
        ask.setText(
            "Do you have a separate folder containing the matching label JSON files?\n\n"
            "Choose 'Yes' to point to that folder, or 'No' if labels are next to the images (or there are none)."
        )
        ask.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        ask.setDefaultButton(QMessageBox.Yes)
        if ask.exec() == QMessageBox.Yes:
            chosen = QFileDialog.getExistingDirectory(
                self, "Select label JSON folder", str(Path(paths[0]).parent),
            )
            if chosen:
                json_dir = Path(chosen)

        imported, errors, label_count = import_images(self.label_id, [Path(p) for p in paths], json_dir=json_dir)
        self._reset_dataset_image_index()
        self._refresh_images(force=True)
        msg = (
            f"Imported {len(imported)} image(s) into the {self.label_id} dataset.\n"
            f"Imported {label_count} sidecar label file(s)."
        )
        if errors:
            msg += f"\n\nSkipped {len(errors)}:\n" + "\n".join(f"• {e}" for e in errors[:10])
            QMessageBox.warning(self, "Import Images", msg)
        else:
            QMessageBox.information(self, "Import Images", msg)
        self.status.showMessage(
            f"Imported {len(imported)} image(s), {label_count} label file(s).", 8000
        )

    def import_background_images(self) -> None:
        """Bulk-import negative images -- empty conveyor, bare fixture.

        Marked background as they land, because there is nothing to draw on
        them: making the operator open a few hundred empty frames one at a time
        just to click "no objects here" is the reason negatives never get added.
        """
        exts = " ".join(f"*{e}" for e in IMPORT_IMAGE_EXTS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import background images (no objects)", "",
            f"Images ({exts});;All files (*)",
        )
        if not paths:
            return

        reply = QMessageBox.question(
            self, "Import Backgrounds",
            f"Import {len(paths)} image(s) as backgrounds?\n\n"
            "Each will be marked as containing no objects and exported as an "
            "empty label file. Any existing labels for these files are not "
            "affected -- they are copied in as new images.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        imported, errors, label_count = import_images(
            self.label_id, [Path(p) for p in paths], as_background=True
        )
        self._reset_dataset_image_index()
        self._refresh_images(force=True)
        self._update_dataset_summary()
        msg = (
            f"Imported {len(imported)} background image(s) into the "
            f"{self.label_id} dataset.\n"
            f"Marked {label_count} as containing no objects."
        )
        if errors:
            msg += f"\n\nSkipped {len(errors)}:\n" + "\n".join(f"• {e}" for e in errors[:10])
            QMessageBox.warning(self, "Import Backgrounds", msg)
        else:
            QMessageBox.information(self, "Import Backgrounds", msg)
        self.status.showMessage(f"Imported {len(imported)} background image(s).", 8000)

    def change_data_folder(self) -> None:
        """Point the image library at another folder, e.g. a shared drive.

        Takes effect on restart: DATA_DIR and everything derived from it are
        module-level constants imported across the app, so re-pointing them in a
        running process would leave stale paths behind.
        """
        env_override = os.environ.get(storage_mod.DATA_DIR_ENV, "").strip()
        configured = storage_mod.read_configured_data_dir()

        lines = [f"Current library:\n{DATA_DIR}", ""]
        if env_override:
            lines.append(
                f"Note: {storage_mod.DATA_DIR_ENV} is set in the environment and "
                "overrides whatever is chosen here."
            )
            lines.append("")
        lines.append(
            "Choose a folder to hold captures, labels, recipes and exports. "
            "Point every machine at the same shared folder to share one library."
        )

        box = QMessageBox(self)
        box.setWindowTitle("Data folder")
        box.setIcon(QMessageBox.Information)
        box.setText("Image library location")
        box.setInformativeText("\n".join(lines))
        choose_btn = box.addButton("Choose folder...", QMessageBox.AcceptRole)
        reset_btn = None
        if configured is not None:
            reset_btn = box.addButton("Reset to default", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked is reset_btn and reset_btn is not None:
            storage_mod.write_configured_data_dir(None)
            self._refresh_library_label()
            QMessageBox.information(
                self, "Data folder",
                "Reset to the default location.\n\nRestart the application to apply it.",
            )
            return
        if clicked is not choose_btn:
            return

        chosen = QFileDialog.getExistingDirectory(
            self, "Select the image library folder", str(DATA_DIR),
        )
        if not chosen:
            return

        target = Path(chosen)
        # Verify before saving: a share the operator cannot write to would
        # otherwise only fail on the next launch, after the old path is gone.
        if not storage_mod._ensure_data_dirs(target):
            QMessageBox.warning(
                self, "Data folder",
                f"Cannot create the library folders in:\n{target}\n\n"
                "Check that the drive is connected and that you have permission "
                "to write there. The library was not changed.",
            )
            return

        storage_mod.write_configured_data_dir(target)
        self._refresh_library_label()
        QMessageBox.information(
            self, "Data folder",
            f"Image library set to:\n{target}\n\n"
            "Restart the application to start using it.\n\n"
            "Your existing images were not moved -- copy the contents of\n"
            f"{DATA_DIR}\ninto the new folder if you want to bring them along.",
        )
        self.status.showMessage(f"Data folder set to {target} (restart to apply)", 10000)

    def show_dataset_health(self) -> None:
        """Readiness of every label's dataset, in one table.

        The question it answers is "what do I train next": which labels have
        enough reviewed images, which are short, and which carry stale
        approvals that would quietly poison a run.
        """
        labels = self.library.all()
        if not labels:
            QMessageBox.information(
                self, "Dataset Health",
                "No labels in the library yet. Add one from the Label tab.")
            return

        rows: list[tuple[str, dict, int]] = []
        grand = dataset_health.new_tally()
        for label in labels:
            tally = dataset_health.new_tally()
            for image in list_images(label.label_id):
                data = storage_mod.read_json(
                    storage_mod.annotation_path(label.label_id, image.name))
                dataset_health.add_image(tally, data, label.label_id)
            target = max(1, int(getattr(label, "train_target", 150) or 150))
            rows.append((label.label_id, tally, target))
            dataset_health.merge_tally(grand, tally)

        dlg = QDialog(self)
        dlg.setWindowTitle("Dataset Health")
        dlg.resize(820, 480)
        v = QVBoxLayout(dlg)
        trainable = sum(1 for _id, t, target in rows
                        if dataset_health.export_ready(t) >= target)
        header = QLabel(
            f"{len(rows)} label(s). {trainable} at or above their training target. "
            f"Export-ready overall: {dataset_health.export_ready(grand)} "
            f"of {grand['images']} images."
        )
        header.setWordWrap(True)
        v.addWidget(header)

        cols = ["Label", "Images", "Ready", "Forced", "Problem", "Needs review",
                "Background", "Unlabeled", "Export-ready", "Target", "%"]
        table = QTableWidget(0, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        def add_row(name: str, t: dict, target: int, *, emphasis: bool = False) -> None:
            row = table.rowCount()
            table.insertRow(row)
            ready = dataset_health.export_ready(t)
            pct = int(round(dataset_health.readiness(t, target) * 100)) if target else 100
            values = [
                name, t["images"], t["ready"], t["forced"], t["problem"],
                t["needs_review"], t["background"], t["unlabeled"], ready,
                target or "-", f"{pct}%",
            ]
            for c, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if c >= 1:
                    item.setTextAlignment(Qt.AlignCenter)
                if emphasis:
                    f = item.font(); f.setBold(True); item.setFont(f)
                    item.setForeground(QColor("#bfdbfe"))
                # Stale approvals are the only column that is wrong rather than
                # merely unfinished, so it is the only one coloured.
                elif c == 4 and t["problem"]:
                    item.setForeground(QColor("#f87171"))
                table.setItem(row, c, item)

        for label_id, tally, target in rows:
            add_row(label_id, tally, target)
        add_row("— all labels —", grand, 0, emphasis=True)

        v.addWidget(table)
        note = QLabel(
            "Export-ready = reviewed + force-reviewed + background images, which is "
            "exactly what a reviewed-only export includes. Problem images are "
            "approvals left behind by a later edit and are excluded until re-reviewed."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #94a3b8;")
        v.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        v.addWidget(buttons)
        dlg.exec()
    def check_variable_regions(self) -> None:
        """Say whether each label's date codes and serials really differ.

        The question this answers is whether a varying region is a problem at
        all. One that differs across the dataset carries no signal for the class
        and the model already ignores it. One that is the same picture in every
        image is a shortcut waiting to be learned -- and the day a new value
        appears, detection breaks for reasons nothing in the labels explains.
        """
        library = self.library
        entries = []
        for label in library.all():
            for image in list_images(label.label_id):
                data = storage_mod.read_json(
                    storage_mod.annotation_path(label.label_id, image.name))
                if not data or not review_logic.export_ready(
                        review_logic.annotation_status(data, label.label_id)):
                    continue
                entries.append(dataset_logic.entry_from_annotation(
                    label.label_id, str(image), data))

        if not entries:
            QMessageBox.information(
                self, "Variable Regions",
                "Nothing reviewed yet, so there is nothing to measure.")
            return

        self.status.showMessage("Measuring variable regions...", 2000)
        QApplication.processEvents()
        reports = augment_logic.scan_entries(entries, library)

        dlg = QDialog(self)
        dlg.setWindowTitle("Variable Regions")
        dlg.resize(760, 420)
        v = QVBoxLayout(dlg)
        body = QTextEdit()
        body.setReadOnly(True)
        body.setPlainText(augment_logic.scan_text(reports))
        v.addWidget(body)
        note = QLabel(
            "Capturing across more lots is the real fix. Variable-region copies at "
            "export are the stopgap when the images you have all came from one "
            "session."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #94a3b8;")
        v.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        v.addWidget(buttons)
        dlg.exec()
        self.status.showMessage(
            f"Checked {len(reports)} variable region(s) across {len(entries)} images",
            6000)
    def show_shortcuts_reference(self) -> None:
        """Cheat-sheet of keyboard shortcuts (the menu bar is hidden)."""
        groups = [
            ("Editing", [
                ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                ("Delete", "Delete selected annotation"),
                ("Shift+Delete", "Delete captured image"),
                ("Arrows", "Nudge selected box (Shift = 10px)"),
            ]),
            ("File", [
                ("Ctrl+O", "Open image"),
                ("Ctrl+S", "Save labels"),
            ]),
            ("View", [
                ("Ctrl + / Ctrl -", "Zoom in / out"),
                ("Ctrl+0", "Fit image to window"),
                ("Ctrl+F5", "Refresh recipe index"),
                ("Mouse wheel", "Zoom; Middle/Alt-drag pans"),
            ]),
            ("Class", [
                ("B", "Select battery class"),
                ("U", "Select bung class"),
                ("R", "Select retainer class"),
            ]),
            ("Navigate", [
                ("N / P", "Next / Previous image"),
                ("Ctrl+U", "Find next unreviewed"),
                ("Ctrl+Shift+R", "Mark current reviewed"),
                ("Ctrl+Shift+F", "Force review current"),
            ]),
            ("Tools", [
                ("Ctrl+L", "Auto-label current (model)"),
                ("Ctrl+Shift+P", "Pre-label unlabeled && review (model)"),
                ("Ctrl+Shift+V", "Validate current image"),
                ("Ctrl+Shift+N", "Next in review queue"),
                ("C", "Capture adjusted frame"),
                ("F1", "Show this shortcut reference"),
            ]),
        ]
        lines = ["<table cellpadding='4'>"]
        for title, items in groups:
            lines.append(f"<tr><td colspan='2' style='padding-top:8px'><b style='color:#bfdbfe'>{title}</b></td></tr>")
            for keys, desc in items:
                lines.append(
                    f"<tr><td style='color:#fbbf24'><code>{keys}</code></td>"
                    f"<td>{desc}</td></tr>"
                )
        lines.append("</table>")

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.resize(460, 560)
        v = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml("".join(lines))
        v.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        v.addWidget(buttons)
        dlg.exec()


    def _current_image_index(self) -> int:
        if not self.current_image_path:
            return -1
        images = self._get_dataset_image_paths()
        try:
            return images.index(self.current_image_path)
        except ValueError:
            return -1

    def _load_image_by_index(self, idx: int) -> None:
        images = self._get_dataset_image_paths()
        if not images:
            QMessageBox.information(self, "Images", "No captured images found for this recipe.")
            return
        idx = max(0, min(len(images) - 1, idx))
        self._load_image_path(images[idx])

    def next_image(self) -> None:
        idx = self._current_image_index()
        if idx < 0:
            self._load_image_by_index(0)
        else:
            self._load_image_by_index(idx + 1)

    def previous_image(self) -> None:
        idx = self._current_image_index()
        if idx < 0:
            self._load_image_by_index(0)
        else:
            self._load_image_by_index(idx - 1)

    def save_and_next(self) -> None:
        self.save_labels()
        self.next_image()

    def find_next_problem_image(self) -> None:
        images = self._get_dataset_image_paths()
        if not images:
            QMessageBox.information(self, "QA", "No captured images found for this recipe.")
            return
        start = self._current_image_index()
        order = list(range(max(0, start + 1), len(images))) + list(range(0, max(0, start + 1)))
        for idx in order:
            status, batt, bung = self._image_status(images[idx])
            if status not in ("ready", "forced"):
                self._load_image_path(images[idx])
                self.status.showMessage(f"QA: {status} image loaded ({batt} battery, {bung} bungs)", 6000)
                return
        QMessageBox.information(self, "QA", "No problem images found. All reviewed/force-reviewed images are handled.")

    def reset_adjustments(self) -> None:
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.gamma_slider.setValue(100)
        self.sharpen_slider.setValue(0)
        self.clahe_check.setChecked(False)
        self.clahe_clip_slider.setValue(20)
        self.clahe_grid_slider.setValue(8)

    def _export_augment(self) -> int:
        if not hasattr(self, "export_augment_spin"):
            return 0
        return int(self.export_augment_spin.value())

    def _browse_live_classifier(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Stage 2 classifier", "", "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.live_classifier_edit.setText(path)

    def _live_crop_px(self, classifier_path: str) -> int:
        """The size stage 2 was trained at, from the export that produced it.

        A classifier fed a different size than it trained on loses accuracy
        silently, and the export already wrote the answer next to the weights,
        so guessing 224 would be choosing to be wrong.
        """
        from pathlib import Path as _P

        if not classifier_path:
            return 224
        for parent in _P(classifier_path).resolve().parents:
            marker = parent / "classes.txt"
            if marker.exists():
                sample = next((parent / "train").rglob("*.jpg"), None)
                if sample is not None:
                    try:
                        import cv2
                        probe = cv2.imread(str(sample))
                        if probe is not None:
                            return int(probe.shape[0])
                    except Exception:
                        break
                break
        return 224

    def show_label_scale_report(self) -> None:
        """Measure how big labels actually are, and say what follows from it.

        The single-stage / two-stage choice turns on a number nobody estimates
        well by eye, and the tool is holding every box needed to compute it.
        """
        from label_detections.core import scale_report, yolo_export

        entries = []
        for label_id in yolo_export.list_datasets():
            entries.extend(yolo_export.collect_entries(label_id, reviewed_only=False))
        imgsz = int(self.test_imgsz_spin.value()) if hasattr(self, "test_imgsz_spin") else 640
        scales = scale_report.measure(entries)
        text = (scale_report.advise(scales, self.library, imgsz=imgsz) + "\n\n"
                + "-" * 70 + "\n\nWORKING\n\n"
                + scale_report.full_report(scales, self.library, imgsz=imgsz))

        box = QMessageBox(self)
        box.setWindowTitle("Label scale")
        box.setText(
            f"How big your labels really are at imgsz {imgsz}, and what that means "
            f"for single-stage vs two-stage.\n\n"
            f"imgsz comes from the Test Models tab -- set it there to match what "
            f"you actually train and run at, or these numbers describe a pipeline "
            f"you do not have.")
        box.setDetailedText(text)
        box.setIcon(QMessageBox.Information)
        box.exec()

    def export_region_crops(self) -> None:
        """Crops of the read-regions themselves, at full resolution.

        The stage that separates labels differing only in fine print. Nothing
        at detector resolution reaches a revision letter, and a whole-label
        crop reaches it less; the region cropped from the original frame keeps
        every pixel it ever had.
        """
        from label_detections.core import classify_export

        try:
            out = classify_export.export_region_crops(
                reviewed_only=self._export_reviewed_only(), library=self.library)
        except FileNotFoundError as exc:
            QMessageBox.information(self, "Export Region Crops", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Export Region Crops", f"Export failed:\n{exc}")
            return

        rows = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]
        native = [float(r.split(",")[4]) for r in rows if len(r.split(",")) > 4]
        classes = (out / "classes.txt").read_text(encoding="utf-8").split()
        span = (f"{min(native):.0f}-{max(native):.0f} px native" if native else "-")
        QMessageBox.information(
            self, "Export complete",
            f"Read-region crops:\n{out}\n\n"
            f"{len(rows)} crop(s) over {len(classes)} label(s), {span}.\n\n"
            f"Cropped from the full-resolution frames, so these keep detail no "
            f"detector input and no whole-label crop can reach. Train a small "
            f"classifier on them to separate labels that differ only in fine "
            f"print.\n\nSuggested command:\n"
            f"  yolo classify train data={out} model=yolo11n-cls.pt imgsz=224")
        self.status.showMessage(f"Exported {len(rows)} region crop(s) to {out}", 8000)

    def export_two_stage(self) -> None:
        """Write a detector dataset and a classifier crop dataset together.

        Offered as a pair rather than two buttons because exporting them
        separately is how the two halves end up with different splits, and a
        classifier validated on crops of batteries the detector trained on
        reports an accuracy that does not survive the line.
        """
        from label_detections.core import classify_export

        task = self._export_task()
        try:
            detect_dir, classify_dir = classify_export.export_two_stage(
                task=task, reviewed_only=self._export_reviewed_only(),
                library=self.library)
        except FileNotFoundError as exc:
            QMessageBox.information(self, "Export Two-Stage", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Export Two-Stage", f"Export failed:\n{exc}")
            return

        classes = (classify_dir / "classes.txt").read_text(encoding="utf-8").split()
        crop_px = 224
        first = next(classify_dir.rglob("*.jpg"), None)
        if first is not None:
            import cv2
            probe = cv2.imread(str(first))
            if probe is not None:
                crop_px = int(probe.shape[0])
        QMessageBox.information(
            self, "Export complete",
            f"Detector (finds where a label is):\n{detect_dir}\n"
            f"  {detect_dir / 'data.yaml'}\n\n"
            f"Classifier ({len(classes)} label(s), {crop_px} px crops):\n{classify_dir}\n\n"
            f"Both share one split and seed, so they hold out the same "
            f"batteries.\n\nSuggested commands:\n"
            f"  yolo {task} train data={detect_dir / 'data.yaml'} model=yolo11s-obb.pt\n"
            f"  yolo classify train data={classify_dir} model=yolo11s-cls.pt imgsz={crop_px}"
        )
        self.status.showMessage(f"Exported two-stage datasets to {classify_dir.parent}", 8000)

    def _export_task(self) -> str:
        return self.export_task_combo.currentData() if hasattr(self, "export_task_combo") else "obb"

    def _export_reviewed_only(self) -> bool:
        # Reviewed-only export is intentionally hardcoded. There is no UI option to include unreviewed imports.
        return True

    def _export_count_summary(self, out: Path) -> str:
        return export_report.count_summary(out)

    def export_yolo(self) -> None:
        """Export just the active label's dataset, for checking it in isolation."""
        task = self._export_task()
        reviewed_only = self._export_reviewed_only()
        if not self.label_id:
            QMessageBox.information(self, "Export", "Open a label first.")
            return
        try:
            out = export_label_yolo(self.label_id, task=task, reviewed_only=reviewed_only,
                                    library=self.library, augment=self._export_augment())
        except Exception as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        train_hint = ("yolo obb train model=yolo11s-obb.pt data=data.yaml ..."
                      if task == "obb" else
                      "yolo detect train model=yolo11s.pt data=data.yaml ...")
        summary = self._export_count_summary(out)
        QMessageBox.information(
            self,
            "Export complete",
            f"YOLO dataset exported to:\n{out}\n\nTraining file:\n{out / 'data.yaml'}\n\n"
            f"{summary}\n\n"
            f"Task: {task}\nClasses: label ids\nReview filter: reviewed only\n\n"
            f"This is one label on its own, for debugging a single dataset. Export "
            f"All is the normal path -- one detector over every label. A model from "
            f"this export has never seen a competing label.\n\n"
            f"Suggested command:\n{train_hint}"
        )
        self.status.showMessage(f"Exported YOLO {task} dataset: {out}", 8000)
    def export_all_yolo(self) -> None:
        """Export every label's dataset into one training set.

        This is the normal export: labels are gathered one at a time but trained
        together, and a model trained on a single label has nothing to tell it
        apart from.
        """
        task = self._export_task()
        reviewed_only = self._export_reviewed_only()
        try:
            out = export_all_labels_yolo(task=task, reviewed_only=reviewed_only,
                                          library=self.library,
                                          augment=self._export_augment())
        except Exception as e:
            QMessageBox.warning(self, "Export All Labels", str(e))
            return
        data_yaml = out / "data.yaml"
        manifest = out / "manifest.csv"
        summary = self._export_count_summary(out)
        QMessageBox.information(
            self,
            "Export All complete",
            f"Combined YOLO dataset exported to:\n{out}\n\n"
            f"Training file:\n{data_yaml}\nManifest:\n{manifest}\n"
            f"Split report:\n{out / 'split_report.txt'}\n\n"
            f"{summary}\n\n"
            f"Task: {task}\nClasses: label ids\nReview filter: reviewed only"
        )
        self.status.showMessage(f"Exported combined YOLO {task} dataset: {out}", 8000)
def main() -> None:
    app = QApplication(sys.argv)
    # Windows enables the combo-box open animation by default (it follows the
    # OS "animate controls" setting). Qt animates the popup open by painting it
    # progressively, which under a stylesheet flashes the window behind before
    # the rows appear -- the popup looks correct once open, but the opening
    # does not. Nothing here depends on the animation, so turn it off.
    QApplication.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    win = MainWindow()
    win.show()
    # A configured library that could not be opened (typically an offline
    # network share) silently relocates the app's data. Say so rather than let
    # an operator label into an unexpected folder.
    if storage_mod.DATA_DIR_FALLBACK_REASON:
        QMessageBox.warning(win, "Data folder", storage_mod.DATA_DIR_FALLBACK_REASON)
    sys.exit(app.exec())
