"""A window for watching a training run, instead of a log in a 340 px rail.

Training is the longest thing this application starts and the least visible
while it runs. Everything worth knowing was already on screen -- it was in a
scrolling wall of yolo output, in a pane narrow enough that a single progress
line wrapped three times, under a chart small enough to be decorative.

So the same information, arranged as the questions somebody actually asks:

  How far through is it?   A bar, and "epoch 34 of 100".
  How long has that taken? Elapsed, from the moment the process started.
  How much longer?         Straight-line from the rate so far.
  Is it still improving?   The best epoch, how long ago, and the patience that
                           will end the run if nothing beats it.

The raw output is still there, folded away, because it is the only thing that
says anything useful when a run fails to start at all.

One tab per stage, because Train Both is two runs and they are not comparable.
A detector run and a classifier run report different metrics, take different
lengths of time, and fail for different reasons -- and the first is still worth
reading after the second has started, which is exactly when a single shared
panel has thrown it away.

Modeless on purpose. A run takes minutes to hours and there is no reason to
stop labelling while it happens; the window can be pushed aside and comes back
with the run's state, not a fresh one.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (QDialog, QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QScrollArea,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ..core import training as training_logic


class StagePanel(QWidget):
    """One training run's own progress, curves and output."""

    OUTPUT_CLOSED = "\u25b8  Output"
    OUTPUT_OPEN = "\u25be  Output"

    def __init__(self, chart: QWidget, parent=None) -> None:
        super().__init__(parent)
        self._total_epochs = 0
        self._gpu_gb = 0.0
        self.weights: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.what_label = QLabel("Preparing...")
        self.what_label.setWordWrap(True)
        self.what_label.setStyleSheet("font-size: 11pt; font-weight: 700;")
        layout.addWidget(self.what_label)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #9aa4b2;")
        layout.addWidget(self.detail_label)

        self.bar = QProgressBar()
        # Indeterminate until the first epoch line says how many there are.
        # Ultralytics prints the count itself, so guessing it from the settings
        # would be a second answer to a question with one source.
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(True)
        self.bar.setFormat("")
        layout.addWidget(self.bar)

        self.progress_label = QLabel("Starting...")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        self.best_label = QLabel("")
        self.best_label.setWordWrap(True)
        self.best_label.setStyleSheet("color: #bfdbfe;")
        layout.addWidget(self.best_label)

        chart_box = QGroupBox("Curves")
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        self.chart = chart
        chart_layout.addWidget(self.chart)
        layout.addWidget(chart_box, 1)

        self.matrix_box = QGroupBox("Confusion matrix")
        self.matrix_box.setToolTip(
            "Which labels the model mistook for which, written by Ultralytics "
            "at the end of the run.\n\n"
            "The one question a headline metric cannot answer. mAP or top-1 "
            "says how often it was right; this says WHICH pair it gets wrong, "
            "which is what decides the next label to photograph more of and "
            "what belongs in a label's confusable_with.")
        matrix_layout = QVBoxLayout(self.matrix_box)
        matrix_layout.setContentsMargins(8, 8, 8, 8)
        self.matrix_label = QLabel("")
        self.matrix_label.setAlignment(Qt.AlignCenter)
        matrix_scroll = QScrollArea()
        matrix_scroll.setWidgetResizable(True)
        matrix_scroll.setWidget(self.matrix_label)
        matrix_layout.addWidget(matrix_scroll)
        self.matrix_box.setVisible(False)
        layout.addWidget(self.matrix_box, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Training output appears here.")
        self.log_box = QGroupBox(self.OUTPUT_CLOSED)
        self.log_box.setCheckable(True)
        self.log_box.setChecked(False)
        self.log_box.setToolTip(
            "Everything the yolo command printed, exactly as it printed it.\n\n"
            "Folded away because the four lines above are that wall's useful "
            "content, and opened by itself when a run fails -- a run that dies "
            "before its first epoch produces no progress, no curves and no "
            "metrics, and this is the only thing that says why.\n\n"
            "Also where to look for what Ultralytics reports once and never "
            "again: the dataset it actually scanned, the device it chose, and "
            "the directory it is writing to.")
        log_layout = QVBoxLayout(self.log_box)
        log_layout.setContentsMargins(8, 4, 8, 8)
        log_layout.addWidget(self.log)
        self.log.setVisible(False)
        self.log_box.toggled.connect(self.log.setVisible)
        # The same arrow the Train tab's Advanced sections use. Without it a
        # closed group box is just an empty rectangle with a word on it.
        self.log_box.toggled.connect(
            lambda open_: self.log_box.setTitle(
                self.OUTPUT_OPEN if open_ else self.OUTPUT_CLOSED))
        layout.addWidget(self.log_box)

    # --- while it runs -----------------------------------------------------

    def begin(self, stage: str, model: str, data: str) -> None:
        self.weights = None
        self._total_epochs = 0
        self._gpu_gb = 0.0
        self.matrix_label.setText("")
        self.matrix_box.setVisible(False)
        self.what_label.setText(f"Training the {stage}")
        self.detail_label.setText(f"{model}\n{data}" if data else str(model))
        self.bar.setRange(0, 0)
        self.bar.setFormat("")
        self.progress_label.setText("Starting -- the first epoch also warms up "
                                    "and caches, so it is the slow one.")
        self.best_label.setText("")
        self.log.clear()
        self.log_box.setChecked(False)

    def append_output(self, text: str) -> None:
        if not text:
            return
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
        found = training_logic.parse_epoch(text)
        if found:
            self.set_epoch(*found)
        gpu = training_logic.parse_gpu_gb(text)
        if gpu:
            self._gpu_gb = gpu

    def set_epoch(self, done: int, total: int) -> None:
        if total and total != self._total_epochs:
            self._total_epochs = int(total)
            self.bar.setRange(0, int(total))
            self.bar.setFormat("%v of %m epochs")
        if self._total_epochs:
            self.bar.setValue(int(done))

    def set_progress(self, elapsed: float) -> None:
        line = training_logic.progress_text(
            self.bar.value(), self._total_epochs, elapsed)
        if self._gpu_gb:
            # An out-of-memory at epoch 40 costs forty epochs, and it is
            # visible climbing long before it happens.
            line += f"  \u00b7  {self._gpu_gb:.2f} GB on the GPU"
        self.progress_label.setText(line)

    def set_metrics(self, rows: list[dict], patience: int = 0) -> None:
        if hasattr(self.chart, "set_data"):
            self.chart.set_data(training_logic.metric_series(rows, "epoch"),
                                training_logic.chart_series(rows))
        self.best_label.setText(training_logic.stall_note(rows, patience))
        # results.csv is the authority on epochs completed: stdout can be
        # buffered away for a while, and a bar that stalls while the curves
        # move is a bar nobody believes again.
        done = training_logic.summarize_results(rows).get("epochs", 0)
        if done and self._total_epochs:
            self.bar.setValue(min(int(done), self._total_epochs))

    # Ultralytics writes both; the normalised one reads better on a dataset
    # whose classes have very different image counts, which is every dataset
    # here while it is being collected.
    MATRIX_NAMES = ("confusion_matrix_normalized.png", "confusion_matrix.png")

    def show_confusion_matrix(self, run_dir: Path | None) -> bool:
        """Show the matrix the finished run wrote, if it wrote one."""
        for name in self.MATRIX_NAMES:
            path = Path(run_dir) / name if run_dir else None
            if path is None or not path.is_file():
                continue
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            self.matrix_label.setPixmap(pixmap)
            self.matrix_box.setTitle(f"Confusion matrix -- {name}")
            self.matrix_box.setVisible(True)
            return True
        return False

    def finish(self, headline: str, detail: str, weights: Path | None,
               failed: bool = False, run_dir: Path | None = None) -> None:
        if failed:
            # The one case where the output IS the answer: a run that dies
            # before its first epoch has no progress, no curves and no metrics.
            self.log_box.setChecked(True)
        self.what_label.setText(headline)
        self.progress_label.setText(detail)
        if self._total_epochs:
            self.bar.setValue(self.bar.maximum())
        else:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        self.show_confusion_matrix(run_dir)
        self.weights = weights if (weights and Path(weights).is_file()) else None
        if self.weights is None and weights:
            # Said rather than left as a disabled button nobody can explain.
            self.detail_label.setText(
                f"No weights at {weights} -- the run did not get far enough to "
                f"write any, or it wrote them somewhere else.")


class TrainingMonitor(QDialog):
    """Follows a training run, or both stages of one. Reused across runs."""

    def __init__(self, chart_factory, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training")
        # Not modal: a run takes minutes to hours, and nothing about labelling
        # has to wait for it.
        self.setModal(False)
        self.setMinimumSize(720, 660)

        self._chart_factory = chart_factory
        self._panels: dict[str, StagePanel] = {}
        self._stage = ""
        self._on_use = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.plan_label = QLabel("")
        self.plan_label.setWordWrap(True)
        self.plan_label.setStyleSheet("color: #9aa4b2;")
        self.plan_label.setVisible(False)
        layout.addWidget(self.plan_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip(
            "Ends the run. Ultralytics writes weights every epoch, so what has "
            "been reached so far is already on disk.")
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)
        self.use_btn = QPushButton("Use This Model")
        self.use_btn.setEnabled(False)
        self.use_btn.setToolTip(
            "Put a finished run's weights into the field the rest of the "
            "application reads -- Test Models for a detector, which is also "
            "where Live Detect gets its detector, and the stage 2 field for a "
            "classifier.\n\n"
            "With both stages finished it takes both at once: a two-stage "
            "pipeline is not usable as one half of itself. With one, it takes "
            "the open tab's.\n\n"
            "It is best.pt, not last.pt: the epoch that validated best rather "
            "than the one that happened to be last.")
        self.use_btn.clicked.connect(self._use_clicked)
        buttons.addWidget(self.use_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        # Which tab is open decides which weights Use This Model offers, so it
        # has to follow the tab rather than the run.
        self.tabs.currentChanged.connect(lambda _i: self._refresh_use_button())

    # --- panels ------------------------------------------------------------

    def panel(self, stage: str = "") -> StagePanel | None:
        """The named stage's panel, or the one currently open."""
        if stage:
            return self._panels.get(stage)
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, StagePanel) else None

    @property
    def current(self) -> StagePanel | None:
        """The panel the RUN is writing to, which is not always the open tab."""
        return self._panels.get(self._stage)

    def _stage_panel(self, stage: str) -> StagePanel:
        panel = self._panels.get(stage)
        if panel is None:
            panel = StagePanel(self._chart_factory(), self)
            self._panels[stage] = panel
            self.tabs.addTab(panel, stage.capitalize())
        return panel

    # --- starting ----------------------------------------------------------

    def begin(self, stage: str, model: str, data: str, fresh: bool = True,
              plan: str = "") -> None:
        """Start a stage. ``fresh`` clears the other stages with it.

        A queued second stage must NOT be fresh: the classifier continues the
        detector's work, and wiping the tab that recorded it is exactly what
        the queue's ``fresh=False`` exists to prevent. The detector's curves
        are also the first thing anybody wants to look at while the classifier
        runs, which is when a single shared panel has just discarded them.
        """
        if fresh:
            self.tabs.clear()
            self._panels = {}
        self._stage = stage
        panel = self._stage_panel(stage)
        panel.begin(stage, model, data)
        self.tabs.setCurrentWidget(panel)
        self.plan_label.setText(plan)
        self.plan_label.setVisible(bool(plan))
        self.stop_btn.setEnabled(True)
        self._refresh_use_button()

    # --- while it runs -----------------------------------------------------

    def append_output(self, text: str) -> None:
        panel = self.current
        if panel is not None:
            panel.append_output(text)

    def clear_output(self) -> None:
        panel = self.current
        if panel is not None:
            panel.log.clear()

    def set_progress(self, elapsed: float) -> None:
        panel = self.current
        if panel is not None:
            panel.set_progress(elapsed)

    def set_metrics(self, rows: list[dict], patience: int = 0) -> None:
        panel = self.current
        if panel is not None:
            panel.set_metrics(rows, patience)

    # --- finishing ---------------------------------------------------------

    def finish(self, headline: str, detail: str, weights: Path | None,
               failed: bool = False, run_dir: Path | None = None) -> None:
        self.stop_btn.setEnabled(False)
        panel = self.current
        if panel is not None:
            panel.finish(headline, detail, weights, failed, run_dir)
        self._refresh_use_button()

    def finished_stages(self) -> list[tuple[str, Path]]:
        """Every stage that produced weights, in the order they were run."""
        return [(name, panel.weights) for name, panel in self._panels.items()
                if panel.weights is not None]

    def _refresh_use_button(self) -> None:
        """One button, two meanings, and the text says which.

        With both stages finished it takes both -- a two-stage pipeline is not
        usable as one half, so making somebody visit each tab to apply them
        separately is two clicks to express one decision. Until then it is the
        open tab's, because that is the only one there is.
        """
        done = self.finished_stages()
        if len(done) > 1:
            self.use_btn.setText("Use Both Models")
            self.use_btn.setEnabled(True)
            return
        self.use_btn.setText("Use This Model")
        panel = self.panel()
        self.use_btn.setEnabled(panel is not None and panel.weights is not None)

    def _use_clicked(self) -> None:
        if self._on_use is None:
            return
        done = self.finished_stages()
        if len(done) > 1:
            self._on_use([(str(path), name) for name, path in done])
            return
        panel = self.panel()
        if panel is None or panel.weights is None:
            return
        stage = next((name for name, p in self._panels.items() if p is panel), "")
        self._on_use([(str(panel.weights), stage)])

    def set_use_handler(self, handler) -> None:
        self._on_use = handler

    def show_run(self) -> None:
        """Bring it up, or back up if it was pushed behind the main window."""
        self.show()
        self.raise_()
        self.activateWindow()
