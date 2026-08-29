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

Modeless on purpose. A run takes minutes to hours and there is no reason to
stop labelling while it happens; the window can be pushed aside and comes back
with the run's state, not a fresh one.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QDialog, QGroupBox, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QTextEdit,
                               QVBoxLayout, QWidget)

from ..core import training as training_logic


class TrainingMonitor(QDialog):
    """Follows one training run. Reused across runs rather than rebuilt.

    Held by the main window for its whole life so the log and the chart are
    always real widgets -- a monitor created per run would leave every caller
    that appends a line guarding against its absence.
    """

    def __init__(self, chart: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training")
        # Not modal: a run takes minutes to hours, and nothing about labelling
        # has to wait for it.
        self.setModal(False)
        self.setMinimumSize(720, 620)

        self._weights: Path | None = None
        self._on_use = None
        self._total_epochs = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
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

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Training output appears here.")
        self.log_box = QGroupBox("Output")
        self.log_box.setCheckable(True)
        self.log_box.setChecked(False)
        self.log_box.setToolTip(
            "Everything the yolo command printed. Folded away because it is a "
            "wall, and opened when a run fails to start -- which is the one "
            "thing only this can explain.")
        log_layout = QVBoxLayout(self.log_box)
        log_layout.setContentsMargins(8, 4, 8, 8)
        log_layout.addWidget(self.log)
        self.log.setVisible(False)
        self.log_box.toggled.connect(self.log.setVisible)
        layout.addWidget(self.log_box)

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
            "Put the weights this run produced into Test Models, which is also "
            "where Live Detect reads its detector from.\n\n"
            "It is best.pt, not last.pt: the epoch that validated best rather "
            "than the one that happened to be last.")
        self.use_btn.clicked.connect(self._use_clicked)
        buttons.addWidget(self.use_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    # --- starting ----------------------------------------------------------

    def begin(self, stage: str, model: str, data: str) -> None:
        """Reset for a new run and say what it is."""
        self._weights = None
        self._total_epochs = 0
        self.what_label.setText(f"Training the {stage}")
        self.detail_label.setText(f"{model}\n{data}" if data else str(model))
        self.bar.setRange(0, 0)
        self.bar.setFormat("")
        self.progress_label.setText("Starting -- the first epoch also warms up "
                                    "and caches, so it is the slow one.")
        self.best_label.setText("")
        self.log.clear()
        self.use_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    # --- while it runs -----------------------------------------------------

    def append_output(self, text: str) -> None:
        """Add a chunk of the process's stdout, and read the epoch off it."""
        if not text:
            return
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
        found = training_logic.parse_epoch(text)
        if found:
            self._set_epoch(*found)

    def _set_epoch(self, done: int, total: int) -> None:
        if total and total != self._total_epochs:
            self._total_epochs = int(total)
            self.bar.setRange(0, int(total))
            self.bar.setFormat("%v of %m epochs")
        if self._total_epochs:
            self.bar.setValue(int(done))

    def set_progress(self, elapsed: float) -> None:
        """Refresh the elapsed/remaining line. Driven by the caller's timer."""
        self.progress_label.setText(training_logic.progress_text(
            self.bar.value(), self._total_epochs, elapsed))

    def set_metrics(self, rows: list[dict], patience: int = 0) -> None:
        """Hand over the parsed results.csv: the curves and the best epoch."""
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

    # --- finishing ---------------------------------------------------------

    def finish(self, headline: str, detail: str, weights: Path | None) -> None:
        self.stop_btn.setEnabled(False)
        self.what_label.setText(headline)
        self.progress_label.setText(detail)
        if self._total_epochs:
            self.bar.setValue(self.bar.maximum())
        else:
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
        self._weights = weights if (weights and Path(weights).is_file()) else None
        self.use_btn.setEnabled(self._weights is not None)
        if self._weights is None and weights:
            # Said rather than left as a disabled button nobody can explain.
            self.detail_label.setText(
                f"No weights at {weights} -- the run did not get far enough to "
                f"write any, or it wrote them somewhere else.")

    def _use_clicked(self) -> None:
        if self._weights is not None and self._on_use is not None:
            self._on_use(str(self._weights))

    def set_use_handler(self, handler) -> None:
        self._on_use = handler

    def show_run(self) -> None:
        """Bring it up, or back up if it was pushed behind the main window."""
        self.show()
        self.raise_()
        self.activateWindow()
