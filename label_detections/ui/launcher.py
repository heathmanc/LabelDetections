"""A minimal home window: the label library, the recipes, and the two wizards.

Deliberately small. The labeling canvas, camera capture and training tabs are
the next pieces of work; this exists so the wizards and the library can be
driven end to end today.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ..core import persistence, review, storage
from ..version import APP_TITLE
from . import wizards


class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(960, 620)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        location = QLabel(f"Library: {storage.DATA_DIR}")
        location.setStyleSheet("color: #9aa4b2;")
        root.addWidget(location)
        if storage.DATA_DIR_FALLBACK_REASON:
            warning = QLabel(storage.DATA_DIR_FALLBACK_REASON)
            warning.setStyleSheet("color: #fbbf24;")
            warning.setWordWrap(True)
            root.addWidget(warning)

        columns = QHBoxLayout()
        root.addLayout(columns, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel("Labels"))
        self.labels_list = QListWidget()
        left.addWidget(self.labels_list, 1)
        add_label_button = QPushButton("Add label...")
        add_label_button.clicked.connect(self._add_label)
        left.addWidget(add_label_button)
        columns.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Recipes"))
        self.recipes_list = QListWidget()
        self.recipes_list.itemDoubleClicked.connect(self._edit_recipe)
        right.addWidget(self.recipes_list, 1)
        new_recipe_button = QPushButton("New recipe...")
        new_recipe_button.clicked.connect(self._new_recipe)
        right.addWidget(new_recipe_button)
        columns.addLayout(right, 1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.refresh()

    def refresh(self) -> None:
        library = persistence.load_library()
        self.labels_list.clear()
        for label in library.all():
            statuses = list(persistence.dataset_statuses(label.label_id).values())
            ready = sum(1 for s in statuses if review.export_ready(s))
            self.labels_list.addItem(
                f"{label.label_id}  [{label.family}]  {ready}/{label.train_target} ready")

        self.recipes_list.clear()
        recipes = persistence.list_recipes()
        for recipe in recipes:
            missing = sorted(i for i in recipe.label_ids() if i not in library)
            suffix = f"  -- {len(missing)} label(s) not in the library" if missing else ""
            item = f"{recipe.safe_name}  ({len(recipe.views)} views){suffix}"
            self.recipes_list.addItem(item)
            self.recipes_list.item(self.recipes_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole, recipe.safe_name)

        self.status_label.setText(
            f"{len(library)} labels, {len(recipes)} recipes. "
            "Each label trains on its own dataset; a recipe only assembles labels "
            "that already exist."
        )

    def _add_label(self) -> None:
        if wizards.add_label(self):
            self.refresh()

    def _new_recipe(self) -> None:
        if wizards.new_recipe(self):
            self.refresh()

    def _edit_recipe(self, item) -> None:
        safe_name = item.data(Qt.ItemDataRole.UserRole)
        existing = next((r for r in persistence.list_recipes() if r.safe_name == safe_name), None)
        if existing is None:
            QMessageBox.warning(self, "Recipe", "That recipe could not be reloaded.")
            return
        if wizards.new_recipe(self, existing=existing):
            self.refresh()
