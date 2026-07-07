"""Dataset page for PySide6 GUI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


from src.config.config import get_project_paths

from app.pages._base_page import BasePage


class DatasetPage(BasePage):
    """Show dataset preview and basic stats."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._paths = get_project_paths()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Dataset")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self._row_limit = QSpinBox()
        self._row_limit.setRange(5, 500)
        self._row_limit.setValue(50)
        layout.addWidget(QLabel("Preview rows:"))
        layout.addWidget(self._row_limit)

        btn = QPushButton("Load preview")
        btn.clicked.connect(self._load_preview)
        layout.addWidget(btn)

        self._table = QTableWidget()
        self._table.setColumnCount(0)
        self._table.setRowCount(0)
        self._table.setMinimumHeight(400)
        layout.addWidget(self._table)

        self._status = QLabel("")
        layout.addWidget(self._status)

        # auto load
        self._load_preview()

    def _load_preview(self) -> None:
        try:
            csv_path = self._paths.root / "data" / "apartment_rent.csv"
            if not csv_path.exists():
                # fallback to existing file name
                csv_path = Path("data/apartment_rent.csv")

            df = pd.read_csv(csv_path)
            df = df.head(int(self._row_limit.value()))

            self._table.clear()
            self._table.setRowCount(df.shape[0])
            self._table.setColumnCount(df.shape[1])
            self._table.setHorizontalHeaderLabels(list(df.columns))

            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iat[r, c]))
                    self._table.setItem(r, c, item)

            self._status.setText(f"Loaded: {csv_path} | Preview shape: {df.shape}")
        except Exception as exc:
            self._status.setText(f"Error loading dataset: {exc}")

