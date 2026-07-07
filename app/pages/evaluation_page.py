"""Evaluation page for PySide6 GUI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from app.pages._base_page import BasePage
from src.config.config import get_project_paths


class EvaluationPage(BasePage):
    """Load and display metrics from latest evaluation outputs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._paths = get_project_paths()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Evaluation")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        btn = QPushButton("Load metrics")
        btn.clicked.connect(self._load_metrics)
        layout.addWidget(btn)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self._table)

        self._status = QLabel("")
        layout.addWidget(self._status)

        self._load_metrics()

    def _load_metrics(self) -> None:
        try:
            metrics_txt = self._paths.output_reports / "metrics.txt"
            if not metrics_txt.exists():
                self._status.setText("No metrics found. Run training/evaluation first.")
                return

            # parse metrics.txt lines: key: value
            lines = [ln.strip() for ln in metrics_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            items = []
            for ln in lines:
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                items.append((k.strip(), v.strip()))

            self._table.setRowCount(len(items))
            for r, (k, v) in enumerate(items):
                self._table.setItem(r, 0, QTableWidgetItem(k))
                self._table.setItem(r, 1, QTableWidgetItem(v))

            self._status.setText(f"Loaded {len(items)} metrics from {metrics_txt}")
        except Exception as exc:
            self._status.setText(f"Error: {exc}")

