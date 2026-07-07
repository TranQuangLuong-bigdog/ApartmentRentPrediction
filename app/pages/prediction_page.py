"""Prediction page for PySide6 GUI."""

from __future__ import annotations

from typing import List

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from app.pages._base_page import BasePage
from src.prediction.prediction_service import PredictionService


class PredictionPage(BasePage):
    """Single prediction UI.

    Note: This UI expects the user to input numeric features in the same order
    as the model's training feature_columns.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Prediction")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(QLabel("Input features (comma separated floats):"))
        self._input = QLineEdit()
        self._input.setPlaceholderText("e.g. 50,2,1, ...")
        layout.addWidget(self._input)

        btn = QPushButton("Predict")
        btn.clicked.connect(self._predict)
        layout.addWidget(btn)

        self._output = QLabel("")
        self._output.setWordWrap(True)
        layout.addWidget(self._output)

        self._service = PredictionService()

    def _predict(self) -> None:
        try:
            raw = self._input.text().strip()
            if not raw:
                self._output.setText("Please enter input features.")
                return

            parts: List[str] = [p.strip() for p in raw.split(",") if p.strip()]
            x = [float(v) for v in parts]

            res = self._service.predict_single(x)
            y = float(res.y_pred[0]) if res.y_pred.size else None
            self._output.setText(f"Prediction: {y}\nModel: {res.prediction_model_name}\n{res.message}")
        except Exception as exc:
            self._output.setText(f"Error: {exc}")

