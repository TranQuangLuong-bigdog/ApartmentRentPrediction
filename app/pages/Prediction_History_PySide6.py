"""Prediction History page (PySide6).

This module replaces the Streamlit skeleton so that prediction history,
search/filter/delete/export are available inside the desktop app.

Notes:
- ANN pipeline remains unchanged; this is only UI + history presentation.
- History is stored by src/prediction/prediction_history.py as CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.pages._base_page import BasePage
from src.prediction.prediction_history import PredictionHistory


@dataclass(frozen=True)
class HistoryFilter:
    query_text: str = ""
    model_used: Optional[str] = None


class PredictionHistoryPage(BasePage):
    """Display prediction history and allow CRUD + export."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._history = PredictionHistory()
        self._rows: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Prediction History")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # Filter widgets
        layout.addWidget(QLabel("Tìm kiếm (session/model/preview):"))
        self._query = QLineEdit()
        self._query.setPlaceholderText("Nhập từ khóa...")
        layout.addWidget(self._query)

        layout.addWidget(QLabel("Lọc model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("(Tất cả)")
        self._model_combo.setEditable(False)
        layout.addWidget(self._model_combo)

        self._btn_refresh = QPushButton("Tải lại")
        self._btn_refresh.clicked.connect(self._load)
        layout.addWidget(self._btn_refresh)

        btn_row = QVBoxLayout()
        layout.addLayout(btn_row)

        self._btn_delete = QPushButton("Xóa theo Session ID")
        self._btn_delete.clicked.connect(self._delete_selected_session)
        layout.addWidget(self._btn_delete)

        self._session_id_input = QLineEdit()
        self._session_id_input.setPlaceholderText("Nhập session_id để xóa (từ bảng hoặc copy)")
        layout.addWidget(self._session_id_input)

        self._btn_export = QPushButton("Xuất CSV")
        self._btn_export.clicked.connect(self._export_csv)
        layout.addWidget(self._btn_export)

        # Table
        self._table = QTableWidget()
        self._table.setMinimumHeight(450)
        layout.addWidget(self._table)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._load()

    def _parse_history_predictions_preview(self, row: Dict[str, Any]) -> str:
        preds = row.get("predictions")
        # predictions currently stored as json string in csv loader
        if isinstance(preds, str):
            # Keep short preview only
            try:
                vals = pd.io.json.loads(preds)
                if isinstance(vals, list) and vals:
                    return f"{vals[0]:.4f} ... ({len(vals)} mẫu)"
            except Exception:
                pass
            return preds[:60]
        return str(preds)

    def _load(self) -> None:
        try:
            rows = self._history.load_history()
            self._rows = rows

            # Populate model combo
            models = sorted({r.get("model_used") for r in rows if r.get("model_used")})
            current = self._model_combo.currentText()
            self._model_combo.blockSignals(True)
            self._model_combo.clear()
            self._model_combo.addItem("(Tất cả)")
            for m in models:
                self._model_combo.addItem(str(m))
            # Restore selection if possible
            idx = self._model_combo.findText(current)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
            self._model_combo.blockSignals(False)

            self._apply_filters()
        except Exception as exc:
            self._status.setText(f"Error loading history: {exc}")

    def _apply_filters(self) -> None:
        query = self._query.text().strip().lower()
        selected = self._model_combo.currentText()
        model_used = None if selected == "(Tất cả)" else selected

        filtered = []
        for r in self._rows:
            if model_used and r.get("model_used") != model_used:
                continue
            if query:
                hay = " ".join(
                    [
                        str(r.get("session_id", "")),
                        str(r.get("model_used", "")),
                        str(r.get("prediction_time_iso", "")),
                        self._parse_history_predictions_preview(r),
                    ]
                ).lower()
                if query not in hay:
                    continue
            filtered.append(r)

        # Render table
        self._table.clear()
        headers = [
            "Thời gian",
            "Session ID",
            "Model",
            "Dataset",
            "Prediction Count",
            "Giá dự đoán (preview)",
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(filtered))

        for i, r in enumerate(filtered):
            self._table.setItem(i, 0, QTableWidgetItem(str(r.get("prediction_time_iso", ""))))
            self._table.setItem(i, 1, QTableWidgetItem(str(r.get("session_id", ""))))
            self._table.setItem(i, 2, QTableWidgetItem(str(r.get("model_used", ""))))
            self._table.setItem(i, 3, QTableWidgetItem(str(r.get("dataset_used", ""))))
            self._table.setItem(i, 4, QTableWidgetItem(str(r.get("prediction_count", ""))))
            self._table.setItem(i, 5, QTableWidgetItem(self._parse_history_predictions_preview(r)))

        self._status.setText(f"Loaded: {len(filtered)} / {len(self._rows)}")

    def _selected_session_id(self) -> Optional[str]:
        sel = self._table.selectedItems()
        if not sel:
            return None
        # Session ID is column 1
        # Find item in col=1 for first selected row
        for item in sel:
            row = item.row()
            col = item.column()
            if col == 1:
                return item.text().strip() or None
            # If not session col, still use row+1 column
            if col != 1 and self._table.item(row, 1) is not None:
                return self._table.item(row, 1).text().strip() or None
        return None

    def _delete_selected_session(self) -> None:
        session_id = self._session_id_input.text().strip() or self._selected_session_id()
        if not session_id:
            self._status.setText("Vui lòng chọn/nhập session_id để xóa.")
            return

        try:
            deleted = self._history.delete_history(session_id=session_id)
            self._status.setText(f"Đã xóa {deleted} dòng cho session_id={session_id}")
            self._session_id_input.clear()
            self._load()
        except Exception as exc:
            self._status.setText(f"Error delete history: {exc}")

    def _export_csv(self) -> None:
        try:
            # export currently filtered view
            query = self._query.text().strip().lower()
            selected = self._model_combo.currentText()
            model_used = None if selected == "(Tất cả)" else selected

            filtered = []
            for r in self._rows:
                if model_used and r.get("model_used") != model_used:
                    continue
                if query:
                    hay = " ".join(
                        [
                            str(r.get("session_id", "")),
                            str(r.get("model_used", "")),
                            str(r.get("prediction_time_iso", "")),
                            self._parse_history_predictions_preview(r),
                        ]
                    ).lower()
                    if query not in hay:
                        continue
                filtered.append(r)

            df = pd.DataFrame(filtered)
            if df.empty:
                self._status.setText("Không có dữ liệu để xuất.")
                return

            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", "prediction_history_export.csv", "CSV (*.csv)"
            )
            if not out_path:
                return

            df.to_csv(out_path, index=False, encoding="utf-8")
            self._status.setText(f"Đã xuất CSV: {out_path}")
        except Exception as exc:
            self._status.setText(f"Error export: {exc}")

