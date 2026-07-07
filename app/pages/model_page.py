"""Model page for PySide6 GUI."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.pages._base_page import BasePage
from src.config.config import get_project_paths


class ModelPage(BasePage):
    """Show trained model registry and basic info."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._paths = get_project_paths()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Model Manager")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self._label = QLabel("")
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        btn = QPushButton("Load registry.json")
        btn.clicked.connect(self._load_registry)
        layout.addWidget(btn)

        self._load_registry()

    def _load_registry(self) -> None:
        try:
            reg_path = self._paths.trained_models / "registry.json"
            if not reg_path.exists():
                reg_path = Path("trained_models/registry.json")

            if not reg_path.exists():
                self._label.setText("No registry.json found.")
                return

            data = json.loads(reg_path.read_text(encoding="utf-8"))
            self._label.setText(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            self._label.setText(f"Error: {exc}")

