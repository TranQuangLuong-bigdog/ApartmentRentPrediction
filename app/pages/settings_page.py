"""Settings page for PySide6 GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.pages._base_page import BasePage


class SettingsPage(BasePage):
    """Basic settings placeholder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(QLabel("No PySide6 settings implemented yet."))
        layout.addWidget(QLabel("(Next step: add model path, logging level, etc.)"))

