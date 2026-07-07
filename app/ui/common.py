"""UI common helpers for PySide6 app."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QWidget


def set_wait_cursor(widget: QWidget, is_waiting: bool) -> None:
    """Toggle wait cursor for a widget."""
    if is_waiting:
        widget.setCursor(Qt.CursorShape.WaitCursor)
    else:
        widget.unsetCursor()


def show_error(parent: Optional[QWidget], message: str, title: str = "Error") -> None:
    """Show an error message box."""
    QMessageBox.critical(parent, title, message)


def set_title(label: QLabel, text: str) -> None:
    """Small helper for consistent title style."""
    label.setText(text)

