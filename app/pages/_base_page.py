"""Base page abstractions for PySide6 GUI pages."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QWidget


class BasePage(QWidget):
    """Base class for all pages in the PySide6 app."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)

    def page_title(self) -> str:
        """Return a human readable page title."""
        return self.__class__.__name__

