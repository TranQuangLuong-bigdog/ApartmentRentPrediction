"""Sidebar menu for the PySide6 desktop GUI."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QVBoxLayout, QWidget
from PySide6.QtWidgets import QPushButton


class Sidebar(QWidget):
    """Left navigation sidebar."""

    pageChanged: Signal = Signal(str)

    def __init__(self, pages: List[str], parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._pages = pages

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for idx, name in enumerate(self._pages):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, n=name: self.set_current(n))
            self._group.addButton(btn, idx)
            layout.addWidget(btn)

        layout.addStretch(1)

    def set_current(self, page_name: str) -> None:
        """Mark current page and emit signal."""
        for b in self._group.buttons():
            if b.text() == page_name:
                b.setChecked(True)
                break
        self.pageChanged.emit(page_name)

