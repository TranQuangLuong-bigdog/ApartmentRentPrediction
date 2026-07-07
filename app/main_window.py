"""Main window for the PySide6 desktop GUI.

Chạy bằng:
    python app/app.py
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from app.sidebar import Sidebar
from app.pages.dataset_page import DatasetPage
from app.pages.evaluation_page import EvaluationPage
from app.pages.model_page import ModelPage
from app.pages.prediction_page import PredictionPage
from app.pages.settings_page import SettingsPage
from app.pages.Prediction_History_PySide6 import PredictionHistoryPage

from app.pages.training_page import TrainingPage


class MainWindow(QMainWindow):
    """Application main window with page router."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Apartment Rent Prediction (ANN) - Desktop")
        self.setMinimumSize(QSize(1100, 700))

        self._page_widgets: Dict[str, QWidget] = {}
        self._pages: Dict[str, Type[QWidget]] = {
            "Dataset": DatasetPage,
            "Training": TrainingPage,
            "Prediction": PredictionPage,
            "Evaluation": EvaluationPage,
            "Model": ModelPage,
            "Prediction History": PredictionHistoryPage,
            "Settings": SettingsPage,
        }


        # Sidebar auto builds from _pages keys
        self.sidebar = Sidebar(list(self._pages.keys()))

        self.sidebar.pageChanged.connect(self._on_page_changed)

        self._current_page: Optional[QWidget] = None

        self._root = QWidget(self)
        self.setCentralWidget(self._root)

        self._layout = QHBoxLayout(self._root)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._layout.addWidget(self.sidebar)

        # Content area where pages are injected
        self._content = QWidget(self._root)
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 12, 12, 12)
        self._layout.addWidget(self._content, stretch=1)

        self.sidebar.set_current("Dataset")
        self._on_page_changed("Dataset")

    def _on_page_changed(self, page_name: str) -> None:
        """Switch to selected page."""

        page_cls = self._pages.get(page_name)
        if page_cls is None:
            return

        if self._current_page is not None:
            self._current_page.setParent(None)
            self._current_page = None

        widget = self._page_widgets.get(page_name)
        if widget is None:
            widget = page_cls(parent=self._content)
            self._page_widgets[page_name] = widget

        self._content_layout.addWidget(widget)
        self._current_page = widget

