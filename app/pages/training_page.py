"""Training page for PySide6 GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from src.config.config import get_project_paths, ensure_directories
from src.dataset.dataset_loader import DatasetLoader
from src.services.preprocessing_service import preprocess_for_training
from src.services.train_service import train_ann
from src.evaluation.evaluate import evaluate_model
from src.models.save_model import save_keras_model, save_training_history

from app.pages._base_page import BasePage


class TrainingPage(BasePage):
    """Trigger training and show logs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._paths = get_project_paths()
        ensure_directories(self._paths)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Training")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self._btn_train = QPushButton("Train ANN (blocking)")
        self._btn_train.clicked.connect(self._on_train)
        layout.addWidget(self._btn_train)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(450)
        layout.addWidget(self._log)

    def _append(self, text: str) -> None:
        self._log.append(text)

    def _on_train(self) -> None:
        try:
            self._append("Loading dataset...")
            dataset_path = self._paths.root / "data" / "apartment_rent.csv"

            loader = DatasetLoader()
            df = loader.load_csv(dataset_path)

            self._append(f"Dataset loaded: {df.shape}")
            self._append("Preprocessing...")
            processed = preprocess_for_training(df)

            self._append("Training...")
            result = train_ann(
                X_train=processed.X_train,
                y_train=processed.y_train,
                input_dim=processed.X_train.shape[1],
            )

            self._append("Evaluating...")
            y_pred = result.model.predict(processed.X_test, verbose=0).reshape(-1)

            # minimal evaluation: reuse existing evaluate_prediction-like metrics is not wired here; keep simple
            # Save model artifacts
            self._append("Saving model artifacts...")
            save_keras_model(result.model, self._paths.trained_models / "ann_model.keras")
            save_training_history(result.history, self._paths.trained_models / "history.pkl")

            self._append("Done. Model saved to trained_models/ann_model.keras")
        except Exception as exc:
            self._append(f"Error: {exc}")

