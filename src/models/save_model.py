"""Lưu model và history huấn luyện."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.utils.file_manager import create_folder


def save_keras_model(model: Any, path: Path) -> None:
    """Lưu model Keras ra file.

    Args:
        model: Model Keras.
        path: đường dẫn đích file .keras
    """

    create_folder(path.parent)
    model.save(path)


def save_training_history(history: Any, path: Path) -> None:
    """Lưu history huấn luyện dưới dạng pickle.

    Args:
        history: history từ model.fit.
        path: đường dẫn file pickle.
    """

    create_folder(path.parent)
    joblib.dump(history.history, path)

