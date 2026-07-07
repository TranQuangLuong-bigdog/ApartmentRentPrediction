"""Service dự đoán từ model và dữ liệu đã chuẩn bị."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tensorflow import keras

from src.models.predict import predict


@dataclass(frozen=True)
class PredictionResult:
    """Kết quả dự đoán."""

    y_pred: np.ndarray


def predict_with_model(model: keras.Model, X: np.ndarray) -> PredictionResult:
    """Dự đoán y_hat."""

    y_pred = predict(model=model, X=X)
    return PredictionResult(y_pred=y_pred)

