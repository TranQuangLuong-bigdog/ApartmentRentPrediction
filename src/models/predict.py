"""Dự đoán sử dụng model đã lưu.

Chưa bao gồm preprocessing; module chỉ dự đoán trên X đã sẵn sàng.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorflow import keras


def load_keras_model(model_path: Path) -> keras.Model:
    """Load model Keras từ file."""

    return keras.models.load_model(model_path)


def predict(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Tạo dự đoán.

    Args:
        model: keras.Model.
        X: ma trận đặc trưng (n_samples, n_features)

    Returns:
        y_pred: mảng dự đoán (n_samples,)
    """

    preds = model.predict(X, verbose=0)
    return preds.reshape(-1)


def save_predictions_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Lưu kết quả dự đoán ra CSV.

    Args:
        y_true: giá trị thực.
        y_pred: giá trị dự đoán.
        out_path: đường dẫn CSV.
    """

    create_df = pd.DataFrame(
        {
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    create_df.to_csv(out_path, index=False)

