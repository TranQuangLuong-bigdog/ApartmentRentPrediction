"""Xây dựng mô hình ANN (Keras).

Giai đoạn 4: chỉ triển khai modules model/training/predict/save.
Chưa bao gồm preprocessing.

Lưu ý:
- Mô hình nhận đầu vào dạng số đặc trưng đã chuẩn hoá.
- Output là giá thuê (regression).
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class ANNConfig:
    """Cấu hình kiến trúc ANN."""

    input_dim: int
    learning_rate: float
    epochs: int = 100
    batch_size: int = 32
    model_name: str = "ann.keras"


def build_ann_model(cfg: ANNConfig) -> keras.Model:
    """Tạo mô hình ANN Keras.

    Args:
        cfg: ANNConfig.

    Returns:
        keras.Model: mô hình đã compile.
    """

    model = keras.Sequential(
        [
            layers.Input(shape=(cfg.input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name="ann_rent_predictor",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return model

