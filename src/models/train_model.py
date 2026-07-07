"""Huấn luyện ANN model và trả về model + history.

Giai đoạn 4: module train_model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from tensorflow import keras

from src.models.ann_model import ANNConfig, build_ann_model


@dataclass(frozen=True)
class TrainConfig:
    """Cấu hình huấn luyện."""

    input_dim: int
    learning_rate: float
    epochs: int
    batch_size: int


def train_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[keras.Model, Any]:
    """Huấn luyện mô hình ANN.

    Args:
        X_train: đặc trưng train.
        y_train: nhãn train (n_samples, 1) hoặc (n_samples,)
        X_val: đặc trưng validation.
        y_val: nhãn validation.
        cfg: cấu hình huấn luyện.

    Returns:
        (model, history)
    """

    ann_cfg = ANNConfig(
        input_dim=cfg.input_dim,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )

    model = build_ann_model(ann_cfg)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
    )

    return model, history

