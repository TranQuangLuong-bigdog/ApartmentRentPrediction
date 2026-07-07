"""Service huấn luyện model ANN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from tensorflow import keras

from src.config.constants import BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.models.train_model import TrainConfig, train_ann_model




@dataclass(frozen=True)
class TrainServiceConfig:
    """Cấu hình service huấn luyện."""

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE


@dataclass(frozen=True)
class TrainServiceResult:
    """Kết quả huấn luyện."""

    model: keras.Model
    history: Any


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    cfg: TrainServiceConfig = TrainServiceConfig(),
) -> TrainServiceResult:
    """Huấn luyện ANN với validation split đơn giản."""

    # Chia validation từ đầu train (đơn giản, có thể thay bằng train_test_split cho đồng đều)
    val_ratio = 0.2
    val_size = int(len(X_train) * val_ratio)

    X_val = X_train[:val_size]
    y_val = y_train[:val_size]

    X_train2 = X_train[val_size:]
    y_train2 = y_train[val_size:]

    train_cfg = TrainConfig(
        input_dim=input_dim,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )

    model, history = train_ann_model(
        X_train=X_train2,
        y_train=y_train2,
        X_val=X_val,
        y_val=y_val,
        cfg=train_cfg,
    )

    return TrainServiceResult(model=model, history=history)

