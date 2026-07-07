"""Chia train/test.

Module chỉ dùng sklearn train_test_split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    """Cấu hình chia dữ liệu."""

    test_size: float = 0.2
    random_state: int = 42


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    config: SplitConfig = SplitConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia dữ liệu thành train/test.

    Args:
        df: DataFrame đã chuẩn bị (đã encode & scale).
        target_column: Cột mục tiêu.
        config: Cấu hình.

    Returns:
        (X_train, X_test, y_train, y_test)
    """

    if df.empty:
        raise ValueError("Input df is empty")

    X = df.drop(columns=[target_column]).values
    y = df[[target_column]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test

