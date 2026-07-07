"""Chuẩn hoá (scaling) dữ liệu.

Dùng StandardScaler cho các đặc trưng số.

Module trả về scaler để có thể lưu và dùng cho inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScalingConfig:
    """Cấu hình scaling."""

    with_mean: bool = True
    with_std: bool = True


def scale_features(
    df: pd.DataFrame,
    target_column: str,
    config: ScalingConfig = ScalingConfig(),
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale các cột đặc trưng.

    Args:
        df: DataFrame đã encode (có thể gồm số & dummy variables).
        target_column: Tên cột mục tiêu.
        config: Cấu hình.

    Returns:
        (df_scaled, scaler)
    """

    if df.empty:
        raise ValueError("Input df is empty")

    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    scaler = StandardScaler(with_mean=config.with_mean, with_std=config.with_std)
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    scaled_df[target_column] = y[target_column].values

    return scaled_df, scaler

