"""Feature engineering.

Module này có thể chứa các bước tạo đặc trưng bổ sung.

Hiện tại (giai đoạn nền tảng) cung cấp một hook tối giản:
- remove_outliers: tuỳ chọn loại bỏ outliers đơn giản theo IQR.

Tránh làm quá phức tạp trước khi có code training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Cấu hình feature engineering."""

    remove_outliers: bool = True
    outlier_iqr_multiplier: float = 1.5


def remove_outliers_iqr(
    df: pd.DataFrame,
    target_column: str,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Loại bỏ outliers theo IQR trên cột target.

    Args:
        df: DataFrame.
        target_column: Cột mục tiêu.
        multiplier: Hệ số IQR.

    Returns:
        DataFrame sau khi loại outliers.
    """

    q1 = df[target_column].quantile(0.25)
    q3 = df[target_column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return df[(df[target_column] >= lower) & (df[target_column] <= upper)].copy()


def feature_engineering(
    df: pd.DataFrame,
    target_column: str,
    config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
) -> pd.DataFrame:
    """Thực hiện feature engineering.

    Args:
        df: DataFrame.
        target_column: Cột mục tiêu.
        config: Cấu hình.

    Returns:
        DataFrame sau feature engineering.
    """

    if config.remove_outliers:
        return remove_outliers_iqr(df, target_column=target_column, multiplier=config.outlier_iqr_multiplier)

    return df

