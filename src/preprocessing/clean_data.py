"""Làm sạch dữ liệu thô.

Chỉ thực hiện các bước làm sạch cơ bản như:
- Chuẩn hoá kiểu dữ liệu (nếu cần)
- Xử lý missing values ở mức cơ bản

Giai đoạn này chưa thực hiện encoding/scaling/feature engineering nâng cao.

Lưu ý:
- Module chỉ chịu trách nhiệm cho việc làm sạch.
- Logic chính pipeline nằm ở train/evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleanConfig:
    """Cấu hình làm sạch dữ liệu."""

    numeric_fill_strategy: str = "median"  # 'median' | 'mean' | 'zero'
    categorical_fill_strategy: str = "most_frequent"  # 'most_frequent' | 'missing'


def clean_data(df: pd.DataFrame, config: CleanConfig = CleanConfig()) -> pd.DataFrame:
    """Làm sạch DataFrame.

    Args:
        df: DataFrame đầu vào.
        config: Cấu hình chiến lược xử lý missing.

    Returns:
        DataFrame sau khi làm sạch.
    """

    if df.empty:
        return df

    out = df.copy()

    # Xử lý missing cho cột số
    numeric_cols = out.select_dtypes(include=["number"]).columns
    if config.numeric_fill_strategy == "median":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].median())
    elif config.numeric_fill_strategy == "mean":
        for col in numeric_cols:
            out[col] = out[col].fillna(out[col].mean())
    elif config.numeric_fill_strategy == "zero":
        for col in numeric_cols:
            out[col] = out[col].fillna(0)

    # Xử lý missing cho cột phân loại
    cat_cols = out.columns.difference(numeric_cols)
    if config.categorical_fill_strategy == "most_frequent":
        for col in cat_cols:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].mode(dropna=True).iloc[0])
    elif config.categorical_fill_strategy == "missing":
        for col in cat_cols:
            out[col] = out[col].fillna("missing")

    return out

