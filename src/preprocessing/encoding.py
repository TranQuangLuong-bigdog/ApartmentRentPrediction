"""Mã hoá (encoding) các biến phân loại.

Module này dùng One-Hot Encoding.

Yêu cầu:
- Không hard-code tên cột.
- Hàm trả về encoded DataFrame và object encoder để lưu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class EncodingConfig:
    """Cấu hình encoding."""

    handle_unknown: str = "ignore"
    sparse_output: bool = False  # sklearn>=1.2


def get_categorical_columns(df: pd.DataFrame, target_column: str) -> List[str]:
    """Lấy danh sách cột phân loại (loại trừ target)."""

    cat_cols = []
    for col in df.columns:
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols


def encode_one_hot(
    df: pd.DataFrame,
    target_column: str,
    config: EncodingConfig = EncodingConfig(),
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """Encode one-hot cho các cột phân loại.

    Args:
        df: DataFrame đầu vào.
        target_column: Tên cột mục tiêu.
        config: Cấu hình.

    Returns:
        (encoded_df, preprocessor)
    """

    if df.empty:
        raise ValueError("Input df is empty")

    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    cat_cols = get_categorical_columns(df, target_column)

    # ColumnTransformer: one-hot cho cat, pass-through cho còn lại
    ohe = OneHotEncoder(handle_unknown=config.handle_unknown, sparse_output=config.sparse_output)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
        ],
        remainder="passthrough",
    )

    # Fit_transform
    X_encoded = preprocessor.fit_transform(X)

    # Tạo cột tên
    # Lấy feature names từ OneHotEncoder
    ohe_feature_names = []
    if cat_cols:
        ohe_feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))

    # remainder là các cột còn lại theo thứ tự trong X
    remainder_cols = [c for c in X.columns if c not in cat_cols]
    feature_names = ohe_feature_names + remainder_cols

    encoded_df = pd.DataFrame(X_encoded, columns=feature_names, index=df.index)
    encoded_df[target_column] = y[target_column].values

    return encoded_df, preprocessor

