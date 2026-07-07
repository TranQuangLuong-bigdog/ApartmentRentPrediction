"""Service tiền xử lý dữ liệu (clean -> feature engineering -> encoding -> scaling -> split)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.config.constants import TEST_SIZE, TARGET_COLUMN, RANDOM_STATE
from src.exceptions.custom_exception import ValidationError
from src.preprocessing.clean_data import CleanConfig, clean_data
from src.preprocessing.encoding import EncodingConfig, encode_one_hot
from src.preprocessing.feature_engineering import (
    FeatureEngineeringConfig,
    feature_engineering,
)
from src.preprocessing.scaling import ScalingConfig, scale_features
from src.preprocessing.split_data import SplitConfig, split_train_test


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Các đối tượng artifact phục vụ inference."""

    encoder: Any
    scaler: Any
    feature_columns: list[str]


@dataclass(frozen=True)
class PreprocessedData:
    """Đầu ra sau preprocessing."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    artifacts: PreprocessingArtifacts


def preprocess_for_training(df: pd.DataFrame) -> PreprocessedData:
    """Preprocess dataset để huấn luyện.

    Args:
        df: DataFrame raw.

    Returns:
        PreprocessedData.
    """

    if df is None or df.empty:
        raise ValidationError("Dataset rỗng.")

    df_clean = clean_data(df, CleanConfig())
    df_fe = feature_engineering(
        df_clean,
        target_column=TARGET_COLUMN,
        config=FeatureEngineeringConfig(),
    )

    df_encoded, encoder = encode_one_hot(
        df_fe,
        target_column=TARGET_COLUMN,
        config=EncodingConfig(),
    )

    df_scaled, scaler = scale_features(
        df_encoded,
        target_column=TARGET_COLUMN,
        config=ScalingConfig(),
    )

    X_train, X_test, y_train, y_test = split_train_test(
        df_scaled,
        target_column=TARGET_COLUMN,
        config=SplitConfig(test_size=TEST_SIZE, random_state=RANDOM_STATE),
    )

    feature_columns = list(df_scaled.drop(columns=[TARGET_COLUMN]).columns)

    artifacts = PreprocessingArtifacts(
        encoder=encoder,
        scaler=scaler,
        feature_columns=feature_columns,
    )

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        artifacts=artifacts,
    )

