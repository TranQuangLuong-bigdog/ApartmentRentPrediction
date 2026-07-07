"""Dataset profiler.

Generate DatasetSummary dataclass.
Include:
- Rows
- Columns
- Missing
- Duplicates
- Memory Usage
- Categorical
- Numerical
- Feature Schema
- Target Column
- Dataset Hash

DatasetSummary dataclass hiện có trong src/contracts/dataset_summary.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.contracts.dataset_summary import DatasetSummary
from src.dataset.dataset_hash import DatasetHash
from src.utils.logger import get_logger


class DatasetProfiler:
    """Generate summary/profile from a dataframe."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)

    def profile(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        file_path: Optional[Path] = None,
    ) -> DatasetSummary:
        """Return DatasetSummary based on provided dataframe."""
        dataset_rows = int(df.shape[0])
        dataset_columns = int(df.shape[1])

        missing_value_counts = df.isna().sum().to_dict()

        categorical_cols = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)
        numerical_cols = [c for c in df.columns if c not in categorical_cols]

        # DatasetSummary contract hiện tại không có field duplicates/memory/hash/feature schema.
        # Vì PR要求 chỉ tạo dataset layer và không thay đổi pipeline, ta trả về đúng contract.
        return DatasetSummary(
            dataset_rows=dataset_rows,
            dataset_columns=dataset_columns,
            target_column=target_column,
            feature_count=int(df.shape[1]),
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols,
            missing_value_counts=missing_value_counts,
        )

    def profile_extended(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        file_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Profile đầy đủ để exporter ghi ra file dataset_profile.json."""

        memory_usage = int(df.memory_usage(deep=True).sum())
        duplicates = int(df.duplicated().sum())

        categorical_cols = list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)
        numerical_cols = [c for c in df.columns if c not in categorical_cols]

        feature_schema = {str(col): str(df[col].dtype) for col in df.columns}

        dataset_hash = DatasetHash.compute_all(file_path) if file_path else {"sha256": None, "md5": None}

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing": df.isna().sum().to_dict(),
            "duplicates": duplicates,
            "memory_usage_bytes": memory_usage,
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
            "feature_schema": feature_schema,
            "target_column": target_column,
            "dataset_hash": dataset_hash,
        }

