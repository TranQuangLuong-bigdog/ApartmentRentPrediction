"""Dataset validation.

Validate:
- Empty dataset
- Missing target
- Duplicate rows
- Null values
- Invalid data types
- Unsupported encoding
- Invalid column names

Return DatasetValidationResult.

Chỉ tạo Dataset Layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger


@dataclass(frozen=True)
class DatasetValidationResult:
    """Kết quả validate dataset."""

    success: bool
    errors: List[str]
    warnings: List[str]


class DatasetValidator:
    """Validate dataset theo một số rule phổ biến."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)

    def validate_empty_dataset(self, df: pd.DataFrame) -> List[str]:
        if df.empty:
            return ["Dataset is empty"]
        return []

    def validate_missing_target(self, df: pd.DataFrame, target_column: str) -> List[str]:
        if target_column not in df.columns:
            return [f"Missing target column: {target_column}"]
        return []

    def validate_duplicate_rows(self, df: pd.DataFrame) -> List[str]:
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            return [f"Dataset contains duplicate rows: {dup_count}"]
        return []

    def validate_null_values(self, df: pd.DataFrame) -> List[str]:
        null_count = int(df.isna().sum().sum())
        if null_count > 0:
            return [f"Dataset contains null values: {null_count}"]
        return []

    def validate_invalid_column_names(self, df: pd.DataFrame) -> List[str]:
        errors: List[str] = []
        for c in df.columns:
            if c is None:
                errors.append("Found None column name")
                continue
            name = str(c).strip()
            if not name:
                errors.append("Found empty column name")
        return errors

    def validate_invalid_data_types(
        self, df: pd.DataFrame, allowed_dtypes: Optional[List[str]] = None
    ) -> List[str]:
        """Validate đơn giản theo kiểu dữ liệu."""
        allowed_dtypes = allowed_dtypes or ["number", "object", "bool"]
        errors: List[str] = []

        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                kind = "number"
            elif pd.api.types.is_bool_dtype(s):
                kind = "bool"
            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                kind = "object"
            else:
                kind = str(s.dtype)

            if kind not in allowed_dtypes:
                errors.append(f"Invalid dtype for column {col}: {s.dtype}")
        return errors

    def validate_unsupported_encoding(self, encoding: Optional[str]) -> List[str]:
        if encoding is None:
            return []
        supported = {"utf-8-sig", "utf-8", "cp1252", "latin1", "utf-16"}
        if encoding not in supported:
            return [f"Unsupported encoding: {encoding}"]
        return []

    def validate(
        self,
        df: pd.DataFrame,
        target_column: str,
        *,
        file_path: Optional[Path] = None,
        encoding: Optional[str] = None,
        allowed_dtypes: Optional[List[str]] = None,
    ) -> DatasetValidationResult:
        """Run all validations."""
        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(self.validate_empty_dataset(df))
        errors.extend(self.validate_invalid_column_names(df))
        errors.extend(self.validate_missing_target(df, target_column))
        errors.extend(self.validate_duplicate_rows(df))
        errors.extend(self.validate_null_values(df))
        errors.extend(self.validate_invalid_data_types(df, allowed_dtypes=allowed_dtypes))
        errors.extend(self.validate_unsupported_encoding(encoding))

        success = len(errors) == 0
        if success:
            self.logger.info("Dataset validation passed")
        else:
            self.logger.warning("Dataset validation failed: %d errors", len(errors))

        return DatasetValidationResult(success=success, errors=errors, warnings=warnings)

