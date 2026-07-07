"""Data contract cho tóm tắt dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional



@dataclass(frozen=True)
class DatasetSummary:
    """Tóm tắt dataset phục vụ reproducibility."""

    dataset_rows: Optional[int] = None
    dataset_columns: Optional[int] = None
    target_column: Optional[str] = None

    feature_count: Optional[int] = None
    categorical_columns: Optional[list[str]] = None
    numerical_columns: Optional[list[str]] = None

    missing_value_counts: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sang JSON-compatible dict."""
        return {
            "dataset_rows": self.dataset_rows,
            "dataset_columns": self.dataset_columns,
            "target_column": self.target_column,
            "feature_count": self.feature_count,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "missing_value_counts": self.missing_value_counts,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DatasetSummary":
        """Deserialize từ JSON-compatible dict."""
        return DatasetSummary(
            dataset_rows=data.get("dataset_rows"),
            dataset_columns=data.get("dataset_columns"),
            target_column=data.get("target_column"),
            feature_count=data.get("feature_count"),
            categorical_columns=data.get("categorical_columns"),
            numerical_columns=data.get("numerical_columns"),
            missing_value_counts=data.get("missing_value_counts"),
        )


