"""Data contract cho thông tin mô hình."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelInfo:
    """Metadata cấp cao về mô hình."""

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    training_time_seconds: Optional[float] = None
    model_size_mb: Optional[float] = None

    tensorflow_version: Optional[str] = None

    dataset_name: Optional[str] = None
    dataset_hash: Optional[str] = None

    preprocessing_version: Optional[str] = None
    feature_schema_version: Optional[str] = None

    pipeline_version: Optional[str] = None
    target_column: Optional[str] = None

    best: Optional[bool] = None

    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sang JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_time_seconds": self.training_time_seconds,
            "model_size_mb": self.model_size_mb,
            "tensorflow_version": self.tensorflow_version,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "preprocessing_version": self.preprocessing_version,
            "feature_schema_version": self.feature_schema_version,
            "pipeline_version": self.pipeline_version,
            "target_column": self.target_column,
            "best": self.best,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelInfo":
        """Deserialize từ JSON-compatible dict."""
        return ModelInfo(
            model_name=data.get("model_name"),
            model_version=data.get("model_version"),
            training_time_seconds=data.get("training_time_seconds"),
            model_size_mb=data.get("model_size_mb"),
            tensorflow_version=data.get("tensorflow_version"),
            dataset_name=data.get("dataset_name"),
            dataset_hash=data.get("dataset_hash"),
            preprocessing_version=data.get("preprocessing_version"),
            feature_schema_version=data.get("feature_schema_version"),
            pipeline_version=data.get("pipeline_version"),
            target_column=data.get("target_column"),
            best=data.get("best"),
            extra=data.get("extra"),
        )


