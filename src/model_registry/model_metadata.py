"""Model metadata schema cho Model Registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelMetrics:
    """Các metric dùng để so sánh best model."""

    r2: float
    rmse: float
    mae: float
    mse: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mse": self.mse,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelMetrics":
        return ModelMetrics(
            r2=float(d.get("r2")),
            rmse=float(d.get("rmse")),
            mae=float(d.get("mae")),
            mse=float(d["mse"]) if d.get("mse") is not None else None,
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata của 1 version mô hình."""

    version: str
    model_file: str

    created_at: str

    training_time_seconds: float

    dataset_hash: str
    dataset_name: str
    rows: int
    columns: int
    feature_count: int

    pipeline_version: str
    preprocessing_version: str
    tensorflow_version: str
    python_version: str

    feature_schema: Dict[str, Any]
    metrics: ModelMetrics

    is_best: bool

    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_file": self.model_file,
            "created_at": self.created_at,
            "training_time": self.training_time_seconds,
            "dataset_hash": self.dataset_hash,
            "dataset_name": self.dataset_name,
            "rows": self.rows,
            "columns": self.columns,
            "feature_count": self.feature_count,
            "pipeline_version": self.pipeline_version,
            "preprocessing_version": self.preprocessing_version,
            "tensorflow_version": self.tensorflow_version,
            "python_version": self.python_version,
            "feature_schema": self.feature_schema,
            "metrics": self.metrics.to_dict(),
            "is_best": self.is_best,
            "extra": self.extra or {},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelMetadata":
        return ModelMetadata(
            version=str(d["version"]),
            model_file=str(d["model_file"]),
            created_at=str(d["created_at"]),
            training_time_seconds=float(d["training_time"]),
            dataset_hash=str(d["dataset_hash"]),
            dataset_name=str(d["dataset_name"]),
            rows=int(d["rows"]),
            columns=int(d["columns"]),
            feature_count=int(d["feature_count"]),
            pipeline_version=str(d["pipeline_version"]),
            preprocessing_version=str(d["preprocessing_version"]),
            tensorflow_version=str(d["tensorflow_version"]),
            python_version=str(d["python_version"]),
            feature_schema=dict(d.get("feature_schema") or {}),
            metrics=ModelMetrics.from_dict(d["metrics"]),
            is_best=bool(d.get("is_best", False)),
            extra=dict(d.get("extra") or {}),
        )

