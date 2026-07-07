"""Data contract cho kết quả dự đoán."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PredictionResult:

    """Kết quả dự đoán.

    notes:
    - prediction_model_name dùng cho UI badge (best/last).
    """

    y_pred: np.ndarray
    success: bool = True
    message: str = ""

    prediction_model_path: Optional[Path] = None
    prediction_model_name: Optional[str] = None
    inference_time_seconds: Optional[float] = None

    artifacts: Optional[Dict[str, Any]] = None
    prediction_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sang JSON-compatible dict."""
        return {
            "y_pred": self.y_pred.tolist(),
            "success": self.success,
            "message": self.message,
            "prediction_model_path": str(self.prediction_model_path) if self.prediction_model_path is not None else None,
            "prediction_model_name": self.prediction_model_name,
            "inference_time_seconds": self.inference_time_seconds,
            "artifacts": self.artifacts,
            "prediction_path": str(self.prediction_path) if self.prediction_path is not None else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PredictionResult":
        """Deserialize từ JSON-compatible dict."""
        return PredictionResult(
            y_pred=np.array(data.get("y_pred", []), dtype=float),
            success=bool(data.get("success", True)),
            message=str(data.get("message", "")),
            prediction_model_path=Path(data["prediction_model_path"]) if data.get("prediction_model_path") else None,
            prediction_model_name=data.get("prediction_model_name"),
            inference_time_seconds=data.get("inference_time_seconds"),
            artifacts=data.get("artifacts"),
            prediction_path=Path(data["prediction_path"]) if data.get("prediction_path") else None,
        )


