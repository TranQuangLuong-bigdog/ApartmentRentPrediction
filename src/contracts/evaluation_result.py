"""Data contract cho kết quả evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:


    """Kết quả evaluation trên tập test."""

    metrics: Dict[str, float]
    residuals: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray

    report_dir: Optional[Path] = None
    figure_dir: Optional[Path] = None

    model_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sang JSON-compatible dict."""
        return {
            "metrics": self.metrics,
            "residuals": self.residuals.tolist(),
            "y_true": self.y_true.tolist(),
            "y_pred": self.y_pred.tolist(),
            "report_dir": str(self.report_dir) if self.report_dir is not None else None,
            "figure_dir": str(self.figure_dir) if self.figure_dir is not None else None,
            "model_metadata": self.model_metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EvaluationResult":
        """Deserialize từ JSON-compatible dict."""
        return EvaluationResult(
            metrics=data.get("metrics", {}),
            residuals=np.array(data.get("residuals", []), dtype=float),
            y_true=np.array(data.get("y_true", []), dtype=float),
            y_pred=np.array(data.get("y_pred", []), dtype=float),
            report_dir=Path(data["report_dir"]) if data.get("report_dir") else None,
            figure_dir=Path(data["figure_dir"]) if data.get("figure_dir") else None,
            model_metadata=data.get("model_metadata"),
        )


