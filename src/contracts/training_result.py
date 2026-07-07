"""Data contract cho kết quả huấn luyện.

Chỉ chứa dataclass/TypedDict để truyền dữ liệu giữa các tầng.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingResult:
    """Kết quả trả về sau khi train."""

    success: bool
    message: str

    model_path: Optional[Path] = None
    metrics: Optional[Dict[str, float]] = None
    history: Optional[Any] = None

    report_dir: Optional[Path] = None
    figure_dir: Optional[Path] = None

    training_time_seconds: Optional[float] = None
    artifacts: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sang JSON-compatible dict."""
        return {
            "success": self.success,
            "message": self.message,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "metrics": self.metrics,
            "history": self.history,
            "report_dir": str(self.report_dir) if self.report_dir is not None else None,
            "figure_dir": str(self.figure_dir) if self.figure_dir is not None else None,
            "training_time_seconds": self.training_time_seconds,
            "artifacts": self.artifacts,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingResult":
        """Deserialize từ JSON-compatible dict."""
        return TrainingResult(
            success=bool(data.get("success")),
            message=str(data.get("message")),
            model_path=Path(data["model_path"]) if data.get("model_path") else None,
            metrics=data.get("metrics"),
            history=data.get("history"),
            report_dir=Path(data["report_dir"]) if data.get("report_dir") else None,
            figure_dir=Path(data["figure_dir"]) if data.get("figure_dir") else None,
            training_time_seconds=data.get("training_time_seconds"),
            artifacts=data.get("artifacts"),
        )


