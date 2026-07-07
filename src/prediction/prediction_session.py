"""Prediction session.

Store session id, prediction time, model used, dataset used, count, execution time.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PredictionSession:
    """Prediction session information."""

    session_id: str
    prediction_time_iso: str
    model_used: str
    dataset_used: Optional[str]
    prediction_count: int
    execution_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prediction_time_iso": self.prediction_time_iso,
            "model_used": self.model_used,
            "dataset_used": self.dataset_used,
            "prediction_count": self.prediction_count,
            "execution_time_seconds": self.execution_time_seconds,
        }


def create_session(*, model_used: str, dataset_used: Optional[str], prediction_count: int, execution_time_seconds: float) -> PredictionSession:
    return PredictionSession(
        session_id=str(uuid.uuid4()),
        prediction_time_iso=datetime.utcnow().isoformat() + "Z",
        model_used=model_used,
        dataset_used=dataset_used,
        prediction_count=prediction_count,
        execution_time_seconds=execution_time_seconds,
    )

