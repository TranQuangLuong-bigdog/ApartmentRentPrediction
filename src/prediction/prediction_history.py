"""PredictionHistory.

Automatically save every prediction.
Support:
- append_prediction()
- load_history()
- delete_history()
- clear_history()
- search()
- statistics()
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger


def get_logger(name: str):
    return setup_logger(name=name, log_dir=Path("logs"))


@dataclass(frozen=True)
class PredictionHistoryRow:
    session_id: str
    prediction_time_iso: str
    model_used: str
    dataset_used: Optional[str]
    prediction_count: int
    predictions: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prediction_time_iso": self.prediction_time_iso,
            "model_used": self.model_used,
            "dataset_used": self.dataset_used,
            "prediction_count": self.prediction_count,
            "predictions": self.predictions,
        }


class PredictionHistory:
    """Manage prediction history persisted on disk."""

    def __init__(self, history_csv: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)
        self.history_csv = history_csv or Path("output") / "prediction" / "prediction_history.csv"

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_csv.exists():
            return []
        with self.history_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    def append_prediction(
        self,
        *,
        session_id: str,
        prediction_time_iso: str,
        model_used: str,
        dataset_used: Optional[str],
        predictions: List[float],
    ) -> Path:
        self.history_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["session_id", "prediction_time_iso", "model_used", "dataset_used", "prediction_count", "predictions"]

        is_new = not self.history_csv.exists()
        with self.history_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    "session_id": session_id,
                    "prediction_time_iso": prediction_time_iso,
                    "model_used": model_used,
                    "dataset_used": dataset_used,
                    "prediction_count": len(predictions),
                    "predictions": json.dumps(predictions, ensure_ascii=False),
                }
            )

        self.logger.info("Appended prediction history row to %s", self.history_csv)
        return self.history_csv

    def delete_history(self, *, session_id: str) -> int:
        rows = self.load_history()
        kept = [r for r in rows if r.get("session_id") != session_id]
        deleted = len(rows) - len(kept)
        if deleted <= 0:
            return 0

        if self.history_csv.exists():
            self.history_csv.unlink()
        for r in kept:
            self.append_prediction(
                session_id=r["session_id"],
                prediction_time_iso=r["prediction_time_iso"],
                model_used=r["model_used"],
                dataset_used=r.get("dataset_used") or None,
                predictions=json.loads(r["predictions"]),
            )
        return deleted

    def clear_history(self) -> None:
        if self.history_csv.exists():
            self.history_csv.unlink()

    def search(self, *, model_used: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.load_history()
        if model_used:
            rows = [r for r in rows if r.get("model_used") == model_used]
        return rows

    def statistics(self) -> Dict[str, Any]:
        rows = self.load_history()
        counts = sum(int(r.get("prediction_count", 0)) for r in rows)
        return {"history_rows": len(rows), "prediction_count_total": counts}

