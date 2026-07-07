"""PredictionExporter.

Export outputs:
- CSV
- Excel
- JSON

Output structure:
output/prediction/
- prediction_latest.csv
- prediction_history.csv (from PredictionHistory)
- prediction_statistics.json
- prediction_session.json
- prediction_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import setup_logger


def get_logger(name: str):
    # fallback nhẹ cho module prediction
    return setup_logger(name=name, log_dir=Path("logs"))


class PredictionExporter:
    """Export prediction results to files."""

    def __init__(self, output_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)
        self.output_dir = Path(output_dir) if output_dir is not None else Path("output") / "prediction"

    def export_latest_csv(self, y_pred: List[float], feature_columns: Optional[List[str]] = None) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "prediction_latest.csv"
        df = pd.DataFrame({"y_pred": y_pred})
        df.to_csv(path, index=False)
        self.logger.info("Exported latest predictions to %s", path)
        return path

    def export_timestamp_csv(self, y_pred: List[float]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"prediction_{ts}.csv"
        df = pd.DataFrame({"y_pred": y_pred})
        df.to_csv(path, index=False)
        self.logger.info("Exported predictions to %s", path)
        return path

    def export_json(self, payload: Dict[str, Any], filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.info("Exported json to %s", path)
        return path

