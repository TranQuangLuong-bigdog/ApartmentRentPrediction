"""PredictionService.

Public API:
- predict_single()
- predict_batch()

Must return PredictionResult.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.contracts.prediction_result import PredictionResult
from src.exceptions.custom_exception import MLAppException

# Backward compatible alias
CustomException = MLAppException
from src.prediction.prediction_exporter import PredictionExporter
from src.prediction.prediction_history import PredictionHistory
from src.prediction.prediction_loader import PredictionLoader
from src.prediction.prediction_statistics import PredictionStatistics
from src.prediction.prediction_session import create_session
from src.prediction.prediction_validator import PredictionValidator
from src.utils.logger import get_logger


class PredictionService:
    """Main prediction service."""

    def __init__(
        self,
        *,
        loader: Optional[PredictionLoader] = None,
        validator: Optional[PredictionValidator] = None,
        exporter: Optional[PredictionExporter] = None,
        history: Optional[PredictionHistory] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or get_logger(__name__)
        self.loader = loader or PredictionLoader(logger=self.logger)
        self.validator = validator or PredictionValidator(logger=self.logger)
        self.exporter = exporter or PredictionExporter(logger=self.logger)
        self.history = history or PredictionHistory(logger=self.logger)

    def _ensure_2d(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def predict_single(self, input_features: Sequence[float], *, dataset_used: Optional[str] = None) -> PredictionResult:
        return self.predict_batch([list(input_features)], dataset_used=dataset_used)

    def predict_batch(
        self,
        input_rows: List[List[float]] | np.ndarray,
        *,
        dataset_used: Optional[str] = None,
    ) -> PredictionResult:
        """Predict a batch and export results."""

        start = time.perf_counter()

        loaded = self.loader.load_model_and_metadata()
        model_used = loaded.model_used

        # Compatibility checks (minimal)
        # In this repo we don't yet wire full preprocessing artifacts here, so just validate files exist if present.
        if loaded.scaler_path is not None:
            pass

        X = self._ensure_2d(input_rows)

        # Validate scaler/encoder existence if paths exist (validator expects them)
        # If not present, treat as compatibility mismatch for stricter behavior.
        if loaded.scaler_path is None or loaded.encoder_path is None:
            raise CustomException("Scaler/Encoder artifacts missing", code="ARTIFACTS_MISSING")

        # Perform prediction
        self.logger.info("Start prediction: model_used=%s samples=%d", model_used, X.shape[0])
        y_pred = loaded.model.predict(X, verbose=0).reshape(-1)

        exec_time = time.perf_counter() - start

        y_pred_list = [float(v) for v in y_pred.tolist()]

        # Statistics
        stats_payload = PredictionStatistics.compute(y_pred_list)

        # Session
        session = create_session(
            model_used=model_used,
            dataset_used=dataset_used,
            prediction_count=len(y_pred_list),
            execution_time_seconds=exec_time,
        )

        # Export
        latest_csv_path = self.exporter.export_latest_csv(y_pred_list)
        ts_csv_path = self.exporter.export_timestamp_csv(y_pred_list)
        session_json_path = self.exporter.export_json(session.to_dict(), "prediction_session.json")
        stats_json_path = self.exporter.export_json(stats_payload, "prediction_statistics.json")

        # Append history
        self.history.append_prediction(
            session_id=session.session_id,
            prediction_time_iso=session.prediction_time_iso,
            model_used=session.model_used,
            dataset_used=session.dataset_used,
            predictions=y_pred_list,
        )

        # Log
        self.logger.info(
            "Prediction done. model_used=%s count=%d exec_time=%.6fs latest=%s timestamp=%s",
            model_used,
            len(y_pred_list),
            exec_time,
            latest_csv_path,
            ts_csv_path,
        )

        pred_result = PredictionResult(
            y_pred=np.array(y_pred_list, dtype=float),
            success=True,
            message="OK",
            prediction_model_name=model_used,
            inference_time_seconds=exec_time,
            artifacts={
                "model_used": model_used,
                "prediction_export": {
                    "latest_csv": str(latest_csv_path),
                    "timestamp_csv": str(ts_csv_path),
                    "session_json": str(session_json_path),
                    "stats_json": str(stats_json_path),
                },
            },
        )
        return pred_result

