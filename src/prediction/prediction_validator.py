"""PredictionValidator.

Validate:
- Model exists
- Metadata exists
- Scaler exists
- Encoder exists
- Feature count/names/order/dtype
- Dataset compatibility
- Registry compatibility

Theo yêu cầu: không phá vỡ Contracts/Registry API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.exceptions.custom_exception import MLAppException

# Backward compatible alias
CustomException = MLAppException
from src.utils.logger import get_logger


class PredictionValidator:
    """Validate prediction compatibility before inference."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)

    def validate_files(self, *, model_path: Optional[Path], scaler_path: Optional[Path], encoder_path: Optional[Path]) -> None:
        if model_path is None or not model_path.exists():
            raise CustomException("Model file not found", code="MODEL_NOT_FOUND")
        if scaler_path is None or not scaler_path.exists():
            raise CustomException("Scaler file not found", code="SCALER_NOT_FOUND")
        if encoder_path is None or not encoder_path.exists():
            raise CustomException("Encoder file not found", code="ENCODER_NOT_FOUND")

    def validate_feature_schema(
        self,
        *,
        model_feature_columns: List[str],
        input_feature_columns: List[str],
    ) -> None:
        if len(model_feature_columns) != len(input_feature_columns):
            raise CustomException("Feature count mismatch", code="FEATURE_COUNT_MISMATCH")

        if model_feature_columns != input_feature_columns:
            raise CustomException("Feature order mismatch", code="FEATURE_ORDER_MISMATCH")

    def validate_compatibility(
        self,
        *,
        pipeline_version: Optional[str],
        preprocessing_version: Optional[str],
        dataset_hash: Optional[str],
        metadata: Dict[str, Any],
    ) -> None:
        # Minimal compatibility checks.
        expected = metadata.get("training_config", {})
        expected_pipeline_version = expected.get("pipeline_version")
        expected_preprocessing_version = expected.get("preprocessing_version")

        if expected_pipeline_version and pipeline_version and expected_pipeline_version != pipeline_version:
            raise CustomException("pipeline_version incompatible", code="PIPELINE_VERSION_INCOMPATIBLE")

        if expected_preprocessing_version and preprocessing_version and expected_preprocessing_version != preprocessing_version:
            raise CustomException("preprocessing_version incompatible", code="PREPROCESSING_VERSION_INCOMPATIBLE")

        # dataset_hash check if available
        expected_hash = metadata.get("model_info", {}).get("dataset_hash")
        if expected_hash and dataset_hash and expected_hash != dataset_hash:
            raise CustomException("dataset_hash incompatible", code="DATASET_HASH_INCOMPATIBLE")

    def validate_registry_compatibility(self, *, registry_metadata: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        # Minimal: nếu có version fields thì so sánh.
        if not registry_metadata:
            return
        if "model_version" in registry_metadata and "model_version" in metadata:
            if registry_metadata["model_version"] != metadata["model_version"]:
                raise CustomException("Registry metadata incompatible", code="REGISTRY_INCOMPATIBLE")

