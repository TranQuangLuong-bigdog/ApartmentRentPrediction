"""PredictionLoader: load model + metadata + artifacts used for inference.

Theo yêu cầu:
- best_model.keras / fallback last_model.keras
- feature_columns.json
- label_metadata.json
- training_config.json
- model_info.json

Chỉ thuộc prediction layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from src.utils.logger import get_logger


@dataclass(frozen=True)
class LoadedArtifacts:
    """Container lưu trữ artifacts phục vụ prediction."""

    model: tf.keras.Model
    model_used: str  # BEST or LAST
    metadata: Dict[str, Any]

    scaler_path: Optional[Path]
    encoder_path: Optional[Path]
    feature_columns_path: Optional[Path]


class PredictionLoader:
    """Load prediction artifacts from trained_models/ and output/ folders."""

    def __init__(self, base_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)
        self.base_dir = Path(base_dir) if base_dir is not None else Path("trained_models")

    def _load_keras_model(self, model_path: Path) -> tf.keras.Model:
        self.logger.info("Loading Keras model: %s", model_path)
        return tf.keras.models.load_model(model_path)

    def _read_json_if_exists(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_model_and_metadata(
        self,
        *,
        default_model_name: str = "best_model.keras",
        fallback_model_name: str = "last_model.keras",
    ) -> LoadedArtifacts:
        """Load model and metadata with fallback."""

        best_path = self.base_dir / default_model_name
        last_path = self.base_dir / fallback_model_name

        if best_path.exists():
            model = self._load_keras_model(best_path)
            model_used = "BEST"
            used_path = best_path
        elif last_path.exists():
            model = self._load_keras_model(last_path)
            model_used = "LAST"
            used_path = last_path
        else:
            raise FileNotFoundError(f"Neither {best_path} nor {last_path} exists")

        # Load metadata json files if present
        feature_columns_path = self.base_dir / "feature_columns.json"
        label_metadata_path = self.base_dir / "label_metadata.json"
        training_config_path = self.base_dir / "training_config.json"
        model_info_path = self.base_dir / "model_info.json"

        metadata: Dict[str, Any] = {
            "model_used": model_used,
            "model_path": str(used_path),
            "feature_columns": self._read_json_if_exists(feature_columns_path),
            "label_metadata": self._read_json_if_exists(label_metadata_path),
            "training_config": self._read_json_if_exists(training_config_path),
            "model_info": self._read_json_if_exists(model_info_path),
        }

        # Scaler/encoder typically stored as pkl in this repo, but không ném error nếu không có.
        scaler_path = self.base_dir / "scaler.pkl"
        encoder_path = self.base_dir / "encoder.pkl"

        return LoadedArtifacts(
            model=model,
            model_used=model_used,
            metadata=metadata,
            scaler_path=scaler_path if scaler_path.exists() else None,
            encoder_path=encoder_path if encoder_path.exists() else None,
            feature_columns_path=feature_columns_path if feature_columns_path.exists() else None,
        )

