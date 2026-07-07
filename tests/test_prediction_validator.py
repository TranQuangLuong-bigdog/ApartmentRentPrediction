import pytest

from src.prediction.prediction_validator import PredictionValidator
from src.exceptions.custom_exception import CustomException


def test_validate_files_missing_model(tmp_path):
    v = PredictionValidator()
    with pytest.raises(CustomException):
        v.validate_files(
            model_path=tmp_path / "missing.keras",
            scaler_path=tmp_path / "scaler.pkl",
            encoder_path=tmp_path / "encoder.pkl",
        )


def test_validate_feature_schema_mismatch():
    v = PredictionValidator()
    with pytest.raises(CustomException):
        v.validate_feature_schema(model_feature_columns=["a"], input_feature_columns=["b"])

