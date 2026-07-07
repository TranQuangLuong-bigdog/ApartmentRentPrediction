import numpy as np
import pytest

from src.prediction.prediction_service import PredictionService
from src.exceptions.custom_exception import CustomException


def test_predict_service_artifacts_missing_raises(monkeypatch):
    service = PredictionService()

    class DummyLoaded:
        model_used = "BEST"
        scaler_path = None
        encoder_path = None

        @property
        def model(self):
            class M:
                def predict(self, X, verbose=0):
                    return np.array([[1.0] for _ in range(len(X))])

            return M()

    class DummyLoader:
        def load_model_and_metadata(self):
            return DummyLoaded()

    service.loader = DummyLoader()

    with pytest.raises(CustomException):
        service.predict_single([0.1, 0.2])

