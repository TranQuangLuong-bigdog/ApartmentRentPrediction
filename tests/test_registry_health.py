import json
from pathlib import Path

from src.model_registry.health import RegistryHealthManager
from src.model_registry.model_metadata import ModelMetadata, ModelMetrics


def _make_metadata(version: str, model_file: str, *, r2: float, rmse: float, mae: float) -> dict:
    meta = ModelMetadata(
        version=version,
        model_file=model_file,
        created_at="2024-01-01T00:00:00+00:00",
        training_time_seconds=1.0,
        dataset_hash="hash1",
        dataset_name="dataset1",
        rows=10,
        columns=10,
        feature_count=3,
        pipeline_version="1.0.0",
        preprocessing_version="1.0.0",
        tensorflow_version="unknown",
        python_version="unknown",
        feature_schema={"numerical_columns": ["a", "b", "c"], "categorical_columns": []},
        metrics=ModelMetrics(r2=r2, rmse=rmse, mae=mae, mse=None),
        is_best=False,
        extra={},
    )
    return meta.to_dict()


def test_validate_healthy_registry(tmp_path: Path) -> None:
    # Arrange
    registry_path = tmp_path / "registry.json"
    model_path = tmp_path / "best.keras"
    model_path.write_text("dummy", encoding="utf-8")

    state = {
        "registry_version": "2",
        "active_model": "best.keras",
        "latest_model": "best.keras",
        "best_model": "best.keras",
        "models": [
            _make_metadata("1.0.0", "best.keras", r2=0.5, rmse=10.0, mae=8.0),
        ],
    }
    registry_path.write_text(json.dumps(state), encoding="utf-8")

    # Act
    mgr = RegistryHealthManager(registry_path)
    health = mgr.validate_registry()

    # Assert
    assert health.valid is True
    assert health.best_model == "best.keras"
    assert health.missing_models == []


def test_repair_registry_missing_model_file(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"

    state = {
        "registry_version": "2",
        "active_model": "missing.keras",
        "latest_model": "missing.keras",
        "best_model": "missing.keras",
        "models": [
            _make_metadata("1.0.0", "missing.keras", r2=0.5, rmse=10.0, mae=8.0),
        ],
    }
    registry_path.write_text(json.dumps(state), encoding="utf-8")

    mgr = RegistryHealthManager(registry_path)
    repair = mgr.repair_registry()

    assert repair.success is True
    assert repair.repaired is True

    # Registry should now have empty models and pointers cleared
    new_state = json.loads(registry_path.read_text(encoding="utf-8"))
    assert new_state["models"] == []
    assert new_state["latest_model"] is None
    assert new_state["best_model"] is None

