import json
from pathlib import Path

import pytest

from src.model_registry.registry import ModelRegistry
from src.model_registry.model_metadata import ModelMetadata, ModelMetrics


def _make_metadata(version: str, model_file: str, *, r2: float, rmse: float, mae: float) -> ModelMetadata:
    return ModelMetadata(
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


def test_migrate_registry_default_fields(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"

    # minimal old format (no top-level pointers)
    reg_path.write_text(json.dumps({"models": []}), encoding="utf-8")

    reg = ModelRegistry(reg_path)

    assert reg_path.exists()
    state = json.loads(reg_path.read_text(encoding="utf-8"))

    assert state["active_model"] is None
    assert state["latest_model"] is None
    assert state["best_model"] is None
    assert isinstance(state["models"], list)


def test_register_model_updates_best_latest(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg = ModelRegistry(reg_path)

    m1 = _make_metadata("1.0.0", "m1.keras", r2=0.5, rmse=10.0, mae=8.0)
    reg.register_model(metadata=m1)

    m2 = _make_metadata("1.0.1", "m2.keras", r2=0.6, rmse=12.0, mae=9.0)
    reg.register_model(metadata=m2)

    best = reg.get_best_model()
    latest = reg.get_latest_model()

    assert best is not None
    assert latest is not None
    assert best.model_file == "m2.keras"  # higher r2
    assert latest.model_file == "m2.keras"  # last registered

