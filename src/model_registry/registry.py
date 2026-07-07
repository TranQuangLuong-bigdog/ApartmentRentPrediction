"""Model Registry quản lý phiên bản mô hình."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.model_registry._registry_lock import registry_file_lock
from src.model_registry.model_metadata import ModelMetadata, ModelMetrics
from src.model_registry.versioning import next_version, utc_now_iso


class ModelRegistry:
    """Registry lưu nhiều version model và metadata.

    Designed để:
    - không đổi prediction pipeline hiện tại (PR6 chỉ thêm layer registry)
    - hỗ trợ best/latest + rollback.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._state: Dict[str, Any] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Protect concurrent readers/writers.
        with registry_file_lock(self.registry_path):
            if self.registry_path.exists():
                self._state = json.loads(self.registry_path.read_text(encoding="utf-8"))
                self.migrate_registry()
                return


            self._state = {
                "registry_version": "2",
                "active_model": None,
                "latest_model": None,
                "best_model": None,
                "models": [],
            }
            self._persist()

    def _persist(self) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    def _metrics_tuple(self, m: ModelMetrics) -> Tuple[float, float, float]:
        # Primary: r2 (max)
        # Secondary: rmse (min)
        # Tertiary: mae (min)
        return (m.r2, m.rmse, m.mae)

    def _is_better(self, new: ModelMetrics, best: ModelMetrics) -> bool:
        new_r2, new_rmse, new_mae = self._metrics_tuple(new)
        best_r2, best_rmse, best_mae = self._metrics_tuple(best)

        if new_r2 > best_r2:
            return True
        if new_r2 < best_r2:
            return False

        if new_rmse < best_rmse:
            return True
        if new_rmse > best_rmse:
            return False

        return new_mae < best_mae

    def _find_best_metadata(self) -> Optional[ModelMetadata]:
        best_file = self._state.get("best_model")
        if not best_file:
            return None
        for m in self._state.get("models", []):
            if m.get("model_file") == best_file and m.get("is_best") is True:
                return ModelMetadata.from_dict(m)
        return None

    def register_model(
        self,
        *,
        metadata: ModelMetadata,
        model_file: Optional[str] = None,
        force_activate: bool = False,
    ) -> ModelMetadata:
        """Register 1 model version.

        Quy ước cập nhật best:
        - Chỉ dùng metrics trên tập TEST (r2/rmse/mae trong metadata.metrics)
        - Primary: r2 max; Secondary: rmse min; Tertiary: mae min
        """

        model_file_final = model_file or metadata.model_file
        metadata_dict = metadata.to_dict()
        metadata_dict["model_file"] = model_file_final

        # Tạo best flag theo tiêu chí
        best_meta = self._find_best_metadata()
        if best_meta is None:
            # first model => set best
            metadata_dict["is_best"] = True
            self._state["best_model"] = model_file_final
        else:
            if self._is_better(metadata.metrics, best_meta.metrics):
                # replace best
                for m in self._state.get("models", []):
                    if m.get("model_file") == best_meta.model_file:
                        m["is_best"] = False
                metadata_dict["is_best"] = True
                self._state["best_model"] = model_file_final
            else:
                metadata_dict["is_best"] = False

        # latest always updates
        self._state["latest_model"] = model_file_final

        # active model updates only if requested or if empty
        if self._state.get("active_model") is None or force_activate:
            self._state["active_model"] = model_file_final

        # upsert by version
        models: List[Dict[str, Any]] = self._state.get("models", [])
        replaced = False
        for i, m in enumerate(models):
            if str(m.get("version")) == str(metadata.version) and m.get("model_file") == model_file_final:
                models[i] = metadata_dict
                replaced = True
                break
        if not replaced:
            models.append(metadata_dict)

        self._state["models"] = models
        self._persist()

        # Return metadata object normalized
        return ModelMetadata.from_dict(metadata_dict)

    def list_models(self) -> List[ModelMetadata]:
        return [ModelMetadata.from_dict(m) for m in self._state.get("models", [])]

    def get_latest_model(self) -> Optional[ModelMetadata]:
        latest_file = self._state.get("latest_model")
        if not latest_file:
            return None
        for m in self._state.get("models", []):
            if m.get("model_file") == latest_file:
                return ModelMetadata.from_dict(m)
        return None

    def get_best_model(self) -> Optional[ModelMetadata]:
        best_file = self._state.get("best_model")
        if not best_file:
            return None
        for m in self._state.get("models", []):
            if m.get("model_file") == best_file:
                return ModelMetadata.from_dict(m)
        return None

    def migrate_registry(self) -> None:
        """Migrate legacy registry.json state in-place.

        This method is intentionally minimal to avoid breaking existing flows.
        Full enhancements (compatibility/search/export/upgrade) will be added later.
        """

        if not isinstance(self._state, dict):
            self._state = {}

        self._state.setdefault("registry_version", "2")
        self._state.setdefault("active_model", None)
        self._state.setdefault("latest_model", None)
        self._state.setdefault("best_model", None)
        self._state.setdefault("models", [])

        # Ensure each model entry has expected keys for ModelMetadata.from_dict.
        migrated_models: List[Dict[str, Any]] = []
        for m in self._state.get("models", []) or []:
            if not isinstance(m, dict):
                continue
            m.setdefault("dataset_hash", "unknown")
            m.setdefault("dataset_name", "unknown")
            m.setdefault("rows", 0)
            m.setdefault("columns", 0)
            m.setdefault("feature_count", 0)
            m.setdefault("pipeline_version", "1.0.0")
            m.setdefault("preprocessing_version", "1.0.0")
            m.setdefault("tensorflow_version", "unknown")
            m.setdefault("python_version", "unknown")
            m.setdefault("feature_schema", {})
            if "metrics" not in m or not isinstance(m.get("metrics"), dict):
                m["metrics"] = {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "mse": None}
            m.setdefault("is_best", False)
            m.setdefault("extra", {})
            migrated_models.append(m)

        self._state["models"] = migrated_models
        self._persist()

    def upgrade_registry(self) -> None:
        """Upgrade registry to current format.

        For now, it delegates to migrate_registry() to keep behavior consistent.
        """

        self.migrate_registry()

    def delete_model(self, model_file: str) -> None:

        models = self._state.get("models", [])
        new_models = [m for m in models if m.get("model_file") != model_file]
        self._state["models"] = new_models

        # update pointers
        if self._state.get("latest_model") == model_file:
            self._state["latest_model"] = None
        if self._state.get("best_model") == model_file:
            self._state["best_model"] = None
        if self._state.get("active_model") == model_file:
            self._state["active_model"] = None

        self._persist()

    def rollback_model(self, version: str) -> Optional[ModelMetadata]:
        """Set active_model = model_file của version (nếu tồn tại)."""

        for m in self._state.get("models", []):
            if str(m.get("version")) == str(version):
                self._state["active_model"] = m.get("model_file")
                self._persist()
                return ModelMetadata.from_dict(m)
        return None

    @staticmethod
    def ensure_minimum_metadata(
        dataset_info: Dict[str, Any],
        feature_schema: Dict[str, Any],
        metrics: ModelMetrics,
        *,
        created_at: Optional[str] = None,
        model_file: str,
        version: Optional[str] = None,
    ) -> ModelMetadata:
        """Tự sinh metadata nếu project chưa cung cấp đủ.

        - không để null nếu có thể tự tính/tự sinh
        """

        created_at_final = created_at or utc_now_iso()

        # safe defaults
        dataset_hash = dataset_info.get("dataset_hash") or "unknown"
        dataset_name = dataset_info.get("dataset_name") or dataset_info.get("dataset_file") or "unknown"

        rows = int(dataset_info.get("rows") or dataset_info.get("dataset_rows") or 0)
        columns = int(dataset_info.get("columns") or dataset_info.get("dataset_columns") or 0)
        feature_count = int(dataset_info.get("feature_count") or 0)

        preprocessing_version = dataset_info.get("preprocessing_version") or "1.0.0"
        pipeline_version = dataset_info.get("pipeline_version") or "1.0.0"

        # runtime info (best-effort)
        try:
            import tensorflow as tf  # noqa: WPS433

            tf_version = getattr(tf, "__version__", None) or "unknown"
        except Exception:
            tf_version = "unknown"

        import platform

        python_version = platform.python_version()

        tensorflow_version = tf_version

        v = version or "1.0.0"

        training_time_seconds = float(dataset_info.get("training_time_seconds") or 0.0)

        return ModelMetadata(
            version=v,
            model_file=model_file,
            created_at=created_at_final,
            training_time_seconds=training_time_seconds,
            dataset_hash=str(dataset_hash),
            dataset_name=str(dataset_name),
            rows=rows,
            columns=columns,
            feature_count=feature_count if feature_count > 0 else len(feature_schema) if isinstance(feature_schema, dict) else 0,
            pipeline_version=str(pipeline_version),
            preprocessing_version=str(preprocessing_version),
            tensorflow_version=str(tensorflow_version),
            python_version=str(python_version),
            feature_schema=feature_schema,
            metrics=metrics,
            is_best=False,
            extra={},
        )

