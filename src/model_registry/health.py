from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.model_registry.model_metadata import ModelMetadata
from src.model_registry._registry_lock import registry_file_lock
from src.model_registry.health_result import RegistryHealth
from src.model_registry.repair_result import RepairResult


def _safe_list(x: Any) -> List[str]:
    return list(x) if isinstance(x, list) else []


class RegistryHealthManager:
    """Quản lý validate/repair cho trained_models/registry.json."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path

    def _load_registry_state(self) -> Tuple[Dict[str, Any], Optional[str]]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        try:
            raw = self.registry_path.read_text(encoding="utf-8")
            state = json.loads(raw)
            if not isinstance(state, dict):
                raise ValueError("Registry JSON must be an object")
            registry_version = str(state.get("registry_version") or "unknown")
            return state, registry_version
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted registry JSON: {e}") from e

    def validate_registry(self) -> RegistryHealth:
        """Validate registry.json.

        Returns:
            RegistryHealth
        """

        missing_models: List[str] = []
        broken_paths: List[str] = []
        orphan_models: List[str] = []
        duplicated_entries: List[str] = []
        invalid_metadata: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        try:
            state, registry_version = self._load_registry_state()
        except Exception as e:
            return RegistryHealth(
                valid=False,
                registry_version="unknown",
                active_model=None,
                latest_model=None,
                best_model=None,
                missing_models=[],
                broken_paths=[],
                orphan_models=[],
                duplicated_entries=[],
                invalid_metadata=[],
                warnings=[],
                errors=[str(e)],
            )

        active_model = state.get("active_model")
        latest_model = state.get("latest_model")
        best_model = state.get("best_model")

        models = _safe_list(state.get("models"))

        # basic top-level pointer checks
        pointers = {
            "active_model": active_model,
            "latest_model": latest_model,
            "best_model": best_model,
        }

        # Validate model file existence and metadata correctness
        for key, model_file in pointers.items():
            if model_file is None:
                continue
            model_path = self._model_path(model_file)
            if not model_path.exists():
                broken_paths.append(str(model_path))
                missing_models.append(str(model_file))

        # Validate model list entries
        seen_keys: set[str] = set()
        for entry in models:
            if not isinstance(entry, dict):
                invalid_metadata.append("non_dict_model_entry")
                continue

            version = entry.get("version")
            model_file = entry.get("model_file")
            entry_key = f"{version}:{model_file}"
            if entry_key in seen_keys:
                duplicated_entries.append(entry_key)
            else:
                seen_keys.add(entry_key)

            # Check metadata can be parsed
            try:
                _ = ModelMetadata.from_dict(entry)
            except Exception:
                invalid_metadata.append(entry_key)

            if model_file:
                model_path = self._model_path(str(model_file))
                if not model_path.exists():
                    missing_models.append(str(model_file))
                    broken_paths.append(str(model_path))

        # Orphan models: model entries that are not referenced by pointers
        referenced = {x for x in [active_model, latest_model, best_model] if x}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            mf = entry.get("model_file")
            if mf and mf not in referenced:
                # considered orphan only if file exists but not referenced; if file missing it will be handled above
                orphan_models.append(str(mf))

        # Determine validity
        valid = len(errors) == 0 and len(invalid_metadata) == 0
        return RegistryHealth(
            valid=valid,
            registry_version=str(registry_version),
            active_model=active_model,
            latest_model=latest_model,
            best_model=best_model,
            missing_models=sorted(set(missing_models)),
            broken_paths=sorted(set(broken_paths)),
            orphan_models=sorted(set(orphan_models)),
            duplicated_entries=sorted(set(duplicated_entries)),
            invalid_metadata=sorted(set(invalid_metadata)),
            warnings=sorted(set(warnings)),
            errors=sorted(set(errors)),
        )

    def repair_registry(self) -> RepairResult:
        """Repair registry.json.

        Strategy (best-effort):
        - Backup registry before changes.
        - Remove invalid model pointer entries.
        - Rebuild latest/best/active from remaining valid entries.
        - Remove orphan entries (entries whose model_file isn't referenced) only if they are invalid/unreachable.

        Returns:
            RepairResult
        """

        removed_models: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        try:
            state, _ = self._load_registry_state()
        except Exception as e:
            return RepairResult(
                success=False,
                repaired=False,
                removed_models=[],
                rebuilt_pointers=False,
                regenerated_registry=False,
                warnings=[],
                errors=[str(e)],
            )

        models = _safe_list(state.get("models"))

        # Backup
        backup_dir = self.registry_path.parent / "registry_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = "registry_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        backup_path = backup_dir / backup_name
        shutil.copy2(self.registry_path, backup_path)

        rebuilt_pointers = False

        # Filter valid entries that have metadata parseable and model file exists
        valid_entries: List[Dict[str, Any]] = []
        for entry in models:
            if not isinstance(entry, dict):
                continue
            model_file = entry.get("model_file")
            if not model_file:
                removed_models.append("<missing model_file>")
                continue

            model_path = self._model_path(str(model_file))
            if not model_path.exists():
                # remove unreachable/orphan missing file
                removed_models.append(str(model_file))
                continue

            try:
                _ = ModelMetadata.from_dict(entry)
            except Exception:
                removed_models.append(str(model_file))
                continue

            valid_entries.append(entry)

        # Regenerate pointers
        # latest: last registered entry order (preserve order)
        latest_file = None
        if valid_entries:
            latest_file = valid_entries[-1].get("model_file")

        # best: maximize r2, then minimize rmse, then minimize mae
        best_file = None
        best_meta: Optional[ModelMetadata] = None
        for entry in valid_entries:
            meta = ModelMetadata.from_dict(entry)
            if best_meta is None:
                best_meta = meta
                best_file = meta.model_file
                continue

            # compare
            if (meta.metrics.r2, -meta.metrics.rmse, -meta.metrics.mae) > (
                best_meta.metrics.r2,
                -best_meta.metrics.rmse,
                -best_meta.metrics.mae,
            ):
                best_meta = meta
                best_file = meta.model_file

        # active: keep existing if points to a valid entry else fallback to best
        active_file = state.get("active_model")
        existing_files = {str(e.get("model_file")) for e in valid_entries if e.get("model_file")}
        if active_file not in existing_files:
            active_file = best_file

        state["models"] = valid_entries
        state["latest_model"] = latest_file
        state["best_model"] = best_file
        state["active_model"] = active_file

        state["registry_version"] = str(state.get("registry_version") or "2")
        rebuilt_pointers = True

        # Persist
        with registry_file_lock(self.registry_path):
            tmp = self.registry_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.registry_path)

        return RepairResult(
            success=True,
            repaired=True,
            removed_models=sorted(set(removed_models)),
            rebuilt_pointers=rebuilt_pointers,
            regenerated_registry=True,
            warnings=warnings,
            errors=errors,
        )

    def _model_path(self, model_file: str) -> Path:
        # trained_models/<model_file>
        return self.registry_path.parent / model_file


# Convenience functions

def validate_registry(registry_path: Path) -> RegistryHealth:
    return RegistryHealthManager(registry_path).validate_registry()


def repair_registry(registry_path: Path) -> RepairResult:
    return RegistryHealthManager(registry_path).repair_registry()

