from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegistryInfo:
    """Thông tin top-level của model registry."""

    active_model: Optional[str] = None
    best_model: Optional[str] = None
    latest_model: Optional[str] = None
    registry_version: Optional[str] = None

    # Danh sách model entries (JSON-compatible)
    models: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_model": self.active_model,
            "best_model": self.best_model,
            "latest_model": self.latest_model,
            "registry_version": self.registry_version,
            "models": self.models,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RegistryInfo":
        return RegistryInfo(
            active_model=data.get("active_model"),
            best_model=data.get("best_model"),
            latest_model=data.get("latest_model"),
            registry_version=data.get("registry_version"),
            models=data.get("models"),
        )

