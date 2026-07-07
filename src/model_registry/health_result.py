from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RegistryHealth:
    """Kết quả validate health của Model Registry."""

    valid: bool
    registry_version: str

    active_model: Optional[str]
    latest_model: Optional[str]
    best_model: Optional[str]

    missing_models: List[str]
    broken_paths: List[str]
    orphan_models: List[str]

    duplicated_entries: List[str]
    invalid_metadata: List[str]

    warnings: List[str]
    errors: List[str]

