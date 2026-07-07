from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RepairResult:
    """Kết quả repair của Model Registry."""

    success: bool
    repaired: bool

    removed_models: List[str]
    rebuilt_pointers: bool
    regenerated_registry: bool

    warnings: List[str]
    errors: List[str]

