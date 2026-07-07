"""Versioning cho Model Registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def utc_now_iso() -> str:
    """ISO timestamp UTC (không phụ thuộc locale)."""

    return datetime.now(timezone.utc).isoformat()


def increment_patch(version: str) -> str:
    """Tăng patch của semantic version dạng x.y.z."""

    try:
        major, minor, patch = (int(x) for x in version.split("."))
    except Exception:
        return "1.0.0"

    return str(SemanticVersion(major=major, minor=minor, patch=patch + 1))


def next_version(latest_version: Optional[str]) -> str:
    """Sinh version kế tiếp dựa trên latest_version."""

    if not latest_version:
        return "1.0.0"

    return increment_patch(latest_version)

