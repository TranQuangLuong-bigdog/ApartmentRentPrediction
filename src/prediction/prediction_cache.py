"""Prediction cache.

Mục tiêu PR9: chỉ implement Prediction management.

Hiện tại module này giữ interface nhỏ để tương lai mở rộng caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PredictionCacheEntry:
    """Entry cache cho kết quả prediction."""

    key: str
    y_pred: Any
    artifacts: Optional[Dict[str, Any]] = None

