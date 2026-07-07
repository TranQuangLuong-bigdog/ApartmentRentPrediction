"""Cache manager (hook).

Giai đoạn hiện tại để chuẩn bị cho future improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheManager:
    """Quản lý cache (placeholder)."""

    enabled: bool = False

    def get(self, key: str) -> Optional[Any]:
        """Lấy dữ liệu từ cache."""
        return None

    def set(self, key: str, value: Any) -> None:
        """Lưu dữ liệu vào cache."""
        return None

