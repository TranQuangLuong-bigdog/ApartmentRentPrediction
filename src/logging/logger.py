"""Wrapper logging cho production.

Giai đoạn hiện tại tái sử dụng src.utils.logger.setup_logger.
"""

from __future__ import annotations

from pathlib import Path
from logging import Logger

import logging

from src.utils.logger import setup_logger


def get_logger(name: str, log_dir: Path, level: int = logging.INFO) -> Logger:
    """Lấy logger."""

    return setup_logger(name=name, log_dir=log_dir, level=level)

