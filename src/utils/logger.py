"""Thiết lập logging cho project."""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path


def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> Logger:
    """Tạo logger ghi ra console và file.

    Args:
        name: Tên logger.
        log_dir: Thư mục lưu file log.
        level: Mức độ logging.

    Returns:
        Logger: instance logger.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh gắn handler trùng lặp khi gọi nhiều lần.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """Backward compatible alias.

    Nhiều module trong project đang import `get_logger`.
    Hàm này giữ nguyên API cũ bằng cách gọi `setup_logger`.
    """

    return setup_logger(name=name, log_dir=Path("logs"), level=level)

