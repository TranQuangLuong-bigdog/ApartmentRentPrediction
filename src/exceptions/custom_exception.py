"""Ngoại lệ tùy chỉnh cho project."""

from __future__ import annotations


class MLAppException(Exception):
    """Ngoại lệ cơ sở cho ứng dụng ML."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


class ValidationError(MLAppException):
    """Ngoại lệ khi dữ liệu input không hợp lệ."""


class DataNotFoundError(MLAppException):
    """Ngoại lệ khi không tìm thấy dữ liệu."""


# Backward compatible alias
CustomException = MLAppException


