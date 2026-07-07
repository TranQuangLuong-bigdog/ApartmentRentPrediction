"""Validator cho dữ liệu input từ user/file upload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.config.constants import TARGET_COLUMN
from src.exceptions.custom_exception import DataNotFoundError, ValidationError


@dataclass(frozen=True)
class InputValidationResult:
    """Kết quả validate input."""

    ok: bool
    errors: List[str]


def validate_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> InputValidationResult:
    """Validate DataFrame dataset.

    Args:
        df: DataFrame.
        target_column: tên cột nhãn.

    Returns:
        InputValidationResult.
    """

    errors: List[str] = []

    if df is None or df.empty:
        errors.append("Dataset rỗng hoặc None.")

    if df is not None and target_column not in df.columns:
        errors.append(f"Không tìm thấy cột target '{target_column}' trong dataset.")

    # Kiểm tra kiểu dữ liệu cơ bản
    if df is not None and df.shape[0] < 10:
        errors.append("Dataset quá nhỏ. Cần ít nhất 10 dòng để huấn luyện ổn định.")

    ok = len(errors) == 0
    return InputValidationResult(ok=ok, errors=errors)


def validate_file_exists(path: "object") -> None:
    """Validate sự tồn tại của file.

    Args:
        path: Path-like.
    """

    if path is None:
        raise DataNotFoundError("Không nhận được đường dẫn file.")

    if not getattr(path, "exists")():
        raise DataNotFoundError(f"Không tìm thấy file: {path}")

