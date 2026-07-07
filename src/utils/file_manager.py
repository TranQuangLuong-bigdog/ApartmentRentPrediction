"""Tiện ích lưu/đọc file: pickle và CSV."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def create_folder(path: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại.

    Args:
        path: Đường dẫn thư mục.
    """

    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    """Lưu đối tượng vào file pickle.

    Args:
        obj: Đối tượng cần lưu.
        path: Đường dẫn file pickle.
    """

    create_folder(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Đọc đối tượng pickle.

    Args:
        path: Đường dẫn file pickle.

    Returns:
        Đối tượng đã được lưu.
    """

    with path.open("rb") as f:
        return pickle.load(f)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Lưu DataFrame ra CSV.

    Args:
        df: DataFrame cần lưu.
        path: Đường dẫn file CSV.
        index: Có lưu index hay không.
    """

    create_folder(path.parent)
    df.to_csv(path, index=index)


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Đọc CSV thành DataFrame.

    Args:
        path: Đường dẫn file CSV.
        **kwargs: Tham số truyền cho pandas.read_csv.

    Returns:
        DataFrame.
    """

    return pd.read_csv(path, **kwargs)

