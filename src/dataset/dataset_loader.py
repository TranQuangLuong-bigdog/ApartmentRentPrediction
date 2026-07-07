"""Dataset loader.

DatasetLoader:
- load_csv()
- load_excel()
- detect_encoding()
- detect_separator()

Chỉ dataset layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger


class DatasetLoader:
    """Load dataset file (CSV/Excel)."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)

    def detect_encoding(self, file_path: Path, sample_size: int = 200_000) -> str:
        """Detect encoding by trying common candidates."""
        import codecs

        candidates = ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1"]
        raw = file_path.read_bytes()[:sample_size]

        for enc in candidates:
            try:
                raw.decode(enc)
                self.logger.info("Detected encoding=%s for %s", enc, file_path)
                return enc
            except Exception:
                continue

        self.logger.warning("Fallback to utf-8 for %s", file_path)
        return "utf-8"

    def detect_separator(self, file_path: Path, encoding: Optional[str] = None) -> str:
        """Detect CSV separator by counting delimiters in first line."""
        encoding = encoding or self.detect_encoding(file_path)
        candidates = [",", ";", "\t", "|", " "]

        first_line = file_path.read_text(encoding=encoding, errors="ignore").splitlines()[0:1]
        first_line = first_line[0] if first_line else ""

        best_sep = ","
        best_cnt = -1
        for sep in candidates:
            cnt = first_line.count(sep)
            if cnt > best_cnt:
                best_cnt = cnt
                best_sep = sep

        self.logger.info("Detected separator=%r for %s", best_sep, file_path)
        return best_sep

    def load_csv(
        self,
        file_path: Path,
        *,
        encoding: Optional[str] = None,
        separator: Optional[str] = None,
        **read_csv_kwargs: object,
    ) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        encoding = encoding or self.detect_encoding(file_path)
        separator = separator or self.detect_separator(file_path, encoding=encoding)

        self.logger.info("Loading CSV: %s (encoding=%s, sep=%r)", file_path, encoding, separator)
        return pd.read_csv(file_path, encoding=encoding, sep=separator, **read_csv_kwargs)

    def load_excel(
        self,
        file_path: Path,
        *,
        sheet_name: int | str = 0,
        **read_excel_kwargs: object,
    ) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        self.logger.info("Loading Excel: %s (sheet=%s)", file_path, sheet_name)
        return pd.read_excel(file_path, sheet_name=sheet_name, **read_excel_kwargs)

