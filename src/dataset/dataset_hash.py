"""Dataset hash (DatasetHash).

SHA256
MD5
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


class DatasetHash:
    """Compute hashes for dataset files."""

    @staticmethod
    def sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        file_path = Path(file_path)
        h = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        file_path = Path(file_path)
        h = hashlib.md5()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_all(file_path: Path) -> Dict[str, str]:
        return {"sha256": DatasetHash.sha256(file_path), "md5": DatasetHash.md5(file_path)}

