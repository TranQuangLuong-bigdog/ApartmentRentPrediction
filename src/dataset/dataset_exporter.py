"""Dataset exporter.

Save:
- output/report/dataset_summary.json
- output/report/dataset_statistics.json
- output/report/dataset_profile.json

Chỉ dataset layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger


class DatasetExporter:
    """Export dataset reports to disk."""

    def __init__(self, output_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or get_logger(__name__)
        self.output_dir = Path(output_dir) if output_dir is not None else Path("output") / "report"

    def _write_json(self, file_path: Path, payload: Dict[str, Any]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.info("Exported dataset report: %s", file_path)

    def export(
        self,
        dataset_summary: Dict[str, Any],
        dataset_statistics: Dict[str, Any],
        dataset_profile: Dict[str, Any],
    ) -> None:
        self._write_json(self.output_dir / "dataset_summary.json", dataset_summary)
        self._write_json(self.output_dir / "dataset_statistics.json", dataset_statistics)
        self._write_json(self.output_dir / "dataset_profile.json", dataset_profile)

