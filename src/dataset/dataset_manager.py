"""Dataset manager.

Orchestrate dataset layer:
- load
- validate
- profile
- statistics
- export

Không đụng Train/Predict pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.dataset.dataset_exporter import DatasetExporter
from src.dataset.dataset_hash import DatasetHash
from src.dataset.dataset_loader import DatasetLoader
from src.dataset.dataset_profiler import DatasetProfiler
from src.dataset.dataset_statistics import DatasetStatistics
from src.dataset.dataset_validator import DatasetValidationResult, DatasetValidator
from src.contracts.dataset_summary import DatasetSummary
from src.utils.logger import get_logger


class DatasetManager:
    """Main entry point for dataset layer."""

    def __init__(
        self,
        *,
        loader: Optional[DatasetLoader] = None,
        validator: Optional[DatasetValidator] = None,
        profiler: Optional[DatasetProfiler] = None,
        statistics: Optional[DatasetStatistics] = None,
        exporter: Optional[DatasetExporter] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or get_logger(__name__)
        self.loader = loader or DatasetLoader(logger=self.logger)
        self.validator = validator or DatasetValidator(logger=self.logger)
        self.profiler = profiler or DatasetProfiler(logger=self.logger)
        self.statistics = statistics or DatasetStatistics()
        self.exporter = exporter or DatasetExporter(logger=self.logger)

    def run(
        self,
        file_path: Path,
        target_column: str,
        *,
        sheet_name: int | str = 0,
        encoding: Optional[str] = None,
        separator: Optional[str] = None,
    ) -> DatasetValidationResult:
        """Run dataset workflow and export reports."""
        file_path = Path(file_path)

        if file_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            df = self.loader.load_csv(file_path, encoding=encoding, separator=separator)
        elif file_path.suffix.lower() in {".xlsx", ".xls"}:
            df = self.loader.load_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported dataset file type: {file_path.suffix}")

        validation = self.validator.validate(df=df, target_column=target_column, file_path=file_path, encoding=encoding)

        dataset_summary_obj: DatasetSummary = self.profiler.profile(df, target_column=target_column, file_path=file_path)
        dataset_profile_ext = self.profiler.profile_extended(df, target_column=target_column, file_path=file_path)

        stats_res = self.statistics.compute(df)

        self.exporter.export(
            dataset_summary=dataset_summary_obj.to_dict(),
            dataset_statistics={"statistics": stats_res.statistics, "correlation": stats_res.correlation},
            dataset_profile=dataset_profile_ext,
        )

        return validation

