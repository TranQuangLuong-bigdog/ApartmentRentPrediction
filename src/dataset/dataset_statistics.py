"""Dataset statistics.

Generate:
- Mean
- Median
- Mode
- Std
- Variance
- Min
- Max
- Quartiles
- Correlation

Chỉ dataset layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetStatisticsResult:
    """Kết quả thống kê dataset."""

    statistics: Dict[str, Any]
    correlation: Optional[Dict[str, Any]]


class DatasetStatistics:
    """Compute statistical summaries for numeric columns."""

    def compute(self, df: pd.DataFrame) -> DatasetStatisticsResult:
        numeric_df = df.select_dtypes(include=["number", "bool"]).copy()
        numeric_cols = list(numeric_df.columns)
        if not numeric_cols:
            return DatasetStatisticsResult(statistics={}, correlation=None)

        statistics: Dict[str, Any] = {}

        statistics["mean"] = numeric_df.mean(numeric_only=True).to_dict()
        statistics["median"] = numeric_df.median(numeric_only=True).to_dict()

        mode_series = numeric_df.mode(numeric_only=True)
        mode_dict: Dict[str, Any] = {}
        for col in numeric_cols:
            if col in mode_series.columns and not mode_series[col].empty:
                mode_dict[col] = mode_series[col].iloc[0]
            else:
                mode_dict[col] = None
        statistics["mode"] = mode_dict

        statistics["std"] = numeric_df.std(numeric_only=True).to_dict()
        statistics["variance"] = numeric_df.var(numeric_only=True).to_dict()
        statistics["min"] = numeric_df.min(numeric_only=True).to_dict()
        statistics["max"] = numeric_df.max(numeric_only=True).to_dict()

        q = numeric_df.quantile([0.25, 0.5, 0.75], numeric_only=True)
        statistics["quartiles"] = {
            "25%": q.loc[0.25].to_dict(),
            "50%": q.loc[0.5].to_dict(),
            "75%": q.loc[0.75].to_dict(),
        }

        corr = numeric_df.corr(numeric_only=True)
        correlation = corr.to_dict() if corr is not None else None

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        # sanitize nested dicts
        for k, v in list(statistics.items()):
            if isinstance(v, dict):
                statistics[k] = {kk: _sanitize(vv) for kk, vv in v.items()}

        if correlation is not None:
            correlation = {str(a): {str(b): _sanitize(c) for b, c in inner.items()} for a, inner in correlation.items()}

        return DatasetStatisticsResult(statistics=statistics, correlation=correlation)

