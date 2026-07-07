"""Prediction statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


class PredictionStatistics:
    """Compute prediction distribution metrics."""

    @staticmethod
    def compute(y_pred: List[float]) -> Dict[str, Any]:
        if not y_pred:
            return {
                "count": 0,
                "average": None,
                "min": None,
                "max": None,
                "median": None,
                "std": None,
                "distribution": {},
            }

        arr = np.array(y_pred, dtype=float)
        distribution: Dict[str, int] = {}
        # Simple binning into 10 bins
        bins = np.histogram(arr, bins=10)
        counts = bins[0].astype(int)
        edges = bins[1]
        for i in range(len(counts)):
            key = f"[{edges[i]:.3f},{edges[i+1]:.3f})"
            distribution[key] = int(counts[i])

        return {
            "count": int(len(arr)),
            "average": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
            "distribution": distribution,
        }

