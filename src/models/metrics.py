"""Tính toán các chỉ số đánh giá cho bài toán hồi quy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class MetricsResult:
    """Kết quả các metrics."""

    mae: float
    mse: float
    rmse: float
    r2: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
    """Tính các chỉ số đánh giá hồi quy.

    Args:
        y_true: giá trị thực (n_samples, ) hoặc (n_samples, 1).
        y_pred: giá trị dự đoán (n_samples, ) hoặc (n_samples, 1).

    Returns:
        MetricsResult: chứa mae, mse, rmse, r2.
    """
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_flat, y_pred_flat))

    return MetricsResult(mae=mae, mse=mse, rmse=rmse, r2=r2)


def metrics_to_dict(metrics: MetricsResult) -> Dict[str, float]:
    """Chuyển MetricsResult thành dict để lưu/ghi log.

    Args:
        metrics: MetricsResult.

    Returns:
        Dict[str, float]
    """
    return {
        "MAE": metrics.mae,
        "MSE": metrics.mse,
        "RMSE": metrics.rmse,
        "R2": metrics.r2,
    }
