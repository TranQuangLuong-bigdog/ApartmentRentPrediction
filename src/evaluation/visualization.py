"""Các hàm vẽ biểu đồ phục vụ evaluation.

Giai đoạn 5: Visualization layer.

Tất cả hàm đều:
- dùng matplotlib
- không duplicate logic
- chỉ nhận dữ liệu đầu vào và Path output
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

# Đảm bảo chạy được trong môi trường headless
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _finalize_plot(out_path: Path, title: Optional[str] = None) -> None:
    """Hoàn thiện plot: đặt title, tight_layout và lưu file."""

    if title:
        plt.title(title)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_loss(history: Any, out_path: Path) -> None:
    """Vẽ loss/val_loss theo epoch."""

    epochs = np.arange(len(history.history.get("loss", [])))
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(10, 6))
    if len(loss) > 0:
        plt.plot(epochs, loss, label="Loss")
    if len(val_loss) > 0:
        plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    _finalize_plot(out_path)


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Biểu đồ prediction vs actual."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    _finalize_plot(out_path, title="Prediction vs Actual")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Vẽ residual plot."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    _finalize_plot(out_path, title="Residual Plot")


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Vẽ phân bố residual."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, alpha=0.85)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    _finalize_plot(out_path, title="Residual Distribution")


def plot_correlation_heatmap(corr: np.ndarray, out_path: Path, labels: Optional[Sequence[str]] = None) -> None:
    """Vẽ heatmap tương quan."""

    corr = np.asarray(corr)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation")

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

    _finalize_plot(out_path, title="Correlation Heatmap")


def plot_feature_distribution(values: np.ndarray, out_path: Path, feature_name: str = "Feature") -> None:
    """Vẽ phân bố một feature."""

    values = np.asarray(values).reshape(-1)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40, alpha=0.85)
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    _finalize_plot(out_path, title=f"Distribution of {feature_name}")


def plot_boxplot(values: np.ndarray, out_path: Path) -> None:
    """Vẽ boxplot cho residuals hoặc y_pred."""

    values = np.asarray(values).reshape(-1)
    plt.figure(figsize=(8, 6))
    plt.boxplot(values, vert=True)
    plt.ylabel("Value")
    _finalize_plot(out_path, title="Boxplot")


def plot_scatter_prediction(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Vẽ scatter prediction (alternative)."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    _finalize_plot(out_path, title="Scatter Prediction")


def plot_learning_curve(history: Any, out_path: Path) -> None:
    """Vẽ học tập theo epoch (dùng loss/val_loss)."""

    plot_loss(history=history, out_path=out_path)


def plot_training_summary(history: Any, out_path: Path) -> None:
    """Tóm tắt training (in text lên figure) + lưu ảnh."""

    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    mae = history.history.get("mae", [])
    val_mae = history.history.get("val_mae", [])

    epochs = max(len(loss), len(val_loss))
    final_loss = loss[-1] if loss else None
    final_val_loss = val_loss[-1] if val_loss else None
    final_mae = mae[-1] if mae else None
    final_val_mae = val_mae[-1] if val_mae else None

    plt.figure(figsize=(12, 7))
    plt.axis("off")

    lines = [
        f"Epochs: {epochs}",
        f"Final Loss: {final_loss}",
        f"Final Val Loss: {final_val_loss}",
        f"Final MAE: {final_mae}",
        f"Final Val MAE: {final_val_mae}",
    ]

    plt.text(0.02, 0.95, "Training Summary", fontsize=18, weight="bold", va="top")

    y = 0.85
    for line in lines:
        plt.text(0.02, y, line, fontsize=14, va="top")
        y -= 0.07

    _finalize_plot(out_path)

