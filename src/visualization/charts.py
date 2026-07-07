"""Biểu đồ production-ready (wrapper)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.visualization import plot_training_history


def plot_loss_mae(history: Any, out_path: Path) -> None:
    """Vẽ biểu đồ Loss/MAE và lưu."""

    plot_training_history(history=history, out_path=out_path)

