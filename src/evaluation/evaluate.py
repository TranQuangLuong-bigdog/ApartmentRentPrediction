"""Evaluation layer cho ANN.

Giai đoạn 5: compute metrics, generate figures, and create report artifacts.

Các hàm public:
- evaluate_model()
- evaluate_prediction()
- save_metrics()
- generate_report()
- export_figures()

Tất cả hàm trả về dict (không print).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

from src.config.config import ProjectPaths, get_project_paths
from src.evaluation.visualization import (
    plot_boxplot,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_distribution,
    plot_learning_curve,
    plot_loss,
    plot_prediction_vs_actual,
    plot_residual_distribution,
    plot_residuals,
    plot_scatter_prediction,
    plot_training_summary,
)

from src.evaluation.visualization import plot_heatmap  # type: ignore

from src.models.metrics import compute_metrics

import matplotlib

matplotlib.use("Agg")


@dataclass(frozen=True)
class EvaluationResult:
    """Kết quả evaluation (metrics + extras)."""

    metrics: Dict[str, float]
    residuals: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def _safe_overwrite(path: Path, write_fn: Any) -> None:
    """Ghi file an toàn (tạo parent; ghi đè)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_fn(path)


def _compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính thêm các metrics yêu cầu ngoài MAE/MSE/RMSE/R2."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    eps = 1e-8
    residuals = y_true - y_pred

    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE
    mape = float(np.mean(np.abs(residuals) / (np.abs(y_true) + eps)) * 100.0)

    # Explained variance
    # sklearn: explained_variance_score
    var_y_true = float(np.var(y_true))
    explained_variance = 1.0 - (float(np.var(residuals)) / (var_y_true + eps))

    median_absolute_error = float(np.median(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "ExplainedVariance": explained_variance,
        "MedianAbsoluteError": median_absolute_error,
        "MaxError": max_error,
    }


def evaluate_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Evaluate prediction: trả về dict metrics."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # Dùng compute_metrics có sẵn (MAE/MSE/RMSE/R2)
    base = compute_metrics(y_true=y_true, y_pred=y_pred)
    base_dict = {"MAE": base.mae, "MSE": base.mse, "RMSE": base.rmse, "R2": base.r2}

    extra = _compute_additional_metrics(y_true=y_true, y_pred=y_pred)

    # Gộp: đảm bảo MAE/MSE/RMSE không bị lệch (base đã dùng sklearn)
    base_dict.update({
        "MAPE": extra["MAPE"],
        "ExplainedVariance": extra["ExplainedVariance"],
        "MedianAbsoluteError": extra["MedianAbsoluteError"],
        "MaxError": extra["MaxError"],
    })

    return base_dict


def evaluate_model(
    *,
    model: Any,
    history: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_size: int,
    train_size: int,
    test_size: int,
    training_time_seconds: float,
    paths: Optional[ProjectPaths] = None,
    model_file_path: Optional[Path] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    optimizer_name: Optional[str] = None,
    activation: Optional[str] = None,
    loss_name: Optional[str] = None,
    final_loss: Optional[float] = None,
    final_val_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate model và tạo dict chứa metrics + metadata."""

    metrics = evaluate_prediction(y_true=y_true, y_pred=y_pred)

    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)

    result: Dict[str, Any] = {
        "metrics": metrics,
        "residuals": residuals,
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
        "model_metadata": {
            "ModelName": getattr(model, "name", "unknown"),
            "TrainingTimeSeconds": training_time_seconds,
            "Epochs": epochs,
            "BatchSize": batch_size,
            "LearningRate": learning_rate,
            "Optimizer": optimizer_name,
            "Activation": activation,
            "Loss": loss_name,
            "FinalLoss": final_loss,
            "ValidationLossFinal": final_val_loss,
            "DatasetSize": dataset_size,
            "TrainSize": train_size,
            "TestSize": test_size,
            "ModelFileSizeBytes": model_file_path.stat().st_size if model_file_path and model_file_path.exists() else None,
        },
    }

    return result


def save_metrics(metrics: Dict[str, float], *, paths: ProjectPaths) -> Dict[str, str]:
    """Lưu metrics vào output/report."""

    report_dir = paths.output_reports
    metrics_json = report_dir / "metrics.json"
    metrics_csv = report_dir / "metrics.csv"
    metrics_txt = report_dir / "metrics.txt"

    def _write_json(p: Path) -> None:
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_csv(p: Path) -> None:
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        df.to_csv(p, index=False)

    def _write_txt(p: Path) -> None:
        lines = [f"{k}: {v}" for k, v in metrics.items()]
        p.write_text("\n".join(lines), encoding="utf-8")

    _safe_overwrite(metrics_json, _write_json)
    _safe_overwrite(metrics_csv, _write_csv)
    _safe_overwrite(metrics_txt, _write_txt)

    return {
        "metrics.json": str(metrics_json),
        "metrics.csv": str(metrics_csv),
        "metrics.txt": str(metrics_txt),
    }


def export_figures(
    *,
    history: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    features_df: Optional[pd.DataFrame],
    paths: ProjectPaths,
) -> Dict[str, str]:
    """Export các figures theo danh sách yêu cầu."""

    figs_dir = paths.output_figures

    out_files = {
        "loss_curve.png": figs_dir / "loss_curve.png",
        "mae_curve.png": figs_dir / "mae_curve.png",
        "prediction_vs_actual.png": figs_dir / "prediction_vs_actual.png",
        "residual_plot.png": figs_dir / "residual_plot.png",
        "residual_distribution.png": figs_dir / "residual_distribution.png",
        "correlation_heatmap.png": figs_dir / "correlation_heatmap.png",
        "feature_distribution.png": figs_dir / "feature_distribution.png",
        "boxplot.png": figs_dir / "boxplot.png",
        "scatter_prediction.png": figs_dir / "scatter_prediction.png",
        "learning_curve.png": figs_dir / "learning_curve.png",
        "training_summary.png": figs_dir / "training_summary.png",
    }

    # 1) Loss curve + learning curve
    if hasattr(history, "history"):
        plot_loss(history=history, out_path=out_files["loss_curve.png"])
        plot_learning_curve(history=history, out_path=out_files["learning_curve.png"])
        plot_training_summary(history=history, out_path=out_files["training_summary.png"])

        # MAE curve: dùng plot_loss nhưng với mae/val_mae thì hiện tại Visualization chưa có riêng
        # Tạm vẽ dựa trên history (inline nhỏ để không duplicate code quá nhiều)
        import matplotlib.pyplot as plt
        import numpy as np

        mae = history.history.get("mae", [])
        val_mae = history.history.get("val_mae", [])
        epochs = np.arange(len(mae))
        plt.figure(figsize=(10, 6))
        if len(mae) > 0:
            plt.plot(epochs, mae, label="MAE")
        if len(val_mae) > 0:
            plt.plot(epochs, val_mae, label="Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        out_files["mae_curve.png"].parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_files["mae_curve.png"], dpi=300)
        plt.close()

    # 2) Prediction vs actual
    plot_prediction_vs_actual(y_true=y_true, y_pred=y_pred, out_path=out_files["prediction_vs_actual.png"])
    plot_scatter_prediction(y_true=y_true, y_pred=y_pred, out_path=out_files["scatter_prediction.png"])

    # 3) Residual
    plot_residuals(y_true=y_true, y_pred=y_pred, out_path=out_files["residual_plot.png"])
    plot_residual_distribution(y_true=y_true, y_pred=y_pred, out_path=out_files["residual_distribution.png"])

    # 4) Boxplot residual
    plot_boxplot(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1), out_path=out_files["boxplot.png"])

    # 5) Correlation heatmap / feature distribution
    if features_df is not None and not features_df.empty:
        corr = features_df.corr(numeric_only=True).values
        plot_correlation_heatmap(
            corr=corr,
            out_path=out_files["correlation_heatmap.png"],
            labels=list(features_df.columns)[: corr.shape[0]],
        )

        # Feature distribution: lấy feature đầu tiên
        first_col = str(features_df.columns[0])
        plot_feature_distribution(features_df[first_col].values, out_path=out_files["feature_distribution.png"], feature_name=first_col)
    else:
        # không có features_df thì không crash
        # tạo figure trống
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No feature data", ha="center")
        plt.savefig(out_files["correlation_heatmap.png"], dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No feature data", ha="center")
        plt.savefig(out_files["feature_distribution.png"], dpi=300)
        plt.close()

    return {k: str(v) for k, v in out_files.items()}


def generate_report(
    *,
    eval_result: Dict[str, Any],
    history: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: ProjectPaths,
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """Tạo đầy đủ report artifacts theo yêu cầu."""

    report_dir = paths.output_reports

    metrics: Dict[str, float] = eval_result["metrics"]
    model_metadata: Dict[str, Any] = eval_result["model_metadata"]

    # metrics
    save_metrics(metrics=metrics, paths=paths)

    # summary.md
    summary_path = report_dir / "summary.md"

    def _write_summary(p: Path) -> None:
        lines = ["# Evaluation Summary", "", "## Metrics"]
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Model Information"]
        for k, v in model_metadata.items():
            lines.append(f"- {k}: {v}")
        p.write_text("\n".join(lines), encoding="utf-8")

    _safe_overwrite(summary_path, _write_summary)

    # model_information.json
    model_info_path = report_dir / "model_information.json"
    _safe_overwrite(model_info_path, lambda p: p.write_text(json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8"))

    # history.json
    history_path = report_dir / "history.json"
    hist_dict = history.history if hasattr(history, "history") else {}
    _safe_overwrite(history_path, lambda p: p.write_text(json.dumps(hist_dict, indent=2, ensure_ascii=False), encoding="utf-8"))

    # prediction_result.csv
    pred_csv_path = report_dir / "prediction_result.csv"
    pred_df = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1), "y_pred": np.asarray(y_pred).reshape(-1)})
    _safe_overwrite(pred_csv_path, lambda p: pred_df.to_csv(p, index=False))

    # evaluation.xlsx
    excel_path = report_dir / "evaluation.xlsx"
    # Ghi 2 sheet: metrics + predictions
    metrics_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
    pred_df = pd.DataFrame({"y_true": np.asarray(y_true).reshape(-1), "y_pred": np.asarray(y_pred).reshape(-1)})
    with pd.ExcelWriter(excel_path) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        pred_df.to_excel(writer, sheet_name="predictions", index=False)

    # metrics.json already written by save_metrics
    metrics_json_path = report_dir / "metrics.json"

    return {
        "summary.md": str(summary_path),
        "model_information.json": str(model_info_path),
        "history.json": str(history_path),
        "prediction_result.csv": str(pred_csv_path),
        "evaluation.xlsx": str(excel_path),
        "metrics.json": str(metrics_json_path),
    }

