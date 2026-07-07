"""Service cho AI Assistant: sinh giải thích dễ hiểu cho người dùng."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.assistant.assistant_prompt import (
    EXPLAIN_MAE_TEMPLATE,
    EXPLAIN_RMSE_TEMPLATE,
    EXPLAIN_R2_TEMPLATE,
)


@dataclass(frozen=True)
class AssistantInput:
    """Input cho assistant."""

    dataset_info: Dict[str, Any]
    train_info: Dict[str, Any]
    eval_metrics: Dict[str, float]
    history: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class AssistantOutput:
    """Output text giải thích."""

    explanations: Dict[str, str]
    model_comments: str
    improvement_suggestions: str


def _analyze_over_under_fitting(history: Optional[Dict[str, Any]]) -> str:
    """Phân tích overfitting/underfitting dựa trên train/val loss.

    Lưu ý: chỉ là heuristic, không thay thế đánh giá chuyên sâu.
    """

    if not history:
        return "Chưa có history để phân tích overfitting/underfitting."

    loss = history.get("loss") or []
    val_loss = history.get("val_loss") or []
    mae = history.get("mae") or []
    val_mae = history.get("val_mae") or []

    if len(loss) < 2 or len(val_loss) < 2:
        return "Dữ liệu history quá ngắn để phân tích xu hướng."

    # So sánh xu hướng cuối cùng
    loss_end = loss[-1]
    val_loss_end = val_loss[-1]

    # Heuristic đơn giản
    if val_loss_end > loss_end * 1.15:
        return (
            "Có dấu hiệu **overfitting**: loss train giảm/nhỏ hơn rõ rệt so với validation loss. "
            "Mô hình có thể học quá kỹ dữ liệu huấn luyện và kém tổng quát."
        )

    # Underfitting: cả train và val đều cao (heuristic theo mức)
    if loss_end > 1.0 and val_loss_end > 1.0:
        return (
            "Có dấu hiệu **underfitting**: cả train loss và validation loss đều còn cao. "
            "Mô hình có thể chưa đủ năng lực hoặc dữ liệu/feature chưa tốt."
        )

    return (
        "Mô hình có vẻ **phù hợp**: train và validation loss không chênh lệch quá lớn. "
        "Cần thêm biểu đồ/lần chạy khác để xác nhận chắc chắn."
    )


def generate_assistant_report(inp: AssistantInput) -> AssistantOutput:
    """Sinh báo cáo giải thích dễ hiểu."""

    mae = float(inp.eval_metrics.get("MAE", 0.0))
    rmse = float(inp.eval_metrics.get("RMSE", 0.0))
    r2 = float(inp.eval_metrics.get("R2", inp.eval_metrics.get("R2 Score", 0.0)))

    explanations: Dict[str, str] = {
        "MAE": f"MAE = {mae:.6f}. {EXPLAIN_MAE_TEMPLATE}",
        "RMSE": f"RMSE = {rmse:.6f}. {EXPLAIN_RMSE_TEMPLATE}",
        "R2": f"R² = {r2:.6f}. {EXPLAIN_R2_TEMPLATE}",
        "Dataset": (
            "Thông tin dataset: "
            f"{inp.dataset_info}. "
            "Nếu dataset nhỏ hoặc có nhiều giá trị thiếu/ngoại lệ, mô hình thường khó đạt R² cao."
        ),
    }

    overfit_comment = _analyze_over_under_fitting(inp.history)

    model_comments = (
        "Nhận xét mô hình (dễ hiểu):\n"
        f"- {overfit_comment}\n"
        "- Nếu R² thấp, mô hình chưa nắm được mối quan hệ giữa đặc trưng và giá thuê."
    )

    improvement_suggestions = (
        "Gợi ý cải thiện (ưu tiên theo mức tác động lớn):\n"
        "1) Kiểm tra preprocessing: encoding/scaling có đúng với training và inference không.\n"
        "2) Thử tăng/giảm kiến trúc ANN (hidden layers / neurons) hoặc đổi dropout.\n"
        "3) Thử optimizer/learning rate khác, áp dụng EarlyStopping.\n"
        "4) Làm sạch outliers và/hoặc cân bằng dữ liệu nếu có bất thường.\n"
        "5) Nếu dữ liệu nhiều cột categorical, thử thêm feature engineering (ví dụ grouping, target encoding nếu phù hợp)."
    )

    return AssistantOutput(
        explanations=explanations,
        model_comments=model_comments,
        improvement_suggestions=improvement_suggestions,
    )

