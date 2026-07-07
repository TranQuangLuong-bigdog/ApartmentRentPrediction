"""Trang AI Assistant (placeholder)."""

from __future__ import annotations

import streamlit as st

from src.assistant.assistant import explain_training_result


def render() -> None:
    """Hiển thị trang AI Assistant."""

    st.header("AI Assistant")
    st.info(
        "Trang này sẽ hiển thị giải thích kết quả train/evaluate theo ngôn ngữ dễ hiểu. "
        "Hiện tại là skeleton để tích hợp với backend sau."
    )

    # Example placeholders
    if st.button("Demo giải thích metrics (MAE/RMSE/R²)"):
        out = explain_training_result(
            dataset_info={"rows": 100, "features": 10},
            train_info={"epochs": 100, "batch_size": 32, "learning_rate": 0.001},
            eval_metrics={"MAE": 100.0, "MSE": 20000.0, "RMSE": 141.42, "R2": 0.35},
            history={"loss": [1.2, 0.8], "val_loss": [1.3, 1.0], "mae": [0.9, 0.6], "val_mae": [1.0, 0.8]},
        )

        st.subheader("Giải thích chỉ số")
        for k, v in out.explanations.items():
            st.markdown(f"**{k}:** {v}")

        st.subheader("Nhận xét mô hình")
        st.write(out.model_comments)

        st.subheader("Gợi ý cải thiện")
        st.write(out.improvement_suggestions)

