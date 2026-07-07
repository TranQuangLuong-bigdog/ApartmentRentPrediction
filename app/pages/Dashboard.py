"""Trang Dashboard (placeholder UI)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Hiển thị dashboard."""
    st.header("Dashboard")

    st.subheader("Các chỉ số hệ thống/experiment (skeleton)")
    st.caption("Bản skeleton hiển thị placeholder. Giai đoạn sau sẽ gắn dữ liệu từ pipeline/train history.")

    cols = st.columns(4)
    for c in cols:
        c.metric(label="Metric", value="-", delta="")

    st.divider()

    st.subheader("Biểu đồ/Trạng thái (skeleton)")
    st.write("Tổng dữ liệu, missing values, hyperparameters, loss/val_loss cuối, model status, prediction count...")

