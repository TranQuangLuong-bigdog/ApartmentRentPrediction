"""Trang Model Manager (skeleton UI)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Hiển thị UI quản lý model."""

    st.header("Model Manager")
    st.info(
        "Skeleton UI: sẽ tích hợp chức năng lưu/xóa/đổi tên model, đặt model mặc định, "
        "import/export, so sánh model và hiển thị MAE/RMSE/accuracy theo phiên train."
    )

