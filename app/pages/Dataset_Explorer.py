"""Trang Dataset Explorer (skeleton UI)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Hiển thị UI khám phá dataset."""

    st.header("Dataset Explorer")
    st.info(
        "Skeleton UI: ở phiên bản tiếp theo sẽ tích hợp upload CSV, phân trang, thống kê missing/outlier/correlation và biểu đồ."
    )

    st.subheader("Chức năng dự kiến")
    st.markdown(
        """
- Hiển thị bảng dữ liệu
- Search / Filter / Sort / Pagination
- Download CSV
- Missing values, Outlier
- Correlation matrix
- Histogram, Box plot, Pair plot, Distribution
        """
    )

