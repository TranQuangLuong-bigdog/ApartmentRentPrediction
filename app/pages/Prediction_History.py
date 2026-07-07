"""Trang Prediction History (skeleton UI)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Hiển thị lịch sử prediction."""

    st.header("Prediction History")
    st.info(
        "Skeleton UI: trong các bước sau sẽ tích hợp lưu prediction history "
        "(thời gian, input, prediction, model sử dụng) và cho phép search/filter/export/xóa."
    )

