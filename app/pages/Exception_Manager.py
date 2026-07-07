"""Trang Exception Manager (skeleton UI)."""

from __future__ import annotations

import streamlit as st

from src.exceptions.exception_manager import handle_exception


def render() -> None:
    """Hiển thị UI mô tả cách quản lý lỗi."""

    st.header("Exception Manager")
    st.info(
        "Trang skeleton. Backend xử lý lỗi bằng exception_manager sẽ được sử dụng khi tích hợp các bước Train/Predict thực tế."
    )

    if st.button("Demo: tạo lỗi và hiển thị thông báo thân thiện"):
        try:
            raise FileNotFoundError("No such file")
        except Exception as exc:
            friendly = handle_exception(exc)
            st.error(f"{friendly.title}: {friendly.message}")

