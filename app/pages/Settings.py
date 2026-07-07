"""Trang Settings (skeleton UI)."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Hiển thị UI cấu hình."""

    st.header("Settings")
    st.info(
        "Skeleton UI: sẽ tích hợp Theme (dark/light), language, default hyperparameters "
        "(epoch/batch size/learning rate), Auto Save/Load, Clear Cache, Reset Settings."
    )

