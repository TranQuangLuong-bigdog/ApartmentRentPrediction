"""Streamlit app cho Apartment Rent Prediction."""

from __future__ import annotations

import streamlit as st

from app.pages import About, Dashboard, Dataset, Home, Predict, Train_Model, Evaluate
from app.pages import Model_Manager, Prediction_History, Settings


def main() -> None:
    """Entry point Streamlit."""

    st.set_page_config(
        page_title="Apartment Rent Prediction (ANN)",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Apartment Rent Prediction (ANN)")

    st.sidebar.header("Menu")
    st.sidebar.radio(
        "Điều hướng",
        options=[
            "Home",
            "Dataset",
            "Dataset Explorer",
            "Train",
            "Predict",
            "Evaluation",
            "Model Manager",
            "Prediction History",
            "Settings",
            "About",
            "Dashboard",
        ],
    )




    try:
        if page == "Home":
            Home.render()
        elif page == "Dataset":
            Dataset.render()
        elif page == "Dataset Explorer":
            from app.pages import Dataset_Explorer

            Dataset_Explorer.render()
        elif page == "Train":
            Train_Model.render()
        elif page == "Predict":
            Predict.render()
        elif page == "Evaluation":
            Evaluate.render()
        elif page == "About":
            About.render()
        elif page == "Model Manager":
            Model_Manager.render()
        elif page == "Prediction History":
            Prediction_History.render()
        elif page == "Settings":
            Settings.render()
        elif page == "Dashboard":
            Dashboard.render()

    except Exception as exc:
        st.error(f"Có lỗi xảy ra: {exc}")



if __name__ == "__main__":
    main()


