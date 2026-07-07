from pathlib import Path

import numpy as np

# Ensure UTF-8 for Vietnamese UI/logs in some environments
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.config.config import get_project_paths, ensure_directories

from src.dataset.dataset_loader import DatasetLoader
from src.services.preprocessing_service import preprocess_for_training
from src.services.train_service import train_ann
from src.models.save_model import save_keras_model, save_training_history
from src.evaluation.evaluate import evaluate_prediction


def main():
    paths = get_project_paths()
    ensure_directories(paths)

    dataset_path = Path("data/apartment_rent.csv")

    loader = DatasetLoader()
    df = loader.load_csv(dataset_path)

    processed = preprocess_for_training(df)

    result = train_ann(
        X_train=processed.X_train,
        y_train=processed.y_train,
        input_dim=processed.X_train.shape[1],
    )

    y_pred = result.model.predict(processed.X_test, verbose=0).reshape(-1)

    metrics = evaluate_prediction(
        processed.y_test,
        y_pred,
    )

    print(metrics)

    save_keras_model(
    result.model,
    paths.trained_models / "ann_model.keras",
)

    save_training_history(
        result.history,
        paths.trained_models / "history.pkl",
    )


if __name__ == "__main__":
    main()