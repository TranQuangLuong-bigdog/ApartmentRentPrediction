from pathlib import Path

from src.prediction.prediction_history import PredictionHistory


def test_history_append_and_load(tmp_path: Path):
    history_csv = tmp_path / "prediction_history.csv"
    h = PredictionHistory(history_csv=history_csv)

    h.append_prediction(
        session_id="s1",
        prediction_time_iso="t1",
        model_used="BEST",
        dataset_used=None,
        predictions=[1.0, 2.0],
    )

    rows = h.load_history()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"

    stats = h.statistics()
    assert stats["history_rows"] == 1

