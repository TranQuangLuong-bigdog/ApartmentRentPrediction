"""Facade cho AI Assistant."""

from __future__ import annotations

from typing import Dict, Any, Optional

from src.assistant.assistant_service import AssistantInput, AssistantOutput, generate_assistant_report


def explain_training_result(
    dataset_info: Dict[str, Any],
    train_info: Dict[str, Any],
    eval_metrics: Dict[str, float],
    history: Optional[Dict[str, Any]] = None,
) -> AssistantOutput:
    """Giải thích kết quả train/evaluate cho người không chuyên."""

    inp = AssistantInput(
        dataset_info=dataset_info,
        train_info=train_info,
        eval_metrics=eval_metrics,
        history=history,
    )

    return generate_assistant_report(inp)

