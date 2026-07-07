from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ConfigInfo:
    """Thông tin cấu hình huấn luyện."""

    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    validation_split: float
    random_seed: int

    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "random_seed": self.random_seed,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConfigInfo":
        return ConfigInfo(
            epochs=int(data.get("epochs")),
            batch_size=int(data.get("batch_size")),
            optimizer=str(data.get("optimizer")),
            learning_rate=float(data.get("learning_rate")),
            validation_split=float(data.get("validation_split")),
            random_seed=int(data.get("random_seed")),
            extra=data.get("extra"),
        )

