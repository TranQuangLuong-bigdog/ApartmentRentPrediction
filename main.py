

from __future__ import annotations

from src.config.config import get_config
from src.models.train_model import train_and_evaluate


def main() -> None:
    """Run training and evaluation pipeline."""
    cfg = get_config()
    train_and_evaluate(cfg)


if __name__ == "__main__":
    main()

