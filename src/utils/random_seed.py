"""Thiết lập seed ngẫu nhiên để tái lập kết quả."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Đặt seed cho Python, NumPy và TensorFlow.

    Args:
        seed: Giá trị seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        # Giúp giảm độ bất định khi có GPU (tuỳ thuộc hệ thống).
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Không bắt buộc với mọi phiên bản TF.
            pass
    except ImportError:
        # Nếu TensorFlow chưa cài, bỏ qua phần TF.
        return

