import numpy as np
import pytest


def test_placeholder() -> None:
    """Placeholder test to ensure pytest discovery works."""
    assert np.array([1, 2, 3]).sum() == 6

