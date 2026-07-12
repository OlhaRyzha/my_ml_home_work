import numpy as np
import pytest

from ml_homework.metrics import root_mean_squared_error


def test_root_mean_squared_error() -> None:
    assert root_mean_squared_error([1, 2, 3], [1, 2, 5]) == pytest.approx(
        np.sqrt(4 / 3)
    )


def test_root_mean_squared_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        root_mean_squared_error([1, 2], [1])
