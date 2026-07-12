import pytest

from ml_homework.optimization import full_batch_gradient_descent


def test_gradient_descent_fits_simple_line() -> None:
    slope, intercept, errors = full_batch_gradient_descent(
        [0, 1, 2, 3], [1, 3, 5, 7], learning_rate=0.1, epochs=1000
    )
    assert slope == pytest.approx(2, abs=1e-4)
    assert intercept == pytest.approx(1, abs=1e-4)
    assert errors[-1] < errors[0]
