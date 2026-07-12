import pytest

from ml_homework.calculus import forward_derivative


def test_forward_derivative_approximates_linear_slope() -> None:
    assert forward_derivative(lambda x: 3 * x - 2, 10) == pytest.approx(3)


def test_forward_derivative_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        forward_derivative(lambda x: x, 0, step=0)
