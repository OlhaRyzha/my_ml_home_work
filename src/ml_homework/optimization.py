import numpy as np
from numpy.typing import ArrayLike

from ml_homework.metrics import root_mean_squared_error


def full_batch_gradient_descent(
    features: ArrayLike,
    targets: ArrayLike,
    learning_rate: float = 0.0001,
    epochs: int = 1000,
) -> tuple[float, float, list[float]]:
    """Fit a univariate linear model using full-batch gradient descent."""
    x_values = np.asarray(features, dtype=float)
    y_values = np.asarray(targets, dtype=float).reshape(-1)

    if x_values.ndim == 2 and x_values.shape[1] == 1:
        x_values = x_values[:, 0]
    elif x_values.ndim != 1:
        raise ValueError("features must be one-dimensional or have shape (n, 1)")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")
    if x_values.size == 0:
        raise ValueError("features and targets must not be empty")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    slope = 0.0
    intercept = 0.0
    errors: list[float] = []
    sample_count = y_values.size

    for _ in range(epochs):
        predictions = slope * x_values + intercept
        residuals = predictions - y_values
        errors.append(root_mean_squared_error(y_values, predictions))

        slope -= learning_rate * (2 / sample_count) * np.dot(residuals, x_values)
        intercept -= learning_rate * (2 / sample_count) * np.sum(residuals)

    return float(slope), float(intercept), errors
