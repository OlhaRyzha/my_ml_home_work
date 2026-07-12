import numpy as np
from numpy.typing import ArrayLike


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return RMSE for equally shaped numeric arrays."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if actual.size == 0:
        raise ValueError("input arrays must not be empty")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
