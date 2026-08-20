import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return RMSE for equally shaped numeric arrays."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if actual.size == 0:
        raise ValueError("input arrays must not be empty")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compare_regression_metrics(
    models: dict[str, Pipeline],
    train_inputs: pd.DataFrame,
    train_targets: pd.Series,
    test_inputs: pd.DataFrame,
    test_targets: pd.Series,
) -> pd.DataFrame:
    """Compare train and test RMSE and R² for fitted regression pipelines."""
    comparison = {}

    for name, model in models.items():
        train_predictions = model.predict(train_inputs)
        test_predictions = model.predict(test_inputs)

        comparison[name] = [
            root_mean_squared_error(train_targets, train_predictions),
            root_mean_squared_error(test_targets, test_predictions),
            r2_score(train_targets, train_predictions),
            r2_score(test_targets, test_predictions),
        ]

    return pd.DataFrame(
        comparison,
        index=["Train RMSE", "Test RMSE", "Train R²", "Test R²"],
    )
