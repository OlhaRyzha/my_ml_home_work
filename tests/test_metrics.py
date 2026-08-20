import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ml_homework.metrics import compare_regression_metrics, root_mean_squared_error


def test_root_mean_squared_error() -> None:
    assert root_mean_squared_error([1, 2, 3], [1, 2, 5]) == pytest.approx(
        np.sqrt(4 / 3)
    )


def test_root_mean_squared_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        root_mean_squared_error([1, 2], [1])


def test_compare_regression_metrics_builds_model_columns() -> None:
    train_inputs = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    train_targets = pd.Series([1.0, 3.0, 5.0])
    test_inputs = pd.DataFrame({"x": [3.0, 4.0]})
    test_targets = pd.Series([7.0, 9.0])
    model = Pipeline([("regressor", LinearRegression())]).fit(
        train_inputs, train_targets
    )

    result = compare_regression_metrics(
        {"Linear Regression": model},
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    )

    assert result.columns.tolist() == ["Linear Regression"]
    assert result.index.tolist() == [
        "Train RMSE",
        "Test RMSE",
        "Train R²",
        "Test R²",
    ]
    np.testing.assert_allclose(
        result["Linear Regression"],
        [0.0, 0.0, 1.0, 1.0],
        atol=1e-12,
    )
