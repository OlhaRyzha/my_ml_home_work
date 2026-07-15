from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper

from ml_homework.metrics import root_mean_squared_error

FeatureMatrix = pd.DataFrame | NDArray[np.float64]
TargetVector = pd.Series | NDArray[np.float64]
FeatureNames = Iterable[str]


def select_feature_columns(
    train_inputs: pd.DataFrame,
    test_inputs: pd.DataFrame,
    feature_names: FeatureNames,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the same ordered feature columns from train and test data."""
    names = list(feature_names)
    return train_inputs[names], test_inputs[names]


def train_linear_regression_show_rmse(
    train_inputs: FeatureMatrix,
    train_targets: TargetVector,
    test_inputs: FeatureMatrix,
    test_targets: TargetVector,
) -> tuple[LinearRegression, NDArray[np.float64], NDArray[np.float64]]:
    """Fit linear regression, print train/test RMSE, and return predictions."""
    model = LinearRegression()
    model.fit(train_inputs, train_targets)

    train_predictions = np.asarray(model.predict(train_inputs), dtype=float)
    test_predictions = np.asarray(model.predict(test_inputs), dtype=float)

    train_rmse = root_mean_squared_error(train_targets, train_predictions)
    test_rmse = root_mean_squared_error(test_targets, test_predictions)

    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)

    return model, train_predictions, test_predictions


def train_ols_show_summary(
    train_inputs: pd.DataFrame,
    train_targets: TargetVector,
    test_inputs: pd.DataFrame,
) -> tuple[RegressionResultsWrapper, pd.DataFrame, pd.DataFrame]:
    """Fit OLS, print its summary, and return inputs with a constant."""
    train_with_constant = sm.add_constant(train_inputs, has_constant="add")
    test_with_constant = sm.add_constant(test_inputs, has_constant="add")

    results = sm.OLS(train_targets, train_with_constant).fit()
    print(results.summary())

    return results, train_with_constant, test_with_constant


def linear_regression_coefficients(
    model: LinearRegression,
    feature_names: FeatureNames,
    *,
    include_intercept: bool = False,
    sort_by_magnitude: bool = True,
) -> pd.DataFrame:
    """Return a coefficient table for a fitted single-target linear model."""
    names = list(feature_names)
    coefficients = np.asarray(model.coef_, dtype=float)

    if include_intercept:
        names.append("intercept")
        coefficients = np.append(coefficients, float(model.intercept_))

    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    if sort_by_magnitude:
        order = table["coefficient"].abs().sort_values(ascending=False).index
        table = table.loc[order]

    return table.reset_index(drop=True)


def significant_ols_coefficients(
    results: RegressionResultsWrapper,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, list[str]]:
    """Return and print OLS coefficients whose p-values are below alpha."""
    table = pd.DataFrame(
        {
            "coefficient": results.params,
            "p_value": results.pvalues,
        }
    )

    significant = table.loc[
        (table["p_value"] < alpha) & (table.index != "const")
    ].sort_values("p_value")

    feature_names = [str(name) for name in significant.index]
    print("Статистично значущі ознаки:", feature_names)

    return significant, feature_names
