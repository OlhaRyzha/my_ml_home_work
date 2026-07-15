import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from ml_homework.modeling import (
    linear_regression_coefficients,
    select_feature_columns,
    significant_ols_coefficients,
    train_linear_regression_show_rmse,
    train_ols_show_summary,
)


def test_select_feature_columns_preserves_order_and_indices() -> None:
    train_inputs = pd.DataFrame(
        {"first": [1.0, 2.0], "second": [3.0, 4.0]}, index=[10, 11]
    )
    test_inputs = pd.DataFrame({"first": [5.0], "second": [6.0]}, index=[20])

    train_selected, test_selected = select_feature_columns(
        train_inputs,
        test_inputs,
        ["second", "first"],
    )

    assert train_selected.columns.tolist() == ["second", "first"]
    assert test_selected.columns.tolist() == ["second", "first"]
    assert train_selected.index.tolist() == [10, 11]
    assert test_selected.index.tolist() == [20]


def test_train_linear_regression_show_rmse_returns_model_and_predictions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_inputs = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    train_targets = pd.Series([1.0, 3.0, 5.0])
    test_inputs = pd.DataFrame({"x": [3.0, 4.0]})
    test_targets = pd.Series([7.0, 9.0])

    model, train_predictions, test_predictions = train_linear_regression_show_rmse(
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
    )

    assert isinstance(model, LinearRegression)
    np.testing.assert_allclose(train_predictions, train_targets)
    np.testing.assert_allclose(test_predictions, test_targets)
    output_lines = capsys.readouterr().out.splitlines()
    train_label, train_rmse = output_lines[0].split(": ")
    test_label, test_rmse = output_lines[1].split(": ")
    assert train_label == "Train RMSE"
    assert test_label == "Test RMSE"
    assert float(train_rmse) == pytest.approx(0.0, abs=1e-12)
    assert float(test_rmse) == pytest.approx(0.0, abs=1e-12)


def test_train_ols_show_summary_adds_constant_and_returns_fitted_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_inputs = pd.DataFrame({"x": np.arange(8, dtype=float)})
    train_targets = pd.Series([1.0, 3.1, 4.9, 7.2, 9.1, 11.0, 13.2, 14.9])
    test_inputs = pd.DataFrame({"x": [8.0, 9.0]})

    results, train_with_constant, test_with_constant = train_ols_show_summary(
        train_inputs,
        train_targets,
        test_inputs,
    )

    assert train_with_constant.columns.tolist() == ["const", "x"]
    assert test_with_constant.columns.tolist() == ["const", "x"]
    assert results.model.exog_names == ["const", "x"]
    assert "OLS Regression Results" in capsys.readouterr().out


def test_linear_regression_coefficients_sorts_by_absolute_value() -> None:
    inputs = pd.DataFrame({"small": [0.0, 1.0, 0.0], "large": [0.0, 0.0, 1.0]})
    targets = pd.Series([2.0, 3.0, -2.0])
    model = LinearRegression().fit(inputs, targets)

    result = linear_regression_coefficients(model, inputs.columns)

    assert result["feature"].tolist() == ["large", "small"]
    assert result["coefficient"].abs().is_monotonic_decreasing


def test_linear_regression_coefficients_can_include_intercept() -> None:
    inputs = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    targets = pd.Series([1.0, 3.0, 5.0])
    model = LinearRegression().fit(inputs, targets)

    result = linear_regression_coefficients(
        model,
        inputs.columns,
        include_intercept=True,
        sort_by_magnitude=False,
    )

    assert result["feature"].tolist() == ["x", "intercept"]
    np.testing.assert_allclose(result["coefficient"], [2.0, 1.0])


def test_significant_ols_coefficients_uses_default_alpha_and_excludes_constant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = sm.add_constant(pd.DataFrame({"signal": np.arange(10, dtype=float)}))
    targets = pd.Series([0.2, 2.9, 6.1, 8.8, 12.2, 15.1, 17.9, 21.2, 24.1, 26.8])
    results = sm.OLS(targets, inputs).fit()

    table, feature_names = significant_ols_coefficients(results)

    assert feature_names == ["signal"]
    assert table.index.tolist() == ["signal"]
    assert np.asarray(table["p_value"], dtype=float)[0] < 0.05
    assert capsys.readouterr().out.strip() == ("Статистично значущі ознаки: ['signal']")
