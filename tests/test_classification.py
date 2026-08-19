import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_homework.classification import (
    add_age_group,
    compute_auroc_and_build_roc,
    get_f1_score,
    predict_and_plot,
    predict_majority_class,
    predict_raw_df,
)


def test_add_age_group_uses_fixed_boundaries_without_mutating_input() -> None:
    inputs = pd.DataFrame({"Age": [30, 31, 35, 36, 60, 61]})

    result = add_age_group(inputs)

    assert result["AgeGroup"].astype(str).tolist() == [
        "up_to_30",
        "31_35",
        "31_35",
        "36_40",
        "51_60",
        "over_60",
    ]
    assert "AgeGroup" not in inputs.columns


@pytest.fixture
def fitted_classifier() -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    inputs = pd.DataFrame(
        {
            "signal": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "noise": [0.1, 0.2, 0.0, 0.0, 0.2, 0.1],
        }
    )
    targets = pd.Series([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(random_state=42).fit(inputs, targets)
    return model, inputs, targets


def test_predict_and_plot_returns_predictions_and_labeled_plot(
    fitted_classifier: tuple[LogisticRegression, pd.DataFrame, pd.Series],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, inputs, targets = fitted_classifier

    predictions = predict_and_plot(model, inputs, targets, "Validation")
    figure = plt.gcf()
    axis = plt.gca()

    np.testing.assert_array_equal(predictions, targets)
    assert capsys.readouterr().out.strip() == "Accuracy: 100.00%"
    assert axis.get_title() == "Validation Confusion Matrix"
    plt.close(figure)


def test_get_f1_score_returns_score(
    fitted_classifier: tuple[LogisticRegression, pd.DataFrame, pd.Series],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, inputs, targets = fitted_classifier

    score = get_f1_score(model, inputs, targets, "Validation")

    assert score == pytest.approx(1.0)
    assert capsys.readouterr().out.strip() == "F1 score Validation: 100.00%"


def test_compute_auroc_and_build_roc_returns_score_and_plot(
    fitted_classifier: tuple[LogisticRegression, pd.DataFrame, pd.Series],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, inputs, targets = fitted_classifier

    score = compute_auroc_and_build_roc(model, inputs, targets, "Validation")
    figure = plt.gcf()
    axis = plt.gca()

    assert score == pytest.approx(1.0)
    assert capsys.readouterr().out.strip() == "AUROC for Validation: 1.00"
    assert axis.get_xlabel() == "False Positive Rate"
    assert len(axis.lines) == 2
    plt.close(figure)


def test_predict_majority_class_uses_input_length() -> None:
    predictions = predict_majority_class(pd.DataFrame({"x": [1, 2, 3]}), 0)

    np.testing.assert_array_equal(predictions, [0, 0, 0])
    assert predictions.dtype == np.int64


def test_predict_raw_df_transforms_copy_and_preserves_probability_precision() -> None:
    raw_inputs = pd.DataFrame(
        {
            "numeric": [-2.0, -1.0, 1.0, 2.0],
            "category": ["A", "A", "B", "B"],
        }
    )
    targets = pd.Series([0, 0, 1, 1])
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["numeric"]),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["category"],
            ),
        ]
    ).fit(raw_inputs)
    transformed = preprocessor.transform(raw_inputs)
    model = LogisticRegression(random_state=42).fit(transformed, targets)

    probabilities = predict_raw_df(preprocessor, model, raw_inputs)

    expected = model.predict_proba(transformed)[:, 1]
    np.testing.assert_allclose(probabilities, expected)
    assert np.any(probabilities != probabilities.round(2))
    assert raw_inputs.columns.tolist() == ["numeric", "category"]
