import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_homework.classification import (
    add_age_group,
    build_ovr_logistic_pipeline,
    compare_classification_metrics,
    compare_multiclass_predictions,
    compute_auroc,
    compute_auroc_and_build_roc,
    evaluate_multiclass_model,
    get_f1_score,
    max_depth_auroc,
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


def test_compare_classification_metrics_builds_model_columns() -> None:
    inputs = pd.DataFrame({"signal": [-2.0, -1.0, 1.0, 2.0]})
    targets = pd.Series([0, 0, 1, 1])
    model = Pipeline([("classifier", LogisticRegression(random_state=42))]).fit(
        inputs, targets
    )

    result = compare_classification_metrics(
        {"Baseline": model},
        inputs,
        targets,
        inputs,
        targets,
    )

    assert result.columns.tolist() == ["Baseline"]
    assert result.index.tolist() == [
        "Train Accuracy",
        "Validation Accuracy",
        "Train F1",
        "Validation F1",
        "Train AUROC",
        "Validation AUROC",
    ]
    np.testing.assert_allclose(result["Baseline"], 1.0)


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
    assert capsys.readouterr().out.strip() == "Accuracy for Validation: 100.00%"
    assert axis.get_title() == "Validation Confusion Matrix"
    assert {text.get_text() for text in axis.texts} == {
        "TN\n100.0%",
        "FP\n0.0%",
        "FN\n0.0%",
        "TP\n100.0%",
    }
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
    assert capsys.readouterr().out.strip() == "AUROC for Validation: 1.0000"
    assert axis.get_xlabel() == "False Positive Rate"
    assert len(axis.lines) == 2
    plt.close(figure)


def test_compute_auroc_returns_score_without_plot(
    fitted_classifier: tuple[LogisticRegression, pd.DataFrame, pd.Series],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, inputs, targets = fitted_classifier
    plt.close("all")

    score = compute_auroc(model, inputs, targets, "Validation")

    assert score == pytest.approx(1.0)
    assert capsys.readouterr().out.strip() == "AUROC for Validation: 1.0000"
    assert plt.get_fignums() == []


def test_max_depth_auroc_returns_train_and_validation_scores(
    fitted_classifier: tuple[LogisticRegression, pd.DataFrame, pd.Series],
) -> None:
    _, inputs, targets = fitted_classifier

    result = max_depth_auroc(
        2,
        inputs,
        targets,
        inputs,
        targets,
    )

    assert result["Max Depth"] == 2
    assert result["Training AUROC"] == pytest.approx(1.0)
    assert result["Validation AUROC"] == pytest.approx(1.0)


def test_evaluate_multiclass_model_prints_metrics_and_confusion_matrices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    targets = pd.Series(["A", "A", "B", "B", "C", "C"])
    model = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))]).fit(
        inputs, targets
    )

    predictions = evaluate_multiclass_model(model, inputs, targets, "Validation")
    figure = plt.gcf()
    axis = plt.gca()
    output = capsys.readouterr().out

    np.testing.assert_array_equal(predictions, ["A"] * len(targets))
    assert "Validation Accuracy: 0.3333" in output
    assert "Validation Macro F1: 0.1667" in output
    assert "Validation Macro AUROC: 0.5000" in output
    assert "Validation Confusion Matrix — кількість:" in output
    assert "Validation Confusion Matrix — частка:" in output
    assert "Actual A" in output
    assert "Predicted C" in output
    assert axis.get_title() == "Validation Normalized Confusion Matrix"
    plt.close(figure)


def test_build_ovr_logistic_pipeline_supports_optional_preprocessing() -> None:
    raw_inputs = pd.DataFrame(
        {
            "numeric": [-3.0, -2.0, 0.0, 1.0, 3.0, 4.0],
            "category": ["A", "A", "B", "B", "C", "C"],
        }
    )
    targets = pd.Series(["A", "A", "B", "B", "C", "C"])
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["numeric"]),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore"),
                ["category"],
            ),
        ]
    )

    raw_model = build_ovr_logistic_pipeline(preprocessor).fit(raw_inputs, targets)
    numeric_inputs = pd.DataFrame({"numeric": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})
    numeric_model = build_ovr_logistic_pipeline().fit(numeric_inputs, targets)

    assert raw_model.predict(raw_inputs).shape == (len(targets),)
    assert numeric_model.predict(numeric_inputs).shape == (len(targets),)
    np.testing.assert_array_equal(raw_model.classes_, ["A", "B", "C"])
    np.testing.assert_array_equal(numeric_model.classes_, ["A", "B", "C"])
    assert not hasattr(preprocessor, "transformers_")


def test_compare_multiclass_predictions_builds_model_columns() -> None:
    targets = pd.Series(["A", "A", "B", "B", "C", "C"])
    predictions = {
        "Perfect": np.array(["A", "A", "B", "B", "C", "C"], dtype=object),
        "Imperfect": np.array(["A", "A", "A", "B", "C", "B"], dtype=object),
    }

    result = compare_multiclass_predictions(predictions, targets)

    assert result.columns.tolist() == ["Perfect", "Imperfect"]
    assert result.index.tolist() == [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]
    np.testing.assert_allclose(result["Perfect"], 1.0)
    assert result.loc["Macro F1", "Imperfect"] == pytest.approx(0.6555556)


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
