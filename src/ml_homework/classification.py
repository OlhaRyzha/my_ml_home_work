"""Reusable helpers for evaluating classification models."""

from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class _ProbabilityClassifier(Protocol):
    """Classifier interface required by binary probability metrics."""

    def predict_proba(self, inputs: pd.DataFrame, /) -> NDArray[np.float64]: ...


def add_age_group(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with age encoded into fixed, interpretable groups."""
    result = data.copy()
    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[-np.inf, 30, 35, 40, 45, 50, 60, np.inf],
        labels=["up_to_30", "31_35", "36_40", "41_45", "46_50", "51_60", "over_60"],
    )
    return result


def compare_classification_metrics(
    models: dict[str, Pipeline],
    train_inputs: pd.DataFrame,
    train_targets: pd.Series,
    val_inputs: pd.DataFrame,
    val_targets: pd.Series,
) -> pd.DataFrame:
    """Compare train and validation metrics for fitted classifiers."""
    comparison = {}

    for name, model in models.items():
        train_predictions = model.predict(train_inputs)
        val_predictions = model.predict(val_inputs)
        train_probabilities = model.predict_proba(train_inputs)[:, 1]
        val_probabilities = model.predict_proba(val_inputs)[:, 1]

        comparison[name] = [
            accuracy_score(train_targets, train_predictions),
            accuracy_score(val_targets, val_predictions),
            f1_score(train_targets, train_predictions),
            f1_score(val_targets, val_predictions),
            roc_auc_score(train_targets, train_probabilities),
            roc_auc_score(val_targets, val_probabilities),
        ]

    return pd.DataFrame(
        comparison,
        index=[
            "Train Accuracy",
            "Validation Accuracy",
            "Train F1",
            "Validation F1",
            "Train AUROC",
            "Validation AUROC",
        ],
    )


def predict_and_plot(
    model: LogisticRegression,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> NDArray[np.float64]:
    """Predict labels, print accuracy, and plot a normalized confusion matrix."""
    predictions = np.asarray(model.predict(inputs), dtype=float)
    accuracy = accuracy_score(targets, predictions)
    print(f"Accuracy for {name}: {accuracy * 100:.2f}%")

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
        normalize="true",
    )
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annotations = np.array(
        [
            [
                f"{label}\n{value:.1%}"
                for label, value in zip(label_row, value_row, strict=True)
            ]
            for label_row, value_row in zip(labels, matrix, strict=True)
        ]
    )

    _, axis = plt.subplots()
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ax=axis,
    )
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Target")
    axis.set_title(f"{name} Confusion Matrix")
    return predictions


def get_f1_score(
    model: LogisticRegression,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> float:
    """Print and return the positive-class F1 score."""
    predictions = np.asarray(model.predict(inputs), dtype=float)
    score = float(f1_score(targets, predictions, pos_label=1))
    print(f"F1 score {name}: {score * 100:.2f}%")
    return score


def compute_auroc(
    model: _ProbabilityClassifier,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> float:
    """Print and return binary AUROC without building a ROC curve."""
    probabilities = np.asarray(model.predict_proba(inputs), dtype=float)[:, 1]
    score = float(roc_auc_score(targets, probabilities))
    print(f"AUROC for {name}: {score:.4f}")
    return score


def max_depth_auroc(
    max_depth: int,
    train_inputs: pd.DataFrame,
    train_targets: pd.Series,
    val_inputs: pd.DataFrame,
    val_targets: pd.Series,
    *,
    random_state: int = 42,
) -> dict[str, int | float]:
    """Return train and validation AUROC for one decision-tree depth."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    ).fit(train_inputs, train_targets)
    train_probabilities = np.asarray(
        model.predict_proba(train_inputs),
        dtype=float,
    )[:, 1]
    val_probabilities = np.asarray(
        model.predict_proba(val_inputs),
        dtype=float,
    )[:, 1]
    return {
        "Max Depth": max_depth,
        "Training AUROC": float(roc_auc_score(train_targets, train_probabilities)),
        "Validation AUROC": float(roc_auc_score(val_targets, val_probabilities)),
    }


def compute_auroc_and_build_roc(
    model: _ProbabilityClassifier,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> float:
    """Print and return AUROC, and plot its ROC curve."""
    probabilities = np.asarray(model.predict_proba(inputs), dtype=float)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets, probabilities, pos_label=1
    )
    roc_auc = float(auc(false_positive_rate, true_positive_rate))
    print(f"AUROC for {name}: {roc_auc:.4f}")

    _, axis = plt.subplots()
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"ROC curve (area = {roc_auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"Receiver Operating Characteristic (ROC) Curve for {name}")
    axis.legend(loc="lower right")
    return roc_auc


def evaluate_multiclass_model(
    model: Pipeline,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> NDArray[np.object_]:
    """Print multiclass metrics and confusion matrices, and return predictions."""
    predictions = np.asarray(model.predict(inputs), dtype=object)
    probabilities = np.asarray(model.predict_proba(inputs), dtype=float)
    classes = np.asarray(model.classes_, dtype=object)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro")
    macro_auroc = roc_auc_score(
        targets,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="macro",
    )

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Macro F1: {macro_f1:.4f}")
    print(f"{name} Macro AUROC: {macro_auroc:.4f}")

    raw_matrix = confusion_matrix(targets, predictions, labels=classes)
    normalized_matrix = confusion_matrix(
        targets,
        predictions,
        labels=classes,
        normalize="true",
    )
    row_labels = [f"Actual {label}" for label in classes]
    column_labels = [f"Predicted {label}" for label in classes]
    raw_matrix_table = pd.DataFrame(
        raw_matrix,
        index=row_labels,
        columns=column_labels,
    )
    normalized_matrix_table = pd.DataFrame(
        normalized_matrix,
        index=row_labels,
        columns=column_labels,
    )

    print(f"\n{name} Confusion Matrix — кількість:")
    print(raw_matrix_table.to_string())
    print(f"\n{name} Confusion Matrix — частка:")
    print(normalized_matrix_table.round(3).to_string())

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=normalized_matrix,
        display_labels=classes,
    )
    matrix_display.plot(values_format=".2f", cmap="Blues")
    matrix_display.ax_.set_title(f"{name} Normalized Confusion Matrix")

    return predictions


def build_ovr_logistic_pipeline(
    preprocessor: ColumnTransformer | None = None,
    *,
    max_iter: int = 1_000,
    random_state: int = 42,
) -> Pipeline:
    """Build an independent One-vs-Rest logistic regression pipeline."""
    classifier = OneVsRestClassifier(
        LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )
    )

    if preprocessor is None:
        return Pipeline([("classifier", classifier)])

    return Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]
    )


def compare_multiclass_predictions(
    predictions: dict[str, NDArray[np.object_]],
    targets: pd.Series,
) -> pd.DataFrame:
    """Compare the main validation metrics for multiclass predictions."""
    metrics = {}

    for name, model_predictions in predictions.items():
        metrics[name] = [
            accuracy_score(targets, model_predictions),
            precision_score(
                targets,
                model_predictions,
                average="macro",
                zero_division=0,
            ),
            recall_score(
                targets,
                model_predictions,
                average="macro",
                zero_division=0,
            ),
            f1_score(
                targets,
                model_predictions,
                average="macro",
                zero_division=0,
            ),
        ]

    return pd.DataFrame(
        metrics,
        index=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    )


def predict_majority_class(
    inputs: pd.DataFrame, majority_class: int
) -> NDArray[np.int64]:
    """Return a constant majority-class prediction for every input row."""
    return np.full(len(inputs), majority_class, dtype=np.int64)


def predict_raw_df(
    preprocessor: ColumnTransformer,
    model: LogisticRegression,
    input_df: pd.DataFrame,
) -> NDArray[np.float64]:
    """Transform raw features and return unrounded positive-class probabilities."""
    transformed = preprocessor.transform(input_df.copy())
    return np.asarray(model.predict_proba(transformed), dtype=float)[:, 1]
