"""Notebook classification reports combining metrics and figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from ml_homework.classification import _ProbabilityClassifier


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

    plot_binary_confusion_matrix(targets, predictions, name=name)
    return predictions


def compute_auroc_and_build_roc(
    model: _ProbabilityClassifier,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> float:
    """Print and return AUROC, and plot its ROC curve."""
    probabilities = np.asarray(model.predict_proba(inputs), dtype=float)[:, 1]
    roc_auc = float(roc_auc_score(targets, probabilities))
    print(f"AUROC for {name}: {roc_auc:.4f}")
    plot_roc_curve(targets, probabilities, name=name)
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


def plot_binary_confusion_matrix(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    name: str = "",
) -> tuple[Figure, Axes]:
    """Plot normalized binary counts from labels; never predict or print."""
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

    figure, axis = plt.subplots()
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
    return figure, axis


def plot_roc_curve(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    name: str = "",
) -> tuple[Figure, Axes]:
    """Plot a binary ROC curve from positive-class probabilities without printing."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets, probabilities, pos_label=1
    )
    roc_auc = float(auc(false_positive_rate, true_positive_rate))
    figure, axis = plt.subplots()
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
    return figure, axis
