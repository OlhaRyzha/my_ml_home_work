"""Reusable helpers for evaluating binary classification models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve


def add_age_group(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with age encoded into fixed, interpretable groups."""
    result = data.copy()
    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[-np.inf, 30, 35, 40, 45, 50, 60, np.inf],
        labels=["up_to_30", "31_35", "36_40", "41_45", "46_50", "51_60", "over_60"],
    )
    return result


def predict_and_plot(
    model: LogisticRegression,
    inputs: pd.DataFrame,
    targets: pd.Series,
    name: str = "",
) -> NDArray[np.float64]:
    """Predict labels, print accuracy, and plot a normalized confusion matrix."""
    predictions = np.asarray(model.predict(inputs), dtype=float)
    accuracy = accuracy_score(targets, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    matrix = confusion_matrix(targets, predictions, normalize="true")
    _, axis = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=axis)
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


def compute_auroc_and_build_roc(
    model: LogisticRegression,
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
    print(f"AUROC for {name}: {roc_auc:.2f}")

    _, axis = plt.subplots()
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    axis.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(f"Receiver Operating Characteristic (ROC) Curve for {name}")
    axis.legend(loc="lower right")
    return roc_auc


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
