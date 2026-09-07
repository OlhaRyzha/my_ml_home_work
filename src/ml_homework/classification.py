"""Reusable helpers for evaluating classification models."""

from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class _ProbabilityClassifier(Protocol):
    """Classifier interface required by binary probability metrics."""

    def predict_proba(self, inputs: pd.DataFrame, /) -> NDArray[np.float64]: ...


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
