"""Model prediction, decision-region, and depth-comparison plots."""

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


class Classifier(Protocol):
    """Minimal interface required for plotting classifier predictions."""

    def predict(self, features: NDArray[np.float64]) -> ArrayLike:
        """Predict one class label per row."""


DEFAULT_CLASS_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
)


def plot_decision_regions(
    classifier: Classifier,
    features: ArrayLike,
    target: ArrayLike,
    *,
    axis: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Principal Component 1",
    ylabel: str = "Principal Component 2",
    class_names: Mapping[object, str] | None = None,
    colors: Sequence[str] = DEFAULT_CLASS_COLORS,
    grid_resolution: int = 300,
    padding: float = 0.08,
) -> tuple[Figure, Axes]:
    """Plot decision regions and observations for a fitted 2D classifier.

    ``classifier`` must be fitted on the same two features represented by
    ``features``. Class labels may be numeric or strings. The same color is
    used for each class in the translucent decision region and its points.
    """
    feature_array = np.asarray(features, dtype=float)
    target_array = np.asarray(target)

    if feature_array.ndim != 2 or feature_array.shape[1] != 2:
        raise ValueError("features must have shape (n_samples, 2)")
    if target_array.ndim != 1 or len(target_array) != len(feature_array):
        raise ValueError("target must be one-dimensional and match features")
    if len(feature_array) == 0:
        raise ValueError("features and target must not be empty")
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2")

    classes = np.unique(target_array)
    if len(colors) < len(classes):
        raise ValueError("colors must contain at least one color per class")

    x_min, x_max = _axis_limits(feature_array[:, 0], padding)
    y_min, y_max = _axis_limits(feature_array[:, 1], padding)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    predicted = np.asarray(classifier.predict(grid))
    if predicted.shape != (len(grid),):
        raise ValueError("classifier.predict must return one label per grid point")

    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    try:
        region_values = np.array(
            [class_to_index[label] for label in predicted.tolist()], dtype=int
        ).reshape(xx.shape)
    except KeyError as error:
        raise ValueError(
            "classifier predicted a class that is absent from target"
        ) from error

    if axis is None:
        figure, axis = plt.subplots(figsize=(7, 6))
    else:
        figure = cast(Figure, axis.figure)

    class_colors = list(colors[: len(classes)])
    color_map = ListedColormap(class_colors)
    axis.contourf(
        xx,
        yy,
        region_values,
        levels=np.arange(len(classes) + 1) - 0.5,
        cmap=color_map,
        alpha=0.22,
    )
    if len(classes) > 1:
        axis.contour(
            xx,
            yy,
            region_values,
            levels=np.arange(len(classes) - 1) + 0.5,
            colors="#334155",
            linewidths=0.8,
            alpha=0.65,
        )

    for index, label in enumerate(classes):
        mask = target_array == label
        legend_label = (
            class_names.get(label, str(label))
            if class_names is not None
            else str(label)
        )
        axis.scatter(
            feature_array[mask, 0],
            feature_array[mask, 1],
            color=class_colors[index],
            label=legend_label,
            alpha=0.82,
            s=42,
            edgecolor="white",
            linewidth=0.7,
        )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title is not None:
        axis.set_title(title)
    axis.legend(title="Class", frameon=True)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axis


def _axis_limits(values: NDArray[np.float64], padding: float) -> tuple[float, float]:
    """Return plot limits with relative padding, including constant features."""
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    margin = span * padding if span > 0 else 1.0
    return lower - margin, upper + margin


def plot_regression_predictions(
    x: ArrayLike,
    y: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    title: str,
    xlabel: str,
    ylabel: str,
) -> tuple[Figure, Axes]:
    """Scatter actual values and overlay named prediction lines."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.scatter(x, y, alpha=0.5, label="Data points")
    for label, values in predictions.items():
        axis.plot(x, values, label=label)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(True)
    return figure, axis


def plot_auroc_by_max_depth(auroc_df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Plot train and validation AUROC across decision-tree depths."""
    figure, axis = plt.subplots()
    axis.plot(
        auroc_df["Max Depth"],
        auroc_df["Training AUROC"],
        label="Training",
    )
    axis.plot(
        auroc_df["Max Depth"],
        auroc_df["Validation AUROC"],
        label="Validation",
    )
    axis.set_title("Training vs. Validation AUROC")
    axis.set_xticks(auroc_df["Max Depth"])
    axis.set_xlabel("Max Depth")
    axis.set_ylabel("AUROC")
    axis.legend()
    figure.tight_layout()
    return figure, axis
