from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ml_homework.eda import upper_outlier_bound


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


def boxplots_for_columns(
    data: pd.DataFrame, columns: Iterable[str]
) -> list[tuple[Figure, Axes]]:
    """Create one boxplot per numeric column."""
    plots: list[tuple[Figure, Axes]] = []
    for column in columns:
        figure, axis = plt.subplots()
        sns.boxplot(y=data[column], ax=axis)
        axis.set_title(f"Boxplot: {column}")
        plots.append((figure, axis))
    return plots


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


def distribution_boxplot(data: pd.DataFrame, column: str) -> tuple[Figure, list[Axes]]:
    """Plot a histogram with KDE and a boxplot for one column."""
    figure, axes_array = plt.subplots(1, 2, figsize=(16, 6))
    axes = list(axes_array)
    sns.histplot(data=data, x=column, kde=True, ax=axes[0])
    axes[0].ticklabel_format(style="plain", axis="x")
    axes[0].set_title(f"Distribution: {column}")

    sns.boxplot(
        data=data,
        y=column,
        flierprops={"markerfacecolor": "r", "marker": "D"},
        ax=axes[1],
    )
    axes[1].set_title(f"Boxplot: {column}")
    figure.tight_layout()
    return figure, axes


def compare_kde_without_upper_outliers(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_upper_bound: float,
    second_upper_bound: float,
    column: str,
) -> tuple[Figure, Axes]:
    """Compare KDE curves after applying explicit upper bounds."""
    figure, axis = plt.subplots(figsize=(14, 6))
    sns.kdeplot(
        first.loc[first[column] <= first_upper_bound, column],
        label="On-Time Payments",
        ax=axis,
    )
    sns.kdeplot(
        second.loc[second[column] <= second_upper_bound, column],
        label="Payment Difficulties",
        ax=axis,
    )
    axis.ticklabel_format(style="plain", axis="x")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def compare_scatter_without_outliers(
    df0: pd.DataFrame, df1: pd.DataFrame, x_column: str, y_column: str
) -> tuple[Figure, list[Axes]]:
    """Compare two scatterplots after filtering upper IQR outliers."""
    frames = (df0, df1)
    titles = ("On-Time Payments", "Payment Difficulties")
    figure, axes_array = plt.subplots(1, 2, figsize=(14, 6))
    axes = list(axes_array)

    for data, title, axis in zip(frames, titles, axes, strict=True):
        x_upper = upper_outlier_bound(data, x_column)
        y_upper = upper_outlier_bound(data, y_column)
        filtered = data.loc[(data[x_column] <= x_upper) & (data[y_column] <= y_upper)]
        sns.scatterplot(data=filtered, x=x_column, y=y_column, ax=axis)
        axis.set_title(title)
        axis.ticklabel_format(style="plain", axis="both")
        axis.tick_params(axis="x", rotation=45)

    figure.tight_layout(pad=4)
    return figure, axes


def draw_boxplot(
    data: pd.DataFrame,
    categorical: str,
    continuous: str,
    continuous_upper_bound: float,
    title: str,
    hue_column: str,
    axis: Axes,
) -> Axes:
    """Draw a filtered categorical boxplot on an explicit axis."""
    filtered = data.loc[data[continuous] < continuous_upper_bound]
    sns.boxplot(
        data=filtered,
        x=categorical,
        y=continuous,
        hue=hue_column,
        order=sorted(filtered[categorical].dropna().unique(), reverse=True),
        hue_order=sorted(filtered[hue_column].dropna().unique(), reverse=True),
        flierprops={"markerfacecolor": "r", "marker": "D"},
        ax=axis,
    )
    axis.set_title(title)
    axis.ticklabel_format(style="plain", axis="y")
    axis.tick_params(axis="x", rotation=90)
    return axis


def compare_boxplots(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    categorical: str,
    continuous: str,
    df1_upper_bound: float,
    df0_upper_bound: float,
    hue_column: str,
) -> tuple[Figure, list[Axes]]:
    """Compare filtered boxplots for on-time and difficult-payment groups."""
    figure, axes_array = plt.subplots(1, 2, figsize=(16, 10))
    axes = list(axes_array)
    draw_boxplot(
        df1,
        categorical,
        continuous,
        df1_upper_bound,
        "Payment Difficulties",
        hue_column,
        axes[0],
    )
    draw_boxplot(
        df0,
        categorical,
        continuous,
        df0_upper_bound,
        "On-Time Payments",
        hue_column,
        axes[1],
    )
    figure.tight_layout(pad=4)
    return figure, axes


def numeric_vs_categorical_analysis(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    numeric_column: str,
    category_column: str,
    hue_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure]:
    """Summarize and plot a numeric feature against two categorical features."""
    df1_upper_bound = upper_outlier_bound(df1, numeric_column)
    df0_upper_bound = upper_outlier_bound(df0, numeric_column)
    group_columns = [category_column, hue_column]
    df1_summary = df1.groupby(group_columns)[numeric_column].describe()
    df0_summary = df0.groupby(group_columns)[numeric_column].describe()
    figure, _ = compare_boxplots(
        df0,
        df1,
        category_column,
        numeric_column,
        df1_upper_bound,
        df0_upper_bound,
        hue_column,
    )
    return df1_summary, df0_summary, figure


def rotated_countplot(
    data: pd.DataFrame, column: str, rotation: int = 90
) -> tuple[Figure, Axes]:
    """Draw a countplot with rotated x tick labels."""
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.countplot(data=data, x=column, ax=axis)
    axis.tick_params(axis="x", rotation=rotation)
    return figure, axis


def correlation_heatmap(
    corr: pd.DataFrame, lower: float = 0.8, upper: float = 0.9999
) -> tuple[Figure, Axes]:
    """Plot an annotated heatmap of correlations within [lower, upper)."""
    figure, axis = plt.subplots(figsize=(25, 25))
    sns.heatmap(
        data=corr[(corr >= lower) & (corr < upper)],
        annot=True,
        cmap="RdYlGn",
        cbar=True,
        fmt=".2f",
        ax=axis,
    )
    return figure, axis


def category_counts_by_hue(
    data: pd.DataFrame, column: str, hue_column: str
) -> tuple[Figure, list[Axes]]:
    """Plot normalized and absolute value counts of a column split by hue."""
    first_hue_value = data[hue_column].unique()[0]
    figure, axes_array = plt.subplots(1, 2, figsize=(14, 6))
    axes = list(axes_array)

    proportions: pd.Series = (
        data.groupby(hue_column)[column].value_counts(normalize=True) * 100
    ).round(2)
    proportions.unstack(hue_column).sort_values(
        by=first_hue_value, ascending=False
    ).plot.bar(
        ax=axes[0], title=f"Нормалізований розподіл значень за категорією: {column}"
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="{:,.1f}%")

    counts = data.groupby(hue_column)[column].value_counts()
    counts.unstack(hue_column).sort_values(
        by=first_hue_value, ascending=False
    ).plot.bar(ax=axes[1], title=f"Кількість даних за категорією: {column}")
    for container in axes[1].containers:
        axes[1].bar_label(container)

    figure.tight_layout()
    return figure, axes


def compare_category_counts(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    column: str,
    hue_column: str,
) -> tuple[Figure, Figure]:
    """Compare normalized and absolute category counts for two groups."""

    def plot_measure(normalize: bool, title: str, percentage: bool) -> Figure:
        figure, axes = plt.subplots(1, 2, figsize=(14, 4))
        for data, group_title, axis in zip(
            (df1, df0),
            ("Payment Difficulties", "On-Time Payments"),
            axes,
            strict=True,
        ):
            grouped = data.groupby(hue_column)[column]
            values = (
                grouped.value_counts(normalize=True)
                if normalize
                else grouped.value_counts(normalize=False)
            )
            if percentage:
                values = (values * 100).round(2)
            table = values.unstack(hue_column)
            table.plot.bar(ax=axis, title=group_title)
            for container in axis.containers:
                if percentage:
                    axis.bar_label(container, fmt="{:,.1f}%")
                else:
                    axis.bar_label(container)
        figure.suptitle(title)
        figure.tight_layout()
        return figure

    normalized = plot_measure(
        normalize=True,
        title=f"Normalized category distribution: {column}",
        percentage=True,
    )
    absolute = plot_measure(
        normalize=False,
        title=f"Category counts: {column}",
        percentage=False,
    )
    return normalized, absolute


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
