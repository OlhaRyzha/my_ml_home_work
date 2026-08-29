import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from ml_homework.visualization import (
    category_counts_by_hue,
    compare_boxplots,
    compare_category_counts,
    correlation_heatmap,
    distribution_boxplot,
    numeric_vs_categorical_analysis,
    plot_auroc_by_max_depth,
    plot_decision_regions,
    plot_regression_predictions,
)


class ThresholdClassifier:
    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int64]:
        return (features[:, 0] + features[:, 1] > 0).astype(np.int64)


def test_plot_decision_regions_draws_regions_points_and_labels() -> None:
    features = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    target = np.array([0, 0, 1, 1])

    figure, axis = plot_decision_regions(
        ThresholdClassifier(),
        features,
        target,
        title="Decision boundary",
        class_names={0: "Negative", 1: "Positive"},
        grid_resolution=20,
    )

    assert len(axis.collections) >= 3
    assert axis.get_xlabel() == "Principal Component 1"
    assert axis.get_ylabel() == "Principal Component 2"
    assert axis.get_title() == "Decision boundary"
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Negative",
        "Positive",
    ]
    plt.close(figure)


def test_plot_decision_regions_uses_supplied_axis() -> None:
    figure, supplied_axis = plt.subplots()
    features = np.array([[-1.0, -1.0], [1.0, 1.0]])

    returned_figure, returned_axis = plot_decision_regions(
        ThresholdClassifier(), features, [0, 1], axis=supplied_axis
    )

    assert returned_figure is figure
    assert returned_axis is supplied_axis
    plt.close(figure)


@pytest.mark.parametrize(
    ("features", "target"),
    [
        (np.ones((3, 3)), np.array([0, 1, 1])),
        (np.ones((3, 2)), np.array([0, 1])),
    ],
)
def test_plot_decision_regions_rejects_invalid_shapes(
    features: NDArray[np.float64], target: NDArray[np.int64]
) -> None:
    with pytest.raises(ValueError):
        plot_decision_regions(ThresholdClassifier(), features, target)


def test_distribution_boxplot_returns_figure_and_axes() -> None:
    figure, axes = distribution_boxplot(pd.DataFrame({"value": [1, 2, 3]}), "value")
    assert len(axes) == 2
    plt.close(figure)


def test_category_counts_by_hue_returns_two_axes() -> None:
    data = pd.DataFrame({"category": ["A", "B", "A", "A"], "hue": [0, 0, 1, 1]})
    figure, axes = category_counts_by_hue(data, "category", "hue")
    assert len(axes) == 2
    plt.close(figure)


def test_correlation_heatmap_returns_figure() -> None:
    corr = pd.DataFrame({"a": [1.0, 0.9], "b": [0.9, 1.0]}, index=["a", "b"])
    figure, _ = correlation_heatmap(corr)
    plt.close(figure)


def test_plot_regression_predictions_draws_all_lines() -> None:
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    predictions = {"MNK": [1.1, 2.1, 3.1], "Sklearn": [0.9, 1.9, 2.9]}

    figure, axis = plot_regression_predictions(
        x, y, predictions, title="t", xlabel="x", ylabel="y"
    )

    assert len(axis.lines) == len(predictions)
    assert [line.get_label() for line in axis.lines] == list(predictions)
    plt.close(figure)


def test_plot_auroc_by_max_depth_labels_both_curves() -> None:
    scores = pd.DataFrame(
        {
            "Max Depth": [1, 2, 3],
            "Training AUROC": [0.7, 0.8, 0.9],
            "Validation AUROC": [0.68, 0.78, 0.76],
        }
    )

    figure, axis = plot_auroc_by_max_depth(scores)

    assert [line.get_label() for line in axis.lines] == ["Training", "Validation"]
    assert axis.get_xlabel() == "Max Depth"
    assert axis.get_ylabel() == "AUROC"
    plt.close(figure)


def test_composed_visualizations_use_explicit_dataframes() -> None:
    df0 = pd.DataFrame({"category": ["A", "B"], "hue": ["X", "Y"], "value": [1.0, 2.0]})
    df1 = pd.DataFrame({"category": ["A", "B"], "hue": ["X", "Y"], "value": [2.0, 3.0]})

    box_figure, _ = compare_boxplots(df0, df1, "category", "value", 10, 10, "hue")
    first_summary, second_summary, analysis_figure = numeric_vs_categorical_analysis(
        df0, df1, "value", "category", "hue"
    )
    count_figures = compare_category_counts(df0, df1, "category", "hue")

    assert not first_summary.empty
    assert not second_summary.empty
    for figure in (box_figure, analysis_figure, *count_figures):
        plt.close(figure)
