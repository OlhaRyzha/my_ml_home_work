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
    plot_clusters,
    plot_decision_regions,
    plot_regression_predictions,
    prepare_cluster_viz_df,
)


def test_prepare_cluster_viz_df_preserves_input_and_sums_marketing_columns() -> None:
    df = pd.DataFrame(
        {
            "MntWines": [1, 2],
            "MntFruits": [3, 4],
            "MntMeatProducts": [5, 6],
            "MntFishProducts": [7, 8],
            "MntSweetProducts": [9, 10],
            "MntGoldProds": [11, 12],
            "NumWebPurchases": [1, 2],
            "NumCatalogPurchases": [3, 4],
            "NumStorePurchases": [5, 6],
            "NumDealsPurchases": [100, 200],
        },
        index=[4, 9],
    )
    original = df.copy(deep=True)
    result = prepare_cluster_viz_df(df, np.array([2, 0]))
    assert result.index.equals(df.index)
    assert result["Cluster"].tolist() == [2, 0]
    assert result["Total_Spent"].tolist() == [36, 42]
    assert result["Total_Purchases"].tolist() == [9, 12]
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("labels", [[0], [[0], [1]]])
def test_prepare_cluster_viz_df_rejects_misaligned_labels(
    labels: list[int] | list[list[int]],
) -> None:
    with pytest.raises(ValueError, match="one cluster label per dataframe row"):
        prepare_cluster_viz_df(pd.DataFrame(index=[4, 9]), labels)


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


def test_plot_clusters_labels_groups_and_marks_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "Cluster": [-1, 0, 1]})
    plot_clusters(data, "x", "y")
    axis = plt.gca()
    assert axis.get_legend_handles_labels()[1] == [
        "Cluster 0 (n=1)",
        "Cluster 1 (n=1)",
        "Noise (-1, n=1)",
    ]
    np.testing.assert_allclose(
        np.asarray(axis.collections[-1].get_facecolor(), dtype=float)[0, :3],
        np.array([0.5019608] * 3),
    )
    plt.close("all")
