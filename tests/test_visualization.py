import matplotlib.pyplot as plt
import pandas as pd

from ml_homework.visualization import (
    category_counts_by_hue,
    compare_boxplots,
    compare_category_counts,
    correlation_heatmap,
    distribution_boxplot,
    numeric_vs_categorical_analysis,
    plot_auroc_by_max_depth,
    plot_regression_predictions,
)


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
