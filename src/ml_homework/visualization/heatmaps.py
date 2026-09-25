"""Reusable heatmap visualizations."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (16, 12),
    cmap: str = "coolwarm",
    center: float = 0,
    annot: bool = True,
    fmt: str = ".2f",
    axis: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the lower triangle of a correlation matrix as an annotated heatmap."""
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square dataframe")

    if axis is None:
        figure, axis = plt.subplots(figsize=figsize)
    else:
        figure = cast(Figure, axis.figure)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=center,
        ax=axis,
    )
    figure.tight_layout()
    return figure, axis
