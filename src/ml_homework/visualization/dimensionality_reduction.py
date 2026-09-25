"""Plots for PCA and t-SNE embeddings."""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def plot_embedding_clusters(
    embedding: pd.DataFrame,
    labels: Iterable[int],
    *,
    title: str,
    dimensions: Sequence[str] | None = None,
) -> Figure:
    """Plot a two- or three-dimensional embedding colored by cluster."""
    selected_dimensions = list(dimensions or embedding.columns)
    if len(selected_dimensions) not in (2, 3):
        raise ValueError("embedding must contain exactly two or three dimensions")
    if not set(selected_dimensions).issubset(embedding.columns):
        raise ValueError("dimensions must be columns of embedding")

    label_array = np.asarray(list(labels))
    if len(label_array) != len(embedding):
        raise ValueError("labels must contain one value per embedding row")

    plot_data = embedding.loc[:, selected_dimensions].copy()
    plot_data["cluster"] = label_array
    plot_kwargs = {
        "data_frame": plot_data,
        "title": title,
        "color": "cluster",
        "hover_data": ["cluster"],
    }
    if len(selected_dimensions) == 2:
        return px.scatter(
            x=selected_dimensions[0], y=selected_dimensions[1], **plot_kwargs
        )
    return px.scatter_3d(
        x=selected_dimensions[0],
        y=selected_dimensions[1],
        z=selected_dimensions[2],
        **plot_kwargs,
    )
