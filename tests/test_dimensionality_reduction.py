import numpy as np
import pandas as pd
import pytest

from ml_homework.dimensionality_reduction import (
    fit_pca_kmeans,
    fit_tsne_embedding,
    summarize_standardized_clusters,
    top_loadings,
)
from ml_homework.visualization.dimensionality_reduction import (
    plot_embedding_clusters,
)


def test_fit_pca_kmeans_returns_embedding_labels_and_score() -> None:
    features = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "y": [0.0, 0.2, 0.1, 10.0, 10.2, 10.1],
        }
    )

    pipeline, labels, embedding, score = fit_pca_kmeans(
        features,
        n_clusters=2,
        n_components=2,
        random_state=42,
        n_init=10,
    )

    assert pipeline.named_steps["pca"].n_components == 2
    assert labels.shape == (6,)
    assert embedding.shape == (6, 2)
    assert score > 0.8


def test_fit_tsne_embedding_returns_requested_shape() -> None:
    features = pd.DataFrame(np.arange(60).reshape(20, 3))

    embedding = fit_tsne_embedding(
        features,
        perplexity=5,
        random_state=42,
    )

    assert embedding.shape == (20, 2)


def test_summarize_standardized_clusters_returns_cluster_profiles() -> None:
    features = pd.DataFrame({"x": [0.0, 1.0, 10.0, 11.0]})

    result = summarize_standardized_clusters(features, [0, 0, 1, 1])

    assert list(result.index) == [0, 1]
    assert result.loc[0, "x"] == pytest.approx(-1.0, abs=0.01)
    assert result.loc[1, "x"] == pytest.approx(1.0, abs=0.01)


def test_top_loadings_returns_strongest_features_per_component() -> None:
    loadings = pd.DataFrame(
        [[0.1, -0.8, 0.3], [0.7, 0.2, -0.9]],
        index=["PC1", "PC2"],
        columns=["a", "b", "c"],
    )

    result = top_loadings(loadings, top_n=2)

    assert result.to_dict("records") == [
        {"component": "PC1", "feature": "b", "loading": -0.8},
        {"component": "PC1", "feature": "c", "loading": 0.3},
        {"component": "PC2", "feature": "c", "loading": -0.9},
        {"component": "PC2", "feature": "a", "loading": 0.7},
    ]


def test_plot_embedding_clusters_supports_two_dimensions() -> None:
    embedding = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})

    figure = plot_embedding_clusters(embedding, [0, 1], title="Embedding")

    assert figure.layout.title.text == "Embedding"
    assert len(figure.data) == 1
