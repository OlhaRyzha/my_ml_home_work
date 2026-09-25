"""Reusable dimensionality-reduction experiments for notebooks."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pca_kmeans_pipeline(
    *,
    n_clusters: int,
    n_components: int,
    random_state: int,
    n_init: int,
) -> Pipeline:
    """Build a standardized PCA embedding followed by K-Means."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=n_init,
                ),
            ),
        ]
    )


def fit_pca_kmeans(
    features: pd.DataFrame,
    *,
    n_clusters: int,
    n_components: int,
    random_state: int,
    n_init: int,
) -> tuple[Pipeline, NDArray[np.int32], NDArray[np.float64], float]:
    """Fit PCA plus K-Means and return the model, labels, embedding, and score."""
    pipeline = build_pca_kmeans_pipeline(
        n_clusters=n_clusters,
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
    )
    labels = np.asarray(pipeline.fit_predict(features), dtype=np.int32)
    embedding = np.asarray(pipeline[:-1].transform(features), dtype=np.float64)
    score = float(silhouette_score(embedding, labels))
    return pipeline, labels, embedding, score


def fit_tsne_embedding(
    features: pd.DataFrame,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int,
) -> NDArray[np.float64]:
    """Standardize features and fit t-SNE for visualization.

    t-SNE is intentionally not wrapped in ``Pipeline``: it has ``fit_transform``
    but no standalone ``transform`` method.
    """
    scaled_features = StandardScaler().fit_transform(features)
    embedding = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(scaled_features)
    return np.asarray(embedding, dtype=np.float64)


def summarize_standardized_clusters(
    features: pd.DataFrame,
    labels: Iterable[int],
) -> pd.DataFrame:
    """Return per-cluster means in standardized feature units."""
    label_array = np.asarray(list(labels))
    if len(label_array) != len(features):
        raise ValueError("labels must contain one value per feature row")

    scaled = pd.DataFrame(
        StandardScaler().fit_transform(features),
        columns=features.columns,
        index=features.index,
    )
    scaled["cluster"] = label_array
    return scaled.groupby("cluster").mean().round(2)


def top_loadings(loadings: pd.DataFrame, *, top_n: int = 5) -> pd.DataFrame:
    """Return the strongest absolute loadings for every component."""
    if top_n < 1:
        raise ValueError("top_n must be positive")

    rows: list[dict[str, str | float]] = []
    for component in loadings.index:
        features = loadings.loc[component].abs().nlargest(top_n).index
        rows.extend(
            {
                "component": str(component),
                "feature": str(feature),
                "loading": float(loadings.loc[component, feature]),
            }
            for feature in features
        )
    return pd.DataFrame(rows, columns=["component", "feature", "loading"])
