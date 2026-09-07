from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_kmeans(
    features: pd.DataFrame,
    k_values: Iterable[int],
    *,
    silhouette_k_values: Iterable[int],
    random_state: int,
    n_init: int,
) -> pd.DataFrame:
    """Fit once per k and return inertia and requested silhouette scores.

    Unrequested or undefined silhouette scores (e.g. k=1) are NaN.
    Compare results only within the same feature representation.
    """
    silhouette_ks = set(silhouette_k_values)
    rows: list[dict[str, int | float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(
            features
        )
        n_labels = len(np.unique(model.labels_))
        score = float("nan")
        if k in silhouette_ks and 1 < n_labels < len(features):
            score = float(silhouette_score(features, model.labels_))
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": score})
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def fit_kmeans(
    features: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[KMeans, NDArray[np.int32], float]:
    """Return the fitted model, positional labels, and silhouette score."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(features)
    score = float(silhouette_score(features, labels))
    return model, labels, score


def summarize_customer_clusters(viz_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize marketing clusters; cluster -1 represents DBSCAN noise."""
    return viz_df.groupby("Cluster").agg(
        customers=("Income", "size"),
        mean_income=("Income", "mean"),
        mean_spent=("Total_Spent", "mean"),
        mean_purchases=("Total_Purchases", "mean"),
        complain_share=("Complain", "mean"),
    )
