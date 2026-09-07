import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from ml_homework.clustering import (
    evaluate_kmeans,
    fit_kmeans,
    summarize_customer_clusters,
)


def test_kmeans_evaluation_matches_selected_model() -> None:
    features = pd.DataFrame({"x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2]})
    original = features.copy(deep=True)
    results = evaluate_kmeans(
        features, [1, 2, 3], silhouette_k_values=[2], random_state=42, n_init=10
    )
    model, labels, score = fit_kmeans(
        features, n_clusters=2, random_state=42, n_init=10
    )
    assert results["k"].tolist() == [1, 2, 3]
    assert results.loc[[0, 2], "silhouette"].isna().all()
    assert results.loc[1, "inertia"] == pytest.approx(model.inertia_)
    assert results.loc[1, "silhouette"] == pytest.approx(score)
    assert score == pytest.approx(silhouette_score(features, labels))
    assert score > 0.9
    np.testing.assert_array_equal(labels, model.predict(features))
    pd.testing.assert_frame_equal(features, original)


def test_customer_summary_keeps_noise_and_counts_rows() -> None:
    data = pd.DataFrame(
        {
            "Cluster": [-1, 0, 0],
            "Income": [10, 20, 40],
            "Total_Spent": [1, 2, 4],
            "Total_Purchases": [1, 3, 5],
            "Complain": [1, 0, 1],
        }
    )
    result = summarize_customer_clusters(data)
    assert result.loc[-1, "customers"] == 1
    assert result.loc[0, "customers"] == 2
    assert result.loc[0, "mean_income"] == 30
    assert result.loc[0, "complain_share"] == 0.5
