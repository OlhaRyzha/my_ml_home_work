import numpy as np
import pandas as pd

from ml_homework.preprocessing import prepare_clustering_features


def test_clustering_features_preserve_filtered_indices_and_refit() -> None:
    data = pd.DataFrame(
        {"Income": [10.0, 30.0, 1000.0], "Education": ["A", "B", "C"]},
        index=[4, 9, 20],
    )
    original = data.copy(deep=True)
    prepare_clustering_features(data, ["Education"], scale_numeric=True)
    filtered = data.loc[[4, 9]]

    unscaled = prepare_clustering_features(filtered, ["Education"])
    scaled = prepare_clustering_features(filtered, ["Education"], scale_numeric=True)

    assert scaled.index.tolist() == [4, 9]
    assert scaled.columns.tolist() == ["Income", "Education_A", "Education_B"]
    np.testing.assert_allclose(scaled["Income"], [-1.0, 1.0])
    pd.testing.assert_series_equal(unscaled["Income"], filtered["Income"])
    pd.testing.assert_frame_equal(scaled.iloc[:, 1:], unscaled.iloc[:, 1:])
    assert not scaled.isna().any().any()
    pd.testing.assert_frame_equal(data, original)


def test_clustering_features_support_numeric_only_data() -> None:
    data = pd.DataFrame({"amount": [2.0, 6.0]}, index=[3, 8])
    result = prepare_clustering_features(data, [], scale_numeric=True)
    np.testing.assert_allclose(result["amount"], [-1.0, 1.0])
    assert result.index.equals(data.index)
