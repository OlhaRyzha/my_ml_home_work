import numpy as np
import pandas as pd

from ml_homework.process_bank_churn import preprocess_data, preprocess_new_data


def bank_churn_data() -> pd.DataFrame:
    """Return a small balanced dataset with all required raw columns."""
    row_count = 16
    return pd.DataFrame(
        {
            "id": range(row_count),
            "CustomerId": range(10_000, 10_000 + row_count),
            "Surname": [f"Customer {index}" for index in range(row_count)],
            "CreditScore": np.linspace(550, 750, row_count),
            "Geography": ["France", "Germany", "Spain", "France"] * 4,
            "Gender": ["Female", "Male"] * 8,
            "Age": np.linspace(25, 65, row_count),
            "Tenure": [0, 1, 2, 3] * 4,
            "Balance": np.linspace(0, 120_000, row_count),
            "NumOfProducts": [1, 2, 3, 1] * 4,
            "HasCrCard": [0, 1] * 8,
            "IsActiveMember": [1, 0] * 8,
            "EstimatedSalary": np.linspace(30_000, 150_000, row_count),
            "Exited": [0, 1] * 8,
        }
    )


def test_preprocess_data_without_scaling_returns_aligned_subsets() -> None:
    raw_df = bank_churn_data()

    data = preprocess_data(raw_df, scaler_numeric=False)
    X_train = data["X_train"]
    train_targets = data["train_targets"]
    X_val = data["X_val"]
    val_targets = data["val_targets"]
    input_cols = data["input_cols"]
    scaler = data["scaler"]
    encoder = data["encoder"]

    assert scaler is None
    assert X_train.columns.tolist() == input_cols
    assert X_val.columns.tolist() == input_cols
    assert len(X_train) + len(X_val) == len(raw_df)
    assert train_targets.value_counts().nunique() == 1
    assert val_targets.value_counts().nunique() == 1
    assert "Surname" not in input_cols
    assert "id" not in input_cols
    assert "CustomerId" not in input_cols
    assert "AgeGroup_over_60" in input_cols
    assert encoder.feature_names_in_.tolist()[0] == "Geography"
    pd.testing.assert_series_equal(
        X_train["Age"].sort_index(),
        raw_df.loc[X_train.index, "Age"].sort_index(),
        check_names=False,
    )


def test_preprocess_data_fits_optional_scaler_on_train_only() -> None:
    raw_df = bank_churn_data()

    data = preprocess_data(
        raw_df,
        scaler_numeric=True,
    )
    X_train = data["X_train"]
    scaler = data["scaler"]

    assert scaler is not None
    np.testing.assert_allclose(
        X_train[["CreditScore", "Age", "Balance", "EstimatedSalary"]].mean(),
        0.0,
        atol=1e-12,
    )


def test_preprocess_new_data_reuses_fitted_encoder_and_column_order() -> None:
    raw_df = bank_churn_data()
    data = preprocess_data(
        raw_df,
        scaler_numeric=False,
    )
    input_cols = data["input_cols"]
    scaler = data["scaler"]
    encoder = data["encoder"]
    new_df = raw_df.drop(columns="Exited").iloc[[0]].copy()
    new_df["Geography"] = "Italy"

    processed = preprocess_new_data(new_df, input_cols, scaler, encoder)

    assert processed.columns.tolist() == input_cols
    assert processed.index.tolist() == new_df.index.tolist()
    geography_columns = [
        column for column in input_cols if column.startswith("Geography_")
    ]
    assert processed[geography_columns].to_numpy().sum() == 0
