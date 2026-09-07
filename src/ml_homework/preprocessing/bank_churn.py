"""Preprocessing helpers for the bank customer churn dataset."""

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_homework.preprocessing import (
    encode_categorical_features,
    scale_numeric_features,
    split_train_validation,
    transform_features,
)

TARGET_COLUMN = "Exited"
NUMERIC_COLUMNS = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
CATEGORICAL_COLUMNS = [
    "Geography",
    "Gender",
    "AgeGroup",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]


class PreprocessedData(TypedDict):
    """Typed collection of fitted preprocessing outputs."""

    X_train: pd.DataFrame
    train_targets: pd.Series
    X_val: pd.DataFrame
    val_targets: pd.Series
    input_cols: list[str]
    scaler: StandardScaler | None
    encoder: OneHotEncoder


def select_bank_churn_inputs(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create the age group and return only model input columns."""
    inputs = add_age_group(raw_df)
    required_columns = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in inputs.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return inputs.loc[:, required_columns].copy()


def preprocess_data(
    raw_df: pd.DataFrame,
    *,
    scaler_numeric: bool = True,
    validation_size: float = 0.25,
    random_state: int = 42,
) -> PreprocessedData:
    """Split and preprocess bank churn data without a sklearn pipeline."""
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    inputs = select_bank_churn_inputs(raw_df)
    targets = raw_df[TARGET_COLUMN].copy()
    X_train_raw, X_val_raw, train_targets, val_targets = split_train_validation(
        inputs,
        targets,
        validation_size=validation_size,
        random_state=random_state,
    )

    numeric_train, numeric_val, scaler = scale_numeric_features(
        X_train_raw[NUMERIC_COLUMNS],
        X_val_raw[NUMERIC_COLUMNS],
        enabled=scaler_numeric,
    )
    categorical_train, categorical_val, encoder = encode_categorical_features(
        X_train_raw[CATEGORICAL_COLUMNS],
        X_val_raw[CATEGORICAL_COLUMNS],
    )

    X_train = pd.concat([numeric_train, categorical_train], axis="columns")
    X_val = pd.concat([numeric_val, categorical_val], axis="columns")
    input_cols = X_train.columns.tolist()

    data: PreprocessedData = {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "scaler": scaler,
        "encoder": encoder,
    }
    return data


def preprocess_new_data(
    raw_df: pd.DataFrame,
    input_cols: list[str],
    scaler: StandardScaler | None,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Transform new rows with preprocessing objects fitted on training data."""
    inputs = select_bank_churn_inputs(raw_df)
    numeric_inputs = transform_features(inputs[NUMERIC_COLUMNS], scaler)
    categorical_inputs = transform_features(inputs[CATEGORICAL_COLUMNS], encoder)
    processed_inputs = pd.concat([numeric_inputs, categorical_inputs], axis="columns")

    missing_columns = [
        column for column in input_cols if column not in processed_inputs.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing processed columns: {missing_columns}")
    return processed_inputs.loc[:, input_cols]


def add_age_group(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with age encoded into fixed, interpretable groups."""
    result = data.copy()
    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=[-np.inf, 30, 35, 40, 45, 50, 60, np.inf],
        labels=["up_to_30", "31_35", "36_40", "41_45", "46_50", "51_60", "over_60"],
    )
    return result
