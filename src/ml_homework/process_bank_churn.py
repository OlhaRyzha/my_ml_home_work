"""Preprocessing helpers for the bank customer churn dataset."""

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_homework.classification import add_age_group

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


def split_train_validation(
    inputs: pd.DataFrame,
    targets: pd.Series,
    *,
    validation_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible stratified train and validation subsets."""
    X_train, X_val, train_targets, val_targets = train_test_split(
        inputs,
        targets,
        test_size=validation_size,
        random_state=random_state,
        stratify=targets,
    )
    return X_train, X_val, train_targets, val_targets


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """Optionally fit a scaler on train inputs and transform both subsets."""
    if not enabled:
        return train_inputs.copy(), val_inputs.copy(), None

    scaler = StandardScaler()
    scaled_train = pd.DataFrame(
        np.asarray(scaler.fit_transform(train_inputs), dtype=float),
        columns=train_inputs.columns,
        index=train_inputs.index,
    )
    scaled_val = pd.DataFrame(
        np.asarray(scaler.transform(val_inputs), dtype=float),
        columns=val_inputs.columns,
        index=val_inputs.index,
    )
    return scaled_train, scaled_val, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Fit one-hot encoding on train categories and transform both subsets."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_train_values = np.asarray(
        encoder.fit_transform(train_inputs),
        dtype=float,
    )
    encoded_val_values = np.asarray(
        encoder.transform(val_inputs),
        dtype=float,
    )
    encoded_columns = encoder.get_feature_names_out(train_inputs.columns).tolist()

    encoded_train = pd.DataFrame(
        encoded_train_values,
        columns=encoded_columns,
        index=train_inputs.index,
    )
    encoded_val = pd.DataFrame(
        encoded_val_values,
        columns=encoded_columns,
        index=val_inputs.index,
    )
    return encoded_train, encoded_val, encoder


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
    numeric_inputs = inputs[NUMERIC_COLUMNS]
    if scaler is not None:
        numeric_inputs = pd.DataFrame(
            np.asarray(scaler.transform(numeric_inputs), dtype=float),
            columns=NUMERIC_COLUMNS,
            index=inputs.index,
        )
    else:
        numeric_inputs = numeric_inputs.copy()

    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    categorical_inputs = pd.DataFrame(
        np.asarray(
            encoder.transform(inputs[CATEGORICAL_COLUMNS]),
            dtype=float,
        ),
        columns=encoded_columns,
        index=inputs.index,
    )
    processed_inputs = pd.concat([numeric_inputs, categorical_inputs], axis="columns")

    missing_columns = [
        column for column in input_cols if column not in processed_inputs.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing processed columns: {missing_columns}")
    return processed_inputs.loc[:, input_cols]
