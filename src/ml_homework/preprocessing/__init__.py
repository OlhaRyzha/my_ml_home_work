"""Shared column selection, splitting, and train-fitted tabular transforms."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of numerical columns in the DataFrame."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of categorical columns in the DataFrame."""
    return df.select_dtypes(["object", "string", "category"]).columns.tolist()


def get_numerical_and_categorical_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return a tuple of numerical and categorical columns in the DataFrame."""
    return get_numerical_columns(df), get_categorical_columns(df)


def prepare_clustering_features(
    df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    scale_numeric: bool = False,
) -> pd.DataFrame:
    """Fit fresh preprocessing for a clustering experiment, preserving row indices.

    One-hot encode categorical columns and optionally standardize the remaining
    numeric columns. The input must already have missing values and dates handled.
    Call again after filtering rows to refit on the retained data. This helper
    fits on each call; it is not a transform for held-out or inference data.
    """
    numeric_df = df.drop(columns=categorical_columns).copy()
    if scale_numeric and not numeric_df.empty:
        numeric_df = transform_features(numeric_df, StandardScaler().fit(numeric_df))

    if not categorical_columns:
        return numeric_df

    categorical_df = df[categorical_columns]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        categorical_df
    )
    encoded_df = transform_features(categorical_df, encoder)
    return pd.concat([numeric_df, encoded_df], axis=1)


def transform_features(
    inputs: pd.DataFrame,
    transformer: StandardScaler | OneHotEncoder | None,
) -> pd.DataFrame:
    """Transform with a fitted dense encoder/scaler, preserving row indices.

    None returns an independent copy. This function never fits a transformer;
    validation and inference must reuse the object fitted on training rows.
    OneHotEncoder must be configured with sparse_output=False.
    """
    if transformer is None:
        return inputs.copy()
    return pd.DataFrame(
        np.asarray(transformer.transform(inputs), dtype=float),
        columns=transformer.get_feature_names_out(inputs.columns),
        index=inputs.index,
    )


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
    """Fit only on training rows and apply the same scaling to both subsets."""
    scaler = StandardScaler().fit(train_inputs) if enabled else None
    return (
        transform_features(train_inputs, scaler),
        transform_features(val_inputs, scaler),
        scaler,
    )


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """Fit categories on training rows; unseen validation categories become zeros."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_inputs
    )
    return (
        transform_features(train_inputs, encoder),
        transform_features(val_inputs, encoder),
        encoder,
    )
