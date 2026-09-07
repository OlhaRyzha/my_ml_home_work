import pandas as pd
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
        numeric_df = pd.DataFrame(
            StandardScaler().fit_transform(numeric_df),
            columns=numeric_df.columns,
            index=df.index,
        )

    if not categorical_columns:
        return numeric_df

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
        index=df.index,
    )
    return pd.concat([numeric_df, encoded_df], axis=1)
