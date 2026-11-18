"""Train-fitted preprocessing for the Australian weather course notebooks."""

from typing import TypedDict

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ml_homework.preprocessing import get_categorical_columns, transform_features


class WeatherSplits(TypedDict):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class WeatherData(TypedDict):
    train_inputs: pd.DataFrame
    train_targets: pd.Series
    val_inputs: pd.DataFrame
    val_targets: pd.Series
    test_inputs: pd.DataFrame
    test_targets: pd.Series


def split_data_by_year(df: pd.DataFrame, year_col: str) -> WeatherSplits:
    """Use 2015 for validation and later years for testing."""
    year = pd.to_datetime(df[year_col]).dt.year
    return {
        "train": df.loc[year < 2015],
        "val": df.loc[year == 2015],
        "test": df.loc[year > 2015],
    }


def create_inputs_targets(
    splits: WeatherSplits, input_cols: list[str], target_col: str
) -> WeatherData:
    """Copy feature and target columns from each chronological split."""
    return {
        "train_inputs": splits["train"][input_cols].copy(),
        "train_targets": splits["train"][target_col].copy(),
        "val_inputs": splits["val"][input_cols].copy(),
        "val_targets": splits["val"][target_col].copy(),
        "test_inputs": splits["test"][input_cols].copy(),
        "test_targets": splits["test"][target_col].copy(),
    }


def preprocess_data(raw_df: pd.DataFrame) -> WeatherData:
    """Prepare weather features using only training rows to fit transforms."""
    raw_df = raw_df.dropna(subset=["RainToday", "RainTomorrow"])
    splits = split_data_by_year(raw_df, "Date")
    input_cols = list(raw_df.columns)[1:-1]
    data = create_inputs_targets(splits, input_cols, "RainTomorrow")

    train_inputs = data["train_inputs"]
    numeric_cols = train_inputs.select_dtypes(include="number").columns.tolist()
    categorical_cols = get_categorical_columns(train_inputs)

    imputer = SimpleImputer(strategy="mean").fit(train_inputs[numeric_cols])
    scaler = MinMaxScaler().fit(imputer.transform(train_inputs[numeric_cols]))
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_inputs[categorical_cols]
    )

    def transform_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
        numeric_values = imputer.transform(inputs[numeric_cols])
        numeric_df = pd.DataFrame(
            scaler.transform(numeric_values), columns=numeric_cols, index=inputs.index
        )
        categorical_df = transform_features(inputs[categorical_cols], encoder)
        return pd.concat([numeric_df, categorical_df], axis=1)

    data["train_inputs"] = transform_inputs(data["train_inputs"])
    data["val_inputs"] = transform_inputs(data["val_inputs"])
    data["test_inputs"] = transform_inputs(data["test_inputs"])

    return data
