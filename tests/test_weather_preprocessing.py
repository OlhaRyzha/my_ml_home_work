import pandas as pd

from ml_homework.preprocessing.weather import preprocess_data


def test_weather_preprocessing_fits_train_only_and_keeps_split_indices() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2014-01-01", "2014-02-01", "2015-01-01", "2016-01-01"],
            "MinTemp": [0.0, 10.0, 100.0, None],
            "RainToday": ["No", "Yes", "Unknown", "No"],
            "RainTomorrow": ["No", "Yes", "No", "Yes"],
        },
        index=[10, 11, 12, 13],
    )

    data = preprocess_data(raw)

    assert data["train_inputs"].index.tolist() == [10, 11]
    assert data["val_inputs"].index.tolist() == [12]
    assert data["test_inputs"].index.tolist() == [13]
    assert data["train_inputs"]["MinTemp"].tolist() == [0.0, 1.0]
    assert data["val_inputs"]["MinTemp"].iloc[0] == 10.0
    assert data["test_inputs"]["MinTemp"].iloc[0] == 0.5
    assert data["val_inputs"].columns.tolist() == data["train_inputs"].columns.tolist()
    assert data["val_inputs"].filter(like="RainToday_").sum(axis=1).iloc[0] == 0
    assert data["val_targets"].iloc[0] == "No"
