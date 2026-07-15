import pandas as pd
import pytest

from ml_homework.eda import (
    age_group,
    get_columns_summary,
    iqr_bounds,
    null_summary,
    summary_table,
    years_from_days,
)


def test_get_columns_summary_preserves_columns_and_ignores_null_values() -> None:
    data = pd.DataFrame(
        {
            "category": ["a", "b", "a", None],
            "number": [1.0, 1.0, 2.0, None],
        }
    )

    result = get_columns_summary(["number", "category"], data)

    assert result["column"].tolist() == ["number", "category"]
    assert result["nunique"].tolist() == [2, 2]
    assert result["unique_values"].tolist() == [[1.0, 2.0], ["a", "b"]]


@pytest.mark.parametrize(
    ("age", "expected"),
    [(18, "0-20"), (20, "0-20"), (35, "30-40"), (70, "60-70"), (71, "70+")],
)
def test_age_group(age: float, expected: str) -> None:
    assert age_group(age) == expected


def test_iqr_bounds_returns_both_fences() -> None:
    assert iqr_bounds(pd.Series([1, 2, 3, 4, 100])) == pytest.approx((-1, 7))


def test_null_summary_counts_and_percentages() -> None:
    data = pd.DataFrame({"a": [1.0, None, None, None], "b": [1, 2, 3, 4]})
    result = null_summary(data)
    assert list(result["column_name"]) == ["a", "b"]
    assert list(result["null_count"]) == [3, 0]
    assert list(result["null_percentage"]) == [75.0, 0.0]


def test_years_from_days_handles_negative_values() -> None:
    result = years_from_days(pd.Series([-730, 365, 0]))
    assert list(result) == [2.0, 1.0, 0.0]


def test_summary_table_uses_explicit_inputs() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 2.0, None]})
    nulls = pd.DataFrame(
        {"column_name": ["value"], "null_count": [1], "null_percentage": [25.0]}
    )
    result = summary_table(data, nulls, ["value"])
    assert result.loc["value", "mode"] == 2
    assert result.loc["value", "null_count"] == 1
