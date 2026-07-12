from collections.abc import Iterable

import pandas as pd


def column_summary(
    data: pd.DataFrame, null_summary: pd.DataFrame, column: str
) -> dict[str, object]:
    """Build a missing-value and central-tendency summary for one column."""
    if column not in data.columns:
        raise KeyError(f"Unknown data column: {column}")
    required = {"column_name", "null_count", "null_percentage"}
    missing = required.difference(null_summary.columns)
    if missing:
        raise KeyError(f"Missing null-summary columns: {sorted(missing)}")

    row = null_summary.loc[null_summary["column_name"] == column]
    if row.empty:
        raise KeyError(f"No null summary found for column: {column}")

    series = data[column]
    modes = series.mode(dropna=True)
    return {
        "column_name": column,
        "null_count": row["null_count"].iloc[0],
        "null_percentage": row["null_percentage"].iloc[0],
        "mode": modes.iloc[0] if not modes.empty else pd.NA,
        "mean": series.mean(),
        "median": series.median(),
    }


def summary_table(
    data: pd.DataFrame, null_summary: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    """Build column summaries without relying on notebook globals."""
    summaries = [column_summary(data, null_summary, column) for column in columns]
    return pd.DataFrame(summaries).set_index("column_name").round(2)


def null_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column missing-value summary with counts and percentages."""
    null_count = data.isnull().sum()
    return pd.DataFrame(
        {
            "column_name": data.columns,
            "null_count": null_count,
            "null_percentage": ((null_count / len(data)) * 100).round(2),
        }
    ).reset_index(drop=True)


def years_from_days(days: pd.Series) -> pd.Series:
    """Convert a signed day count to whole years."""
    return (days.abs() / 365).round()


def age_group(years: float) -> str:
    """Map a non-negative age to a ten-year category."""
    if years < 0:
        raise ValueError("years must not be negative")
    if years <= 20:
        return "0-20"
    if years > 70:
        return "70+"
    upper = int((years - 1) // 10 + 1) * 10
    lower = upper - 10
    return f"{lower}-{upper}"


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Return lower and upper Tukey fences for a numeric series."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("series must contain numeric values")
    first_quartile = float(numeric.quantile(0.25))
    third_quartile = float(numeric.quantile(0.75))
    spread = third_quartile - first_quartile
    return (
        first_quartile - multiplier * spread,
        third_quartile + multiplier * spread,
    )


def upper_outlier_bound(data: pd.DataFrame, column: str) -> float:
    """Return the upper Tukey fence for a DataFrame column."""
    return iqr_bounds(data[column])[1]
