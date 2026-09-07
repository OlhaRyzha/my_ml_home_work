"""Marketing feature aggregation for inspecting customer clusters."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def prepare_cluster_viz_df(
    df: pd.DataFrame,
    labels: ArrayLike,
) -> pd.DataFrame:
    """Add positional cluster labels and marketing totals to a copy of df."""
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) != len(df):
        raise ValueError("labels must contain one cluster label per dataframe row")

    viz_df = df.copy()

    viz_df["Cluster"] = label_array

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    purchase_cols = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    viz_df["Total_Spent"] = viz_df[spending_cols].sum(axis=1)
    viz_df["Total_Purchases"] = viz_df[purchase_cols].sum(axis=1)

    return viz_df
