"""Cluster assignments and model-selection plots."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_clusters(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "Cluster",
    title: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))

    cluster_ids = sorted(df[cluster_col].unique())
    colors = plt.get_cmap("tab10")
    for position, cluster_id in enumerate(k for k in cluster_ids if k != -1):
        rows = df.loc[df[cluster_col] == cluster_id]
        plt.scatter(
            rows[x_col],
            rows[y_col],
            color=colors(position % 10),
            alpha=0.6,
            label=f"Cluster {cluster_id} (n={len(rows)})",
        )
    noise = df.loc[df[cluster_col] == -1]
    if not noise.empty:
        plt.scatter(
            noise[x_col],
            noise[y_col],
            color="gray",
            marker="x",
            alpha=0.6,
            label=f"Noise (-1, n={len(noise)})",
        )
    plt.legend()

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"Customer Clusters: {x_col} vs {y_col}")

    plt.tight_layout()
    plt.show()


def plot_elbow(results: pd.DataFrame, *, title: str = "Elbow Method") -> None:
    """Plot inertia from evaluate_kmeans results on a new figure."""
    plt.figure()
    plt.plot(results["k"], results["inertia"], "bx-")
    plt.xlabel("K values")
    plt.ylabel("Sum of Squared Distances")
    plt.title(title)
    plt.show()


def plot_customer_clusters(viz_df: pd.DataFrame, *, title: str) -> None:
    """Show the three marketing views for a prepared cluster dataframe."""
    for x_col, y_col, description in [
        ("Income", "Total_Spent", "Income vs Total Spending"),
        ("Income", "Total_Purchases", "Income vs Total Purchases"),
        ("Recency", "Total_Spent", "Recency vs Total Spending"),
    ]:
        plot_clusters(viz_df, x_col=x_col, y_col=y_col, title=f"{title}: {description}")
