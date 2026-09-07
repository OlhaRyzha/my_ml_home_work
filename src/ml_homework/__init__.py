"""Established notebook API; new code should import from the owning module."""

from ml_homework.classification import (
    build_ovr_logistic_pipeline,
    compare_classification_metrics,
    compare_multiclass_predictions,
    compute_auroc,
    get_f1_score,
    max_depth_auroc,
    predict_majority_class,
    predict_raw_df,
)
from ml_homework.clustering import (
    evaluate_kmeans,
    fit_kmeans,
    summarize_customer_clusters,
)
from ml_homework.eda import get_columns_summary
from ml_homework.metrics import compare_regression_metrics
from ml_homework.modeling import (
    linear_regression_coefficients,
    select_feature_columns,
    significant_ols_coefficients,
    train_linear_regression_show_rmse,
    train_ols_show_summary,
)
from ml_homework.paths import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from ml_homework.preprocessing import prepare_clustering_features
from ml_homework.preprocessing.bank_churn import (
    PreprocessedData,
    add_age_group,
    preprocess_data,
    preprocess_new_data,
)
from ml_homework.preprocessing.customer_marketing import prepare_cluster_viz_df
from ml_homework.visualization.classification import (
    compute_auroc_and_build_roc,
    evaluate_multiclass_model,
    predict_and_plot,
)
from ml_homework.visualization.clustering import plot_customer_clusters, plot_elbow
from ml_homework.visualization.models import plot_decision_regions

__all__ = [
    "summarize_customer_clusters",
    "evaluate_kmeans",
    "fit_kmeans",
    "plot_elbow",
    "plot_customer_clusters",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "PreprocessedData",
    "RAW_DATA_DIR",
    "add_age_group",
    "build_ovr_logistic_pipeline",
    "compare_classification_metrics",
    "compare_multiclass_predictions",
    "compute_auroc",
    "compute_auroc_and_build_roc",
    "compare_regression_metrics",
    "get_columns_summary",
    "evaluate_multiclass_model",
    "get_f1_score",
    "linear_regression_coefficients",
    "max_depth_auroc",
    "predict_and_plot",
    "predict_majority_class",
    "predict_raw_df",
    "plot_decision_regions",
    "prepare_cluster_viz_df",
    "preprocess_data",
    "prepare_clustering_features",
    "preprocess_new_data",
    "select_feature_columns",
    "significant_ols_coefficients",
    "train_linear_regression_show_rmse",
    "train_ols_show_summary",
]
