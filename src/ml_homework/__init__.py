"""Public project configuration used by notebooks and Python modules."""

from ml_homework.classification import (
    add_age_group,
    build_ovr_logistic_pipeline,
    compare_classification_metrics,
    compare_multiclass_predictions,
    compute_auroc_and_build_roc,
    evaluate_multiclass_model,
    get_f1_score,
    predict_and_plot,
    predict_majority_class,
    predict_raw_df,
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

__all__ = [
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "add_age_group",
    "build_ovr_logistic_pipeline",
    "compare_classification_metrics",
    "compare_multiclass_predictions",
    "compute_auroc_and_build_roc",
    "compare_regression_metrics",
    "get_columns_summary",
    "evaluate_multiclass_model",
    "get_f1_score",
    "linear_regression_coefficients",
    "predict_and_plot",
    "predict_majority_class",
    "predict_raw_df",
    "select_feature_columns",
    "significant_ols_coefficients",
    "train_linear_regression_show_rmse",
    "train_ols_show_summary",
]
