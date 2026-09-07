# ML Homework

Homework projects in mathematics, statistics, exploratory data analysis, and
machine learning. Analyses live in Jupyter notebooks; reusable logic is
extracted into a tested Python package.

## Notebooks

| # | Topic | Notebook |
| --- | ----- | -------- |
| 01 | Linear algebra | [Matrices and vectors](notebooks/01_linear_algebra/hw_2_1_matrices_and_vectors.ipynb) |
| 02 | Calculus | [Functions and derivatives](notebooks/02_calculus/hw_2_1_functions_and_derivatives.ipynb) |
| 03 | Statistics | [Hypothesis testing](notebooks/03_statistics/hw_2_1_hypothesis_testing.ipynb) |
| 04 | EDA | [Credit EDA](notebooks/04_eda/credit_eda.ipynb) |
| 05 | Linear regression | [Simple linear regression](notebooks/05_linear_regression/hw_2_1_simple_linear_regression.ipynb) |
| 05 | Multiple linear regression | [Multiple linear regression](notebooks/05_linear_regression/hw_2_1_multiple_linear_regression.ipynb) |
| 06 | Logistic regression | [Mathematical formulation](notebooks/06_logistic_regression/hw_2_2_mathematical_formulation_of_logistic_regression.ipynb) |
| 06 | Logistic regression | [Scikit-learn implementation](notebooks/06_logistic_regression/hw_2_2_logistic_regression_with_scikit_learn.ipynb) |
| 06 | Logistic regression | [Polynomial features and pipelines](notebooks/06_logistic_regression/hw_2_2_polynomial_features_and_pipelines.ipynb) |
| 06 | Multiclass classification | [Imbalanced multiclass classification](notebooks/06_logistic_regression/hw_2_2_imbalanced_multiclass_classification.ipynb) |
| 07 | Decision trees | [Decision trees](notebooks/07_decision_tree/hw_2_3_decision_trees.ipynb) |
| 09 | Visualization | [Decision regions demo](notebooks/09_visualization/decision_regions_demo.ipynb) |

## Project Structure

```text
.
|-- data/
|   |-- raw/          # Immutable input datasets (never modified)
|   `-- processed/    # Derived artifacts produced by notebooks
|-- notebooks/        # Jupyter notebooks, one topic per folder
|-- src/
|   `-- ml_homework/  # Reusable, importable project code
|-- tests/            # Pytest suite for the package
|-- Makefile          # Common project commands
|-- pyproject.toml    # Dependencies and tool configuration (uv)
`-- uv.lock           # Locked package versions
```

### The `ml_homework` package

| Module | Purpose |
| ------ | ------- |
| `paths` | Canonical filesystem paths (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`) |
| `eda` | Missing-value summaries, IQR outlier bounds, feature bucketing |
| `metrics` | Evaluation metrics (RMSE) |
| `modeling` | Linear-regression training and coefficient inspection |
| `classification` | Classifier construction, prediction, and score comparisons |
| `preprocessing` | Column selection, stratified splitting, train-fitted scaling/encoding |
| `preprocessing.bank_churn` | Bank-specific feature rules and training/inference workflow |
| `preprocessing.customer_marketing` | Marketing totals and cluster-label preparation |
| `optimization` | Full-batch gradient descent |
| `calculus` | Numerical differentiation helpers |
| `visualization` | Exploratory plots: distributions, categories, outliers |
| `visualization.models` | Decision regions, regression predictions, tree-depth curves |
| `visualization.classification` | Confusion/ROC plots and notebook classification reports |
| `visualization.clustering` | Cluster scatterplots and elbow curves |
| `clustering` | K-means fitting, evaluation, and cluster summaries |

### Choosing where code belongs

Keep the existing `src/ml_homework` package layout. Group code by responsibility
rather than adding a catch-all `helpers` directory. Small, cohesive modules stay
flat; only preprocessing and visualization currently warrant subpackages.
Their `__init__.py` files implement shared operations, not forwarding-only imports.

- Import from the owning module, for example
  `from ml_homework.preprocessing.bank_churn import preprocess_data`.
- Keep dataset column names and feature rules in dataset modules.
- Shared transforms fit on training rows and reuse those fitted objects for
  validation/inference. `transform_features` never fits and preserves row indices.
- `prepare_clustering_features` fits a fresh exploratory transform on every call;
  it must not be used to preprocess held-out data independently.
- Plot functions accept data or predictions. `plot_binary_confusion_matrix` and
  `plot_roc_curve` return `(Figure, Axes)` without printing or predicting.
- Named notebook reports (`predict_and_plot`, `compute_auroc_and_build_roc`,
  `evaluate_multiclass_model`, regression `*_show_*`) deliberately orchestrate
  multiple steps for teaching. Prefer computation/plot primitives for reuse.
- Preserve the established root-level API for existing notebooks. Moved functions
  use new direct module paths; the repository notebooks have been migrated.

Architecture references: [PyPA src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
and [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
The folder boundaries above are a project-specific design choice. For new
cross-validation workflows, prefer scikit-learn pipelines so each fold fits its
own preprocessing.

Anything used by more than one notebook lives here, with a matching test in
`tests/`. Notebooks resolve data paths via `ml_homework.paths` — no hardcoded
relative paths.

## Getting Started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
make setup    # create .venv, install dependencies, install git hooks
make lab      # start Jupyter Lab
```

After `git pull` or a `pyproject.toml` change:

```bash
make sync
```

## Development

```bash
make format   # Ruff autofix + Black
make lint     # Ruff + Black (check only)
make test     # pytest
make check    # Ruff + Black + mypy + pytest
```

Quality gates run automatically:

- `git commit` runs Ruff and Black on changed files.
- `git push` runs the full `make check`.
- If a hook modifies a file, re-stage it and commit again.

## Conventions

- `data/raw/` is read-only; derived artifacts go to `data/processed/`.
- Notebooks are committed with outputs so results are viewable directly on
  GitHub; re-run a notebook top to bottom before committing changes to it.
- Stochastic code (splits, gradient descent) uses explicit random seeds.
- Typed code throughout; `mypy` runs as part of `make check`.
