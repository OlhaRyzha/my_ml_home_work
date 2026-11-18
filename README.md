# ML Homework

Homework projects in mathematics, statistics, exploratory data analysis, and
machine learning. Analyses live in Jupyter notebooks; reusable logic is
extracted into a tested Python package.

## Notebooks

Course examples and homework live together in topic folders. Imported lesson notebooks are in each topic's `course/` subfolder and keep their course titles; existing homework solutions remain in place. The course examples introduce code step by step, so the project lint and notebook structure checks apply to homework notebooks outside `course/`.

| Topic | Folder | Notebooks |
| --- | --- | ---: |
| Python basics | [00_python](notebooks/00_python/) | 7 |
| NumPy | [00_numpy](notebooks/00_numpy/) | 1 |
| Pandas | [00_pandas](notebooks/00_pandas/) | 3 |
| Linear algebra | [01_linear_algebra](notebooks/01_linear_algebra/) | 2 |
| Calculus | [02_calculus](notebooks/02_calculus/) | 2 |
| Statistics | [03_statistics](notebooks/03_statistics/) | 4 |
| EDA | [04_eda](notebooks/04_eda/) | 3 |
| Linear regression | [05_linear_regression](notebooks/05_linear_regression/) | 7 |
| Logistic regression | [06_logistic_regression](notebooks/06_logistic_regression/) | 8 |
| Decision trees | [07_decision_tree](notebooks/07_decision_tree/) | 2 |
| Cross validation and kNN | [08_cross_validation](notebooks/08_cross_validation/) | 3 |
| Visualization | [09_visualization](notebooks/09_visualization/) | 2 |
| Boosting and ensembles | [10_boosting](notebooks/10_boosting/) | 3 |
| Clustering | [11_clustering algorithms](notebooks/11_clustering%20algorithms/) | 10 |
| Dimensionality reduction | [12_dimensionality_reduction](notebooks/12_dimensionality_reduction/) | 2 |
| NLP | [13_nlp](notebooks/13_nlp/) | 7 |
| Deep learning | [14_deep_learning](notebooks/14_deep_learning/) | 1 |
| Explainable AI | [15_explainable_ai](notebooks/15_explainable_ai/) | 1 |

Some imported examples also expect `tip.csv`, `diabetes.csv`,
`estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.csv`,
or `california_housing_prices.csv` in `data/raw/`. These files were not in the
linked Drive folder, so those examples need the course datasets before running.

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
| `preprocessing.weather` | Weather data splits and train-fitted preprocessing |
| `optimization` | Full-batch gradient descent |
| `calculus` | Numerical differentiation helpers |
| `dimensionality_reduction` | Standardized PCA + K-Means, t-SNE embeddings, cluster profiles, and PCA loading summaries |
| `visualization` | Exploratory plots: distributions, categories, outliers |
| `visualization.models` | Decision regions, regression predictions, tree-depth curves |
| `visualization.classification` | Confusion/ROC plots and notebook classification reports |
| `visualization.clustering` | Cluster scatterplots and elbow curves |
| `visualization.dimensionality_reduction` | Reusable 2D and 3D Plotly plots for PCA/t-SNE embeddings |
| `clustering` | K-means fitting, evaluation, and cluster summaries |
| `text` | Vocabulary coverage and train-only Bag-of-Words fitting |

### Dimensionality-reduction helpers

The dimensionality-reduction notebook uses tested helpers instead of repeating
pipeline and plotting code in each cell:

```python
from ml_homework.dimensionality_reduction import (
    fit_pca_kmeans,
    fit_tsne_embedding,
    summarize_standardized_clusters,
    top_loadings,
)
from ml_homework.visualization.dimensionality_reduction import (
    plot_embedding_clusters,
)
```

`fit_pca_kmeans` standardizes features, fits PCA and K-Means, and returns the
pipeline, cluster labels, PCA embedding, and silhouette score. `fit_tsne_embedding`
standardizes features and fits t-SNE for visualization. t-SNE is intentionally
not placed inside a scikit-learn `Pipeline`: it provides `fit_transform`, but no
standalone `transform` method. `top_loadings` returns the strongest absolute
loadings per principal component, while `summarize_standardized_clusters`
returns per-cluster feature means in standardized units.

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
