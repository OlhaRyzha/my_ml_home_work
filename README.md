# ML Homework

Homework projects in mathematics, statistics, exploratory data analysis, and
machine learning. Analyses live in Jupyter notebooks; reusable logic is
extracted into a tested Python package.

## Notebooks

| # | Topic | Notebook |
| --- | ----- | -------- |
| 01 | Linear algebra | `notebooks/01_linear_algebra/hw_2_1_matrices_and_vectors.ipynb` |
| 02 | Calculus | `notebooks/02_calculus/hw_2_1_functions_and_derivatives.ipynb` |
| 03 | Statistics | `notebooks/03_statistics/hw_2_1_hypothesis_testing.ipynb` |
| 04 | EDA | `notebooks/04_eda/credit_eda.ipynb` |
| 05 | Linear regression | `notebooks/05_linear_regression/hw_2_1_simple_linear_regression.ipynb` |
| 05 | Multiple linear regression | `notebooks/05_linear_regression/hw_2_1_multiple_linear_regression.ipynb` |

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
| `optimization` | Full-batch gradient descent |
| `calculus` | Symbolic differentiation helpers |
| `visualization` | Reusable matplotlib/seaborn plotting helpers |

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
