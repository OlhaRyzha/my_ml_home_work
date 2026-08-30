.PHONY: setup sync lab format lint typecheck test check hooks

setup:
	uv sync
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

sync:
	uv sync

lab:
	uv run jupyter lab

format:
	uv run ruff check . --fix
	uv run black .

lint:
	uv run ruff check .
	uv run black --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

hooks:
	uv run pre-commit run --all-files
	uv run pre-commit run --all-files --hook-stage pre-push
