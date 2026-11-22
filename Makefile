.PHONY: install test lint format typecheck run docker-build clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

run:
	dmc-fraud --help

docker-build:
	docker build -t dmc2019-fraud-hydra:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
