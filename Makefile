PYTHON ?= python
CONFIG ?= default

.PHONY: help install dev lint format test test-fast hw run-all tiny benchmark report serve clean

help:  ## list targets
	@grep -E "^[a-zA-Z_-]+:.*?## " $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## install the package
	$(PYTHON) -m pip install -e .

dev:  ## install with dev extras (pytest, ruff)
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## auto-format
	ruff check --fix .
	ruff format .

test:  ## full test suite (tiny config, CPU; GPU tests skip without CUDA)
	$(PYTHON) -m pytest -q

test-fast:  ## unit tests only (no end-to-end pipeline)
	$(PYTHON) -m pytest -q --ignore=tests/test_cli_e2e.py --ignore=tests/test_api.py

hw:  ## print CPU / RAM / GPU information
	$(PYTHON) -m dcforecast hw

run-all:  ## full pipeline (generate -> report), ~10-20 min on a laptop GPU
	$(PYTHON) -m dcforecast -c $(CONFIG) run-all

tiny:  ## the whole pipeline on the tiny CPU profile (~1 min)
	$(PYTHON) -m dcforecast -c tiny run-all

benchmark:  ## CPU vs GPU benchmark only
	$(PYTHON) -m dcforecast -c $(CONFIG) benchmark

report:  ## rebuild docs/index.html and docs/images
	$(PYTHON) -m dcforecast -c $(CONFIG) report

serve:  ## FastAPI service on http://127.0.0.1:8000
	$(PYTHON) -m dcforecast -c $(CONFIG) serve

clean:  ## remove generated data, runs and caches (keeps docs/)
	rm -rf data runs reports .pytest_cache .ruff_cache .coverage htmlcov
