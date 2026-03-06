.PHONY: install dev test test-cov lint format build clean docs

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ─── Setup ───

install: $(VENV)
	$(PIP) install -e .

dev: $(VENV)
	$(PIP) install -e ".[dev]"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

# ─── Testing ───

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=march --cov-report=term-missing --cov-report=html

# ─── Linting & Formatting ───

lint:
	$(PYTHON) -m ruff check march/ tests/

format:
	$(PYTHON) -m ruff format march/ tests/
	$(PYTHON) -m ruff check --fix march/ tests/

# ─── Build ───

build:
	$(PYTHON) -m build --sdist --wheel

# ─── Clean ───

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

# ─── Docs ───

docs:
	@echo "Documentation generation not yet configured."
	@echo "Future: mkdocs build or sphinx-build"
