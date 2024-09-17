.PHONY: all clean test lint fmt install install-dev install-all build docs help

# Specify the default target
.DEFAULT_GOAL := help

# Define variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
COVERAGE := $(VENV)/bin/coverage
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
FLAKE8 := $(VENV)/bin/flake8
MYPY := $(VENV)/bin/mypy

# Activate virtual environment
VENV_ACTIVATE := . $(VENV)/bin/activate

# Help target
help:
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# Install the package
install: $(VENV)/bin/activate ## Install the package
	$(PIP) install .

# Install development dependencies
install-dev: $(VENV)/bin/activate ## Install development dependencies
	$(PIP) install -e '.[dev]'

# Install all dependencies
install-all: $(VENV)/bin/activate ## Install all dependencies
	$(PIP) install -e '.[all]'

# Run tests
test: ## Run tests
	$(VENV_ACTIVATE) && $(PYTEST) -v tests/

# Run linters
lint: ## Run linters
	bash -c "source $(VENV)/bin/activate && $(FLAKE8) src/ && $(MYPY) src/"
# Format code
fmt: ## Format code
	bash -c "source $(VENV)/bin/activate && $(BLACK) src/ && $(ISORT) src/"

# Build the package
build: clean ## Build the package
	$(PYTHON) -m build

# Generate documentation
docs: ## Generate documentation
	$(VENV_ACTIVATE) && $(SPHINX_BUILD) -b html docs/ docs/_build/html

# Clean up
clean: ## Clean up build artifacts and cache
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Run all checks
all: install-dev lint test build docs ## Run all checks

# Additional targets can be added as needed

