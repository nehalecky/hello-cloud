# Cloud Resource Simulator - Development Commands
# Install just: brew install just
# Run: just <command>

# Show available commands
default:
    @just --list

# Install all dependencies (dev, research, docs)
install:
    uv sync --extra dev --extra research --extra docs
    uv run pre-commit install

# Run tests with coverage
test:
    uv run pytest tests/ -v --cov=src/hellocloud --cov-report=term-missing

# Run tests without coverage (faster)
test-fast:
    uv run pytest tests/ -v

# Start Jupyter Lab for interactive development
lab:
    uv run jupyter lab notebooks/

# Start Jupyter Lab (alias)
jupyter-lab: lab

# Start classic Jupyter Notebook
notebook:
    uv run jupyter notebook notebooks/

# Run linter
lint:
    uv run ruff check src/ tests/

# Fix linting issues
lint-fix:
    uv run ruff check --fix src/ tests/

# Format code with black
format:
    uv run black src/ tests/

# Format and fix linting (recommended before commits)
fix:
    uv run black src/ tests/
    uv run ruff check --fix src/ tests/

# Run pre-commit hooks on all files
pre-commit:
    uv run pre-commit run --all-files

# Install pre-commit hooks (run once after clone)
pre-commit-install:
    uv run pre-commit install

# Update pre-commit hook versions
pre-commit-update:
    uv run pre-commit autoupdate

# Generate API reference documentation
docs-api:
    uv run quartodoc build --config docs/_quarto.yml

# Preview documentation (auto-refreshes)
docs-preview:
    quarto preview docs/

# Build documentation
docs-build:
    quarto render docs/

# Clean documentation artifacts
docs-clean:
    rm -rf docs/_site docs/.quarto

# Generate API docs and build site
docs: docs-api docs-build

# Run all checks (lint + test + docs)
check: lint test docs

# Clean all build artifacts
clean:
    rm -rf docs/_site docs/.quarto .pytest_cache .coverage htmlcov/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Full workflow: clean, install, check
all: clean install check
