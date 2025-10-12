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

# Open specific notebook in Jupyter Lab
nb NAME:
    uv run jupyter lab notebooks/{{NAME}}.md

# Test notebook execution (convert + execute)
nb-test NAME:
    uv run jupytext notebooks/{{NAME}}.md --to ipynb --execute --output /dev/null

# Convert notebook to ipynb (for sharing)
nb-convert NAME:
    uv run jupytext notebooks/{{NAME}}.md --to ipynb --output notebooks/_build/{{NAME}}.ipynb

# Execute notebook and save with outputs (for publishing)
nb-execute NAME:
    @echo "Executing {{NAME}}.md → _build/{{NAME}}.ipynb (with outputs)..."
    @mkdir -p notebooks/_build
    uv run jupytext notebooks/{{NAME}}.md --to ipynb --execute --output notebooks/_build/{{NAME}}.ipynb
    @echo "✓ Saved to notebooks/_build/{{NAME}}.ipynb"

# Execute all notebooks (for site publishing)
nb-execute-all:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Executing all notebooks..."
    mkdir -p notebooks/_build
    for notebook in notebooks/*.md; do
        name=$(basename "$notebook" .md)
        echo "  → $name"
        uv run jupytext "$notebook" --to ipynb --execute --output notebooks/_build/$name.ipynb
    done
    echo "✓ All notebooks executed"

# Test all notebooks as smoke tests
nb-test-all:
    uv run pytest tests/test_notebooks.py -m smoke -v

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

# View specific tutorial (opens in browser after building)
tut NAME:
    quarto render docs/tutorials/{{NAME}}.qmd
    open docs/_site/tutorials/{{NAME}}.html

# Preview specific tutorial with live reload
tut-preview NAME:
    quarto preview docs/tutorials/{{NAME}}.qmd

# Run all checks (lint + test + docs)
check: lint test docs

# Clean all build artifacts
clean:
    rm -rf docs/_site docs/.quarto .pytest_cache .coverage htmlcov/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Full workflow: clean, install, check
all: clean install check
