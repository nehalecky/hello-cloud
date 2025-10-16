# Cloud Resource Simulator - Development Commands
# Install just: brew install just
# Run: just <command>

# Internal: Publishing configuration (not documented - for flexibility)
_PUBLISH_DIR := "notebooks/published"
_GITHUB_REPO := "nehalecky/hello-cloud"
_GITHUB_BRANCH := "master"

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

# Setup Jupytext pairing for VSCode/Cursor (one-time setup)
nb-setup-pairing:
    @echo "Setting up Jupytext pairing for all notebooks..."
    @mkdir -p notebooks/_build
    @for ipynb in notebooks/_build/*.ipynb; do \
        if [ -f "$$ipynb" ]; then \
            echo "  → $$(basename $$ipynb)"; \
            uv run jupytext --set-formats notebooks//md:myst,notebooks/_build//ipynb "$$ipynb"; \
        fi \
    done
    @echo "✓ Pairing configured. Open .ipynb files from notebooks/_build/ in VSCode"

# Convert notebook to ipynb (for sharing)
nb-convert NAME:
    uv run jupytext notebooks/{{NAME}}.md --to ipynb --output notebooks/_build/{{NAME}}.ipynb

# Execute notebook and save with outputs (for publishing)
nb-execute NAME:
    @echo "Executing {{NAME}}.md → _build/{{NAME}}.ipynb (with outputs)..."
    @mkdir -p notebooks/_build
    uv run jupytext notebooks/{{NAME}}.md --to ipynb --execute --output notebooks/_build/{{NAME}}.ipynb
    @echo "✓ Saved to notebooks/_build/{{NAME}}.ipynb"

# Execute all published notebooks (for site publishing)
nb-execute-all:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Executing published notebooks..."
    mkdir -p notebooks/_build
    # Only execute notebooks that don't require external data files
    # 03: IOPS (uses embedded benchmark data)
    # 05: PiedPiper (SKIP - requires gitignored data files)
    for notebook in 03_EDA_iops_web_server; do
        echo "  → $notebook"
        uv run jupytext "notebooks/$notebook.md" --to ipynb --execute --output "notebooks/_build/$notebook.ipynb"
    done
    echo "✓ All CI-executable notebooks processed"

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

# Preview documentation with auto-refresh (local development server)
docs-serve:
    uv run mkdocs serve

# Build documentation (static site)
docs-build:
    uv run mkdocs build

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy

# Clean documentation artifacts
docs-clean:
    rm -rf site/

# Build documentation (alias for docs-build)
docs: docs-build

# Run all checks (lint + test + docs)
check: lint test docs

# Clean all build artifacts
clean:
    rm -rf docs/_site docs/.quarto .pytest_cache .coverage htmlcov/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Full workflow: clean, install, check
all: clean install check

# ==============================================================================
# Notebook Publishing (Colab)
# ==============================================================================

# Publish notebook (copy from _build/ - assumes already executed)
# Usage: just nb-publish 06_quickstart_timeseries_loader
nb-publish NAME:
    @echo "📦 Publishing {{NAME}}..."
    @mkdir -p {{_PUBLISH_DIR}}
    @# Sync content first
    @uv run jupytext --sync notebooks/{{NAME}}.md
    @# Copy executed notebook
    @cp notebooks/_build/{{NAME}}.ipynb {{_PUBLISH_DIR}}/
    @echo "✓ Published to {{_PUBLISH_DIR}}/{{NAME}}.ipynb"
    @just _nb-colab-url {{NAME}}

# Publish notebook (execute from scratch for clean rebuild)
# Usage: just nb-publish-clean 06_quickstart_timeseries_loader
nb-publish-clean NAME:
    @echo "📦 Publishing {{NAME}} (clean execution)..."
    @mkdir -p {{_PUBLISH_DIR}}
    @uv run jupytext --sync notebooks/{{NAME}}.md
    @uv run jupyter nbconvert --to notebook --execute \
        notebooks/_build/{{NAME}}.ipynb \
        --output-dir={{_PUBLISH_DIR}}
    @echo "✓ Published to {{_PUBLISH_DIR}}/{{NAME}}.ipynb"
    @just _nb-colab-url {{NAME}}

# Publish all notebooks (copy from _build/)
# Usage: just nb-publish-all
nb-publish-all:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📦 Publishing all notebooks..."
    mkdir -p {{_PUBLISH_DIR}}

    notebooks=(
        "02_guide_workload_signatures_guide"
        "03_EDA_iops_web_server"
        "04_modeling_gaussian_process"
        "05_EDA_piedpiper_data"
        "06_quickstart_timeseries_loader"
        "07_forecasting_comparison"
    )

    for notebook in "${notebooks[@]}"; do
        echo "  → $notebook"
        uv run jupytext --sync "notebooks/$notebook.md"
        cp "notebooks/_build/$notebook.ipynb" {{_PUBLISH_DIR}}/
    done

    echo "✓ Published ${#notebooks[@]} notebooks to {{_PUBLISH_DIR}}/"

# Publish all notebooks (execute from scratch)
# Usage: just nb-publish-all-clean
nb-publish-all-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📦 Publishing all notebooks (clean execution)..."
    mkdir -p {{_PUBLISH_DIR}}

    notebooks=(
        "02_guide_workload_signatures_guide"
        "03_EDA_iops_web_server"
        "04_modeling_gaussian_process"
        "05_EDA_piedpiper_data"
        "06_quickstart_timeseries_loader"
        "07_forecasting_comparison"
    )

    for notebook in "${notebooks[@]}"; do
        echo "  → $notebook"
        uv run jupytext --sync "notebooks/$notebook.md"
        uv run jupyter nbconvert --to notebook --execute \
            "notebooks/_build/$notebook.ipynb" \
            --output-dir={{_PUBLISH_DIR}} \
            2>&1 | grep -v "WARNING" || true
    done

    echo "✓ Published ${#notebooks[@]} notebooks to {{_PUBLISH_DIR}}/"

# Publish and commit all notebooks
# Usage: just nb-publish-commit "docs: update tutorials"
nb-publish-commit MESSAGE="docs: update published notebooks":
    @just nb-publish-all
    @git add {{_PUBLISH_DIR}}/*.ipynb
    @git commit -m "{{MESSAGE}}"
    @echo "✓ Committed. Push with: git push"

# Internal: Generate Colab URL
_nb-colab-url NAME:
    @echo "🔗 https://colab.research.google.com/github/{{_GITHUB_REPO}}/blob/{{_GITHUB_BRANCH}}/{{_PUBLISH_DIR}}/{{NAME}}.ipynb"
