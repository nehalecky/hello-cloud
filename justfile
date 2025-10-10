# Cloud Resource Simulator - Development Commands
# Install just: brew install just
# Run: just <command>

# Show available commands
default:
    @just --list

# Install all dependencies
install:
    uv sync --all-extras
    uv sync --group docs

# Run tests with coverage
test:
    uv run pytest tests/ -v --cov=src/cloud_sim --cov-report=term-missing

# Run tests without coverage (faster)
test-fast:
    uv run pytest tests/ -v

# Run linter
lint:
    uv run ruff check src/ tests/

# Fix linting issues
lint-fix:
    uv run ruff check --fix src/ tests/

# Format code
format:
    uv run black src/ tests/

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
