# Contributing

Thank you for your interest in contributing to hello cloud! This guide will help you get started.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nehalecky/hello-cloud.git
cd hello-cloud

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks
just pre-commit-install

# Run tests
just test

# Start docs server
just docs-serve
```

## Development Workflow

### 1. Set Up Your Environment

```bash
# Create a virtual environment with uv
uv venv

# Install all dependencies
uv sync --all-extras

# Activate the environment
source .venv/bin/activate
```

### 2. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/issue-description
```

### 3. Make Changes

Follow our coding standards:

- **Code Style**: Black (automated via pre-commit)
- **Linting**: Ruff (replaces flake8, isort)
- **Type Hints**: Use them liberally
- **Docstrings**: Google style
- **Tests**: Aim for >70% coverage

### 4. Test Your Changes

```bash
# Run all tests
just test

# Run specific test file
uv run pytest tests/test_your_module.py -v

# Check coverage
just test
```

### 5. Format and Lint

```bash
# Format and fix all issues at once (recommended)
just fix

# Or run individually
just format    # Black formatting
just lint      # Ruff linting
just lint-fix  # Auto-fix ruff issues
```

### 6. Commit Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new forecasting model"
git commit -m "fix: correct GP kernel initialization"
git commit -m "docs: improve quickstart tutorial"
git commit -m "test: add unit tests for TimeSeries loader"
```

**Commit Types:**
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

### 7. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create PR on GitHub
# Use the PR template and link related issues
```

## Contribution Areas

### 🐛 Bug Fixes

Found a bug? Please:

1. Check if an issue already exists
2. If not, create one with reproduction steps
3. Reference the issue in your PR

### ✨ New Features

Before implementing a new feature:

1. Open an issue to discuss the design
2. Get feedback from maintainers
3. Implement with tests and documentation
4. Update relevant tutorials/examples

### 📚 Documentation

Documentation improvements are always welcome:

- Fix typos, clarify explanations
- Add examples to docstrings
- Create new tutorials
- Improve API reference

### 🧪 Testing

Help us improve test coverage:

- Add unit tests for untested code
- Create integration tests
- Add edge case tests
- Improve test clarity

### 🔬 Research

Contribute research insights:

- Analyze new cloud datasets
- Validate/challenge existing findings
- Add literature reviews
- Share production observations

## Code Quality Standards

For detailed tool configurations, workflows, and troubleshooting:

**[→ Code Quality Guide](code-quality.md)**

### Testing

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Coverage**: Aim for >70% (enforced in CI)
- **Test Data**: Use fixtures from `tests/conftest.py`

### Documentation

- **Docstrings**: Google style for all public APIs
- **Type Hints**: Use for function signatures
- **Examples**: Include usage examples in docstrings
- **Notebooks**: Python percent format (`.py` in `examples/`)

### Style Guide

```python
# Good: Clear naming, type hints, docstring
def calculate_cpu_utilization(
    timestamps: pd.DatetimeIndex,
    values: np.ndarray,
    window_size: int = 5
) -> pd.Series:
    """Calculate rolling average CPU utilization.

    Args:
        timestamps: Time series timestamps
        values: CPU percentage values (0-100)
        window_size: Rolling window size in minutes

    Returns:
        Smoothed CPU utilization series

    Example:
        >>> calculate_cpu_utilization(timestamps, values, window_size=10)
    """
    ...
```

## Documentation Formatting

We use Material for MkDocs with many formatting features. See our comprehensive guide:

**[→ Formatting Examples](formatting-examples.md)**

This page shows all available formatting options with live examples.

## Project Structure

```
hello-cloud/
├── src/hellocloud/          # Source code
│   ├── data_generation/     # Synthetic data generation
│   ├── ml_models/           # ML models (GP, hierarchical, foundation)
│   ├── timeseries/          # Time series operations
│   └── etl/                 # Data loaders
├── tests/                   # Test suite
├── notebooks/               # Tutorial notebooks (MyST format)
├── docs/                    # Documentation
├── examples/                # Runnable Python scripts
└── pyproject.toml           # Project configuration
```

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Tag release (`v0.x.0`)
4. Publish to PyPI

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/nehalecky/hello-cloud/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/nehalecky/hello-cloud/issues)
- **Security**: Email maintainers directly

## Code of Conduct

We are committed to providing a welcoming environment for all contributors. Please:

- Be respectful and inclusive
- Welcome newcomers
- Give constructive feedback
- Focus on what's best for the community

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (check `LICENSE` file).

---

**Ready to contribute?** Pick an issue labeled `good first issue` to get started!
