# Contributing to hello cloud

Thank you for your interest in contributing! 🎉

## Quick Start

```bash
# Clone and install
git clone https://github.com/nehalecky/hello-cloud.git
cd hello-cloud
uv sync --all-extras

# Install pre-commit hooks
just pre-commit-install

# Run tests
just test

# Format and lint
just fix
```

## Development Workflow

1. **Create branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow code quality standards below
3. **Test**: `just test` (requires >70% coverage)
4. **Format**: `just fix` (black + ruff auto-fix)
5. **Commit**: Use [Conventional Commits](https://www.conventionalcommits.org/)
6. **Push and PR**: Link to related issues

## Commit Convention

Use conventional commit format:

```
feat: add new forecasting model
fix: correct GP kernel initialization
docs: improve quickstart tutorial
test: add unit tests for TimeSeries loader
refactor: simplify data loading pipeline
chore: update dependencies
```

## Code Quality Standards

- **Black**: Code formatting (automated via pre-commit)
- **Ruff**: Fast linting (replaces flake8, isort)
- **Type hints**: Use liberally for function signatures
- **Docstrings**: Google style for all public APIs
- **Tests**: Aim for >70% coverage (enforced in CI)

Run quality checks:

```bash
# All-in-one: format and fix linting
just fix

# Or individually
just format     # Black formatting
just lint       # Ruff linting
just lint-fix   # Auto-fix ruff issues
```

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/nehalecky/hello-cloud/discussions)
- **Bugs**: [GitHub Issues](https://github.com/nehalecky/hello-cloud/issues)
- **Security**: Email maintainers directly

## Detailed Contributing Guide

For comprehensive guidelines including:
- Testing strategies
- Documentation standards
- Notebook development workflow
- Project architecture
- Release process

Visit: **https://nehalecky.github.io/hello-cloud/contributing/**

## Code of Conduct

Be respectful, inclusive, and welcoming to all contributors. Focus on what's best for the community.

## License

By contributing, you agree that your contributions will be licensed under MIT.
