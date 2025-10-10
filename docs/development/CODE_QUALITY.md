# Code Quality Workflow

This document describes the code quality tools and workflows for the `hellocloud` project.

## Tools Overview

### Black - Code Formatting
[Black](https://black.readthedocs.io/) is an opinionated code formatter that ensures consistent code style across the project.

**Configuration** (`pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ["py311"]
```

**Usage**:
```bash
# Format all code
just format
# or
uv run black src/ tests/
```

### Ruff - Fast Python Linter
[Ruff](https://docs.astral.sh/ruff/) is a fast Python linter that replaces multiple tools (flake8, isort, pyupgrade, etc.).

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort (import sorting)
    "B",   # flake8-bugbear
    "C90", # mccabe complexity
    "UP",  # pyupgrade (modernize Python syntax)
]
ignore = ["E501", "B008", "B905"]
```

**Usage**:
```bash
# Check for issues
just lint
# or
uv run ruff check src/ tests/

# Auto-fix issues
just lint-fix
# or
uv run ruff check --fix src/ tests/
```

### Pre-commit Hooks - Automated Quality Checks
[Pre-commit](https://pre-commit.com/) automatically runs quality checks before each commit.

**Configuration** (`.pre-commit-config.yaml`):
- Black formatting
- Ruff linting
- Basic file checks (trailing whitespace, YAML/TOML validation, etc.)
- Notebook formatting (nbQA)

**Usage**:
```bash
# Install hooks (once per clone)
just pre-commit-install
# or
uv run pre-commit install

# Run hooks manually on all files
just pre-commit
# or
uv run pre-commit run --all-files

# Update hook versions
just pre-commit-update
# or
uv run pre-commit autoupdate
```

## Recommended Workflows

### Daily Development Workflow

**Before committing**:
```bash
# Format and fix all auto-fixable issues
just fix

# Check for remaining issues
just lint
```

The `just fix` command runs both black and ruff with auto-fix, handling ~90% of code quality issues automatically.

### Initial Setup (One-time)

After cloning the repository:
```bash
# Install all dependencies and pre-commit hooks
just install
```

This will:
1. Install all Python dependencies
2. Install documentation dependencies
3. Set up pre-commit hooks

### Pre-commit Hooks (Automatic)

Once installed, pre-commit hooks run automatically on `git commit`:
1. Black formats staged files
2. Ruff checks and fixes staged files
3. Basic file checks run (trailing whitespace, etc.)

If hooks fail:
- Staged files are modified to fix issues
- Commit is aborted
- Review changes with `git diff`
- Stage fixed files with `git add`
- Retry commit

**Bypass hooks** (not recommended):
```bash
git commit --no-verify
```

### Manual Quality Checks

Run all quality checks manually:
```bash
# Format code
just format

# Auto-fix linting issues
just lint-fix

# Check for remaining issues
just lint

# Or do both at once
just fix
```

## Common Issues and Solutions

### Issue: Ruff reports "Import block is un-sorted"
**Solution**: Ruff's isort integration automatically sorts imports. Run:
```bash
just fix
```

### Issue: Ruff reports "Use `dict` instead of `Dict`"
**Solution**: Python 3.9+ supports lowercase type hints. Ruff's pyupgrade rule automatically modernizes these:
```bash
# Before
from typing import Dict, List
def foo() -> Dict[str, List[int]]:
    pass

# After (auto-fixed)
def foo() -> dict[str, list[int]]:
    pass
```

### Issue: Black and Ruff conflict
**Solution**: Black and Ruff are designed to work together. Our configuration ensures compatibility:
- Both use 100-character line length
- Ruff's E501 (line too long) is ignored (Black handles it)
- Run black first, then ruff: `just fix`

### Issue: Pre-commit hooks are slow
**Solution**: Hooks only run on staged files, so they're usually fast. For first-time setup:
```bash
# Run hooks on all files once (installs hook environments)
just pre-commit

# Subsequent commits will be faster
```

### Issue: Complex code quality issues
Some Ruff warnings require manual intervention:
- **C901**: Function too complex → Refactor into smaller functions
- **B007**: Unused loop variable → Rename to `_variable` if intentionally unused
- **F841**: Unused variable → Remove or prefix with `_` if needed for unpacking
- **B006**: Mutable default argument → Use `None` and initialize in function body

## Integration with CI/CD

GitHub Actions runs these checks on every push/PR:
```yaml
- name: Code Quality
  run: |
    uv run black --check src/ tests/
    uv run ruff check src/ tests/
```

Keep local code quality high to avoid CI failures!

## Quick Reference

| Task | Command |
|------|---------|
| Format code | `just format` |
| Check linting | `just lint` |
| Auto-fix linting | `just lint-fix` |
| Format + fix (recommended) | `just fix` |
| Run pre-commit manually | `just pre-commit` |
| Install hooks | `just pre-commit-install` |
| Update hook versions | `just pre-commit-update` |

## Philosophy

**Code quality is automatic, not optional**:
1. Pre-commit hooks catch issues before they're committed
2. CI enforces quality standards
3. Tools auto-fix 90% of issues
4. Manual intervention only needed for complex refactoring

**Benefits**:
- Consistent code style across the project
- Fewer code review comments about style
- Modern Python idioms enforced automatically
- Faster development (no manual formatting)
