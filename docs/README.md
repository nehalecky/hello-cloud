# Hello Cloud Documentation

Welcome to the Hello Cloud documentation workspace! This directory contains all source files for the [live documentation site](https://nehalecky.github.io/hello-cloud).

## Documentation Architecture

**Tech Stack:** [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) • [mkdocstrings](https://mkdocstrings.github.io/) • [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)

### Content Strategy

| Location | Purpose | Audience |
|----------|---------|----------|
| **README.md** (repo root) | Quick start, installation, basic usage | GitHub visitors, new users |
| **docs/index.md** | Comprehensive overview, feature showcase | Documentation readers |
| **docs/notebooks/** | Interactive tutorials with code & outputs | Learners, practitioners |
| **docs/concepts/** | Deep dives, research, architecture | Advanced users, contributors |
| **docs/reference/** | API documentation (auto-generated) | Developers using the library |

## Structure

```
docs/
├── mkdocs.yml           # MkDocs configuration
├── index.md             # Documentation homepage
├── assets/              # Logos, images, static files
│   ├── logo-full-*.png
│   ├── logo-icon-*.png
│   └── favicon.ico
├── stylesheets/         # Custom CSS
│   └── brand.css        # Foundation Horizon brand colors
├── notebooks/
│   ├── index.md         # Notebooks overview
│   └── published/       # Executed .ipynb files (CI builds these)
├── concepts/
│   ├── index.md         # Concepts overview
│   ├── research/        # Research reports, literature reviews
│   └── design/          # Architecture & design documents
└── reference/
    └── index.md         # API reference entry point
```

## Quick Start

```bash
# Install dependencies
uv sync --group docs

# Preview site locally (live reload)
just docs-serve

# Build static site
just docs-build

# Deploy (happens automatically via GitHub Actions)
git push origin master
```

## Adding Content

### 1. Adding a Tutorial Notebook

**Step 1:** Create MyST markdown notebook in `notebooks/`

```bash
# Create new notebook
touch notebooks/08_my_new_tutorial.md
```

**Step 2:** Add YAML frontmatter

```yaml
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```

**Step 3:** Write content with code cells

````markdown
# My Tutorial Title

Introduction paragraph.

```python
# This is a code cell
import pandas as pd
data = pd.DataFrame({'x': [1, 2, 3]})
```

More explanation.
````

**Step 4:** Execute and publish

```bash
# Execute notebook (creates .ipynb with outputs)
just nb-execute 08_my_new_tutorial

# Outputs saved to notebooks/published/08_my_new_tutorial.ipynb
# This file is gitignored but will be rebuilt by CI
```

**Step 5:** Add to navigation in `mkdocs.yml`

```yaml
nav:
  - Notebooks:
      - My Tutorial: notebooks/published/08_my_new_tutorial.ipynb
```

### 2. Adding a Concept Document

**Step 1:** Create markdown file in appropriate directory

```bash
# For research
touch docs/concepts/research/my-research-topic.md

# For design
touch docs/concepts/design/my-design-doc.md
```

**Step 2:** Write content (standard markdown)

```markdown
# My Research Topic

## Overview

Research question and motivation.

## Methodology

...

## Findings

...

## References

[1] Author et al. "Title." Conference, Year.
```

**Step 3:** Add to navigation in `mkdocs.yml`

```yaml
nav:
  - Concepts:
      - Research:
          - My Research: concepts/research/my-research-topic.md
```

### 3. Documenting API Functions

API reference is **auto-generated** from docstrings. Do NOT edit reference/*.md files directly.

**Step 1:** Write Google-style docstrings in source code

```python
def my_function(arg1: str, arg2: int) -> bool:
    """One-line summary of what the function does.

    Longer description providing context and usage examples.

    Args:
        arg1: Description of first argument
        arg2: Description of second argument

    Returns:
        Description of return value

    Example:
        ```python
        result = my_function("hello", 42)
        ```
    """
    return True
```

**Step 2:** Add to reference index in `docs/reference/index.md`

```markdown
## Functions

:::hellocloud.module_name.my_function
```

The `:::` syntax triggers mkdocstrings to extract and render the docstring.

## Theme & Branding

**Foundation Horizon Colors:**
- Primary: Horizon Blue (#4A90E2) - Trust, cloud, forecasting
- Accent: Forecast Orange (#FF6B35) - Energy, insight, action

**Logo Usage:**
- **Header:** Icon-only logo (pairs with site name)
- **Homepage:** Full logo with text (hero section)
- **Favicon:** Multi-size ICO for all browsers

**Custom Styling:**
- Light/dark mode support (auto-switching)
- Jupyter notebook integration
- Code syntax highlighting
- Responsive tables and images

See `docs/stylesheets/brand.css` for implementation.

## Deployment

Documentation deploys automatically to GitHub Pages when you push to `master`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for full details on the CI/CD workflow.

## Development Tips

**Preview locally before pushing:**
```bash
just docs-serve
# Visit http://127.0.0.1:8000
```

**Check for broken links:**
```bash
just docs-build
# Look for warnings in output
```

**Test notebook execution:**
```bash
pytest tests/test_notebooks.py -v
```

**Format code examples:**
- Use triple backticks with language identifier
- Keep code blocks short (< 20 lines)
- Include docstrings for context
- Show expected output when helpful

**Writing style:**
- Clear, concise, active voice
- Code-first approach
- Practical examples over theory
- Link to related concepts

## Troubleshooting

**MkDocs server won't start:**
- Check for syntax errors in markdown files
- Verify `mkdocs.yml` is valid YAML
- Ensure dependencies installed: `uv sync --group docs`

**Notebook not showing up:**
- Execute notebook: `just nb-execute <name>`
- Check file exists in `notebooks/published/`
- Verify path in `mkdocs.yml` navigation

**API reference empty:**
- Check docstring format (must be Google style)
- Verify `:::module.function` syntax in reference/*.md
- Ensure source paths correct in `mkdocs.yml` (`paths: [src]`)

**Styling looks wrong:**
- Clear browser cache
- Check `docs/stylesheets/brand.css` loaded
- Verify CSS selectors match Material theme structure

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Writing Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)

---

**Questions or issues?** Open an issue in the [GitHub repository](https://github.com/nehalecky/hello-cloud).
