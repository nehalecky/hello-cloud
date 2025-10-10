# Cloud Resource Simulator Documentation

**Professional Quarto-based documentation** following the [Diátaxis framework](https://diataxis.fr/).

For developer documentation about working with this docs system, see [`docs/_meta/`](docs/_meta/).

## Quick Links

- 📚 **[Live Site](https://nehalecky.github.io/cloud-resource-simulator)** (after deployment)
- 🚀 **[Getting Started](tutorials/index.qmd)** - Interactive tutorials
- 🎯 **[How-To Guides](how-to/index.qmd)** - Task-oriented recipes
- 💡 **[Concepts](concepts/index.qmd)** - Research & design docs
- 📖 **[API Reference](reference/index.qmd)** - Auto-generated API docs

## Local Development

```bash
# Install dependencies
uv sync --group docs

# Generate API reference
uv run quartodoc build --config docs/_quarto.yml

# Preview locally (auto-refreshes)
quarto preview docs/

# Build static site
quarto render docs/
```

## Structure

```
docs/
├── _quarto.yml          # Main configuration
├── _theme/              # Custom styling (SCSS, CSS)
├── _meta/               # Meta-documentation (for developers)
├── _archive/            # Archived legacy docs
├── index.qmd            # Landing page
├── tutorials/           # Learning-oriented guides
├── how-to/              # Task-oriented guides
├── concepts/            # Understanding-oriented docs
│   ├── research/        # Empirical research (35+ citations)
│   └── design/          # Architecture & design
└── reference/           # API documentation (auto-generated)
```

## Adding Content

See [`_meta/README.md`](_meta/README.md) for detailed developer documentation.

Quick tips:
- **Tutorials** - Step-by-step learning guides
- **How-To** - Problem/solution format for specific tasks
- **Concepts** - Explanation and deep dives
- **Reference** - Auto-generated from code (don't edit manually)

---

**Built with** [Quarto](https://quarto.org/) • **Inspired by** [Ibis Project](https://ibis-project.org/)
