# Notebook-Centric Documentation Refactor

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Transform notebooks into the primary documentation artifacts, eliminate duplicate tutorials, and migrate to a simpler documentation architecture with MkDocs-Material.

**Architecture:** Notebooks become authoritative analysis artifacts with enhanced pedagogical structure. Research reports stay as standalone markdown. How-to guides remain as quick reference. Quarto removed in favor of MkDocs-Material with mkdocstrings for API docs.

**Tech Stack:**
- MyST Markdown (notebooks)
- MkDocs-Material (documentation site)
- mkdocstrings (API reference generation)
- great-tables (Python) for notebook table presentation
- Jupytext (notebook ↔ markdown sync)

**Success Criteria:**
- Single source of truth for tutorials (notebooks only)
- Reduced maintenance burden (no duplication)
- Simpler docs tooling (MkDocs vs Quarto)
- Enhanced notebook readability (tables, prose, callouts)

---

## Phase 1: Enhance Notebooks with Pedagogical Structure

### Task 1: Add MyST Admonitions to Notebooks

**Goal:** Enhance notebooks with callout boxes (tips, notes, warnings) using MyST markdown syntax.

**Files:**
- Modify: `notebooks/04_modeling_gaussian_process.md`
- Modify: `notebooks/02_guide_workload_signatures_guide.md`
- Modify: `notebooks/03_EDA_iops_web_server.md`
- Modify: `notebooks/05_EDA_piedpiper_data.md`
- Modify: `notebooks/06_quickstart_timeseries_loader.md`
- Modify: `notebooks/07_forecasting_comparison.md`

**Step 1: Review existing tutorial callouts**

Look at `docs/tutorials/gaussian-processes.qmd` lines 20-34, 206-217, 363-377 for examples of Quarto callouts we want to preserve.

**Step 2: Convert Quarto callouts to MyST admonitions**

MyST syntax (compatible with Jupyter):
```markdown
:::{note}
## Optional Title
Content here
:::

:::{tip}
## Optional Title
Content here
:::

:::{warning}
## Optional Title
Content here
:::
```

**Step 3: Add learning objectives to each notebook**

Add after title, before first section:

```markdown
## Learning Objectives

By the end of this notebook, you will:
- Understand X
- Build Y
- Evaluate Z

**Prerequisites:** Link to prerequisite notebooks
**Estimated time:** NN minutes
```

**Step 4: Add "Next Steps" section to each notebook**

Add before final section:

```markdown
## Next Steps

**Related Notebooks:**
- [Notebook Name](path/to/notebook.ipynb) - Brief description

**Deep Dives:**
- [Research Report](../docs/concepts/research/report-name.md) - Context

**Quick Reference:**
- [How-To Guide](../docs/how-to/guide-name.md) - Task recipes
```

**Step 5: Test notebooks render in Jupyter Lab**

```bash
cd .worktrees/notebook-centric-docs
uv run jupyter lab notebooks/
```

Verify:
- Admonitions render properly
- Links work
- No formatting breakage

**Step 6: Commit**

```bash
git add notebooks/
git commit -m "docs(notebooks): add learning objectives and MyST admonitions

- Add learning objectives section to each notebook
- Convert conceptual explanations to MyST tip/note/warning boxes
- Add 'Next Steps' navigation at end of each notebook
- Estimated time and prerequisites added to headers

Improves pedagogical structure while keeping notebooks as single source of truth."
```

---

### Task 2: Add great-tables for Enhanced Table Presentation

**Goal:** Replace verbose print statements with publication-quality tables using great-tables library.

**Files:**
- Modify: `pyproject.toml` (add great-tables dependency)
- Modify: `notebooks/07_forecasting_comparison.md` (add example table)

**Step 1: Add great-tables dependency**

Edit `pyproject.toml`:
```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "great-tables>=0.15.0",
]
```

**Step 2: Update environment**

```bash
uv sync
```

**Step 3: Add great-tables example to forecasting notebook**

In `notebooks/07_forecasting_comparison.md`, find the baseline comparison table section (around line 585-635).

Replace:
```python
print(comparison_df.to_string(index=False))
```

With:
```python
from great_tables import GT

GT(comparison_df) \
    .tab_header(
        title="Baseline Model Comparison",
        subtitle="62-step forecast horizon"
    ) \
    .fmt_number(
        columns=["MAE", "RMSE", "MAPE", "MASE"],
        decimals=3
    ) \
    .tab_style(
        style=style.fill(color="lightgreen"),
        locations=loc.body(rows=0)  # Highlight best model
    )
```

**Step 4: Test rendering**

```bash
uv run jupyter lab notebooks/07_forecasting_comparison.md
```

Execute the cells with great-tables code. Verify table renders attractively.

**Step 5: Document pattern in notebook**

Add a markdown cell documenting the pattern:

```markdown
### Using great-tables for Results

We use [great-tables](https://github.com/posit-dev/great-tables) for publication-quality table formatting:

- **Formatting:** Numbers, percentages, colors
- **Highlighting:** Best results stand out
- **Export:** Tables work in Jupyter, Colab, static HTML

```

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock notebooks/07_forecasting_comparison.md
git commit -m "feat(notebooks): add great-tables for enhanced table presentation

- Add great-tables dependency
- Replace pandas string formatting with GT tables in forecasting notebook
- Add documentation pattern for future table usage
- Tables now have titles, formatting, and highlighting

Improves readability and professional appearance of notebook outputs."
```

---

## Phase 2: Remove Duplicate Tutorials

### Task 3: Delete Redundant Tutorial Files

**Goal:** Remove tutorials that duplicate notebook content.

**Files:**
- Delete: `docs/tutorials/gaussian-processes.qmd`
- Delete: `docs/tutorials/workload-signatures.qmd`
- Keep: `docs/tutorials/data-exploration.qmd` (convert to notebook in Task 4)
- Keep: `docs/how-to/generate-synthetic-data.qmd` (unique quick reference)
- Keep: `docs/how-to/train-gp-models.qmd` (unique quick reference)

**Step 1: Verify notebooks cover tutorial content**

Check that `notebooks/04_modeling_gaussian_process.md` covers all concepts from `docs/tutorials/gaussian-processes.qmd`.

Check that `notebooks/02_guide_workload_signatures_guide.md` covers all concepts from `docs/tutorials/workload-signatures.qmd`.

**Step 2: Delete duplicate tutorials**

```bash
cd .worktrees/notebook-centric-docs
git rm docs/tutorials/gaussian-processes.qmd
git rm docs/tutorials/workload-signatures.qmd
```

**Step 3: Update docs/tutorials/index.qmd**

Edit `docs/tutorials/index.qmd` to remove references to deleted tutorials.

**Step 4: Verify Quarto still builds**

```bash
quarto render docs/
```

Expected: Build succeeds, no broken links.

**Step 5: Commit**

```bash
git add docs/tutorials/
git commit -m "docs: remove duplicate tutorials (gaussian-processes, workload-signatures)

- Delete gaussian-processes.qmd (covered by notebook 04)
- Delete workload-signatures.qmd (covered by notebook 02)
- Update tutorials index to remove references

Notebooks are now the single source of truth for these topics.
Eliminates maintenance burden of keeping two versions in sync."
```

---

### Task 4: Convert data-exploration.qmd to Notebook

**Goal:** Convert the data-exploration tutorial (no notebook equivalent) to a notebook.

**Files:**
- Delete: `docs/tutorials/data-exploration.qmd`
- Create: `notebooks/01_quickstart_data_exploration.md` (new MyST notebook)

**Step 1: Extract content from tutorial**

Read `docs/tutorials/data-exploration.qmd` and identify:
- Learning objectives (lines 14-20)
- Code examples (all Python blocks)
- Explanatory text (markdown between code)

**Step 2: Create notebook structure**

Create `notebooks/01_quickstart_data_exploration.md`:

```markdown
---
jupytext:
  formats: notebooks//md:myst,notebooks/_build//ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Quick Start: Data Exploration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/01_quickstart_data_exploration.ipynb)

Learn to generate and explore realistic cloud resource data using empirically-grounded patterns.

## Learning Objectives

By the end of this notebook, you will:
- Generate realistic cloud workload data for different application types
- Validate simulated data against research findings
- Visualize resource utilization patterns
- Identify inefficiencies and waste

**Prerequisites:** None (this is the starting point!)
**Estimated time:** 15 minutes

---

## 0. Auto-Reload Configuration

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
```

## 1. Environment Setup

```{code-cell} ipython3
# Environment Setup
try:
    import hellocloud
except ImportError:
    !pip install -q git+https://github.com/nehalecky/hello-cloud.git
    import hellocloud
```

[... rest of content from tutorial ...]
```

**Step 3: Adapt content from tutorial to notebook**

- Copy all Python code blocks as `{code-cell}` cells
- Copy markdown text as markdown cells
- Add output hints after code cells: `# Expected output: ...`
- Convert Quarto callouts to MyST admonitions

**Step 4: Test notebook execution**

```bash
uv run jupyter lab notebooks/01_quickstart_data_exploration.md
```

Run all cells. Verify:
- All cells execute without errors
- Plots render correctly
- Library imports work

**Step 5: Delete old tutorial**

```bash
git rm docs/tutorials/data-exploration.qmd
```

**Step 6: Commit**

```bash
git add notebooks/01_quickstart_data_exploration.md docs/tutorials/
git commit -m "docs: convert data-exploration tutorial to notebook

- Create notebooks/01_quickstart_data_exploration.md from tutorial
- Delete docs/tutorials/data-exploration.qmd
- Add Colab badge for cloud execution
- Notebooks now cover all tutorial content

Completes migration to notebook-centric documentation."
```

---

## Phase 3: Migrate from Quarto to MkDocs-Material

### Task 5: Install MkDocs-Material

**Goal:** Add MkDocs-Material as documentation system.

**Files:**
- Modify: `pyproject.toml` (add docs dependencies)
- Create: `mkdocs.yml` (configuration file)

**Step 1: Add dependencies to pyproject.toml**

```toml
[project.optional-dependencies]
docs = [
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.28.0",
    "mkdocs-jupyter>=0.25.0",  # Render notebooks
    "mkdocs-gen-files>=0.5.0",
    "mkdocs-literate-nav>=0.6.0",
]
```

**Step 2: Install dependencies**

```bash
uv sync --group docs
```

**Step 3: Create mkdocs.yml**

Create `mkdocs.yml` in project root:

```yaml
site_name: Hello Cloud
site_description: Time series forecasting and anomaly detection for cloud resources
site_url: https://nehalecky.github.io/hello-cloud
repo_url: https://github.com/nehalecky/hello-cloud
repo_name: nehalecky/hello-cloud

theme:
  name: material
  palette:
    # Light mode
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    # Dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: google
            show_source: false
            show_root_heading: true
  - mkdocs-jupyter:
      include_source: true
      execute: false  # Don't execute notebooks during build

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.arithmatex:
      generic: true
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Notebooks:
    - notebooks/index.md
    - Quick Start: notebooks/published/01_quickstart_data_exploration.ipynb
    - Workload Signatures: notebooks/published/02_guide_workload_signatures_guide.ipynb
    - EDA - IOPS: notebooks/published/03_EDA_iops_web_server.ipynb
    - GP Modeling: notebooks/published/04_modeling_gaussian_process.ipynb
    - EDA - PiedPiper: notebooks/published/05_EDA_piedpiper_data.ipynb
    - TimeSeries Quickstart: notebooks/published/06_quickstart_timeseries_loader.ipynb
    - Forecasting: notebooks/published/07_forecasting_comparison.ipynb
  - Research:
    - concepts/research/cloud-resource-patterns-research.md
    - concepts/research/cloud-resource-correlations-report.md
    - concepts/research/timeseries-anomaly-datasets-review.md
    - concepts/research/opentslm-foundation-model-evaluation.md
  - How-To:
    - how-to/index.md
    - how-to/generate-synthetic-data.md
    - how-to/train-gp-models.md
  - API Reference: reference/

extra_css:
  - stylesheets/extra.css

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

**Step 4: Create docs/stylesheets/extra.css**

```bash
mkdir -p docs/stylesheets
```

Create `docs/stylesheets/extra.css`:
```css
/* Additional styling for great-tables output */
.gt_table {
    margin: 1rem 0;
}

/* Notebook cell output styling */
.jp-OutputArea-output {
    margin: 0.5rem 0;
}
```

**Step 5: Test MkDocs build**

```bash
uv run mkdocs serve
```

Visit http://127.0.0.1:8000/ and verify:
- Site loads
- Navigation works
- Notebooks render (from `published/` directory)

**Step 6: Commit**

```bash
git add pyproject.toml mkdocs.yml docs/stylesheets/
git commit -m "feat(docs): add MkDocs-Material for documentation

- Add mkdocs-material, mkdocstrings, mkdocs-jupyter dependencies
- Create mkdocs.yml configuration with navigation structure
- Include notebooks from published/ directory
- Configure Python API reference with mkdocstrings

Replaces Quarto with simpler, more maintainable documentation system."
```

---

### Task 6: Convert Research Reports from QMD to MD

**Goal:** Convert research reports from Quarto (.qmd) to plain markdown (.md).

**Files:**
- Modify: `docs/concepts/research/cloud-resource-patterns-research.qmd` → `.md`
- Modify: `docs/concepts/research/cloud-resource-correlations-report.qmd` → `.md`
- Modify: `docs/concepts/research/timeseries-anomaly-datasets-review.qmd` → `.md`
- Modify: `docs/concepts/research/opentslm-foundation-model-evaluation.qmd` → `.md`

**Step 1: Strip Quarto YAML frontmatter**

For each `.qmd` file, remove the YAML header (lines 1-7 typically):
```yaml
---
title: "..."
subtitle: "..."
execute:
  eval: false
---
```

Keep just the title as markdown:
```markdown
# Title Here
```

**Step 2: Convert Quarto-specific syntax**

Replace Quarto callouts:
```
::: {.callout-note}
Content
:::
```

With standard markdown:
```
> **Note:** Content
```

Or MyST admonitions if complex:
```
:::{note}
Content
:::
```

**Step 3: Rename files**

```bash
cd docs/concepts/research/
git mv cloud-resource-patterns-research.qmd cloud-resource-patterns-research.md
git mv cloud-resource-correlations-report.qmd cloud-resource-correlations-report.md
git mv timeseries-anomaly-datasets-review.qmd timeseries-anomaly-datasets-review.md
git mv opentslm-foundation-model-evaluation.qmd opentslm-foundation-model-evaluation.md
```

**Step 4: Verify markdown renders in MkDocs**

```bash
uv run mkdocs serve
```

Navigate to research reports and verify rendering.

**Step 5: Commit**

```bash
git add docs/concepts/research/
git commit -m "docs: convert research reports from Quarto to markdown

- Remove Quarto-specific YAML frontmatter
- Convert Quarto callouts to standard markdown or MyST admonitions
- Rename .qmd → .md for all research reports

Makes research reports portable and removes Quarto dependency."
```

---

### Task 7: Convert How-To Guides from QMD to MD

**Goal:** Convert how-to guides from Quarto to markdown.

**Files:**
- Modify: `docs/how-to/generate-synthetic-data.qmd` → `.md`
- Modify: `docs/how-to/train-gp-models.qmd` → `.md`

**Step 1: Strip Quarto YAML and convert syntax**

Same process as Task 6:
- Remove YAML frontmatter
- Convert Quarto-specific syntax to standard markdown
- Keep code blocks as plain markdown code blocks

**Step 2: Rename files**

```bash
cd docs/how-to/
git mv generate-synthetic-data.qmd generate-synthetic-data.md
git mv train-gp-models.qmd train-gp-models.md
```

**Step 3: Verify rendering**

```bash
uv run mkdocs serve
```

Check how-to guides render properly.

**Step 4: Commit**

```bash
git add docs/how-to/
git commit -m "docs: convert how-to guides from Quarto to markdown

- Remove Quarto-specific YAML and syntax
- Rename .qmd → .md
- Maintain task-oriented structure

How-to guides now portable markdown without Quarto dependency."
```

---

### Task 8: Configure MkDocs API Reference

**Goal:** Auto-generate API reference from Python docstrings using mkdocstrings.

**Files:**
- Create: `docs/reference/index.md`
- Create: `docs/reference/generation.md`
- Create: `docs/reference/gaussian-process.md`
- Create: `docs/reference/forecasting.md`

**Step 1: Create API reference index**

Create `docs/reference/index.md`:
```markdown
# API Reference

Auto-generated API documentation from Python source code.

## Modules

- [Data Generation](generation.md) - Synthetic workload data generation
- [Gaussian Process](gaussian-process.md) - GP models for time series
- [Forecasting](forecasting.md) - Baseline and foundation model forecasting
```

**Step 2: Create generation module docs**

Create `docs/reference/generation.md`:
```markdown
# Data Generation

::: hellocloud.generation.WorkloadPatternGenerator
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.generation.CloudMetricsSimulator
    options:
      show_root_heading: true
      show_source: false
```

**Step 3: Create gaussian-process module docs**

Create `docs/reference/gaussian-process.md`:
```markdown
# Gaussian Process Models

::: hellocloud.ml_models.gaussian_process.SparseGPModel
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.ml_models.gaussian_process.CompositePeriodicKernel
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.ml_models.gaussian_process.train_gp_model
    options:
      show_root_heading: true
      show_source: false
```

**Step 4: Create forecasting module docs**

Create `docs/reference/forecasting.md`:
```markdown
# Forecasting Models

## Baseline Methods

::: hellocloud.modeling.forecasting.NaiveForecaster
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.modeling.forecasting.SeasonalNaiveForecaster
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.modeling.forecasting.MovingAverageForecaster
    options:
      show_root_heading: true
      show_source: false

## ARIMA

::: hellocloud.modeling.forecasting.ARIMAForecaster
    options:
      show_root_heading: true
      show_source: false

## Foundation Models

::: hellocloud.modeling.forecasting.foundation.TimesFMForecaster
    options:
      show_root_heading: true
      show_source: false
```

**Step 5: Verify API docs render**

```bash
uv run mkdocs serve
```

Navigate to API Reference section. Verify:
- Classes documented
- Methods shown with signatures
- Docstrings render properly

**Step 6: Commit**

```bash
git add docs/reference/
git commit -m "docs: add mkdocstrings API reference

- Create API reference index and module pages
- Use mkdocstrings to auto-generate from docstrings
- Organize by module: generation, gaussian-process, forecasting

Replaces quartodoc with simpler mkdocstrings approach."
```

---

### Task 9: Create New docs/index.md

**Goal:** Create landing page for MkDocs site.

**Files:**
- Create: `docs/index.md`

**Step 1: Write landing page**

Create `docs/index.md`:
```markdown
# Hello Cloud

Time series forecasting and anomaly detection for cloud resources.

## Overview

Hello Cloud is a Python library for modeling cloud resource utilization patterns, forecasting future usage, and detecting anomalies in operational metrics.

**Key Features:**
- **Empirically grounded:** Based on research showing 12-15% average CPU utilization
- **Multiple models:** Gaussian Processes, ARIMA, foundation models (TimesFM)
- **Production-ready:** 92% test coverage on GP library, comprehensive evaluation

## Getting Started

### Installation

```bash
pip install git+https://github.com/nehalecky/hello-cloud.git
```

### Quick Start

Generate synthetic cloud workload data:

```python
from hellocloud.generation import WorkloadPatternGenerator, WorkloadType

generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    interval_minutes=60
)
```

See the [Quick Start notebook](notebooks/published/01_quickstart_data_exploration.ipynb) for a complete walkthrough.

## Documentation Structure

**[Notebooks](notebooks/index.md)** - Interactive tutorials and analysis
Start here for hands-on learning with executable code.

**[Research](concepts/research/cloud-resource-patterns-research.md)** - Empirical foundations
Understanding the research that informs the models.

**[How-To Guides](how-to/index.md)** - Task-oriented recipes
Quick reference for common operations.

**[API Reference](reference/index.md)** - Complete API documentation
Auto-generated from source code docstrings.

## Research Context

The library is grounded in empirical research showing:
- **CPU Utilization:** 12-15% average across cloud infrastructure
- **Memory Utilization:** 18-25% average
- **Resource Waste:** 25-35% of cloud spending
- **Temporal Autocorrelation:** 0.7-0.8 (strong patterns)

These findings inform all synthetic data generation parameters. See [Cloud Resource Patterns Research](concepts/research/cloud-resource-patterns-research.md) for details.

## Contributing

Contributions welcome! Please see the [GitHub repository](https://github.com/nehalecky/hello-cloud) for development setup.

## License

MIT License - see LICENSE file for details.
```

**Step 2: Test rendering**

```bash
uv run mkdocs serve
```

Verify landing page looks good.

**Step 3: Commit**

```bash
git add docs/index.md
git commit -m "docs: create MkDocs landing page

- Overview and key features
- Quick start example
- Documentation structure guide
- Research context summary

Replaces Quarto index with MkDocs-friendly landing page."
```

---

### Task 10: Remove Quarto Files

**Goal:** Delete Quarto configuration and generated files.

**Files:**
- Delete: `docs/_quarto.yml`
- Delete: `docs/_site/` (build output)
- Delete: `docs/_theme/` (custom Quarto theme)
- Delete: `docs/notebooks/index.qmd` (replaced by published/ notebooks)
- Delete: `docs/tutorials/index.qmd`
- Delete: `docs/tutorials/iops-eda.qmd` (if exists)
- Delete: `docs/how-to/index.qmd`
- Delete: `docs/concepts/index.qmd`
- Delete: `docs/concepts/design/gaussian-process-design.qmd` (convert first if valuable)

**Step 1: Backup any valuable content**

Check if `docs/concepts/design/gaussian-process-design.qmd` has unique content not in notebooks.

If yes, convert to markdown:
```bash
git mv docs/concepts/design/gaussian-process-design.qmd docs/concepts/design/gaussian-process-design.md
# Strip YAML, convert syntax as in previous tasks
```

**Step 2: Remove Quarto configuration**

```bash
git rm docs/_quarto.yml
git rm -r docs/_site/  # If tracked
git rm -r docs/_theme/
```

**Step 3: Remove Quarto index files**

```bash
git rm docs/notebooks/index.qmd
git rm docs/tutorials/index.qmd
git rm docs/how-to/index.qmd
git rm docs/concepts/index.qmd
```

**Step 4: Create replacement index files for MkDocs**

Create `docs/notebooks/index.md`:
```markdown
# Notebooks

Interactive tutorials and analysis artifacts. All notebooks are executable in Jupyter Lab or Google Colab.

## Getting Started

- [Quick Start: Data Exploration](published/01_quickstart_data_exploration.ipynb) - 15 min intro
- [TimeSeries Quickstart](published/06_quickstart_timeseries_loader.ipynb) - New TimeSeries API

## Exploratory Data Analysis

- [IOPS Web Server EDA](published/03_EDA_iops_web_server.ipynb) - TSB-UAD dataset
- [PiedPiper Data EDA](published/05_EDA_piedpiper_data.ipynb) - Hierarchical time series

## Modeling

- [Gaussian Process Modeling](published/04_modeling_gaussian_process.ipynb) - Production-ready GPs
- [Forecasting Comparison](published/07_forecasting_comparison.ipynb) - Baselines, ARIMA, TimesFM

## Running Notebooks

**Local:**
```bash
uv run jupyter lab notebooks/
```

**Google Colab:**
Click the "Open in Colab" badge at the top of any notebook.
```

Create `docs/how-to/index.md`:
```markdown
# How-To Guides

Task-oriented quick reference for common operations.

## Data Generation

- [Generate Synthetic Data](generate-synthetic-data.md) - Create realistic cloud workload datasets

## Model Training

- [Train Gaussian Process Models](train-gp-models.md) - Production recipes for GP forecasting
```

**Step 5: Update .gitignore**

Add to `.gitignore`:
```
# MkDocs build output
/site/

# Quarto artifacts (legacy, can remove after migration)
/.quarto/
```

**Step 6: Remove quartodoc dependency**

Edit `pyproject.toml`, remove:
```toml
[project.optional-dependencies]
docs = [
    # "quartodoc>=0.11.1",  # REMOVED - replaced by mkdocstrings
]
```

**Step 7: Commit**

```bash
git add .gitignore pyproject.toml docs/
git commit -m "refactor(docs): remove Quarto, complete migration to MkDocs

- Delete Quarto configuration (_quarto.yml, _theme/)
- Remove Quarto index files (.qmd)
- Create MkDocs-compatible index files (.md)
- Update .gitignore for MkDocs build output
- Remove quartodoc dependency

Documentation now fully migrated to MkDocs-Material."
```

---

### Task 11: Update GitHub Actions for MkDocs

**Goal:** Update CI/CD to build and deploy MkDocs instead of Quarto.

**Files:**
- Modify: `.github/workflows/docs.yml` (or create if doesn't exist)

**Step 1: Create or update docs workflow**

Create `.github/workflows/docs.yml`:
```yaml
name: Deploy Documentation

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: |
          uv sync --group docs

      - name: Build docs
        run: |
          uv run mkdocs build --strict

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/master'
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

**Step 2: Update README with new docs link**

Edit `README.md`, find the documentation link and update:

Old:
```markdown
Documentation: https://nehalecky.github.io/hello-cloud (built with Quarto)
```

New:
```markdown
Documentation: https://nehalecky.github.io/hello-cloud
```

**Step 3: Commit**

```bash
git add .github/workflows/docs.yml README.md
git commit -m "ci: update GitHub Actions for MkDocs deployment

- Replace Quarto build with mkdocs build
- Deploy to GitHub Pages on push to master
- Use uv for dependency management

Documentation deployment now automated with MkDocs."
```

---

### Task 12: Update CLAUDE.md

**Goal:** Document the new documentation architecture in project instructions.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add documentation section**

Add to `CLAUDE.md` after the "Development Commands" section:

```markdown
### Documentation

**Architecture:** Notebook-centric with MkDocs-Material

```bash
# Build and serve documentation
uv run mkdocs serve  # Preview at http://127.0.0.1:8000

# Build static site
uv run mkdocs build  # Output: site/
```

**Documentation Structure:**

- **Notebooks** (`notebooks/*.md`): Primary tutorials and analysis
  - MyST markdown format (executable in Jupyter)
  - Published to `notebooks/published/*.ipynb` for Colab
  - Single source of truth for tutorials

- **Research Reports** (`docs/concepts/research/*.md`): Standalone markdown
  - Literature reviews and design decisions
  - Not notebook-driven

- **How-To Guides** (`docs/how-to/*.md`): Quick reference recipes
  - Task-oriented, not tutorial format
  - Standalone markdown

- **API Reference** (`docs/reference/*.md`): Auto-generated
  - Uses mkdocstrings to extract from docstrings
  - Don't edit directly - update source code docstrings

**Adding New Notebooks:**

1. Create `notebooks/NN_topic_name.md` in MyST format
2. Add learning objectives, prerequisites, estimated time
3. Use MyST admonitions (:::{note}, :::{tip}, :::{warning})
4. Add "Next Steps" section with links
5. Publish: `just nb-publish NN_topic_name`
6. Add to `mkdocs.yml` nav under "Notebooks"

**Adding New How-To Guides:**

1. Create `docs/how-to/task-name.md`
2. Task-oriented structure (not tutorial)
3. Include validation checklist, troubleshooting section
4. Add to `mkdocs.yml` nav under "How-To"

**Deprecated:**
- ~~Quarto~~ (removed in favor of MkDocs)
- ~~quartodoc~~ (replaced by mkdocstrings)
- ~~Duplicate tutorials in docs/tutorials/~~ (notebooks are now authoritative)
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with new documentation architecture

- Document MkDocs-Material setup
- Explain notebook-centric approach
- Provide guidance for adding new content
- Mark Quarto as deprecated

Project instructions now reflect current documentation system."
```

---

## Phase 4: Testing and Verification

### Task 13: Full Documentation Build Test

**Goal:** Verify complete documentation builds without errors.

**Files:** None (verification only)

**Step 1: Clean build**

```bash
rm -rf site/  # Remove any existing build
uv run mkdocs build --strict
```

Expected: Build succeeds with no warnings or errors.

**Step 2: Verify notebook rendering**

Open `site/notebooks/published/*.html` in a browser. Check:
- Notebooks render with outputs
- Code cells visible
- Plots/tables display correctly
- Colab badges present and functional

**Step 3: Verify research reports**

Open `site/concepts/research/*.html`. Check:
- Markdown renders properly
- Links work (internal and external)
- No broken references

**Step 4: Verify API reference**

Open `site/reference/*.html`. Check:
- Classes documented
- Methods have signatures and docstrings
- Navigation between modules works

**Step 5: Verify search functionality**

Use search box in built site. Search for:
- "Gaussian Process"
- "ARIMA"
- "WorkloadPatternGenerator"

Verify results are relevant and link correctly.

**Step 6: Test local development server**

```bash
uv run mkdocs serve
```

Make a small change to `docs/index.md`. Verify:
- Server auto-reloads
- Change appears immediately
- No errors in terminal

**Step 7: Document verification results**

Create checklist in PR description:
```markdown
## Verification Checklist

- [ ] MkDocs build completes without errors
- [ ] All notebooks render with outputs in published/ directory
- [ ] Research reports display correctly
- [ ] API reference generates from docstrings
- [ ] Search functionality works
- [ ] Local development server auto-reloads on changes
- [ ] GitHub Pages deployment configured
```

---

## Success Criteria

**Phase 1 Complete When:**
- [x] All notebooks have learning objectives and "Next Steps" sections
- [x] MyST admonitions used for pedagogical callouts
- [x] great-tables integrated for enhanced table presentation
- [x] Pattern documented for future notebook authors

**Phase 2 Complete When:**
- [x] Duplicate tutorials deleted (gaussian-processes, workload-signatures)
- [x] data-exploration tutorial converted to notebook
- [x] No Quarto tutorial files remain
- [x] Notebooks cover all tutorial content

**Phase 3 Complete When:**
- [x] MkDocs-Material installed and configured
- [x] All research reports converted to markdown (.qmd → .md)
- [x] How-to guides converted to markdown
- [x] API reference configured with mkdocstrings
- [x] Landing page created
- [x] Quarto files deleted
- [x] GitHub Actions updated for MkDocs deployment
- [x] CLAUDE.md documents new architecture

**Phase 4 Complete When:**
- [x] Full documentation builds without errors
- [x] All content renders correctly (notebooks, research, how-tos, API)
- [x] Search works
- [x] GitHub Pages deployment functional
- [x] Verification checklist complete

---

## Rollback Plan

If issues arise during migration:

1. **Revert to previous commit:**
   ```bash
   git revert HEAD~N  # N = number of commits to revert
   ```

2. **Quarto still works** (until Phase 3 Task 10 completes):
   ```bash
   quarto render docs/
   ```

3. **Notebooks unaffected** - MyST format works with both Quarto and MkDocs

4. **Incremental deployment** - Can pause after any phase, commit, and continue later

---

## Notes for Future Maintainers

**Why notebooks as primary docs?**
- Single source of truth eliminates duplication
- Executed outputs prove code works
- Users can run immediately in Colab
- Easier to keep in sync with library changes

**Why MkDocs-Material over Quarto?**
- Simpler configuration (one mkdocs.yml vs multiple .qmd files)
- Better Python integration (mkdocstrings)
- Faster build times
- More familiar to Python developers

**When to add a notebook vs how-to guide?**
- **Notebook:** Tutorial, exploration, analysis (tell a story)
- **How-to:** Quick reference, recipes, troubleshooting (solve a problem)

**Maintaining great-tables consistency:**
- Use GT() with tab_header() for all result tables
- Apply fmt_number() for metrics (3 decimals)
- Highlight best results with tab_style()
- Document pattern in notebooks as examples for others
