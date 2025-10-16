# Notebook-Centric Documentation Refactor

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

> **Note on Plans:** This plan document is ephemeral. It should be committed at the start of the PR for context during code review, then deleted after the PR is merged. Plans live in `./plans/` (not `./docs/plans/` or repo root).

**Goal:** Delete duplication, use Material theme defaults, display executed notebooks from `./published/`, inline API docs from docstrings only.

**Architecture:** Notebooks (from `./published/*.ipynb`) ARE the documentation. Research reports stay as standalone markdown. API reference uses inline `:::` syntax with mkdocstrings. Zero customization.

**Tech Stack:**
- MkDocs-Material (default theme, no customization)
- mkdocstrings (inline `:::` API docs from docstrings)
- mkdocs-jupyter (render published notebooks)

**Success Criteria:**
- Notebooks from `./published/` displayed with outputs
- Zero duplication (delete tutorials/, how-to/, reference/*.qmd)
- ~30-line mkdocs.yml with defaults only
- CI builds and deploys via mkdocs gh-deploy

---

## Phase 0: Cleanup Obsolete Files

### Task 0.1: Delete Old Plan Files in Repo Root

**Goal:** Remove completed plan files from repo root.

**Files to Delete:**
- `JUPYTEXT_MIGRATION_PLAN.md` (completed)
- `MLFLOW_REFACTOR_PLAN.md` (obsolete - related to dropped MLflow commit)

**Commands:**
```bash
git rm JUPYTEXT_MIGRATION_PLAN.md MLFLOW_REFACTOR_PLAN.md
git commit -m "chore: remove obsolete plan files from repo root"
```

**Verification:** `ls *.md` should not show any `*PLAN*.md` files.

---

### Task 0.2: Delete Scripts Directory

**Goal:** Remove PySpark migration scripts directory (migration complete).

**Directory to Delete:**
- `scripts/` (8 files: conversion scripts, READMEs, verification tools)

**Rationale:**
- PySpark migration is complete
- Scripts were one-time migration utilities
- No ongoing utility

**Commands:**
```bash
git rm -r scripts/
git commit -m "chore: remove PySpark migration scripts (migration complete)"
```

**Verification:** `ls scripts/` should fail (directory should not exist).

---

### Task 0.3: Move Legacy Plan File from docs/plans/

**Goal:** Consolidate legacy plan file into `./plans/` directory.

**File to Move:**
- `docs/plans/2025-10-13-timeseries-loader.md` → `plans/2025-10-13-timeseries-loader.md`

**Commands:**
```bash
mv docs/plans/2025-10-13-timeseries-loader.md plans/
rmdir docs/plans/
git add -A
git commit -m "refactor: consolidate legacy plan into ./plans/"
```

**Verification:**
- `ls docs/plans/` should fail (directory deleted)
- `ls plans/` should show consolidated plan files

---

## Phase 1: Delete Duplicate Content

### Task 1.1: Delete Duplicate Tutorials

**Goal:** Remove tutorials that duplicate notebook content.

**Files to Delete:**
- `docs/tutorials/gaussian-processes.qmd` (duplicate of notebook 04)
- `docs/tutorials/workload-signatures.qmd` (duplicate of notebook 02)
- `docs/tutorials/iops-eda.qmd` (duplicate of notebook 03)
- `docs/tutorials/data-exploration.qmd` (unique, but notebooks ARE tutorials now)
- `docs/tutorials/index.qmd`

**Commands:**
```bash
git rm -r docs/tutorials/
git commit -m "docs: delete duplicate tutorials (notebooks are documentation)"
```

**Verification:** `ls docs/tutorials/` should fail.

---

### Task 1.2: Delete How-To Guides

**Goal:** Remove how-to guides entirely (not converting to notebooks).

**Files to Delete:**
- `docs/how-to/generate-synthetic-data.qmd`
- `docs/how-to/train-gp-models.qmd`
- `docs/how-to/index.qmd`

**Rationale:** Notebooks already show how to do everything. How-to guides add duplication.

**Commands:**
```bash
git rm -r docs/how-to/
git commit -m "docs: delete how-to guides (notebooks cover all use cases)"
```

**Verification:** `ls docs/how-to/` should fail.

---

### Task 1.3: Delete quartodoc-Generated API Reference

**Goal:** Remove quartodoc-generated API files.

**Files to Delete:**
- `docs/reference/*.qmd` (10 quartodoc files)
- Keep `docs/reference/` directory for new mkdocstrings docs

**Commands:**
```bash
git rm docs/reference/*.qmd
git commit -m "docs: delete quartodoc API files (replacing with mkdocstrings)"
```

**Verification:** `ls docs/reference/*.qmd` should fail.

---

## Phase 2: Ultra-Minimal MkDocs Migration

### Task 2.1: Create Minimal mkdocs.yml

**Goal:** Create absolute minimal mkdocs.yml pointing to `./published/` notebooks.

**File to Create:** `mkdocs.yml`

**Content:**
```yaml
site_name: Hello Cloud
site_url: https://nehalecky.github.io/hello-cloud
repo_url: https://github.com/nehalecky/hello-cloud

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - search.suggest

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: google
            show_source: false
  - mkdocs-jupyter:
      include_source: true
      execute: false

nav:
  - Home: index.md
  - Notebooks:
      - notebooks/index.md
      - Workload Signatures: published/02_guide_workload_signatures_guide.ipynb
      - IOPS Analysis: published/03_EDA_iops_web_server.ipynb
      - Gaussian Processes: published/04_modeling_gaussian_process.ipynb
      - PiedPiper Data: published/05_EDA_piedpiper_data.ipynb
      - TimeSeries Quickstart: published/06_quickstart_timeseries_loader.ipynb
      - Forecasting: published/07_forecasting_comparison.ipynb
  - Concepts:
      - concepts/index.md
      - Research:
          - concepts/research/cloud-resource-patterns-research.md
          - concepts/research/cloud-resource-correlations-report.md
          - concepts/research/timeseries-anomaly-datasets-review.md
          - concepts/research/opentslm-foundation-model-evaluation.md
      - Design:
          - concepts/design/gaussian-process-design.md
  - Reference:
      - reference/index.md
```

**Commands:**
```bash
git add mkdocs.yml
git commit -m "feat(docs): add minimal mkdocs.yml with Material defaults

- Point to published/*.ipynb for executed notebooks with outputs
- Use mkdocstrings for inline API docs
- Use mkdocs-jupyter to render notebooks
- Zero customization, all defaults"
```

---

### Task 2.2: Convert Concepts from .qmd to .md

**Goal:** Strip Quarto frontmatter, keep content as plain markdown.

**Files to Convert:**
- `docs/concepts/research/*.qmd` → `.md` (4 files)
- `docs/concepts/design/*.qmd` → `.md` (1 file)

**Process for each file:**
1. Remove YAML frontmatter (--- ... ---)
2. Keep title as `# Title`
3. Convert Quarto callouts to markdown blockquotes: `> **Note:** ...`
4. Rename .qmd → .md

**Commands:**
```bash
# Convert research reports
cd docs/concepts/research/
for f in *.qmd; do
  # Strip YAML, convert callouts (manual edits)
  git mv "$f" "${f%.qmd}.md"
done

# Convert design docs
cd ../design/
for f in *.qmd; do
  git mv "$f" "${f%.qmd}.md"
done

git add docs/concepts/
git commit -m "docs: convert concepts from Quarto to plain markdown

- Strip YAML frontmatter
- Convert Quarto callouts to standard markdown
- Rename .qmd → .md"
```

---

### Task 2.3: Create Inline API Reference

**Goal:** Create minimal API reference using inline `:::` syntax.

**File to Create:** `docs/reference/index.md`

**Content:**
```markdown
# API Reference

Auto-generated documentation from Python docstrings.

## Data Generation

::: hellocloud.generation.WorkloadPatternGenerator
    options:
      show_root_heading: true
      show_source: false
      members:
        - generate_time_series

::: hellocloud.generation.WorkloadType
    options:
      show_root_heading: true

## TimeSeries API

::: hellocloud.timeseries.TimeSeries
    options:
      show_root_heading: true
      show_source: false

::: hellocloud.io.PiedPiperLoader
    options:
      show_root_heading: true
      show_source: false
```

**Commands:**
```bash
git add docs/reference/index.md
git commit -m "docs: create inline API reference with mkdocstrings

- Use ::: syntax to pull docstrings at build time
- Only document actively-used modules
- No separate file generation needed"
```

---

### Task 2.4: Update pyproject.toml Dependencies

**Goal:** Remove quartodoc, add mkdocs stack.

**File to Edit:** `pyproject.toml`

**Changes:**
```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.28.0",
    "mkdocs-jupyter>=0.25.0",
]
```

**Remove:**
- Any quartodoc references

**Commands:**
```bash
uv sync --group docs
git add pyproject.toml uv.lock
git commit -m "build: replace quartodoc with mkdocs stack

- Add mkdocs, mkdocs-material, mkdocstrings, mkdocs-jupyter
- Remove quartodoc
- Update lock file"
```

---

### Task 2.5: Update justfile with MkDocs Commands

**Goal:** Replace quarto commands with mkdocs.

**File to Edit:** `justfile`

**Find and replace:**
```makefile
# OLD
docs-api:
    quartodoc build --config docs/_quarto.yml

docs-preview:
    quarto preview docs/

docs-build:
    quarto render docs/

# NEW
docs-serve:
    uv run mkdocs serve

docs-build:
    uv run mkdocs build

docs-deploy:
    uv run mkdocs gh-deploy
```

**Commands:**
```bash
git add justfile
git commit -m "build: replace quarto commands with mkdocs in justfile

- docs-serve: preview documentation
- docs-build: build static site
- docs-deploy: deploy to GitHub Pages"
```

---

### Task 2.6: Delete Quarto Artifacts

**Goal:** Remove Quarto configuration files and theme.

**Files to Delete:**
- `docs/_quarto.yml`
- `docs/_sidebar.yml`
- `docs/_theme/` (directory)
- `docs/index.qmd` (create new index.md)
- `docs/objects.json` (quartodoc metadata)

**Create New:** `docs/index.md`

**Content:**
```markdown
# Hello Cloud

Time series forecasting and anomaly detection for cloud resources.

## Overview

Hello Cloud is a Python library for modeling cloud resource utilization patterns, forecasting future usage, and detecting anomalies in operational metrics.

**Key Features:**
- Empirically grounded (12-15% average CPU utilization)
- Multiple models (Gaussian Processes, ARIMA, foundation models)
- Production-ready (92% test coverage on GP library)

## Getting Started

### Installation

```bash
pip install git+https://github.com/nehalecky/hello-cloud.git
```

### Quick Start

```python
from hellocloud.generation import WorkloadPatternGenerator, WorkloadType

generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    interval_minutes=60
)
```

## Documentation

**[Notebooks](notebooks/index.md)** - Interactive tutorials (executed with outputs)

**[Concepts](concepts/index.md)** - Research reports and design docs

**[API Reference](reference/index.md)** - Auto-generated from docstrings

## Research Context

- CPU Utilization: 12-15% average
- Memory Utilization: 18-25% average
- Resource Waste: 25-35% of cloud spending
- Temporal Autocorrelation: 0.7-0.8

See [Cloud Resource Patterns Research](concepts/research/cloud-resource-patterns-research.md).
```

**Create:** `docs/notebooks/index.md`

```markdown
# Notebooks

Interactive analysis notebooks. All notebooks are executed and published to `./published/` with outputs.

## Tutorials

- [Workload Signatures](../published/02_guide_workload_signatures_guide.ipynb) - Understanding cloud resource patterns
- [IOPS Analysis](../published/03_EDA_iops_web_server.ipynb) - TSB-UAD dataset exploration
- [Gaussian Process Modeling](../published/04_modeling_gaussian_process.ipynb) - Production GP forecasting
- [PiedPiper Data](../published/05_EDA_piedpiper_data.ipynb) - Hierarchical time series
- [TimeSeries API](../published/06_quickstart_timeseries_loader.ipynb) - New TimeSeries loader
- [Forecasting Comparison](../published/07_forecasting_comparison.ipynb) - Baseline vs ARIMA vs TimesFM

## Running Locally

```bash
uv run jupyter lab notebooks/
```

All notebooks have Colab badges for cloud execution.
```

**Create:** `docs/concepts/index.md`

```markdown
# Concepts

Research reports and design documentation.

## Research

Literature reviews and empirical analysis informing the library:

- [Cloud Resource Patterns](research/cloud-resource-patterns-research.md)
- [Resource Correlations](research/cloud-resource-correlations-report.md)
- [Anomaly Datasets](research/timeseries-anomaly-datasets-review.md)
- [OpenTSLM Evaluation](research/opentslm-foundation-model-evaluation.md)

## Design

Architecture and modeling decisions:

- [Gaussian Process Design](design/gaussian-process-design.md)
```

**Commands:**
```bash
git rm docs/_quarto.yml docs/_sidebar.yml docs/index.qmd docs/objects.json
git rm -r docs/_theme/
git add docs/index.md docs/notebooks/index.md docs/concepts/index.md
git commit -m "refactor(docs): delete Quarto artifacts, create MkDocs index files

- Delete Quarto config and theme
- Create minimal index files for MkDocs navigation
- Point to published/*.ipynb for executed notebooks"
```

---

## Phase 3: CI/CD Integration

### Task 3.1: Update GitHub Actions for MkDocs

**Goal:** Build docs in CI, deploy to GitHub Pages.

**File to Modify:** `.github/workflows/ci.yml`

**Add after test job:**
```yaml
  docs:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/master'
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --group docs

      - name: Build docs
        run: uv run mkdocs build --strict

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

**Commands:**
```bash
git add .github/workflows/ci.yml
git commit -m "ci: add mkdocs build and GitHub Pages deploy

- Build docs after tests pass on master
- Use mkdocs gh-deploy action
- Replaces Quarto deployment"
```

---

### Task 3.2: Update CLAUDE.md with New Workflow

**Goal:** Document new documentation workflow.

**File to Modify:** `CLAUDE.md`

**Update Documentation section:**
```markdown
### Documentation

```bash
# Quick commands with just (recommended)
just docs-serve     # Preview documentation
just docs-build     # Build static site
just docs-deploy    # Deploy to GitHub Pages

# Or use commands directly
uv run mkdocs serve    # http://127.0.0.1:8000
uv run mkdocs build    # Output: site/
```

### Documentation Architecture

**Notebooks (./published/*.ipynb)** - Executed notebooks with outputs ARE the documentation
- Source: `notebooks/*.md` (MyST format, for development)
- Published: `./published/*.ipynb` (executed, displayed in docs)
- CI executes notebooks → publishes → builds docs

**Concepts (docs/concepts/)** - Research reports and design docs (standalone markdown)

**API Reference (docs/reference/)** - Auto-generated from docstrings
- Uses mkdocstrings with `:::` syntax
- DO NOT edit reference/*.md - update source code docstrings instead

**Tech Stack:**
- MkDocs-Material (theme)
- mkdocstrings (API docs)
- mkdocs-jupyter (notebook rendering)
```

**Commands:**
```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with MkDocs workflow

- Document mkdocs serve/build/deploy commands
- Explain notebook → published → docs flow
- Clarify that published/*.ipynb are displayed"
```

---

## Phase 4: Testing

### Task 4.1: Test MkDocs Serve Locally

**Goal:** Verify docs build and display correctly.

**Commands:**
```bash
uv run mkdocs serve
```

**Verification:**
- Navigate to http://127.0.0.1:8000
- Check notebooks from `./published/` display with outputs
- Check concepts render as markdown
- Check API reference shows docstrings
- Test search functionality

**No commit needed** (verification only).

---

### Task 4.2: Verify Internal Links

**Goal:** Ensure all links work.

**Process:**
1. Click through navigation
2. Test links in index.md
3. Test links in notebooks/index.md
4. Test links in concepts/index.md
5. Test API reference cross-references

**Fix any broken links**, commit changes.

---

### Task 4.3: Test CI Build

**Goal:** Verify GitHub Actions builds and deploys.

**Process:**
1. Push to master (or merge PR)
2. Watch GitHub Actions run
3. Verify docs job succeeds
4. Check https://nehalecky.github.io/hello-cloud updates

**If issues:** Debug and fix, re-push.

---

## Success Criteria

**Phase 0 Complete:**
- ✅ Old plan files deleted
- ✅ Scripts directory deleted
- ✅ Legacy plan moved to ./plans/

**Phase 1 Complete:**
- ✅ docs/tutorials/ deleted
- ✅ docs/how-to/ deleted
- ✅ docs/reference/*.qmd deleted

**Phase 2 Complete:**
- ✅ Minimal mkdocs.yml created (points to published/*.ipynb)
- ✅ Concepts converted to plain markdown
- ✅ Inline API reference created
- ✅ pyproject.toml updated (mkdocs stack)
- ✅ justfile updated
- ✅ Quarto artifacts deleted

**Phase 3 Complete:**
- ✅ CI builds docs after tests
- ✅ CI deploys to GitHub Pages
- ✅ CLAUDE.md documents workflow

**Phase 4 Complete:**
- ✅ mkdocs serve works locally
- ✅ All links verified
- ✅ CI deploys successfully

---

## Rollback Plan

If issues arise:
1. Revert commits: `git revert HEAD~N`
2. Incremental execution: Can pause after any task
3. Notebooks unaffected: MyST format works independently

---

## Notes

**Why published/*.ipynb in docs?**
- Executed notebooks with outputs are what users see
- Source .md files in notebooks/ are for development only
- CI executes → publishes → mkdocs displays

**Why delete how-to guides?**
- Notebooks already show how to do everything
- No value in maintaining separate quick-reference docs
- Reduces duplication and maintenance burden

**Why minimal config?**
- Material defaults are excellent
- Customization adds complexity
- Can always enhance later if needed
