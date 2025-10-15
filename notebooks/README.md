# Notebook Architecture: MyST Canonical Source

This repository uses **MyST Markdown** as the canonical source for all notebooks, with automatic `.ipynb` generation for interactive work.

## 🎯 Quick Start (TL;DR)

```bash
# Edit .md files in Cursor, sync to .ipynb for execution
uv run jupytext --sync notebooks/YOUR_NOTEBOOK.md

# Open the generated .ipynb in Cursor or Jupyter Lab
# Kernel should auto-select as "Python 3 (ipykernel)"
```

**Workflow:** Edit `.md` (clean, version-controlled) → Sync → Execute `.ipynb` (gitignored artifact)

## 📁 Directory Structure

```
notebooks/
├── README.md                          # This file - MyST architecture overview
├── WORKFLOW.md                        # 🚀 Hot reload workflow guide (read this!)
├── _build/                            # Generated .ipynb files (gitignored)
│   ├── *.ipynb                        # Converted from .md source
│   └── README.md                      # Explains _build/ purpose
├── 01_data_exploration.md             # ✅ Source (MyST) - Data exploration
├── 02_workload_signatures_guide.md    # ✅ Source (MyST) - Workload taxonomy
├── 03_iops_web_server_eda.md          # ✅ Source (MyST) - Anomaly detection EDA
└── 04_gaussian_process_modeling.md    # ✅ Source (MyST) - GP library runbook
```

## 🎯 Canonical Source: MyST Markdown (.md)

**✅ What we version control:**
- `notebooks/*.md` - MyST markdown notebooks (source of truth)

**❌ What we ignore (gitignored):**
- `notebooks/_build/*.ipynb` - Generated notebooks (recreated from .md)
- `.ipynb_checkpoints/` - Jupyter temporary files

## 🔄 Jupytext Workflow (MyST ↔ Jupyter)

**Configuration:** All pairing handled by `pyproject.toml` → `[tool.jupytext.formats]`

```toml
[tool.jupytext.formats]
"notebooks/" = "md:myst"        # Source (version controlled)
"notebooks/_build/" = "ipynb"   # Generated (gitignored)
```

### Understanding Content vs Outputs

`★ Insight ─────────────────────────────────────`
**What Jupytext Syncs (Bidirectional):**
- ✅ Code cells (Python code)
- ✅ Markdown cells (documentation)
- ✅ Cell metadata (tags, settings)

**What Jupytext Does NOT Sync:**
- ❌ Cell outputs (plots, tables, print statements)
- ❌ Execution counts (`[1]`, `[2]`, etc.)
- ❌ Kernel state

**Mental Model:** `.md` = source code | `.ipynb` = source + execution results

Outputs only appear when you **execute cells** in Jupyter (Lab/Notebook/VSCode).
`─────────────────────────────────────────────────`

### Workflow Options

**Option 1: VSCode/Cursor** ⭐ **Open .ipynb, Edit .md**
```bash
# 1. Ensure pairing is set up (one-time setup)
uv run jupytext --set-formats notebooks//md:myst,notebooks/_build//ipynb notebooks/_build/*.ipynb

# 2. Open the .ipynb file from _build/ in VSCode
# File → Open → notebooks/_build/06_quickstart_timeseries_loader.ipynb

# 3. Edit and execute in VSCode's notebook interface
# 4. When you save, Jupytext auto-syncs changes back to the .md file

# 5. To manually sync if needed:
uv run jupytext --sync notebooks/_build/06_quickstart_timeseries_loader.ipynb
```

**IMPORTANT:** Open `.ipynb` files from `_build/`, NOT the `.md` files directly. VSCode's Jupyter extension doesn't recognize `.md` files as notebooks.

**Option 2: Jupyter Lab** ⭐ **Auto-Sync on Save**
```bash
# Jupyter Lab has Jupytext integration built-in
uv run jupyter lab

# Open .md file directly - Jupyter shows it as notebook
# Syncs to _build/*.ipynb automatically on save
```

**Option 3: Manual CLI Sync**
```bash
# Sync specific notebook
uv run jupytext --sync notebooks/06_quickstart.md

# Sync all notebooks
uv run jupytext --sync notebooks/*.md
```

**Option 4: VSCode Task (Keyboard Shortcut)** *(optional setup)*
Add `.vscode/tasks.json` for **Cmd+Shift+B** sync shortcut (see [WORKFLOW.md](WORKFLOW.md)).

### Hot Reload Pattern ⚡ (Eliminates Kernel Restarts)

Add this to the **top of your notebook** to auto-reload library code when edited with Claude Code:

```python
%load_ext autoreload
%autoreload 2
```

**Now you can:**
1. Edit `src/hellocloud/` modules with Claude Code
2. Changes are picked up automatically
3. **No kernel restart needed** - keeps trained models, loaded data intact!

**👉 See [WORKFLOW.md](WORKFLOW.md) for detailed development patterns.**

## 🧬 MyST Format Benefits

### Version Control Friendly

```markdown
---
kernelspec:
  display_name: Python 3 (cloud-sim)
  name: python3
---

# Analysis Title

## Section 1

```{code-cell} ipython3
import polars as pl
df = pl.read_csv("data.csv")
```

Clean markdown with code cells, perfect for git diffs.
```

### Rich Documentation
- Supports all Jupyter features (plots, widgets, etc.)
- Better markdown rendering than .ipynb
- Executable as notebooks or scripts
- Clean, readable source format

## 📚 Notebook-Library Integration

### Gaussian Process Modeling (Notebook 04)

**Notebook**: `04_gaussian_process_modeling.md`
**Library**: `cloud_sim.ml_models.gaussian_process`
**Design Doc**: `docs/modeling/gaussian-process-design.md`

This notebook demonstrates **library-first development** where:
- Production code lives in `src/cloud_sim/ml_models/gaussian_process/`
- Notebook serves as a **runbook** showing library usage patterns
- Educational narrative preserved in design documentation
- 92% test coverage ensures production readiness

**Key modules:**
- `kernels.py` - Composite periodic kernel for multi-scale patterns
- `models.py` - Sparse variational GP for scalability
- `training.py` - Mini-batch training with numerical stability
- `evaluation.py` - Comprehensive metrics (accuracy + calibration)

**Usage pattern:**
```python
from hellocloud.modeling.gaussian_process import (
    SparseGPModel, CompositePeriodicKernel,
    train_gp_model, compute_metrics
)
```

## 🚀 Quick Start

**Open a notebook:**
```bash
# Launch Jupyter Lab with any MyST notebook
uv run jupyter lab notebooks/04_gaussian_process_modeling.md

# Or launch Jupyter Lab and select from file browser
uv run jupyter lab
```

**Create a new notebook:**
```bash
# Copy an existing notebook as template
cp notebooks/02_workload_signatures_guide.md notebooks/my_analysis.md

# Edit the frontmatter (kernelspec, title)
# Then open in Jupyter Lab
uv run jupyter lab notebooks/my_analysis.md
```

**Convert existing .ipynb to MyST:**
```bash
uv run jupytext --to md:myst existing_notebook.ipynb
mv existing_notebook.md notebooks/
```

## 🔧 Configuration

**Primary Config:** `pyproject.toml` (standard Python PEP 518)

```toml
[tool.jupytext]
notebook_metadata_filter = "-all"
cell_metadata_filter = "-all"

[tool.jupytext.formats]
"notebooks/" = "md:myst"        # Source directory
"notebooks/_build/" = "ipynb"   # Build directory
```

**Why `pyproject.toml`?**
- ✅ Works with Jupytext CLI, Jupyter Lab, AND VSCode extensions
- ✅ Standard Python project config (one file for everything)
- ✅ Recognized by all modern Python tools

**Legacy Config Files (NOT USED):**
- `.jupytext.toml` - CLI-only, doesn't work with Jupyter (deprecated)
- `jupytext.toml` - Redundant with `pyproject.toml`

## 📋 File Lifecycle

1. **Create**: Write MyST markdown (`.md`)
2. **Develop**: Sync to `.ipynb`, work in Jupyter
3. **Execute**: Run cells → outputs appear in `.ipynb` only
4. **Commit**: Only MyST (`.md`) goes to git
5. **Publish**: CI/Quarto executes notebooks and generates outputs at build time

**Result**: Clean git history, reproducible notebooks, no merge conflicts! 🎉

## 📤 Publishing Notebooks with Outputs

**Key Insight:** Outputs (plots, tables) are **execution artifacts**, not source code.

### Recommended Approach: Execute at Build Time

**Our setup (Quarto):**
```bash
quarto render docs/
# Quarto reads .md files, executes them, captures outputs, renders HTML
```

**Benefits:**
- ✅ Git tracks `.md` only (clean, small commits)
- ✅ Outputs always up-to-date (regenerated on each build)
- ✅ No stale outputs (code and results always match)

**Alternative: Execute on CI/CD:**
```bash
# In GitHub Actions
uv run jupyter nbconvert --to notebook --execute notebooks/*.md
# Deploys executed .ipynb files to GitHub Pages
```

### Local Verification Workflow

```bash
# 1. Edit .md file in Cursor
vim notebooks/06_quickstart.md

# 2. Sync to .ipynb
uv run jupytext --sync notebooks/06_quickstart.md

# 3. Execute in Jupyter Lab (verify outputs look good)
uv run jupyter lab notebooks/_build/06_quickstart.ipynb

# 4. Discard .ipynb (gitignored, will be regenerated)
git status  # Only .md file shows as modified

# 5. Commit .md only
git add notebooks/06_quickstart.md
git commit -m "docs: update quickstart analysis"

# 6. CI/Quarto regenerates outputs at publish time
```

**Why this works:** The `.ipynb` in `_build/` is ephemeral - used for local execution, discarded, regenerated fresh at publish time.

---

## 📤 Publishing for Google Colab

For interactive tutorials, publish executed notebooks to `published/` directory.

### Quick Publish (Default)

Assumes notebooks already executed in Jupyter:

```bash
# Publish single notebook (copy from _build/)
just nb-publish 06_quickstart_timeseries_loader

# Or publish all
just nb-publish-all

# Commit and push
git add notebooks/*.md notebooks/published/*.ipynb
git commit -m "docs: update tutorials"
git push
```

### Clean Rebuild (Optional)

Execute from scratch for guaranteed fresh outputs:

```bash
# Clean execution + publish
just nb-publish-clean 06_quickstart_timeseries_loader

# Or all notebooks
just nb-publish-all-clean
```

**When to use:** First publication, CI/CD, or verifying clean execution.

### Library Access in Colab

All notebooks include setup cells that auto-install `hellocloud`:

```python
# Environment Setup (in all notebooks)
try:
    import hellocloud
except ImportError:
    !pip install -q git+https://github.com/nehalecky/hello-cloud.git
    import hellocloud
```

**Users click badge → Opens with outputs → Library auto-installs → Works!**

### Published Directory Structure

```
notebooks/
├── *.md                    # Source (edit these!)
├── _build/*.ipynb          # Working files (gitignored)
└── published/*.ipynb       # Colab distribution (Git tracked)
```

**IMPORTANT:** `published/*.ipynb` are generated artifacts. Always edit `.md` source files!
