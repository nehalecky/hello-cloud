# Notebook Architecture: MyST Canonical Source

This repository uses **MyST Markdown** as the canonical source for all notebooks, with automatic `.ipynb` generation for interactive work.

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

## 🔄 MyST → Jupyter Workflow

**👉 See [WORKFLOW.md](WORKFLOW.md) for the complete hot reload workflow!**

### Quick Start

```bash
# Use the helper script
./scripts/notebook-workflow.sh lab

# Or launch directly
uv run jupyter lab notebooks/
```

### Hot Reload Pattern ⚡ (Eliminates Kernel Restarts)

Add this to the **top of your notebook** to auto-reload library code when edited with Claude Code:

```python
%load_ext autoreload
%autoreload 2
```

**Now you can:**
1. Edit `src/cloud_sim/` modules with Claude Code
2. Changes are picked up automatically
3. **No kernel restart needed** - keeps trained models, loaded data intact!

Jupytext manages `.md` ↔ `.ipynb` syncing automatically via `jupytext.toml`.

## 🧬 MyST Format Benefits

### Version Control Friendly

```markdown
---
kernelspec:
  display_name: Python 3 (cloud-sim)
  name: cloud-sim
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
from hellocloud.ml_models.gaussian_process import (
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

- **Global**: `.jupytext.toml` - Project defaults
- **Per-file**: YAML frontmatter in MyST files - Kernel selection
- **Pairing**: Set with `jupytext --set-formats md:myst,ipynb file.md`

## 📋 File Lifecycle

1. **Create**: Write MyST markdown
2. **Develop**: Generate .ipynb, work in Jupyter
3. **Commit**: Only MyST (.md) goes to git
4. **Share**: Others run `jupytext --to ipynb` to get .ipynb
5. **Collaborate**: Everyone works from same MyST source

**Result**: Clean git history, reproducible notebooks, no merge conflicts! 🎉
