# Notebook Development Workflow

**Goal**: Edit code with Claude Code while keeping Jupyter kernel running - no restarts needed!

## Quick Start

```bash
# Start Jupyter Lab
./scripts/notebook-workflow.sh lab

# Or use the shortcut
uv run jupyter lab notebooks/
```

## The Hot Reload Pattern ⚡

This eliminates kernel restarts when editing library code with Claude Code.

### Setup (One-time per notebook session)

Add this cell **at the top** of your notebook:

```python
# Hot reload configuration - auto-reloads library code on changes
%load_ext autoreload
%autoreload 2

# Optional: Show what's being reloaded
# %aimport hellocloud
```

### How It Works

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Claude Code    │         │  src/cloud_sim/  │         │  Jupyter Kernel │
│                 │ edits   │                  │ reloads │                 │
│  Edit library   ├────────>│  Library code    ├────────>│  %autoreload 2  │
│  code           │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                                                    │
                                                                    │ NO RESTART!
                                                                    │
                                                                    ▼
                                                          ┌─────────────────┐
                                                          │  Run cells      │
                                                          │  using updated  │
                                                          │  code           │
                                                          └─────────────────┘
```

**What gets auto-reloaded:**
- ✅ Functions in `src/cloud_sim/` modules
- ✅ Class definitions (updated for NEW instances)
- ✅ Constants and global variables
- ❌ Already-instantiated objects (create new ones to pick up changes)

**Example workflow:**

1. In Jupyter: Run a cell that uses `SparseGPModel`
2. In Claude Code: Edit `src/cloud_sim/ml_models/gaussian_process/models.py`
3. In Jupyter: Re-run the cell → **picks up changes automatically!**

No kernel restart, no losing computed state (like trained models).

## Editing Notebook Content

### Option A: Edit in Jupyter UI (Recommended)

- Edit cells, markdown, code in Jupyter Lab
- **On save**: Jupytext automatically updates the `.md` file
- ✅ Best for: Iterative development, narrative + code changes
- ✅ Preserves kernel state

### Option B: Edit .md File with Claude Code

- Edit `notebooks/*.md` directly
- In Jupyter browser: File → Reload Notebook from Disk
- ⚠️ **Kernel state is preserved**, but cell outputs are reloaded from disk
- ✅ Best for: Bulk changes, refactoring, documentation updates

## Directory Structure

```
notebooks/
├── *.md                          # Source files (version controlled)
├── _build/                       # Generated artifacts (gitignored)
│   ├── *.ipynb                   # Converted notebooks for execution
│   └── README.md                 # Explanation of _build/
└── WORKFLOW.md                   # This file
```

## Common Tasks

### Open a Notebook

```bash
./scripts/notebook-workflow.sh open 04_gaussian_process_modeling
# OR
uv run jupyter lab notebooks/04_gaussian_process_modeling.md
```

### Convert Without Executing

```bash
./scripts/notebook-workflow.sh convert 04_gaussian_process_modeling
```

### Execute Notebook (Testing)

```bash
./scripts/notebook-workflow.sh execute 04_gaussian_process_modeling
```

### Clean Generated Files

```bash
./scripts/notebook-workflow.sh clean
```

### Publish All Notebooks (Future - Binder)

```bash
./scripts/notebook-workflow.sh publish
```

## Advanced: When to Restart Kernel

You **must** restart the kernel when:

1. **Changing function signatures** that affect already-loaded code paths
2. **Modifying C extensions** or compiled dependencies
3. **Changing import structure** (new modules, renamed imports)
4. **Kernel becomes unstable** (memory leaks, corrupted state)

Otherwise, `%autoreload 2` handles it!

## Binder Publishing (Future)

For deploying notebooks to Binder:

1. Execute all notebooks: `./scripts/notebook-workflow.sh publish`
2. Commit `_build/` to a separate branch: `gh-pages`
3. Configure Binder to use that branch
4. Or use `postBuild` script to convert .md → .ipynb on Binder launch

## References

- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [IPython Autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html)
- [MyST Notebooks](https://myst-nb.readthedocs.io/)
