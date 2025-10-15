# Notebook Development Workflow

**Goal**: Edit code with Claude Code while keeping Jupyter kernel running - no restarts needed!

## Quick Start

```bash
# Start Jupyter Lab
./scripts/notebook-workflow.sh lab

# Or use the shortcut
uv run jupyter lab notebooks/
```

## Jupytext + Cursor Integration

### Understanding the Architecture

**Source Format:** MyST Markdown (`.md`) - clean, version-controlled
**Execution Format:** Jupyter Notebook (`.ipynb`) - generated in `_build/`

```
notebooks/06_quickstart.md         # Edit this (Git tracks)
      ↕ (sync)
notebooks/_build/06_quickstart.ipynb  # Execute this (gitignored)
```

**Configuration:** All managed in `pyproject.toml`:

```toml
[tool.jupytext.formats]
"notebooks/" = "md:myst"        # Source directory
"notebooks/_build/" = "ipynb"   # Build directory
```

### Content vs Outputs: What Gets Synced?

`★ Insight ─────────────────────────────────────`
**Jupytext syncs SOURCE CODE, not EXECUTION RESULTS**

**What syncs (bidirectional):**
- ✅ Code cells (Python code)
- ✅ Markdown cells (documentation)
- ✅ Cell metadata (tags, settings)

**What does NOT sync:**
- ❌ Cell outputs (plots, tables, print statements)
- ❌ Execution counts (`[1]`, `[2]`, etc.)
- ❌ Kernel state

**Mental model:** `.md` = source code | `.ipynb` = source + execution results

To get outputs in `.ipynb`, you must **execute cells** in Jupyter (Lab/Notebook/Cursor).
`─────────────────────────────────────────────────`

### Four Ways to Work with Notebooks

#### **Option 1: Cursor + vscode-jupytext-sync Extension** ⭐ **AUTO-SYNC**

```bash
# If extension installed: vscode-jupytext-sync by caenrigen
# Just edit and save - no manual sync needed!
vim notebooks/06_quickstart.md
# Press Cmd+S → auto-syncs to _build/*.ipynb
```

**How it works:**
- Extension watches file saves (`.md` and `.ipynb`)
- Automatically runs `uv run jupytext --sync` on save
- **Identical behavior to Jupyter Lab** - fully automatic!

**To verify extension is working:**
```bash
# Edit .md file, save, then check:
ls -l notebooks/_build/06_quickstart.ipynb
# Timestamp should be updated!
```

- ✅ Best for: Full Cursor workflow with zero manual steps
- ✅ Bidirectional: Edit `.md` OR `.ipynb`, both auto-sync

#### **Option 2: Jupyter Lab** ⭐ **AUTO-SYNC**

```bash
uv run jupyter lab
```

- **Open `.md` files directly** - Jupyter Lab renders them as notebooks
- **Auto-sync on save** - Changes to `.md` sync to `.ipynb` automatically
- **Bidirectional** - Edit in Jupyter UI, changes sync to `.md`
- ✅ Best for: Iterative development, running code, seeing outputs

#### **Option 3: Manual CLI Sync** (No Extension Required)

```bash
# 1. Edit .md file in Cursor
vim notebooks/06_quickstart.md

# 2. Manually sync to .ipynb
uv run jupytext --sync notebooks/06_quickstart.md

# 3. Open .ipynb in Cursor's Jupyter extension
# 4. Execute cells with project kernel (Python 3 in .venv)
```

- **No extension required** - Works out of the box
- **Explicit control** - You decide when to sync
- ✅ Best for: Minimal setup, explicit workflow

#### **Option 4: VSCode Task (Keyboard Shortcut)** *(Optional Setup)*

Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "JupyText Sync",
      "type": "shell",
      "command": "uv run jupytext --sync ${file}",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "never"
      }
    }
  ]
}
```

Then press **Cmd+Shift+B** (Mac) or **Ctrl+Shift+B** (Windows/Linux) to sync!

- **One-key sync** - Fastest workflow
- **Silent execution** - No terminal clutter
- ✅ Best for: Rapid iteration in Cursor

### Kernel Selection in Cursor

When opening `.ipynb` files in Cursor:

1. Click "Select Kernel" in top-right
2. Choose **"Python 3"** or **"Python 3.12.x (.venv)"**
3. Kernel is project-local (`.venv/share/jupyter/kernels/python3`)

**Why it works:**
- `ipykernel` installed in `.venv/` registers kernel automatically
- Cursor's Jupyter extension discovers it
- No manual `ipykernel install` needed!

**Config:** Ensure `.vscode/settings.json` has:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "notebook.defaultKernel": "python3"
}
```

### How Jupytext Sync is Triggered

**Jupytext is NOT a background daemon** - it's a CLI tool that runs when explicitly invoked:

| Option | Trigger | Who Invokes Jupytext | Auto or Manual? |
|--------|---------|---------------------|-----------------|
| **vscode-jupytext-sync** | File save | VSCode extension | ✅ Automatic |
| **Jupyter Lab** | File save | Jupyter contents manager | ✅ Automatic |
| **Manual CLI** | You type command | Shell | ⚠️ Manual |
| **VSCode Task** | You press hotkey | VSCode shell task | ⚠️ Manual |
| **Pre-commit Hook** | `git commit` | Git pre-commit framework | ✅ Automatic |

**Key insight:** No continuous background sync - files can drift out of sync between triggers.

### Automation Options

#### **Option A: Auto-Sync on Save** ⭐ **Recommended**

**Use vscode-jupytext-sync extension OR Jupyter Lab:**
- ✅ **Zero manual steps** - Just edit and save
- ✅ **Bidirectional** - Edit `.md` or `.ipynb`, both stay in sync
- ✅ **Familiar workflow** - Identical to regular file editing

**Trade-offs:**
- ⚠️ Files sync even if you didn't want them to (e.g., WIP notebooks)
- ⚠️ Extension dependency (but Jupyter Lab built-in)

#### **Option B: Manual Sync** (Current Setup if No Extension)

**Use CLI or VSCode task:**
```bash
uv run jupytext --sync notebooks/06_quickstart.md
# Or press Cmd+Shift+B (if task configured)
```

- ✅ **Full control** - Sync only when you want
- ✅ **No surprises** - Explicit, predictable
- ✅ **Works without extension** - Just CLI tool

**Trade-offs:**
- ⚠️ Must remember to sync before executing `.ipynb`
- ⚠️ Must remember to sync before committing

#### **Option C: Pre-commit Hook** (Team/CI Workflows)

Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/mwouts/jupytext
    rev: v1.17.3
    hooks:
      - id: jupytext
        args: [--sync]
```

- ✅ **Enforced consistency** - Cannot commit out-of-sync notebooks
- ✅ **Works with any editing workflow** - Syncs at commit time
- ✅ **Team safety net** - Catches forgotten syncs

**Trade-offs:**
- ⚠️ Slower commits (~1-3 seconds per commit)
- ⚠️ Always syncs all notebooks (can't selectively skip)

**Recommendation:**
- **Solo dev, iterating:** Use Option A (auto-sync extension) or B (manual)
- **Team, CI/CD:** Use Option A or B + Option C (pre-commit hook as safety net)

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
