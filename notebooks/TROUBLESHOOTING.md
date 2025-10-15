# Notebook Troubleshooting Guide

## Common Issues

### ❌ "The editor could not be opened because the file was not found"

**Problem:** VSCode shows this error when trying to open `.md` notebook files.

**Root Cause:** VSCode's Jupyter extension expects `.ipynb` files, not `.md` files.

**Solution:** Open the paired `.ipynb` file instead:

```bash
# ❌ DON'T: Open .md files directly in VSCode
File → Open → notebooks/05_EDA_piedpiper_data.md

# ✅ DO: Open .ipynb files from _build/ directory
File → Open → notebooks/_build/05_EDA_piedpiper_data.ipynb
```

**Why this works:** Jupytext pairing ensures edits to the `.ipynb` file automatically sync to the `.md` source file on save.

---

### ❌ "Pairing not detected" or changes don't sync

**Problem:** Editing `.ipynb` doesn't update the `.md` file.

**Root Cause:** Pairing metadata missing from `.ipynb` files.

**Solution:** Run the pairing setup command:

```bash
# Setup pairing for all notebooks (one-time)
just nb-setup-pairing

# Or manually for a specific notebook
uv run jupytext --set-formats notebooks//md:myst,notebooks/_build//ipynb \
    notebooks/_build/05_EDA_piedpiper_data.ipynb
```

**Verification:**
```bash
# Check if pairing metadata exists
uv run python -c "
import json
nb = json.load(open('notebooks/_build/05_EDA_piedpiper_data.ipynb'))
print('Formats:', nb['metadata']['jupytext'].get('formats', 'NOT SET'))
"

# Should output: notebooks//md:myst,notebooks/_build//ipynb
```

---

### ❌ `.ipynb` files don't exist in `_build/`

**Problem:** `notebooks/_build/` directory is empty or missing `.ipynb` files.

**Root Cause:** Initial conversion from `.md` to `.ipynb` wasn't done.

**Solution:** Convert all markdown notebooks to ipynb:

```bash
# Create _build directory if it doesn't exist
mkdir -p notebooks/_build

# Convert a specific notebook
uv run jupytext --to ipynb \
    --output notebooks/_build/05_EDA_piedpiper_data.ipynb \
    notebooks/05_EDA_piedpiper_data.md

# Or sync all notebooks (creates .ipynb from .md)
for md in notebooks/*.md; do
    if [[ ! "$md" =~ (README|WORKFLOW|TROUBLESHOOTING) ]]; then
        uv run jupytext --sync "$md"
    fi
done

# Then setup pairing
just nb-setup-pairing
```

---

### ❌ Edits in VSCode don't appear in `.md` files

**Problem:** You edit and save in VSCode, but `git status` shows no changes to `.md` files.

**Possible Causes:**

1. **Pairing not configured** - See "Pairing not detected" above
2. **Editing the wrong file** - Make sure you're editing `.ipynb` from `_build/`, not `.md` directly
3. **Auto-save disabled** - Jupytext syncs on save, ensure auto-save or manual save is working

**Verification:**
```bash
# 1. Edit a cell in VSCode notebook (notebooks/_build/*.ipynb)
# 2. Save the file (Cmd+S or Ctrl+S)
# 3. Check git status
git status notebooks/

# You should see the .md file as modified, NOT the .ipynb
```

---

### ❌ Kernel not found or wrong Python version

**Problem:** VSCode shows "Kernel not found" or uses wrong Python version.

**Solution:** Select the correct kernel:

```bash
# 1. In VSCode, open a notebook from _build/
# 2. Click kernel selector (top-right of notebook)
# 3. Select "Python Environments..."
# 4. Choose ".venv (Python 3.12.x)"

# Or verify kernel from command line
uv run python -c "import sys; print(sys.executable)"
# Should output: /Users/.../hello-cloud/.venv/bin/python
```

---

### ❌ Changes lost when switching between `.md` and `.ipynb`

**Problem:** Edits disappear or conflict when editing both files.

**Root Cause:** Editing both `.md` source and `.ipynb` execution files simultaneously.

**Solution:** **Never edit both files**. Choose one workflow:

**Option A: VSCode/Cursor workflow (Recommended)**
- Edit `.ipynb` files from `_build/`
- Jupytext auto-syncs to `.md` on save
- Commit only `.md` files

**Option B: Text editor workflow**
- Edit `.md` files directly in vim/emacs
- Run `uv run jupytext --sync notebooks/*.md` before opening in Jupyter
- Use `.ipynb` files only for execution

**Option C: Jupyter Lab workflow**
- Jupyter Lab handles pairing automatically
- Open `.md` files directly - Jupyter shows as notebook
- No manual syncing needed

---

### ❌ Published notebooks missing outputs

**Problem:** Notebooks in `published/` directory don't show plots or outputs.

**Solution:** Execute notebooks before publishing:

```bash
# Execute single notebook with outputs
just nb-publish-clean 06_quickstart_timeseries_loader

# Or execute all notebooks
just nb-publish-all-clean

# Then commit
git add notebooks/published/*.ipynb
git commit -m "docs: update published notebooks with outputs"
```

---

## Quick Reference: Correct Workflows

### VSCode/Cursor Development
```bash
# 1. Setup pairing (one-time)
just nb-setup-pairing

# 2. Open notebook from _build/
#    File → Open → notebooks/_build/05_EDA_piedpiper_data.ipynb

# 3. Edit and execute in VSCode
#    Changes auto-sync to .md on save

# 4. Commit only .md files
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "docs: update analysis"
```

### Jupyter Lab Development
```bash
# 1. Launch Jupyter Lab
uv run jupyter lab

# 2. Open .md file directly
#    Jupyter Lab handles pairing automatically

# 3. Edit and execute normally

# 4. Commit only .md files
git add notebooks/*.md
git commit -m "docs: update notebooks"
```

### Command Line Workflow
```bash
# 1. Edit .md file with text editor
vim notebooks/05_EDA_piedpiper_data.md

# 2. Sync to .ipynb
uv run jupytext --sync notebooks/05_EDA_piedpiper_data.md

# 3. Execute (optional)
uv run jupyter nbconvert --to notebook --execute \
    notebooks/_build/05_EDA_piedpiper_data.ipynb

# 4. Commit .md only
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "docs: update analysis"
```

---

## Still Stuck?

1. **Check pairing status:**
   ```bash
   uv run jupytext --paired notebooks/_build/05_EDA_piedpiper_data.ipynb
   # Should output: notebooks/05_EDA_piedpiper_data.md
   ```

2. **Verify Jupytext is installed:**
   ```bash
   uv run jupytext --version
   ```

3. **Check VSCode Jupyter extension:**
   - Extension ID: `ms-toolsai.jupyter`
   - Should be installed and enabled

4. **Re-run pairing setup:**
   ```bash
   just nb-setup-pairing
   ```

5. **Check pyproject.toml config:**
   ```bash
   grep -A 5 "tool.jupytext" pyproject.toml
   ```

If issues persist, [open an issue](https://github.com/nehalecky/hello-cloud/issues) with:
- VSCode version
- Output of `uv run jupytext --version`
- Steps to reproduce
