# Tutorial Notebooks

This directory contains interactive tutorial notebooks that demonstrate the capabilities of **hello cloud** for time series forecasting and cloud resource analysis.

## 📚 Available Tutorials

### Getting Started
- **[TimeSeries Quickstart](published/06_quickstart_timeseries_loader.ipynb)** - Quick introduction to loading and analyzing hierarchical time series data with the PiedPiperLoader (~15 minutes)

### Data Analysis & Exploration
- **[Workload Signatures Guide](published/02_guide_workload_signatures_guide.ipynb)** - Understanding and generating realistic workload patterns for cloud resources
- **[IOPS Analysis](published/03_EDA_iops_web_server.ipynb)** - Exploratory data analysis of web server I/O patterns
- **[PiedPiper Data Analysis](published/05_EDA_piedpiper_data.ipynb)** - Deep dive into hierarchical cloud cost data with temporal and entity-based analysis

### Machine Learning Models
- **[Gaussian Process Modeling](published/04_modeling_gaussian_process.ipynb)** - Time series forecasting with sparse variational Gaussian processes (GPyTorch)
- **[Forecasting Comparison](published/07_forecasting_comparison.ipynb)** - Comparing multiple forecasting approaches including foundation models

## 🔍 Viewing Notebooks

### On GitHub
Click any of the links above to view fully-executed notebooks with outputs directly on GitHub. GitHub natively renders `.ipynb` files with rich formatting, plots, and results.

### In Documentation
All tutorials are also available on the [hello cloud documentation site](https://nehalecky.github.io/hello-cloud) with integrated navigation and search.

## 💻 Running Notebooks Locally

### Source Format
Notebooks are stored as **Python percent format** (`.py` files) for clean git diffs and IDE-friendly editing. The `.py` files contain:
- Python code in regular syntax
- Markdown cells as comments with `# %% [markdown]` markers
- Executable as Python scripts or convertible to `.ipynb`

### Execution Workflow
1. **Edit source:** Modify `.py` files in your IDE with full linting and type checking
2. **Convert to notebook:** Use jupytext to create `.ipynb` from `.py` source
3. **Execute:** Run in Jupyter Lab to generate outputs
4. **Publish:** Executed notebooks saved to `published/*.ipynb`

See **[WORKFLOW.md](WORKFLOW.md)** for detailed instructions on the notebook development workflow.

### Prerequisites
```bash
# Install dependencies
uv sync --all-extras

# Optional: Start Jupyter Lab
uv run jupyter lab examples/
```

### Converting and Executing
```bash
# Convert single notebook
cd examples
uv run jupytext --execute --to ipynb --output published/notebook.ipynb notebook.py

# Or bulk convert all notebooks
uv run jupytext --execute --to ipynb --output-dir published/ *.py
```

## 📖 Documentation

- **[WORKFLOW.md](WORKFLOW.md)** - Complete guide to editing and executing notebooks
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## 🏗️ Structure

```
examples/
├── *.py                     # Source notebooks (Python percent format)
├── published/               # Executed notebooks with outputs
│   └── *.ipynb             # GitHub renders these natively
├── README.md               # This file
├── WORKFLOW.md             # Editing and execution guide
└── TROUBLESHOOTING.md      # Common issues
```

## 🎯 Design Philosophy

**Why Python percent format as source?**
- ✅ **Clean git diffs:** Line-based changes, not JSON
- ✅ **IDE support:** Full linting, type checking, refactoring
- ✅ **Executable:** Run directly as Python scripts
- ✅ **Convertible:** jupytext converts to `.ipynb` when needed
- ✅ **Review-friendly:** Easier to review code changes in PRs

**Why keep executed `.ipynb` in git?**
- ✅ **GitHub rendering:** Native notebook viewing without setup
- ✅ **Documentation:** MkDocs renders them in the website
- ✅ **Reproducibility:** Shows expected outputs
- ✅ **Discoverability:** Users can browse and learn

## 🤝 Contributing

When contributing tutorial notebooks:

1. **Edit the `.py` source file** (not `.ipynb`)
2. **Test your changes** by executing the notebook
3. **Regenerate `published/*.ipynb`** with outputs
4. **Commit both** `.py` source and `.ipynb` output
5. **Verify on GitHub** that the notebook renders correctly

See [WORKFLOW.md](WORKFLOW.md) for detailed contribution guidelines.
