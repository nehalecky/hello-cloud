# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,kernelspec,language_info,-widgets,-toc
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.12
# ---

# %% [markdown]
# # Quick Start: TimeSeries Loader
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/06_quickstart_timeseries_loader.ipynb)
#
# ## Overview
#
# This notebook demonstrates the `TimeSeries` loader for hierarchical time series data. Learn how to:
#
# - Load PiedPiper billing data in 3 lines
# - Filter, sample, and aggregate entities
# - Visualize time series with publication-quality plots
# - Compute summary statistics across entities
#
# **Target audience**: Data scientists working with hierarchical time series (billing, metrics, IoT)
#
# **Prerequisites**: PiedPiper dataset (or substitute your own hierarchical time series data)
#
# ---
#
# ## Setup

# %%
# Environment Setup
# Local: Uses installed hellocloud
# Colab: Installs from GitHub
try:
    import hellocloud
except ImportError:
    # !pip install -q git+https://github.com/nehalecky/hello-cloud.git
    pass

# %%
# Auto-reload: Picks up library changes without kernel restart
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_formats = ['png', 'retina']

# %%
# Standard imports
from pathlib import Path

import seaborn as sns

# PySpark and hellocloud
import hellocloud as hc

# Set seaborn theme for publication-quality plots

sns.set_theme()

# Get Spark session
spark = hc.spark.get_spark_session(app_name="quickstart-timeseries")

# %% [markdown]
# ## 1. Loading Data
#
# The `PiedPiperLoader` applies EDA-informed defaults to clean and structure billing data:
#
# - **Column renames**: `usage_date` → `date`, `materialized_cost` → `cost`
# - **Drops low-info columns**: UUIDs, redundant cost variants (4 removed)
# - **Default hierarchy**: `provider → account → region → product → usage_type`

# %%
from hellocloud.io import PiedPiperLoader

# Load raw data
# data_path = Path("../data/piedpiper_processed/piedpiper_clean")  # Adjust to your data location
# /cloudzero/hello-cloud/data/piedpiper_optimized_daily.parquet
data_path = Path("../../data/piedpiper_optimized_daily.parquet")
raw_df = spark.read.parquet(str(data_path))

print(f"Raw data: {raw_df.count():,} records, {len(raw_df.columns)} columns")

# %%
# Load into TimeSeries with defaults
# Loader logs all transformations (filtering, renaming, dropping columns)
ts = PiedPiperLoader.load(raw_df)

# %% [markdown]
# ## Temporal Observation Density Analysis
#
#   One of the first diagnostic checks for any time series dataset is **observation density** -
#   how consistently are records captured over time?
#
#   ### Why This Matters
#
#   Real-world data collection is messy. Systems fail, APIs timeout, data pipelines have gaps.
#   Before modeling or analysis, you need to understand:
#
#   1. **Data Completeness**: Are there missing dates or sparse periods?
#   2. **Collection Consistency**: Does observation frequency change over time?
#   3. **Quality Issues**: Do sudden drops signal upstream problems?
#
#   ### What the Plot Shows
#
#   The temporal density plot displays:
#   - **Top panel**: Record count per day (with shaded area for visual weight)
#   - **Bottom panel** (optional): Day-over-day percent change
#     - 🟢 Green bars = increases in observations
#     - 🔴 Red bars = decreases in observations
#
#   ### Interpretation Guide
#
#   **Healthy patterns:**
#   - Steady observation counts (flat line)
#   - Small day-to-day variations (<10%)
#   - No sudden drops or gaps
#
#   **Warning signs:**
#   - Sharp drops (>30-50%) suggest data quality issues
#   - Increasing trends may indicate growing system coverage
#   - Periodic spikes/drops might be business cycle effects (weekends, holidays)

# %%
# Overall record density over time.
ts.plot_temporal_density(show_pct_change=True)

# %% [markdown]
# **Observation**: We observe a sharp drop (> 30%) on 2025-10-06, and with data in future. We'll filter the time series to only consider data prior to this date.

# %%
ts = ts.filter_time(end="2025-10-06")
ts.plot_temporal_density(show_pct_change=True)

# %%

# %% [markdown]
# Now we check record density across the additional distinct keys.

# %%
ts.plot_density_by_grain(["region", "product", "usage", "provider"], show_pct_change=True)

# %% [markdown]
# We note some loss of distinct entities in the temporal density plot in the product, usage and provieder grains, however, overall data appears to be complete.

# %% [markdown]
# ## Cost Analysis

# %%
ts.plot_cost_treemap(["provider", "region"], top_n=30)

# %%
# 1. Summary statistics (DataFrame)
stats = ts.cost_summary_by_grain(["region"])
stats.toPandas().sort_values("total_cost", ascending=False)

# %%
# 2. Box plot - Daily cost distributions
ts.plot_cost_distribution(["provider"], top_n=15, min_cost=10, log_scale=True)

# %%
# 3. Time series trends - Top spenders over time
ts.plot_cost_trends(["region"], top_n=5, show_total=True, log_scale=True)

# %%

# %% [markdown]
# **Key pattern**: Create the `ax` object, customize as needed, return `ax` for further manipulation.
#
# ---
#
# ## 5. Next Steps
#
# ### Deeper Analysis
# - **Notebook 05**: Full EDA with grain discovery, entity persistence analysis
# - **Hierarchical forecasting**: Use aggregate/filter to build multi-level models
# - **Anomaly detection**: Compute z-scores with `summary_stats()`, flag outliers
#
# ### TimeSeries API
# - **More operations**: See `hellocloud.timeseries.TimeSeries` for complete API
# - **Transformations**: Use `hellocloud.transforms` for percent change, normalization
# - **Custom grains**: Mix and match hierarchy levels for your analysis needs
#
# ### Data Sources
# - **Extend PiedPiperLoader**: Add custom column mappings, filters
# - **New loaders**: Create loaders for other datasets following the same pattern
# - **Real-time data**: Integrate with streaming PySpark DataFrames
#
# ---
#
# ## Summary
#
# **What we learned:**
# - ✅ Load hierarchical time series data with `PiedPiperLoader`
# - ✅ Filter, sample, and aggregate using `TimeSeries` methods
# - ✅ Compute summary statistics across entities
# - ✅ Create publication-quality plots with automatic date formatting
# - ✅ Customize plots with matplotlib pass-through
#
# **Key insight**: The `TimeSeries` class keeps the full dataset in memory once. Operations like `filter()`, `sample()`, and `aggregate()` return new instances with filtered/aggregated DataFrames—leveraging PySpark's distributed engine while providing a domain-specific API.
#
# **Architecture**: `TimeSeries` → PySpark DataFrame → Distributed processing
