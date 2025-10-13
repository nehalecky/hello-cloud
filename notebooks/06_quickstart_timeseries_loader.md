---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Quick Start: TimeSeries Loader

## Overview

This notebook demonstrates the `TimeSeries` loader for hierarchical time series data. Learn how to:

- Load PiedPiper billing data in 3 lines
- Filter, sample, and aggregate entities
- Visualize time series with publication-quality plots
- Compute summary statistics across entities

**Target audience**: Data scientists working with hierarchical time series (billing, metrics, IoT)

**Prerequisites**: PiedPiper dataset (or substitute your own hierarchical time series data)

---

## Setup

```{code-cell} ipython3
# Standard imports
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from loguru import logger

# PySpark and hellocloud
from pyspark.sql import functions as F
import hellocloud as hc

# Set seaborn theme for publication-quality plots
sns.set_theme(style="whitegrid")

# Get Spark session
spark = hc.spark.get_spark_session(app_name="quickstart-timeseries")
```

## 1. Loading Data

The `PiedPiperLoader` applies EDA-informed defaults to clean and structure billing data:

- **Column renames**: `usage_date` → `date`, `materialized_cost` → `cost`
- **Drops low-info columns**: UUIDs, redundant cost variants (4 removed)
- **Default hierarchy**: `provider → account → region → product → usage_type`

```{code-cell} ipython3
from hellocloud.io import PiedPiperLoader
from hellocloud.timeseries import TimeSeries

# Load raw data
data_path = Path("data/piedpiper_clean")  # Adjust to your data location
raw_df = spark.read.parquet(str(data_path))

print(f"Raw data: {raw_df.count():,} records, {len(raw_df.columns)} columns")
```

```{code-cell} ipython3
# Load into TimeSeries with defaults
ts = PiedPiperLoader.load(raw_df)

print(f"TimeSeries created:")
print(f"  Hierarchy: {ts.hierarchy}")
print(f"  Metric: {ts.metric_col}")
print(f"  Time: {ts.time_col}")
print(f"  Records: {ts.df.count():,}")
```

**Key insight**: The `TimeSeries` object wraps the full dataset - no entity splitting. Operations create filtered views on-demand.

---

## 2. Basic Operations

### 2.1 Filter to Specific Entity

```{code-cell} ipython3
# Filter to specific account + region
ts_filtered = ts.filter(cloud_account_id="123", region="us-east-1")

print(f"Filtered to 1 entity:")
print(f"  Records: {ts_filtered.df.count():,}")
print(f"  Date range: {ts_filtered.df.agg(F.min('date'), F.max('date')).collect()[0]}")
```

### 2.2 Sample Random Entities

```{code-cell} ipython3
# Sample 10 random account+region combinations
ts_sample = ts.sample(grain=["cloud_account_id", "region"], n=10)

print(f"Sampled 10 entities:")
print(f"  Unique entities: {ts_sample.df.select('cloud_account_id', 'region').distinct().count()}")
print(f"  Total records: {ts_sample.df.count():,}")
```

### 2.3 Aggregate to Coarser Grain

```{code-cell} ipython3
# Roll up from account+region+product to just account level
ts_account = ts.aggregate(grain=["cloud_account_id"])

print(f"Aggregated to account level:")
print(f"  Unique accounts: {ts_account.df.select('cloud_account_id').distinct().count()}")
print(f"  Hierarchy preserved: {ts_account.hierarchy}")
print(f"  Columns removed: region, product_family, usage_type")
```

### 2.4 Summary Statistics

```{code-cell} ipython3
# Compute stats across all accounts
stats = ts_account.summary_stats()

# Show top 5 accounts by mean cost
stats_pd = stats.toPandas().sort_values('mean', ascending=False).head()
print("\nTop 5 accounts by average daily cost:")
print(stats_pd[['cloud_account_id', 'count', 'mean', 'std', 'min', 'max']])
```

---

## 3. Visualization

### 3.1 Single Entity Time Series

```{code-cell} ipython3
# Get one entity for visualization
ts_single = ts.sample(grain=["cloud_account_id", "region"], n=1)

# Convert to pandas for plotting
pdf = ts_single.df.toPandas()
pdf['date'] = pd.to_datetime(pdf['date'])

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(pdf['date'], pdf['cost'], linewidth=2, color='steelblue')

# Automatic date formatting (adapts to date range)
locator = ax.xaxis.get_major_locator()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylabel('Cost ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Daily Cost - Single Entity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Date formatting**: `ConciseDateFormatter` automatically adapts labels based on date range:
- **< 1 week**: Shows days (Mon, Tue, ...)
- **1-4 weeks**: Shows days + month
- **1-12 months**: Shows months
- **> 1 year**: Shows years

### 3.2 Multiple Entities Comparison

```{code-cell} ipython3
# Sample 5 accounts for comparison
ts_multi = ts.aggregate(grain=["cloud_account_id"]).sample(grain=["cloud_account_id"], n=5)

# Convert to pandas
pdf_multi = ts_multi.df.toPandas()
pdf_multi['date'] = pd.to_datetime(pdf_multi['date'])

# Create plot with one line per account
fig, ax = plt.subplots(figsize=(12, 6))

for account_id in pdf_multi['cloud_account_id'].unique():
    account_data = pdf_multi[pdf_multi['cloud_account_id'] == account_id]
    ax.plot(account_data['date'], account_data['cost'],
            label=f"Account {account_id}", linewidth=2, alpha=0.7)

# Date formatting
locator = ax.xaxis.get_major_locator()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylabel('Cost ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Daily Cost - Multiple Accounts', fontsize=14, fontweight='bold')
ax.legend(title='Account', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 3.3 Stacked Area Plot (Hierarchical View)

```{code-cell} ipython3
# Aggregate to top 5 accounts, pivot for stacking
ts_top5 = ts.aggregate(grain=["cloud_account_id"]).sample(grain=["cloud_account_id"], n=5)
pdf_top5 = ts_top5.df.toPandas()
pdf_top5['date'] = pd.to_datetime(pdf_top5['date'])

# Pivot to wide format for stacking
pivot = pdf_top5.pivot(index='date', columns='cloud_account_id', values='cost').fillna(0)

# Create stacked area plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(pivot.index, *[pivot[col] for col in pivot.columns],
             labels=pivot.columns, alpha=0.7)

# Date formatting
locator = ax.xaxis.get_major_locator()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylabel('Cost ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Stacked Daily Cost - Top 5 Accounts', fontsize=14, fontweight='bold')
ax.legend(title='Account', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. Advanced: Custom Plotting with Pass-Through

The plotting examples above use standard matplotlib. You can customize further by:

```{code-cell} ipython3
# Example: Custom styling with matplotlib kwargs
fig, ax = plt.subplots(figsize=(12, 6))
pdf_single = ts_single.df.toPandas()
pdf_single['date'] = pd.to_datetime(pdf_single['date'])

# Plot with custom styling
ax.plot(pdf_single['date'], pdf_single['cost'],
        color='darkred',           # Custom color
        linewidth=3,               # Thicker line
        linestyle='--',            # Dashed line
        marker='o',                # Add markers
        markersize=4,              # Marker size
        alpha=0.8,                 # Transparency
        label='Daily Cost')

# Add reference lines
mean_cost = pdf_single['cost'].mean()
ax.axhline(y=mean_cost, color='gray', linestyle=':',
           label=f'Mean: ${mean_cost:.2f}', alpha=0.7)

# Date formatting
locator = ax.xaxis.get_major_locator()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylabel('Cost ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Custom Styled Plot', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Key pattern**: Create the `ax` object, customize as needed, return `ax` for further manipulation.

---

## 5. Next Steps

### Deeper Analysis
- **Notebook 05**: Full EDA with grain discovery, entity persistence analysis
- **Hierarchical forecasting**: Use aggregate/filter to build multi-level models
- **Anomaly detection**: Compute z-scores with `summary_stats()`, flag outliers

### TimeSeries API
- **More operations**: See `hellocloud.timeseries.TimeSeries` for complete API
- **Transformations**: Use `hellocloud.transforms` for percent change, normalization
- **Custom grains**: Mix and match hierarchy levels for your analysis needs

### Data Sources
- **Extend PiedPiperLoader**: Add custom column mappings, filters
- **New loaders**: Create loaders for other datasets following the same pattern
- **Real-time data**: Integrate with streaming PySpark DataFrames

---

## Summary

**What we learned:**
- ✅ Load hierarchical time series data with `PiedPiperLoader`
- ✅ Filter, sample, and aggregate using `TimeSeries` methods
- ✅ Compute summary statistics across entities
- ✅ Create publication-quality plots with automatic date formatting
- ✅ Customize plots with matplotlib pass-through

**Key insight**: The `TimeSeries` class keeps the full dataset in memory once. Operations like `filter()`, `sample()`, and `aggregate()` return new instances with filtered/aggregated DataFrames—leveraging PySpark's distributed engine while providing a domain-specific API.

**Architecture**: `TimeSeries` → PySpark DataFrame → Distributed processing
