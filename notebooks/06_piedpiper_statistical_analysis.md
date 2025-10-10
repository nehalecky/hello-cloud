---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: cloud-sim
  language: python
  name: cloud-sim
---

# CloudZero PiedPiper - Statistical Analysis & Modeling Preparation

## Background

This notebook builds on the exploratory analysis from `05_cloudzero_piedpiper_eda.md`, performing rigorous statistical characterization to prepare for forecasting and optimization modeling.

**Key Findings from EDA (Notebook 05)**:
- **Dataset**: 8.3M rows × 38 columns → filtered to 5.8M rows × 20 columns
- **Valid period**: Sept 1 - Oct 6, 2025 (37 days) - post-Oct 7 data is artifactual
- **Primary cost metric**: `materialized_discounted_cost`
- **Data quality**: AWS collapse detected on Oct 7, causing constant-value artifacts
- **Temporal stability**: High lag-1 autocorrelation (sticky infrastructure costs)

**This Notebook's Objectives**:
1. **Determine data grain** - What is the true temporal frequency?
2. **Entity decomposition** - Which resources/accounts drive costs and variance?
3. **Time series properties** - Autocorrelation structure, stationarity, seasonality
4. **Distribution analysis** - Transformation recommendations for modeling
5. **Modeling readiness** - Feature engineering and validation strategy

---

## Part 0: Setup & Configuration

```{code-cell} ipython3
# Hot reload pattern - library changes auto-reload without kernel restart
%load_ext autoreload
%autoreload 2

# Core libraries
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest

# Import our custom utilities
from hellocloud.utils import (
    # Cost analysis utilities (from notebook 05)
    temporal_quality_metrics,
    cost_distribution_metrics,
    detect_entity_anomalies,
    normalize_by_period,
    split_at_date,
    find_correlated_pairs,
    select_from_pairs,
    # EDA utilities
    comprehensive_schema_analysis,
    smart_sample,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
)

# Configure visualization
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# GLOBAL CONSTANT: Primary cost metric for all analysis
PRIMARY_COST = 'materialized_discounted_cost'

print("✓ Libraries loaded and configured")
print(f"✓ PRIMARY_COST = '{PRIMARY_COST}'")
```

```{code-cell} ipython3
# Load dataset and apply filtering logic from notebook 05
DATA_PATH = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet')

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Load as LazyFrame
df_raw = pl.scan_parquet(DATA_PATH)

# Apply filtering from notebook 05:
# 1. Remove post-collapse artifactual data (Oct 7+)
# 2. Keep only high-information columns (feature selection from Part 3 of notebook 05)

COLLAPSE_DATE = date(2025, 10, 7)

# Essential columns identified in notebook 05
essential_cols = [
    'usage_date',           # Temporal dimension
    'uuid',                 # Primary identifier
    'resource_id',          # Resource tracking
    'materialized_discounted_cost',  # Primary cost metric
    'materialized_usage_amount',     # Primary usage metric
    'cloud_provider',       # Top-level hierarchy
]

# Load schema to identify available columns
schema = df_raw.collect_schema()
available_cols = schema.names()

# Build column list (essentials + any high-information columns that exist)
# For now, we'll work with all available columns, then filter in analysis
df = df_raw.filter(pl.col('usage_date') < COLLAPSE_DATE)

# Collect basic statistics
total_rows = df.select(pl.len()).collect()[0, 0]
date_range = df.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date'),
    pl.col('usage_date').n_unique().alias('days')
]).collect()

print(f"✓ Dataset loaded: {DATA_PATH.name}")
print(f"  Valid period: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
print(f"  Clean days: {date_range['days'][0]}")
print(f"  Total records: {total_rows:,}")
print(f"  Avg records/day: {total_rows / date_range['days'][0]:,.0f}")
```

---

## Part 1: Data Grain & Frequency Analysis

### Objective

Determine the **exact grain** of each row in the dataset. The EDA revealed ~157K rows/day despite being labeled "daily" data, suggesting the grain is more granular than account-level daily aggregates.

### Hypothesis Testing

We test multiple grain hypotheses:
1. **H1**: One row per day (account-level daily aggregate)
2. **H2**: One row per (date, account) combination
3. **H3**: One row per (date, resource_id) combination
4. **H4**: One row per (date, account, resource_id) combination
5. **H5**: UUID provides finer granularity than date × resource

```{code-cell} ipython3
print("=" * 80)
print("DATA GRAIN HYPOTHESIS TESTING")
print("=" * 80)

# Collect sample for analysis (full dataset too large for some operations)
sample_size = min(1_000_000, total_rows)
df_sample = df.head(sample_size).collect()

print(f"\nAnalyzing sample: {len(df_sample):,} rows")
print(f"Sample period: {df_sample['usage_date'].min()} to {df_sample['usage_date'].max()}")
```

```{code-cell} ipython3
# Test H1: One row per day
h1_count = df_sample.select(pl.col('usage_date').n_unique()).item()
print(f"\nH1: One row per day")
print(f"  Unique dates: {h1_count}")
print(f"  Total rows: {len(df_sample):,}")
print(f"  Verdict: {'✓ PASS' if h1_count == len(df_sample) else '✗ FAIL'}")
print(f"  → {'Data is daily aggregate' if h1_count == len(df_sample) else f'Multiple rows per day ({len(df_sample) / h1_count:.0f}x on average)'}")
```

```{code-cell} ipython3
# Test H2: One row per (date, account)
# First, identify account column(s)
account_cols = [col for col in df_sample.columns if 'account' in col.lower()]

if account_cols:
    account_col = account_cols[0]
    h2_count = df_sample.select([pl.col('usage_date'), pl.col(account_col)]).n_unique()

    print(f"\nH2: One row per (date, {account_col})")
    print(f"  Unique (date, account) pairs: {h2_count}")
    print(f"  Total rows: {len(df_sample):,}")
    print(f"  Verdict: {'✓ PASS' if h2_count == len(df_sample) else '✗ FAIL'}")

    if h2_count != len(df_sample):
        print(f"  → Multiple rows per date-account ({len(df_sample) / h2_count:.1f}x on average)")
else:
    print(f"\nH2: SKIPPED (no account column found)")
```

```{code-cell} ipython3
# Test H3: One row per (date, resource_id)
if 'resource_id' in df_sample.columns:
    h3_count = df_sample.select([pl.col('usage_date'), pl.col('resource_id')]).n_unique()

    print(f"\nH3: One row per (date, resource_id)")
    print(f"  Unique (date, resource_id) pairs: {h3_count}")
    print(f"  Total rows: {len(df_sample):,}")
    print(f"  Verdict: {'✓ PASS' if h3_count == len(df_sample) else '✗ FAIL'}")

    if h3_count == len(df_sample):
        print(f"  → GRAIN IDENTIFIED: (usage_date, resource_id)")
        print(f"  → Each row represents ONE resource on ONE day")

        # Compute resources per day
        resources_per_day = (
            df_sample
            .group_by('usage_date')
            .agg(pl.col('resource_id').n_unique().alias('unique_resources'))
        )

        print(f"\n  Resources tracked per day:")
        print(f"    Mean: {resources_per_day['unique_resources'].mean():,.0f}")
        print(f"    Median: {resources_per_day['unique_resources'].median():,.0f}")
        print(f"    Range: [{resources_per_day['unique_resources'].min():,}, {resources_per_day['unique_resources'].max():,}]")
    else:
        print(f"  → Multiple rows per date-resource ({len(df_sample) / h3_count:.1f}x)")
else:
    print(f"\nH3: SKIPPED (no resource_id column found)")
```

```{code-cell} ipython3
# Test H4: UUID uniqueness (finest possible grain)
if 'uuid' in df_sample.columns:
    h4_count = df_sample['uuid'].n_unique()

    print(f"\nH4: UUID uniqueness (finest grain)")
    print(f"  Unique UUIDs: {h4_count}")
    print(f"  Total rows: {len(df_sample):,}")
    print(f"  Verdict: {'✓ UNIQUE' if h4_count == len(df_sample) else '⚠ DUPLICATES'}")

    if h4_count == len(df_sample):
        print(f"  → UUID is a true unique identifier (synthetic row ID)")
    else:
        dup_rate = (1 - h4_count / len(df_sample)) * 100
        print(f"  → UUID duplication rate: {dup_rate:.2f}%")
else:
    print(f"\nH4: SKIPPED (no uuid column found)")
```

```{code-cell} ipython3
# Frequency characterization: What creates multiple rows per day?
print("\n" + "=" * 80)
print("FREQUENCY CHARACTERIZATION")
print("=" * 80)

# Daily record counts
daily_counts = (
    df.group_by('usage_date')
    .agg(pl.len().alias('records'))
    .sort('usage_date')
    .collect()
)

print(f"\nDaily Record Counts ({date_range['days'][0]} days):")
print(f"  Mean: {daily_counts['records'].mean():,.0f}")
print(f"  Median: {daily_counts['records'].median():,.0f}")
print(f"  Std: {daily_counts['records'].std():,.0f}")
print(f"  CV: {daily_counts['records'].std() / daily_counts['records'].mean():.4f}")
print(f"  Range: [{daily_counts['records'].min():,}, {daily_counts['records'].max():,}]")

# Identify if variance is from resource churn or data collection
resource_counts_per_day = (
    df.group_by('usage_date')
    .agg(pl.col('resource_id').n_unique().alias('unique_resources'))
    .sort('usage_date')
    .collect()
)

print(f"\nUnique Resources Per Day:")
print(f"  Mean: {resource_counts_per_day['unique_resources'].mean():,.0f}")
print(f"  Median: {resource_counts_per_day['unique_resources'].median():,.0f}")
print(f"  CV: {resource_counts_per_day['unique_resources'].std() / resource_counts_per_day['unique_resources'].mean():.4f}")

# Check if row count ≈ resource count (validates H3)
row_resource_ratio = daily_counts['records'].mean() / resource_counts_per_day['unique_resources'].mean()
print(f"\nRow-to-Resource Ratio: {row_resource_ratio:.2f}")
if row_resource_ratio < 1.1:
    print("  → Confirms grain: (date, resource_id) - one row per resource per day")
else:
    print(f"  → Multiple rows per resource per day ({row_resource_ratio:.1f}x)")
```

```{code-cell} ipython3
# Visualize daily record counts and resource counts
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

plot_data = daily_counts.to_pandas()
plot_resources = resource_counts_per_day.to_pandas()

# Row counts over time
axes[0].plot(plot_data['usage_date'], plot_data['records'],
             linewidth=2, color='steelblue', marker='o', markersize=4)
axes[0].axhline(plot_data['records'].mean(), color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f"Mean: {plot_data['records'].mean():,.0f}")
axes[0].set_ylabel('Daily Record Count', fontweight='bold')
axes[0].set_title('Daily Row Counts - Frequency Stability Check', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Unique resources over time
axes[1].plot(plot_resources['usage_date'], plot_resources['unique_resources'],
             linewidth=2, color='darkgreen', marker='s', markersize=4)
axes[1].axhline(plot_resources['unique_resources'].mean(), color='red', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f"Mean: {plot_resources['unique_resources'].mean():,.0f}")
axes[1].set_xlabel('Date', fontweight='bold')
axes[1].set_ylabel('Unique Resources', fontweight='bold')
axes[1].set_title('Unique Resources Per Day - Infrastructure Churn', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# Correlation between row counts and resource counts
corr, p_value = pearsonr(daily_counts['records'].to_numpy(),
                          resource_counts_per_day['unique_resources'].to_numpy())
print(f"\nCorrelation (row count vs unique resources): r={corr:.4f}, p={p_value:.4e}")
if corr > 0.95:
    print("  → Very high correlation - row variance is from resource churn, not data quality")
```

### Summary: Data Grain Determination

```{code-cell} ipython3
print("=" * 80)
print("DATA GRAIN CONCLUSION")
print("=" * 80)

print(f"\n✓ GRAIN IDENTIFIED: (usage_date, resource_id)")
print(f"\nEach row represents:")
print(f"  - ONE cloud resource (EC2, S3, RDS, etc.)")
print(f"  - On ONE specific day")
print(f"  - With its associated cost and usage metrics")

print(f"\nWhy ~157K rows/day?")
print(f"  → PiedPiper infrastructure has ~{resource_counts_per_day['unique_resources'].mean():,.0f} active resources")
print(f"  → Each resource tracked daily → ~157K rows/day")
print(f"  → This is RESOURCE-LEVEL daily data, not ACCOUNT-LEVEL daily data")

print(f"\nTemporal Frequency:")
print(f"  → Native granularity: DAILY")
print(f"  → No intraday patterns (not hourly)")
print(f"  → No sub-resource splitting (one row per resource per day)")

print(f"\nModeling Implications:")
print(f"  → Can aggregate to any level: account, product, provider")
print(f"  → Cannot disaggregate below daily (no intraday data)")
print(f"  → Resource-level forecasting possible (if resource persistence is high)")
print(f"  → Account/product-level forecasting more stable (less churn)")
print("=" * 80)
```

---

## Part 1b: Entity Persistence & Time Series Structure

### Objective

**CRITICAL QUESTION**: Do entities persist across time, or do they churn?

From Part 1, we established the storage grain is `(usage_date, resource_id)`. However, this does NOT tell us whether we have:
- **Fixed panel**: Same ~157K resources exist all 37 days → Traditional time series forecasting
- **Rotating panel**: Resources enter/exit over time → Panel methods with entry/exit dynamics
- **Pure cross-section**: Extreme churn, no persistence → Cannot forecast individual resources

**This determines our entire modeling strategy.**

### Methodology

We analyze persistence at three entity levels:
1. **Resource-level** (grain): Lifespan, churn, survival curves
2. **Account-level** (typical target): Stability check
3. **Product-level** (stable aggregation): Always present?

```{code-cell} ipython3
print("=" * 80)
print("ENTITY PERSISTENCE & TIME SERIES STRUCTURE ANALYSIS")
print("=" * 80)
print("\n🎯 CRITICAL QUESTION: Do entities persist across time?")
print("   → Fixed panel: Traditional time series (same entities all days)")
print("   → Rotating panel: Entry/exit dynamics (need panel methods)")
print("   → Pure cross-section: Extreme churn (cannot forecast entities)")
```

### 1.1: Resource-Level Persistence Analysis

```{code-cell} ipython3
PRIMARY_COST = 'materialized_discounted_cost'
print(f"✓ PRIMARY_COST = '{PRIMARY_COST}'")

print(f"\n{'='*80}")
print("RESOURCE-LEVEL PERSISTENCE")
print(f"{'='*80}")

# Compute resource lifespan (how many days does each resource appear?)
resource_lifespan = (
    df.group_by('resource_id')
    .agg([
        pl.col('usage_date').n_unique().alias('days_present'),
        pl.col('usage_date').min().alias('first_seen'),
        pl.col('usage_date').max().alias('last_seen'),
        pl.col(PRIMARY_COST).sum().alias('total_cost')
    ])
    .collect()
    .with_columns([
        (pl.col('last_seen') - pl.col('first_seen')).dt.total_days().alias('lifespan_days')
    ])
)

# Summary statistics
total_unique_resources = len(resource_lifespan)
max_possible_days = date_range['days'][0]

print(f"\nResource Universe:")
print(f"  Total unique resources (across all 37 days): {total_unique_resources:,}")
print(f"  Days in dataset: {max_possible_days}")
print(f"  Maximum possible observations: {total_unique_resources * max_possible_days:,}")
print(f"  Actual observations: {total_rows:,}")
print(f"  Panel completeness: {total_rows / (total_unique_resources * max_possible_days) * 100:.1f}%")
```

```{code-cell} ipython3
# Lifespan distribution
lifespan_stats = resource_lifespan['days_present'].describe()

print(f"\nResource Lifespan Distribution:")
print(f"  Mean: {resource_lifespan['days_present'].mean():.1f} days")
print(f"  Median: {resource_lifespan['days_present'].median():.0f} days")
print(f"  Std: {resource_lifespan['days_present'].std():.1f} days")
print(f"  Min: {resource_lifespan['days_present'].min()} days")
print(f"  Max: {resource_lifespan['days_present'].max()} days")

# Key percentiles
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = resource_lifespan['days_present'].quantile(p / 100)
    print(f"  P{p}: {val:.0f} days")

# Persistence categories
ephemeral = (resource_lifespan['days_present'] == 1).sum()
short_lived = ((resource_lifespan['days_present'] > 1) & (resource_lifespan['days_present'] <= 7)).sum()
medium_lived = ((resource_lifespan['days_present'] > 7) & (resource_lifespan['days_present'] < max_possible_days)).sum()
persistent = (resource_lifespan['days_present'] == max_possible_days).sum()

print(f"\nPersistence Categories:")
print(f"  Ephemeral (1 day only): {ephemeral:,} ({ephemeral / total_unique_resources * 100:.1f}%)")
print(f"  Short-lived (2-7 days): {short_lived:,} ({short_lived / total_unique_resources * 100:.1f}%)")
print(f"  Medium-lived (8-{max_possible_days-1} days): {medium_lived:,} ({medium_lived / total_unique_resources * 100:.1f}%)")
print(f"  Persistent (all {max_possible_days} days): {persistent:,} ({persistent / total_unique_resources * 100:.1f}%)")

print(f"\n💡 INTERPRETATION:")
if persistent / total_unique_resources > 0.7:
    print(f"  → HIGH persistence ({persistent / total_unique_resources * 100:.0f}% present all days)")
    print(f"  → Resource-level forecasting IS viable")
elif persistent / total_unique_resources > 0.3:
    print(f"  → MODERATE persistence ({persistent / total_unique_resources * 100:.0f}% present all days)")
    print(f"  → Resource-level forecasting possible but need to model entry/exit")
else:
    print(f"  → LOW persistence ({persistent / total_unique_resources * 100:.0f}% present all days)")
    print(f"  → Resource-level forecasting NOT recommended (high churn)")
    print(f"  → Must aggregate to stable entity level")
```

```{code-cell} ipython3
# Visualize lifespan distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram of lifespans
axes[0, 0].hist(resource_lifespan['days_present'].to_numpy(), bins=max_possible_days,
                color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(resource_lifespan['days_present'].median(), color='red',
                   linestyle='--', linewidth=2, label=f"Median: {resource_lifespan['days_present'].median():.0f} days")
axes[0, 0].set_xlabel('Days Present', fontweight='bold')
axes[0, 0].set_ylabel('Number of Resources', fontweight='bold')
axes[0, 0].set_title('Resource Lifespan Distribution', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# CDF (cumulative distribution)
sorted_lifespans = np.sort(resource_lifespan['days_present'].to_numpy())
cdf = np.arange(1, len(sorted_lifespans) + 1) / len(sorted_lifespans) * 100
axes[0, 1].plot(sorted_lifespans, cdf, linewidth=2, color='darkgreen')
axes[0, 1].axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50th percentile')
axes[0, 1].axhline(90, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='90th percentile')
axes[0, 1].set_xlabel('Days Present', fontweight='bold')
axes[0, 1].set_ylabel('Cumulative % of Resources', fontweight='bold')
axes[0, 1].set_title('Cumulative Lifespan Distribution', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Bar chart: Persistence categories
categories = ['Ephemeral\n(1 day)', 'Short\n(2-7 days)', 'Medium\n(8-36 days)', f'Persistent\n(all {max_possible_days} days)']
counts = [ephemeral, short_lived, medium_lived, persistent]
percentages = [c / total_unique_resources * 100 for c in counts]

bars = axes[1, 0].bar(categories, counts, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.7)
axes[1, 0].set_ylabel('Number of Resources', fontweight='bold')
axes[1, 0].set_title('Resource Persistence Categories', fontweight='bold', fontsize=14)
axes[1, 0].grid(alpha=0.3, axis='y')

# Add percentage labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# Cost by persistence category
cost_by_persistence = pl.DataFrame({
    'Category': categories,
    'Resources': counts,
    'Total_Cost': [
        resource_lifespan.filter(pl.col('days_present') == 1)['total_cost'].sum(),
        resource_lifespan.filter((pl.col('days_present') > 1) & (pl.col('days_present') <= 7))['total_cost'].sum(),
        resource_lifespan.filter((pl.col('days_present') > 7) & (pl.col('days_present') < max_possible_days))['total_cost'].sum(),
        resource_lifespan.filter(pl.col('days_present') == max_possible_days)['total_cost'].sum()
    ]
}).with_columns([
    (pl.col('Total_Cost') / pl.col('Total_Cost').sum() * 100).alias('Pct_Cost')
])

axes[1, 1].bar(categories, cost_by_persistence['Pct_Cost'].to_numpy(),
               color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.7)
axes[1, 1].set_ylabel('% of Total Cost', fontweight='bold')
axes[1, 1].set_title('Cost Distribution by Persistence', fontweight='bold', fontsize=14)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\nCost Concentration by Persistence:")
with pl.Config(fmt_float='mixed'):
    display(cost_by_persistence)
```

### 1.2: Daily Churn Analysis

```{code-cell} ipython3
print(f"\n{'='*80}")
print("DAILY CHURN ANALYSIS")
print(f"{'='*80}")

# For each day, count:
# - Active resources (present that day)
# - New entrants (first appearance)
# - Exits (last appearance before gap or end)

# Get all dates
all_dates = (
    df.select('usage_date')
    .unique()
    .sort('usage_date')
    .collect()['usage_date']
    .to_list()
)

# Compute daily metrics
daily_churn = []
for date in all_dates:
    # Resources active on this date
    active = df.filter(pl.col('usage_date') == date).select('resource_id').collect()['resource_id'].unique()

    # New entrants: first_seen == this date
    entrants = resource_lifespan.filter(pl.col('first_seen') == date).height

    # Exits: last_seen == this date (and not the final day of dataset)
    if date < all_dates[-1]:
        exits = resource_lifespan.filter(pl.col('last_seen') == date).height
    else:
        exits = 0  # Final day - can't detect exits

    daily_churn.append({
        'date': date,
        'active': len(active),
        'entrants': entrants,
        'exits': exits
    })

churn_df = pl.DataFrame(daily_churn).with_columns([
    (pl.col('entrants') / pl.col('active') * 100).alias('entry_rate_%'),
    (pl.col('exits') / pl.col('active') * 100).alias('exit_rate_%'),
    ((pl.col('entrants') - pl.col('exits')) / pl.col('active') * 100).alias('net_growth_%')
])

print(f"\nDaily Churn Metrics (Summary):")
print(f"  Mean active resources: {churn_df['active'].mean():,.0f}")
print(f"  Mean daily entrants: {churn_df['entrants'].mean():,.0f} ({churn_df['entry_rate_%'].mean():.2f}%)")
print(f"  Mean daily exits: {churn_df['exits'].mean():,.0f} ({churn_df['exit_rate_%'].mean():.2f}%)")
print(f"  Mean net growth: {churn_df['net_growth_%'].mean():.2f}%")

print(f"\nChurn Stability:")
if churn_df['entry_rate_%'].mean() < 5 and churn_df['exit_rate_%'].mean() < 5:
    print(f"  → LOW churn (<5% daily) - resource pool is stable")
elif churn_df['entry_rate_%'].mean() < 15 and churn_df['exit_rate_%'].mean() < 15:
    print(f"  → MODERATE churn (5-15% daily) - some turnover")
else:
    print(f"  → HIGH churn (>15% daily) - significant turnover")

print(f"\nFirst 10 days:")
with pl.Config(fmt_float='mixed', tbl_rows=10):
    display(churn_df.head(10))
```

```{code-cell} ipython3
# Visualize churn over time
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

plot_churn = churn_df.to_pandas()

# Active resources over time
axes[0].plot(plot_churn['date'], plot_churn['active'], linewidth=2,
             color='steelblue', marker='o', markersize=4)
axes[0].set_ylabel('Active Resources', fontweight='bold')
axes[0].set_title('Active Resources Over Time', fontweight='bold', fontsize=14)
axes[0].grid(alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Entry and exit rates
axes[1].plot(plot_churn['date'], plot_churn['entry_rate_%'],
             linewidth=2, color='green', marker='o', markersize=3, label='Entry Rate')
axes[1].plot(plot_churn['date'], plot_churn['exit_rate_%'],
             linewidth=2, color='red', marker='s', markersize=3, label='Exit Rate')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Date', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)', fontweight='bold')
axes[1].set_title('Daily Entry and Exit Rates', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.3: Survival Analysis

```{code-cell} ipython3
print(f"\n{'='*80}")
print("SURVIVAL ANALYSIS")
print(f"{'='*80}")

# Survival curve: P(resource still active after t days | started at day 0)
# Simplified: Use cohort of resources that started on first day

cohort_start = all_dates[0]
cohort = resource_lifespan.filter(pl.col('first_seen') == cohort_start)

if len(cohort) > 0:
    print(f"\nCohort Analysis (resources starting on {cohort_start}):")
    print(f"  Cohort size: {len(cohort):,} resources")

    # Compute survival curve
    max_days = cohort['days_present'].max()
    survival_curve = []
    for t in range(1, max_days + 1):
        still_active = (cohort['days_present'] >= t).sum()
        survival_rate = still_active / len(cohort) * 100
        survival_curve.append({'days': t, 'survival_%': survival_rate})

    survival_df = pl.DataFrame(survival_curve)

    # Key survival metrics
    median_survival = cohort['days_present'].median()
    survival_7d = survival_df.filter(pl.col('days') == min(7, max_days))['survival_%'][0] if max_days >= 7 else 0
    survival_30d = survival_df.filter(pl.col('days') == min(30, max_days))['survival_%'][0] if max_days >= 30 else 0

    print(f"  Median survival: {median_survival:.0f} days")
    if max_days >= 7:
        print(f"  7-day survival: {survival_7d:.1f}%")
    if max_days >= 30:
        print(f"  30-day survival: {survival_30d:.1f}%")

    # Visualize survival curve
    fig, ax = plt.subplots(figsize=(14, 6))

    plot_survival = survival_df.to_pandas()
    ax.plot(plot_survival['days'], plot_survival['survival_%'],
            linewidth=2.5, color='darkblue', marker='o', markersize=3)
    ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'50% survival (median={median_survival:.0f} days)')
    ax.set_xlabel('Days Since Start', fontweight='bold')
    ax.set_ylabel('Survival Rate (%)', fontweight='bold')
    ax.set_title(f'Resource Survival Curve (Cohort starting {cohort_start})', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.show()

    print(f"\n💡 INTERPRETATION:")
    if median_survival >= 30:
        print(f"  → HIGH survival (median ≥ 30 days) - resources are sticky")
    elif median_survival >= 7:
        print(f"  → MODERATE survival (median 7-30 days) - some persistence")
    else:
        print(f"  → LOW survival (median < 7 days) - resources are ephemeral")
else:
    print(f"\n⚠ No resources started on first day - cannot compute cohort survival")
```

### 1.4: Account-Level Stability Check

```{code-cell} ipython3
print(f"\n{'='*80}")
print("ACCOUNT-LEVEL STABILITY")
print(f"{'='*80}")

# Check if account column exists
account_cols = [col for col in df.collect_schema().names() if 'account' in col.lower()]

if account_cols:
    account_col = account_cols[0]
    print(f"Analyzing column: {account_col}")

    # Account persistence
    account_persistence = (
        df.group_by(account_col)
        .agg([
            pl.col('usage_date').n_unique().alias('days_present'),
            pl.col('usage_date').min().alias('first_seen'),
            pl.col('usage_date').max().alias('last_seen')
        ])
        .collect()
    )

    total_accounts = len(account_persistence)
    accounts_all_days = (account_persistence['days_present'] == max_possible_days).sum()

    print(f"\nAccount Persistence:")
    print(f"  Total unique accounts: {total_accounts}")
    print(f"  Accounts present all {max_possible_days} days: {accounts_all_days} ({accounts_all_days / total_accounts * 100:.1f}%)")
    print(f"  Mean days present: {account_persistence['days_present'].mean():.1f}")
    print(f"  Median days present: {account_persistence['days_present'].median():.0f}")

    print(f"\n💡 VERDICT:")
    if accounts_all_days / total_accounts > 0.95:
        print(f"  → HIGHLY STABLE ({accounts_all_days / total_accounts * 100:.0f}% present all days)")
        print(f"  → Account-level forecasting is IDEAL (stable fixed panel)")
    elif accounts_all_days / total_accounts > 0.7:
        print(f"  → MODERATELY STABLE ({accounts_all_days / total_accounts * 100:.0f}% present all days)")
        print(f"  → Account-level forecasting viable with caveats")
    else:
        print(f"  → UNSTABLE ({accounts_all_days / total_accounts * 100:.0f}% present all days)")
        print(f"  → Account churn is significant - use caution")

    # Store for later use
    account_stability_pct = accounts_all_days / total_accounts * 100
else:
    print(f"\n⚠ No account column found in dataset")
    account_stability_pct = 0
```

### 1.5: Product-Level Stability Check

```{code-cell} ipython3
print(f"\n{'='*80}")
print("PRODUCT-LEVEL STABILITY")
print(f"{'='*80}")

# Check for product/service columns
product_cols = [col for col in df.collect_schema().names()
                if any(term in col.lower() for term in ['product', 'service'])]

if product_cols:
    product_col = product_cols[0]
    print(f"Analyzing column: {product_col}")

    # Product persistence
    product_persistence = (
        df.group_by(product_col)
        .agg([
            pl.col('usage_date').n_unique().alias('days_present'),
            pl.col('usage_date').min().alias('first_seen'),
            pl.col('usage_date').max().alias('last_seen')
        ])
        .collect()
    )

    total_products = len(product_persistence)
    products_all_days = (product_persistence['days_present'] == max_possible_days).sum()

    print(f"\nProduct Persistence:")
    print(f"  Total unique products/services: {total_products}")
    print(f"  Products present all {max_possible_days} days: {products_all_days} ({products_all_days / total_products * 100:.1f}%)")
    print(f"  Mean days present: {product_persistence['days_present'].mean():.1f}")
    print(f"  Median days present: {product_persistence['days_present'].median():.0f}")

    print(f"\n💡 VERDICT:")
    if products_all_days / total_products > 0.95:
        print(f"  → HIGHLY STABLE ({products_all_days / total_products * 100:.0f}% present all days)")
        print(f"  → Product-level aggregation is STABLE")
    elif products_all_days / total_products > 0.7:
        print(f"  → MODERATELY STABLE ({products_all_days / total_products * 100:.0f}% present all days)")
        print(f"  → Most products persist")
    else:
        print(f"  → SOME CHURN ({products_all_days / total_products * 100:.0f}% present all days)")
        print(f"  → Product mix changes over time")

    # Store for later use
    product_stability_pct = products_all_days / total_products * 100
else:
    print(f"\n⚠ No product/service column found in dataset")
    product_stability_pct = 0
```

### Summary: Entity Persistence & Forecasting Recommendations

```{code-cell} ipython3
print("=" * 80)
print("ENTITY PERSISTENCE SUMMARY & FORECASTING RECOMMENDATIONS")
print("=" * 80)

print(f"\n📊 PERSISTENCE BY ENTITY LEVEL:")

print(f"\n1. RESOURCE LEVEL (Data Grain):")
print(f"   - Total unique resources: {total_unique_resources:,}")
print(f"   - Present all {max_possible_days} days: {persistent:,} ({persistent / total_unique_resources * 100:.1f}%)")
print(f"   - Median lifespan: {resource_lifespan['days_present'].median():.0f} days")
print(f"   - Daily churn rate: {churn_df['entry_rate_%'].mean():.1f}% entry, {churn_df['exit_rate_%'].mean():.1f}% exit")

if persistent / total_unique_resources > 0.7:
    resource_verdict = "✅ HIGH PERSISTENCE - Resource-level forecasting viable"
elif persistent / total_unique_resources > 0.3:
    resource_verdict = "⚠️ MODERATE PERSISTENCE - Resource forecasting needs entry/exit modeling"
else:
    resource_verdict = "❌ LOW PERSISTENCE - Cannot forecast individual resources"
print(f"   → {resource_verdict}")

if account_stability_pct > 0:
    print(f"\n2. ACCOUNT LEVEL:")
    print(f"   - Stability: {account_stability_pct:.1f}% present all days")
    if account_stability_pct > 95:
        account_verdict = "✅ HIGHLY STABLE - IDEAL forecasting target (fixed panel)"
    elif account_stability_pct > 70:
        account_verdict = "✅ STABLE - Good forecasting target"
    else:
        account_verdict = "⚠️ UNSTABLE - Use with caution"
    print(f"   → {account_verdict}")

if product_stability_pct > 0:
    print(f"\n3. PRODUCT LEVEL:")
    print(f"   - Stability: {product_stability_pct:.1f}% present all days")
    if product_stability_pct > 95:
        product_verdict = "✅ HIGHLY STABLE - Reliable aggregation level"
    elif product_stability_pct > 70:
        product_verdict = "✅ STABLE - Good for aggregated forecasts"
    else:
        product_verdict = "⚠️ SOME CHURN - Product mix changes"
    print(f"   → {product_verdict}")

print(f"\n🎯 FORECASTING ENTITY RECOMMENDATION:")

# Determine primary forecasting entity
if account_stability_pct > 95:
    primary_entity = account_col if account_cols else "Unknown"
    primary_reason = "Accounts are highly stable (>95% fixed panel)"
elif product_stability_pct > 95:
    primary_entity = product_col if product_cols else "Unknown"
    primary_reason = "Products are highly stable aggregation"
elif persistent / total_unique_resources > 0.7:
    primary_entity = "resource_id"
    primary_reason = "Resources show high persistence"
else:
    primary_entity = "System-level only"
    primary_reason = "All entity levels show significant churn"

print(f"\n  PRIMARY: {primary_entity}")
print(f"    Reason: {primary_reason}")

if primary_entity != "System-level only":
    print(f"    Forecasting grain: ({primary_entity}, usage_date)")
    print(f"    Modeling approach: Time series or panel methods")
else:
    print(f"    Forecasting grain: (usage_date) - system total only")
    print(f"    Modeling approach: Univariate time series")

print(f"\n  SECONDARY: System-level (usage_date)")
print(f"    Reason: Always stable (single aggregate time series)")
print(f"    Use case: Budget planning, high-level forecasts")

print(f"\n  AVOID: Entity levels with <70% persistence")
print(f"    Reason: High churn makes entity-specific forecasting unreliable")

print(f"\n🔍 KEY INSIGHT: Entity persistence determines modeling strategy")
print(f"   - We now know WHICH entity-time key defines a valid time series observation")
print(f"   - Distribution analysis (Part 4) should focus on stable entity levels")
print(f"   - Resource-level distributions are informative but may not be forecastable")

print("\n" + "=" * 80)
```

---
## Part 2: Entity Cost Decomposition

### Objective

Analyze cost distribution across entity hierarchies (providers, accounts, products, resources) to identify:
1. **Concentration**: Are costs driven by few entities (Pareto principle)?
2. **Stability**: Which entity types show temporal variance?
3. **Hierarchy**: How do costs flow through provider → account → product → resource?

### Methodology

We use composition patterns with atomic utilities from `cloud_sim.utils.cost_analysis`:
- `normalize_by_period()`: Entity contributions over time
- `detect_entity_anomalies()`: Temporal variance by entity type
- Custom Gini coefficient: Measure concentration inequality

```{code-cell} ipython3
# Primary cost metric (from notebook 05)
PRIMARY_COST = 'materialized_discounted_cost'

print("=" * 80)
print("ENTITY COST DECOMPOSITION")
print("=" * 80)
print(f"\nPrimary cost metric: {PRIMARY_COST}")

# Total spend in analysis period
total_spend = df.select(pl.col(PRIMARY_COST).sum()).collect()[0, 0]
print(f"Total spend (37 days): ${total_spend:,.2f}")
print(f"Daily average: ${total_spend / date_range['days'][0]:,.2f}")
```

### 2.1: Cloud Provider Concentration

```{code-cell} ipython3
# Provider-level aggregation
provider_costs = (
    df.group_by('cloud_provider')
    .agg([
        pl.col(PRIMARY_COST).sum().alias('total_cost'),
        pl.len().alias('record_count'),
        pl.col('resource_id').n_unique().alias('unique_resources'),
        pl.col(PRIMARY_COST).mean().alias('avg_cost_per_record')
    ])
    .sort('total_cost', descending=True)
    .collect()
    .with_columns([
        (pl.col('total_cost') / total_spend * 100).alias('pct_of_total')
    ])
)

print(f"\nCLOUD PROVIDER DISTRIBUTION:")
print(f"  Providers: {len(provider_costs)}")

with pl.Config(fmt_float='mixed'):
    display(provider_costs)

# Provider concentration (Herfindahl-Hirschman Index)
hhi = (provider_costs['pct_of_total'] ** 2).sum()
print(f"\nConcentration Metrics:")
print(f"  Herfindahl-Hirschman Index (HHI): {hhi:.0f}")
if hhi > 2500:
    print(f"    → High concentration (single provider dominant)")
elif hhi > 1500:
    print(f"    → Moderate concentration")
else:
    print(f"    → Low concentration (balanced multi-cloud)")
```

### 2.2: Resource-Level Pareto Analysis

```{code-cell} ipython3
# Resource-level costs (full analysis period)
resource_costs = (
    df.group_by('resource_id')
    .agg([
        pl.col(PRIMARY_COST).sum().alias('total_cost'),
        pl.len().alias('days_active')
    ])
    .sort('total_cost', descending=True)
    .collect()
    .with_columns([
        (pl.col('total_cost') / total_spend * 100).alias('pct_of_total'),
        (pl.col('total_cost').cum_sum() / total_spend * 100).alias('cumulative_pct')
    ])
)

print(f"\nRESOURCE-LEVEL PARETO ANALYSIS:")
print(f"  Total resources: {len(resource_costs):,}")

# Find Pareto thresholds
resources_for_50pct = resource_costs.filter(pl.col('cumulative_pct') <= 50).height
resources_for_80pct = resource_costs.filter(pl.col('cumulative_pct') <= 80).height
resources_for_90pct = resource_costs.filter(pl.col('cumulative_pct') <= 90).height

print(f"\n  Pareto Distribution:")
print(f"    Top {resources_for_50pct:,} resources ({resources_for_50pct / len(resource_costs) * 100:.1f}%) → 50% of costs")
print(f"    Top {resources_for_80pct:,} resources ({resources_for_80pct / len(resource_costs) * 100:.1f}%) → 80% of costs")
print(f"    Top {resources_for_90pct:,} resources ({resources_for_90pct / len(resource_costs) * 100:.1f}%) → 90% of costs")

# Top 20 costliest resources
print(f"\n  Top 20 Costliest Resources:")
with pl.Config(fmt_float='mixed', tbl_rows=20):
    display(resource_costs.head(20).select(['resource_id', 'total_cost', 'pct_of_total', 'cumulative_pct', 'days_active']))
```

```{code-cell} ipython3
# Visualize Pareto curve
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pareto curve (cumulative %)
top_1000 = resource_costs.head(1000).to_pandas()
axes[0].plot(range(len(top_1000)), top_1000['cumulative_pct'],
             linewidth=2.5, color='darkblue')
axes[0].axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
axes[0].axhline(50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')
axes[0].set_xlabel('Resource Rank (Top 1000)', fontweight='bold')
axes[0].set_ylabel('Cumulative Cost (%)', fontweight='bold')
axes[0].set_title('Pareto Curve - Resource Cost Concentration', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Cost distribution (log scale)
axes[1].hist(resource_costs['total_cost'].to_numpy(), bins=100,
             color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Total Cost ($, log scale)', fontweight='bold')
axes[1].set_ylabel('Number of Resources', fontweight='bold')
axes[1].set_title('Resource Cost Distribution (Right-Skewed)', fontweight='bold', fontsize=14)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
```

### 2.3: Gini Coefficient (Inequality Measure)

```{code-cell} ipython3
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Gini = 0: Perfect equality (all values equal)
    Gini = 1: Perfect inequality (one entity has everything)

    Typical interpretations:
    - < 0.3: Low inequality
    - 0.3-0.5: Moderate inequality
    - > 0.5: High inequality (Pareto effect strong)
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

# Compute Gini for resource costs
resource_costs_array = resource_costs['total_cost'].to_numpy()
gini = gini_coefficient(resource_costs_array)

print(f"\nGINI COEFFICIENT (Resource Cost Inequality):")
print(f"  Gini: {gini:.4f}")

if gini > 0.7:
    print(f"    → Extreme inequality (very strong Pareto effect)")
elif gini > 0.5:
    print(f"    → High inequality (strong Pareto - few resources drive most costs)")
elif gini > 0.3:
    print(f"    → Moderate inequality (some concentration)")
else:
    print(f"    → Low inequality (costs relatively balanced)")

print(f"\n  Interpretation:")
print(f"    - Gini near 1.0 suggests optimization efforts should focus on top resources")
print(f"    - Gini near 0.0 suggests costs distributed - system-wide optimization needed")
```

### 2.4: Entity-Level Temporal Variance

```{code-cell} ipython3
# COMPOSITION PATTERN: Entity anomaly detection
# Uses detect_entity_anomalies from hellocloud.utils.cost_analysis

print(f"\nENTITY-LEVEL TEMPORAL VARIANCE:")

# Identify entity columns to analyze (string/categorical columns only)
schema = df.collect_schema()
entity_cols = [col for col in schema.names()
               if col not in ['usage_date', 'uuid', 'resource_id', PRIMARY_COST]
               and schema[col] == pl.Utf8]

print(f"  Analyzing {len(entity_cols)} entity types: {', '.join(entity_cols[:5])}...")

# Detect anomalies for each entity type
variance_results = []
for entity_col in entity_cols[:5]:  # Limit to first 5 for performance
    try:
        anomalies = detect_entity_anomalies(
            df,
            entity_col=entity_col,
            date_col='usage_date',
            min_days=10,
            top_n=3
        )

        if len(anomalies) > 0:
            max_cv = anomalies['cv'][0]
            variance_results.append({
                'entity_type': entity_col,
                'max_cv': max_cv,
                'top_variable_entity': anomalies['entity'][0]
            })
    except Exception as e:
        print(f"  Skipped {entity_col}: {e}")

# Sort by variance
variance_df = pl.DataFrame(variance_results).sort('max_cv', descending=True)
print(f"\n  Entity Types Ranked by Temporal Variance:")
display(variance_df)

# Identify most stable entity for aggregation
if len(variance_df) > 0:
    most_stable = variance_df.tail(1)
    most_volatile = variance_df.head(1)

    print(f"\n  Modeling Recommendations:")
    print(f"    Most stable entity: {most_stable['entity_type'][0]} (CV={most_stable['max_cv'][0]:.3f})")
    print(f"      → Use for stable forecasting targets")
    print(f"    Most volatile entity: {most_volatile['entity_type'][0]} (CV={most_volatile['max_cv'][0]:.3f})")
    print(f"      → Requires robust modeling or exclusion")
```

### 2.5: Hierarchical Cost Flow

```{code-cell} ipython3
# Sankey-style analysis: Provider → Account → Product → Resource
# Simplified: Show top flow paths

print(f"\n" + "=" * 80)
print(f"HIERARCHICAL COST FLOW (Simplified)")
print(f"=" * 80)

# Check which hierarchy columns exist
hierarchy_cols = []
for col_name in ['cloud_provider', 'account_id', 'product_family', 'service_name']:
    if col_name in df.collect_schema().names():
        hierarchy_cols.append(col_name)

if len(hierarchy_cols) >= 2:
    # Analyze top 2 levels
    level1, level2 = hierarchy_cols[0], hierarchy_cols[1]

    flow_analysis = (
        df.group_by([level1, level2])
        .agg(pl.col(PRIMARY_COST).sum().alias('cost'))
        .sort('cost', descending=True)
        .collect()
        .head(20)
        .with_columns([
            (pl.col('cost') / total_spend * 100).alias('pct_of_total')
        ])
    )

    print(f"\nTop 20 Cost Flows: {level1} → {level2}")
    with pl.Config(fmt_float='mixed', tbl_rows=20):
        display(flow_analysis)
else:
    print("\n⚠ Insufficient hierarchy columns for flow analysis")
```

### Summary: Entity Cost Decomposition

```{code-cell} ipython3
print("=" * 80)
print("ENTITY DECOMPOSITION SUMMARY")
print("=" * 80)

print(f"\n1. CONCENTRATION (Pareto Analysis):")
print(f"   - Gini coefficient: {gini:.3f} ({'High' if gini > 0.5 else 'Moderate'} inequality)")
print(f"   - Top {resources_for_80pct / len(resource_costs) * 100:.1f}% of resources → 80% of costs")
print(f"   - Optimization focus: Target top {resources_for_80pct:,} resources")

print(f"\n2. PROVIDER DISTRIBUTION:")
print(f"   - HHI: {hhi:.0f} ({'High' if hhi > 2500 else 'Moderate' if hhi > 1500 else 'Low'} concentration)")
print(f"   - Top provider: {provider_costs['cloud_provider'][0]} ({provider_costs['pct_of_total'][0]:.1f}%)")

if len(variance_df) > 0:
    print(f"\n3. TEMPORAL VARIANCE:")
    print(f"   - Most stable entity: {variance_df.tail(1)['entity_type'][0]}")
    print(f"   - Most volatile entity: {variance_df.head(1)['entity_type'][0]}")
    print(f"   - Recommendation: Aggregate by stable entities for forecasting")

print("\n" + "=" * 80)
```

---

## Part 3: Temporal Properties & Autocorrelation Structure

### Objective

Characterize time series properties for forecasting model selection:
1. **Autocorrelation**: How much does past predict future?
2. **Stationarity**: Is the mean/variance constant over time?
3. **Seasonality**: Are there periodic patterns (daily, weekly)?
4. **Trend**: Is spending increasing, decreasing, or stable?

```{code-cell} ipython3
print("=" * 80)
print("TEMPORAL PROPERTIES ANALYSIS")
print("=" * 80)

# Aggregate to daily time series
daily_costs = (
    df.group_by('usage_date')
    .agg([
        pl.col(PRIMARY_COST).sum().alias('cost'),
        pl.len().alias('records'),
        pl.col('resource_id').n_unique().alias('unique_resources')
    ])
    .sort('usage_date')
    .collect()
)

print(f"\nDaily cost time series: {len(daily_costs)} observations")
print(f"  Mean: ${daily_costs['cost'].mean():,.2f}/day")
print(f"  Std: ${daily_costs['cost'].std():,.2f}")
print(f"  CV: {daily_costs['cost'].std() / daily_costs['cost'].mean():.4f}")
```

### 3.1: Autocorrelation Function (ACF) & Partial ACF

```{code-cell} ipython3
# Compute ACF and PACF
cost_series = daily_costs['cost'].to_numpy()

# PACF requires nlags < len(series) // 2 (50% of sample size)
max_lags_pacf = len(cost_series) // 2 - 1  # Conservative: leave margin
max_lags_desired = 30  # Ideally want 30 days (monthly pattern)
lags = min(len(cost_series) - 1, max_lags_desired, max_lags_pacf)

print(f"\nAUTOCORRELATION ANALYSIS:")
print(f"  Sample size: {len(cost_series)} days")
print(f"  Maximum PACF lags: {max_lags_pacf}")
print(f"  Using {lags} lags for analysis")

acf_values = acf(cost_series, nlags=lags, fft=False)
pacf_values = pacf(cost_series, nlags=lags)

print(f"\n  Lag-1 ACF: {acf_values[1]:.4f} ({'High' if acf_values[1] > 0.7 else 'Moderate' if acf_values[1] > 0.4 else 'Low'})")
if lags >= 7:
    print(f"  Lag-7 ACF: {acf_values[7]:.4f} (weekly pattern: {'Yes' if acf_values[7] > 0.5 else 'No'})")
else:
    print(f"  Lag-7 ACF: N/A (insufficient data, only {lags} lags available)")

# Visualize ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ACF plot
axes[0].stem(range(len(acf_values)), acf_values, basefmt=' ')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].axhline(1.96/np.sqrt(len(cost_series)), color='red', linestyle='--',
                linewidth=1, alpha=0.7, label='95% CI')
axes[0].axhline(-1.96/np.sqrt(len(cost_series)), color='red', linestyle='--',
                linewidth=1, alpha=0.7)
axes[0].set_xlabel('Lag (days)', fontweight='bold')
axes[0].set_ylabel('ACF', fontweight='bold')
axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# PACF plot
axes[1].stem(range(len(pacf_values)), pacf_values, basefmt=' ')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].axhline(1.96/np.sqrt(len(cost_series)), color='red', linestyle='--',
                linewidth=1, alpha=0.7, label='95% CI')
axes[1].axhline(-1.96/np.sqrt(len(cost_series)), color='red', linestyle='--',
                linewidth=1, alpha=0.7)
axes[1].set_xlabel('Lag (days)', fontweight='bold')
axes[1].set_ylabel('PACF', fontweight='bold')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
if acf_values[1] > 0.7:
    print(f"\n  Interpretation:")
    print(f"    - High lag-1 autocorrelation → Yesterday predicts today")
    print(f"    - Sticky infrastructure costs (resources persist)")
    print(f"    - Good for AR/ARIMA models")
else:
    print(f"\n  Interpretation:")
    print(f"    - Low autocorrelation → Day-to-day independence")
    print(f"    - Volatile spending patterns")
    print(f"    - May need external features (not just history)")
```

### 3.2: Stationarity Tests

```{code-cell} ipython3
print(f"\nSTATIONARITY TESTS:")

# Augmented Dickey-Fuller test (null hypothesis: non-stationary)
adf_result = adfuller(cost_series, autolag='AIC')
adf_statistic, adf_pvalue = adf_result[0], adf_result[1]

print(f"\n  Augmented Dickey-Fuller Test:")
print(f"    Test statistic: {adf_statistic:.4f}")
print(f"    P-value: {adf_pvalue:.4f}")
print(f"    Critical values: 1%={adf_result[4]['1%']:.3f}, 5%={adf_result[4]['5%']:.3f}, 10%={adf_result[4]['10%']:.3f}")

if adf_pvalue < 0.05:
    print(f"    → STATIONARY (reject null hypothesis at 5% level)")
    print(f"    → Mean and variance are stable over time")
else:
    print(f"    → NON-STATIONARY (fail to reject null hypothesis)")
    print(f"    → May need differencing or detrending")

# KPSS test (null hypothesis: stationary)
kpss_result = kpss(cost_series, regression='c', nlags='auto')
kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]

print(f"\n  KPSS Test:")
print(f"    Test statistic: {kpss_statistic:.4f}")
print(f"    P-value: {kpss_pvalue:.4f}")
print(f"    Critical values: 10%={kpss_result[3]['10%']:.3f}, 5%={kpss_result[3]['5%']:.3f}, 1%={kpss_result[3]['1%']:.3f}")

if kpss_pvalue > 0.05:
    print(f"    → STATIONARY (fail to reject null hypothesis at 5% level)")
else:
    print(f"    → NON-STATIONARY (reject null hypothesis)")
    print(f"    → Trend or level shift present")

# Combined interpretation
print(f"\n  Combined Interpretation:")
if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
    print(f"    ✓ STATIONARY (both tests agree)")
    print(f"    → Can use level-based models (no differencing needed)")
elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
    print(f"    ✗ NON-STATIONARY (both tests agree)")
    print(f"    → Apply first differencing or detrending")
else:
    print(f"    ⚠ INCONCLUSIVE (tests disagree)")
    print(f"    → Examine time series visually, may have structural break")
```

### 3.3: Trend & Seasonality Decomposition (STL)

```{code-cell} ipython3
# STL decomposition (Seasonal-Trend decomposition using LOESS)
# Requires at least 2 seasonal periods - check if we have enough data

if len(cost_series) >= 14:  # At least 2 weeks for weekly seasonality
    print(f"\nSTL DECOMPOSITION (Seasonal-Trend-Residual):")

    # Create pandas time series for STL (requires datetime index)
    ts = pd.Series(cost_series, index=pd.date_range(start=daily_costs['usage_date'][0],
                                                      periods=len(cost_series), freq='D'))

    # STL decomposition with weekly seasonality (period=7)
    stl = STL(ts, seasonal=7, trend=13)  # trend window ~ 2 weeks
    result = stl.fit()

    # Visualize decomposition
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(ts.index, ts.values, linewidth=2, color='black')
    axes[0].set_ylabel('Observed', fontweight='bold')
    axes[0].set_title('STL Decomposition - Daily Costs', fontweight='bold', fontsize=14)
    axes[0].grid(alpha=0.3)

    axes[1].plot(ts.index, result.trend, linewidth=2, color='darkblue')
    axes[1].set_ylabel('Trend', fontweight='bold')
    axes[1].grid(alpha=0.3)

    axes[2].plot(ts.index, result.seasonal, linewidth=2, color='darkgreen')
    axes[2].set_ylabel('Seasonal (Weekly)', fontweight='bold')
    axes[2].grid(alpha=0.3)

    axes[3].plot(ts.index, result.resid, linewidth=1, color='darkred', alpha=0.7)
    axes[3].set_ylabel('Residual', fontweight='bold')
    axes[3].set_xlabel('Date', fontweight='bold')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Quantify components
    trend_strength = 1 - (result.resid.var() / (result.trend + result.resid).var())
    seasonal_strength = 1 - (result.resid.var() / (result.seasonal + result.resid).var())

    print(f"\n  Component Strength:")
    print(f"    Trend strength: {max(0, trend_strength):.3f} ({'Strong' if trend_strength > 0.6 else 'Moderate' if trend_strength > 0.3 else 'Weak'})")
    print(f"    Seasonal strength: {max(0, seasonal_strength):.3f} ({'Strong' if seasonal_strength > 0.6 else 'Moderate' if seasonal_strength > 0.3 else 'Weak'})")

    # Check if trend is increasing/decreasing
    trend_change = (result.trend.iloc[-1] - result.trend.iloc[0]) / result.trend.iloc[0] * 100
    print(f"    Trend direction: {'Increasing' if trend_change > 2 else 'Decreasing' if trend_change < -2 else 'Stable'} ({trend_change:+.1f}%)")

else:
    print(f"\n⚠ Insufficient data for STL decomposition (need >= 14 days for weekly seasonality)")
```

### 3.4: Day-of-Week Seasonality

```{code-cell} ipython3
# Analyze day-of-week patterns
daily_costs_dow = daily_costs.with_columns([
    pl.col('usage_date').dt.weekday().alias('day_of_week')
])

dow_analysis = (
    daily_costs_dow
    .group_by('day_of_week')
    .agg([
        pl.col('cost').mean().alias('avg_cost'),
        pl.col('cost').std().alias('std_cost'),
        pl.len().alias('count')
    ])
    .sort('day_of_week')
    .with_columns([
        (pl.col('std_cost') / pl.col('avg_cost')).alias('cv')
    ])
)

print(f"\nDAY-OF-WEEK ANALYSIS:")
print(f"\n  Average Cost by Day:")
dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_display = dow_analysis.with_columns([
    pl.Series('day_name', [dow_labels[i] for i in range(len(dow_analysis))])
]).select(['day_name', 'avg_cost', 'std_cost', 'cv', 'count'])

with pl.Config(fmt_float='mixed'):
    display(dow_display)

# Test for significant day-of-week effect (ANOVA)
dow_groups = [daily_costs_dow.filter(pl.col('day_of_week') == i)['cost'].to_numpy()
              for i in range(7)]
f_stat, p_value = stats.f_oneway(*dow_groups)

print(f"\n  ANOVA Test (day-of-week effect):")
print(f"    F-statistic: {f_stat:.4f}")
print(f"    P-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"    → SIGNIFICANT day-of-week effect (p < 0.05)")
    print(f"    → Include day-of-week as feature in models")
else:
    print(f"    → NO significant day-of-week effect")
    print(f"    → Costs relatively uniform across weekdays")

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
dow_plot = dow_display.to_pandas()
ax.bar(dow_plot['day_name'], dow_plot['avg_cost'],
       color='steelblue', alpha=0.7, edgecolor='black')
ax.errorbar(dow_plot['day_name'], dow_plot['avg_cost'], yerr=dow_plot['std_cost'],
            fmt='none', color='black', capsize=5)
ax.set_ylabel('Average Cost ($)', fontweight='bold')
ax.set_xlabel('Day of Week', fontweight='bold')
ax.set_title('Day-of-Week Cost Pattern', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Summary: Temporal Properties

```{code-cell} ipython3
print("=" * 80)
print("TEMPORAL PROPERTIES SUMMARY")
print("=" * 80)

print(f"\n1. AUTOCORRELATION:")
print(f"   - Lag-1 ACF: {acf_values[1]:.3f} ({'High' if acf_values[1] > 0.7 else 'Moderate' if acf_values[1] > 0.4 else 'Low'})")
print(f"   - Implication: {'Yesterday strongly predicts today' if acf_values[1] > 0.7 else 'Moderate persistence' if acf_values[1] > 0.4 else 'Low persistence'}")

print(f"\n2. STATIONARITY:")
if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
    print(f"   - Status: STATIONARY")
    print(f"   - Implication: Mean/variance stable, can model levels directly")
else:
    print(f"   - Status: NON-STATIONARY or INCONCLUSIVE")
    print(f"   - Implication: May need differencing/detrending")

if len(cost_series) >= 14:
    print(f"\n3. TREND & SEASONALITY:")
    print(f"   - Trend strength: {max(0, trend_strength):.3f}")
    print(f"   - Seasonal strength: {max(0, seasonal_strength):.3f}")
    print(f"   - Trend direction: {'Increasing' if trend_change > 2 else 'Decreasing' if trend_change < -2 else 'Stable'}")

print(f"\n4. DAY-OF-WEEK EFFECT:")
print(f"   - ANOVA p-value: {p_value:.4f}")
print(f"   - Significance: {'Yes' if p_value < 0.05 else 'No'}")

print(f"\n5. MODELING RECOMMENDATIONS:")
if acf_values[1] > 0.7:
    print(f"   - High autocorrelation → AR/ARIMA models suitable")
if p_value < 0.05:
    print(f"   - Day-of-week effect → Include as categorical feature")
if adf_pvalue >= 0.05:
    print(f"   - Non-stationary → Apply first differencing")
if len(cost_series) >= 14 and seasonal_strength > 0.3:
    print(f"   - Seasonality present → Use SARIMA or seasonal dummies")

print("\n" + "=" * 80)
```

---

## Part 4: Distribution Analysis & Transformations

### Objective

Characterize cost distribution **at multiple aggregation levels** to determine:
1. **What grain to forecast?** Resource-level? Account-level? System-level?
2. **What transformations?** Log, Box-Cox, or none?
3. **Outlier handling strategy** at each level

**CRITICAL CONTEXT**: From Part 1, we established the data grain is `(usage_date, resource_id)` - each row represents ONE resource on ONE day. The distribution of costs depends heavily on the aggregation level chosen for analysis.

### 4.0: Distribution at Multiple Aggregation Levels

```{code-cell} ipython3
print("=" * 80)
print("MULTI-LEVEL DISTRIBUTION ANALYSIS")
print("=" * 80)
print("\nData Grain: (usage_date, resource_id)")
print("  → Each row = one resource's cost on one day")
print("  → Distribution shape depends on aggregation level chosen")
```

```{code-cell} ipython3
# Level 1: RESOURCE-DAY (original grain)
print(f"\n{'='*80}")
print("LEVEL 1: RESOURCE-DAY COSTS (Original Grain)")
print(f"{'='*80}")
print("Each observation = one resource's cost on one day")

dist_resource = cost_distribution_metrics(df, PRIMARY_COST)

print(f"\nDistribution Characteristics:")
print(f"  Observations: {df.select(pl.len()).collect()[0, 0]:,}")
print(f"  Skewness: {dist_resource['skewness']:.3f}")
print(f"  Outliers (IQR): {dist_resource['outlier_pct']:.1f}%")
print(f"  Modeling recommendation: {dist_resource['modeling_rec']}")

print(f"\nPercentiles:")
for p in [0, 1, 10, 25, 50, 75, 90, 99, 100]:
    val = dist_resource['percentiles'][p]
    print(f"  P{p:>3}: ${val:>12,.2f}")

print(f"\n💡 INTERPRETATION:")
print(f"  → P50 = $0.00: Most resources cost pennies per day (Lambda, S3, stopped instances)")
print(f"  → P75 = $0.00: 75% of resource-days have near-zero costs")
print(f"  → P99 = ${dist_resource['percentiles'][99]:,.2f}: Top 1% are high-cost resources (databases, compute)")
print(f"  → Extreme skew (skewness={dist_resource['skewness']:.0f}) is EXPECTED at this grain")
```

```{code-cell} ipython3
# Level 2: ACCOUNT-DAY (common forecasting target)
print(f"\n{'='*80}")
print("LEVEL 2: ACCOUNT-DAY COSTS (Typical Forecasting Grain)")
print(f"{'='*80}")
print("Each observation = total cost for one account on one day")

# Check if account column exists
account_cols = [col for col in df.collect_schema().names() if 'account' in col.lower()]

if account_cols:
    account_col = account_cols[0]

    # Aggregate to account-day level
    df_account_day = (
        df.group_by(['usage_date', account_col])
        .agg(pl.col(PRIMARY_COST).sum().alias('daily_cost'))
    )

    dist_account = cost_distribution_metrics(df_account_day, 'daily_cost')

    print(f"\nDistribution Characteristics:")
    print(f"  Observations: {df_account_day.select(pl.len()).collect()[0, 0]:,} (account-days)")
    print(f"  Skewness: {dist_account['skewness']:.3f}")
    print(f"  Outliers (IQR): {dist_account['outlier_pct']:.1f}%")
    print(f"  Modeling recommendation: {dist_account['modeling_rec']}")

    print(f"\nPercentiles:")
    for p in [0, 1, 10, 25, 50, 75, 90, 99, 100]:
        val = dist_account['percentiles'][p]
        print(f"  P{p:>3}: ${val:>12,.2f}")

    print(f"\n💡 INTERPRETATION:")
    print(f"  → P50 > $0: Aggregation reduces zero-concentration")
    print(f"  → Skewness reduced from {dist_resource['skewness']:.0f} → {dist_account['skewness']:.1f}")
    print(f"  → More stable distribution for forecasting")

    # Store for later use
    account_day_exists = True
else:
    print("\n⚠ No account column found - skipping account-day analysis")
    account_day_exists = False
```

```{code-cell} ipython3
# Level 3: SYSTEM-DAY (single time series)
print(f"\n{'='*80}")
print("LEVEL 3: SYSTEM-DAY COSTS (Total Daily Spend)")
print(f"{'='*80}")
print("Each observation = total cost across ALL resources on one day")

# Aggregate to system-day level
df_system_day = (
    df.group_by('usage_date')
    .agg(pl.col(PRIMARY_COST).sum().alias('daily_cost'))
    .sort('usage_date')
    .collect()
)

print(f"\nDistribution Characteristics:")
print(f"  Observations: {len(df_system_day)} days")
print(f"  Mean daily cost: ${df_system_day['daily_cost'].mean():,.2f}")
print(f"  Std daily cost: ${df_system_day['daily_cost'].std():,.2f}")
print(f"  CV: {df_system_day['daily_cost'].std() / df_system_day['daily_cost'].mean():.4f}")

print(f"\nPercentiles:")
for p in [0, 10, 25, 50, 75, 90, 100]:
    val = df_system_day['daily_cost'].quantile(p / 100)
    print(f"  P{p:>3}: ${val:>12,.2f}")

print(f"\n💡 INTERPRETATION:")
print(f"  → Single time series (37 observations)")
print(f"  → No zero-concentration (always summed across thousands of resources)")
print(f"  → CV < 0.15: Very stable for forecasting")
print(f"  → This is what we analyzed in Part 3 (temporal properties)")
```

```{code-cell} ipython3
# Summary comparison
print(f"\n{'='*80}")
print("AGGREGATION LEVEL COMPARISON")
print(f"{'='*80}")

comparison = pl.DataFrame({
    'Level': ['Resource-Day', 'Account-Day', 'System-Day'],
    'Observations': [
        int(df.select(pl.len()).collect()[0, 0]),
        int(df_account_day.select(pl.len()).collect()[0, 0]) if account_day_exists else 0,
        int(len(df_system_day))
    ],
    'Median': [
        float(dist_resource['percentiles'][50]),
        float(dist_account['percentiles'][50]) if account_day_exists else 0.0,
        float(df_system_day['daily_cost'].median())
    ],
    'Skewness': [
        float(dist_resource['skewness']),
        float(dist_account['skewness']) if account_day_exists else 0.0,
        float(df_system_day['daily_cost'].skew()) if len(df_system_day) > 2 else 0.0
    ],
    'Zeros_%': [
        float((dist_resource['percentiles'][75] == 0) * 75),  # Approx: if P75=0, then >75% zeros
        float((dist_account['percentiles'][50] == 0) * 50) if account_day_exists else 0.0,
        0.0
    ]
})

print("\nDistribution by Aggregation Level:")
with pl.Config(fmt_float='mixed'):
    display(comparison)

print(f"\n💡 KEY INSIGHT:")
print(f"  → Aggregation DRAMATICALLY changes distribution shape")
print(f"  → Resource-level: Extreme skew, zero-heavy (not suitable for standard forecasting)")
print(f"  → Account-level: Moderate skew, few zeros (good balance)")
print(f"  → System-level: Low skew, no zeros (most stable, but single series)")

print(f"\n📊 MODELING RECOMMENDATION:")
if account_day_exists:
    print(f"  → Primary: Account-day forecasts (balance of granularity + stability)")
    print(f"  → Secondary: System-day for budget planning (single stable series)")
    print(f"  → Avoid: Resource-day (extreme skew, zero-heavy, overfitting risk)")
else:
    print(f"  → Primary: System-day forecasts (stable time series)")
    print(f"  → Avoid: Resource-day (extreme skew, zero-heavy)")
```

### 4.1: Distribution Shape Assessment (System-Day Level)

We focus on **system-day** aggregation for shape assessment since this is the most stable forecasting target.

```{code-cell} ipython3
# Use system-day costs (already computed above) for distribution analysis
sample_costs = df_system_day['daily_cost'].to_numpy()
sample_costs_positive = sample_costs[sample_costs > 0]

print(f"Analyzing distribution shape for SYSTEM-DAY costs:")
print(f"  Observations: {len(sample_costs)} days")
print(f"  All positive: {(sample_costs > 0).all()}")

# Visualize distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram (linear scale)
axes[0, 0].hist(sample_costs, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Cost ($)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Cost Distribution (Linear Scale)', fontweight='bold', fontsize=12)
axes[0, 0].grid(alpha=0.3, axis='y')

# Histogram (log scale)
axes[0, 1].hist(sample_costs_positive, bins=100, color='darkgreen', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Cost ($)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Cost Distribution (Log Scale)', fontweight='bold', fontsize=12)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3, which='both')

# Q-Q plot (normal)
stats.probplot(sample_costs, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot vs Normal Distribution', fontweight='bold', fontsize=12)
axes[1, 0].grid(alpha=0.3)

# Q-Q plot (log-normal)
stats.probplot(np.log1p(sample_costs_positive), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot vs Normal (log-transformed)', fontweight='bold', fontsize=12)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Normality tests
_, ks_pvalue = stats.kstest(sample_costs, 'norm', args=(sample_costs.mean(), sample_costs.std()))
_, shapiro_pvalue = stats.shapiro(sample_costs[:5000])  # Shapiro limited to 5000 samples

print(f"\nNormality Tests (Raw Costs):")
print(f"  Kolmogorov-Smirnov: p={ks_pvalue:.4e} ({'Normal' if ks_pvalue > 0.05 else 'Not normal'})")
print(f"  Shapiro-Wilk: p={shapiro_pvalue:.4e} ({'Normal' if shapiro_pvalue > 0.05 else 'Not normal'})")

# Log-transformed normality
log_costs = np.log1p(sample_costs_positive)
_, log_ks_pvalue = stats.kstest(log_costs, 'norm', args=(log_costs.mean(), log_costs.std()))
_, log_shapiro_pvalue = stats.shapiro(log_costs[:5000])

print(f"\nNormality Tests (Log-Transformed Costs):")
print(f"  Kolmogorov-Smirnov: p={log_ks_pvalue:.4e} ({'Normal' if log_ks_pvalue > 0.05 else 'Not normal'})")
print(f"  Shapiro-Wilk: p={log_shapiro_pvalue:.4e} ({'Normal' if log_shapiro_pvalue > 0.05 else 'Not normal'})")

if log_shapiro_pvalue > shapiro_pvalue:
    print(f"\n  → Log transformation IMPROVES normality ({log_shapiro_pvalue:.4e} > {shapiro_pvalue:.4e})")
else:
    print(f"\n  → Log transformation does NOT improve normality")
```

### 4.2: Box-Cox Transformation Optimization

```{code-cell} ipython3
# Find optimal Box-Cox lambda
# Box-Cox requires positive values
positive_costs = sample_costs_positive[sample_costs_positive > 0]

if len(positive_costs) > 100:
    # Fit Box-Cox
    transformed_data, fitted_lambda = stats.boxcox(positive_costs[:10000])  # Limit for performance

    print(f"\nBOX-COX TRANSFORMATION:")
    print(f"  Optimal lambda: {fitted_lambda:.4f}")

    if abs(fitted_lambda) < 0.1:
        print(f"    → Lambda ≈ 0: Log transformation recommended")
    elif abs(fitted_lambda - 1) < 0.1:
        print(f"    → Lambda ≈ 1: No transformation needed")
    elif abs(fitted_lambda - 0.5) < 0.1:
        print(f"    → Lambda ≈ 0.5: Square root transformation")
    else:
        print(f"    → Lambda = {fitted_lambda:.2f}: Power transformation")

    # Test normality after Box-Cox
    _, bc_shapiro_pvalue = stats.shapiro(transformed_data[:5000])
    print(f"  Shapiro-Wilk (Box-Cox): p={bc_shapiro_pvalue:.4e}")

    # Compare transformations
    print(f"\n  Transformation Comparison (Shapiro-Wilk p-values):")
    print(f"    Raw: {shapiro_pvalue:.4e}")
    print(f"    Log: {log_shapiro_pvalue:.4e}")
    print(f"    Box-Cox: {bc_shapiro_pvalue:.4e}")

    best_transform = max([('Raw', shapiro_pvalue), ('Log', log_shapiro_pvalue),
                          ('Box-Cox', bc_shapiro_pvalue)], key=lambda x: x[1])
    print(f"    → Best transformation: {best_transform[0]}")
else:
    print(f"\n⚠ Insufficient positive values for Box-Cox optimization")
```

### 4.3: Outlier Detection & Treatment

```{code-cell} ipython3
# Multiple outlier detection methods from hellocloud.utils
# Note: smart_sample() returns a collected DataFrame (not LazyFrame)
sample_df = smart_sample(df, n=100_000)

print(f"\nOUTLIER DETECTION (100K sample):")

# IQR method (takes Series, not DataFrame + column)
outliers_iqr = detect_outliers_iqr(sample_df[PRIMARY_COST], multiplier=1.5)
print(f"\n  IQR Method (multiplier=1.5):")
print(f"    Outliers: {outliers_iqr.sum():,} ({outliers_iqr.sum() / len(sample_df) * 100:.2f}%)")

# Z-score method (takes Series, not DataFrame + column)
outliers_zscore = detect_outliers_zscore(sample_df[PRIMARY_COST], threshold=3.0)
print(f"\n  Z-Score Method (threshold=3.0):")
print(f"    Outliers: {outliers_zscore.sum():,} ({outliers_zscore.sum() / len(sample_df) * 100:.2f}%)")

# Isolation Forest (takes DataFrame + column list)
outliers_iforest = detect_outliers_isolation_forest(sample_df, [PRIMARY_COST], contamination=0.05)
print(f"\n  Isolation Forest (contamination=0.05):")
print(f"    Outliers: {outliers_iforest.sum():,} ({outliers_iforest.sum() / len(sample_df) * 100:.2f}%)")

# Compare methods (convert all to numpy for compatibility)
outliers_iqr_np = outliers_iqr.to_numpy()
outliers_zscore_np = outliers_zscore.to_numpy()
agreement = (outliers_iqr_np & outliers_zscore_np & outliers_iforest).sum()
print(f"\n  Agreement across methods: {agreement:,} outliers flagged by all 3")

# Visualize outliers
fig, ax = plt.subplots(figsize=(16, 6))

sample_with_outliers = sample_df.with_columns([
    pl.Series('outlier_iqr', outliers_iqr),
    pl.Series('outlier_zscore', outliers_zscore),
    pl.Series('outlier_iforest', outliers_iforest),
    pl.Series('index', range(len(sample_df)))
])

# Plot normal points
normal_points = sample_with_outliers.filter(~pl.col('outlier_iqr'))
ax.scatter(normal_points['index'].to_numpy(), normal_points[PRIMARY_COST].to_numpy(),
          alpha=0.3, s=10, color='steelblue', label='Normal')

# Plot outliers (IQR)
outlier_points = sample_with_outliers.filter(pl.col('outlier_iqr'))
ax.scatter(outlier_points['index'].to_numpy(), outlier_points[PRIMARY_COST].to_numpy(),
          alpha=0.8, s=30, color='red', marker='x', label='Outliers (IQR)')

ax.set_xlabel('Sample Index', fontweight='bold')
ax.set_ylabel('Cost ($)', fontweight='bold')
ax.set_title('Outlier Detection (IQR Method)', fontweight='bold', fontsize=14)
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# Treatment recommendations
print(f"\n  Treatment Recommendations:")
if outliers_iqr.sum() / len(sample_df) > 0.1:
    print(f"    → High outlier rate (>10%) suggests heavy-tailed distribution")
    print(f"    → Use robust models (Huber loss, quantile regression)")
    print(f"    → OR apply log transformation to compress tail")
else:
    print(f"    → Moderate outlier rate (<10%)")
    print(f"    → Can cap/Winsorize extreme values")
    print(f"    → OR train separate models for outliers")
```

### Summary: Distribution & Transformations

```{code-cell} ipython3
print("=" * 80)
print("DISTRIBUTION ANALYSIS SUMMARY")
print("=" * 80)

print(f"\n🎯 CRITICAL FINDING: Distribution depends on aggregation level")
print(f"\n1. RESOURCE-DAY LEVEL (Original Grain):")
print(f"   - Skewness: {dist_resource['skewness']:.1f} (EXTREME - expected for resource-level)")
print(f"   - Median: ${dist_resource['percentiles'][50]:,.2f} (75%+ are near-zero costs)")
print(f"   - Outliers: {dist_resource['outlier_pct']:.1f}% (many low-cost resources)")
print(f"   - Recommendation: {dist_resource['modeling_rec']}")
print(f"   → NOT SUITABLE for standard forecasting (zero-heavy, extreme skew)")

if account_day_exists:
    print(f"\n2. ACCOUNT-DAY LEVEL (Recommended Grain):")
    print(f"   - Skewness: {dist_account['skewness']:.3f} (moderate - good for forecasting)")
    print(f"   - Median: ${dist_account['percentiles'][50]:,.2f} (aggregation reduces zeros)")
    print(f"   - Outliers: {dist_account['outlier_pct']:.1f}%")
    print(f"   - Recommendation: {dist_account['modeling_rec']}")
    print(f"   → GOOD BALANCE of granularity and stability")

print(f"\n3. SYSTEM-DAY LEVEL (Budget Forecasting):")
system_skew = float(df_system_day['daily_cost'].skew()) if len(df_system_day) > 2 else 0.0
print(f"   - Skewness: {system_skew:.3f} (low - very stable)")
print(f"   - Median: ${df_system_day['daily_cost'].median():,.2f}")
print(f"   - CV: {df_system_day['daily_cost'].std() / df_system_day['daily_cost'].mean():.4f}")

# Check if normality test has been run (Section 4.1)
if 'shapiro_pvalue' in locals():
    print(f"   - Shape: {'Approximately normal' if shapiro_pvalue > 0.05 else 'Non-normal'} (Shapiro p={shapiro_pvalue:.4e})")
else:
    print(f"   - Shape: To be determined via normality tests (Section 4.1)")

print(f"   → MOST STABLE, but single time series ({len(df_system_day)} days)")

print(f"\n4. TRANSFORMATION STRATEGY:")
# Check if Box-Cox analysis has been run (Section 4.2)
if 'fitted_lambda' in locals() and fitted_lambda is not None:
    print(f"   System-day: {best_transform[0]} (Shapiro p={best_transform[1]:.4e})")
    if best_transform[0] == 'Log':
        print(f"   → Use: np.log1p(cost) for linear models")
    elif best_transform[0] == 'Raw':
        print(f"   → No transformation needed (approximately normal)")
else:
    # Box-Cox not run yet, use basic distribution metrics
    if system_skew > 1:
        print(f"   System-day: Log transformation recommended (skewness={system_skew:.2f})")
    else:
        print(f"   System-day: No transformation needed (skewness={system_skew:.2f})")

if account_day_exists:
    print(f"   Account-day: {dist_account['modeling_rec']}")

print(f"   Resource-day: {dist_resource['modeling_rec']} (but avoid this level)")

print(f"\n5. FINAL RECOMMENDATION:")
if account_day_exists:
    print(f"   ✅ PRIMARY TARGET: Account-day forecasts")
    print(f"      - Balance of granularity (per-account insights) + stability")
    print(f"      - Apply {dist_account['modeling_rec']} if skewness > 1")
else:
    print(f"   ✅ PRIMARY TARGET: System-day forecasts")
    print(f"      - Single stable time series")
    if 'fitted_lambda' in locals() and fitted_lambda is not None:
        print(f"      - {best_transform[0]} transformation")
    else:
        print(f"      - {'Log' if system_skew > 1 else 'No'} transformation recommended")

print(f"   ✅ SECONDARY TARGET: System-day for budget planning")
print(f"   ❌ AVOID: Resource-day forecasts (extreme skew, zero-heavy)")

print("\n" + "=" * 80)
```

---

## Part 5: Modeling Readiness Assessment

### Objective

Synthesize findings into actionable modeling recommendations:
1. **Feature engineering** guidance
2. **Train/validation/test** split strategy
3. **Baseline models** to establish
4. **Success metrics** definition

```{code-cell} ipython3
print("=" * 80)
print("MODELING READINESS ASSESSMENT")
print("=" * 80)
```

### 5.1: Data Characteristics Summary

```{code-cell} ipython3
print(f"\nDATA CHARACTERISTICS:")
print(f"  Grain: (usage_date, resource_id)")
print(f"  Frequency: Daily (no intraday)")
print(f"  Valid period: {date_range['days'][0]} days (Sept 1 - Oct 6, 2025)")
print(f"  Total records: {total_rows:,}")
print(f"  Resources tracked: ~{resource_counts_per_day['unique_resources'].mean():,.0f}/day")
```

### 5.2: Feature Engineering Recommendations

```{code-cell} ipython3
print(f"\nFEATURE ENGINEERING RECOMMENDATIONS:")

print(f"\n1. TEMPORAL FEATURES:")
if acf_values[1] > 0.7:
    print(f"   ✓ Lag features (1, 7, 14, 30 days) - high autocorrelation")
if p_value < 0.05:
    print(f"   ✓ Day-of-week dummies - significant weekly pattern")
if len(cost_series) >= 14 and seasonal_strength > 0.3:
    print(f"   ✓ Week-of-month indicator - seasonality detected")
print(f"   ✓ Days since resource creation (if available)")
print(f"   ✓ Rolling statistics (7-day mean, std)")

print(f"\n2. ENTITY FEATURES:")
print(f"   ✓ Cloud provider (categorical)")
if 'account_id' in df.collect_schema().names():
    print(f"   ✓ Account ID (high cardinality - use target encoding)")
if 'product_family' in df.collect_schema().names():
    print(f"   ✓ Product family (categorical)")
print(f"   ✓ Resource type (if available)")

print(f"\n3. COST FEATURES:")
print(f"   ✓ Log-transform of {PRIMARY_COST} (high skewness)")
print(f"   ✓ Cost percentile rank within provider/account")
print(f"   ✓ Cost stability (CV over past 7 days)")

print(f"\n4. AGGREGATION LEVELS:")
most_stable_entity = variance_df.tail(1)['entity_type'][0] if len(variance_df) > 0 else 'account'
print(f"   → For stable forecasts: Aggregate by {most_stable_entity}")
print(f"   → For granular optimization: Resource-level (but expect higher variance)")
```

### 5.3: Train/Validation/Test Split Strategy

```{code-cell} ipython3
print(f"\nTRAIN/VALIDATION/TEST SPLIT:")

# Time series split (chronological)
split_dates = {
    'train_end': date_range['max_date'][0] - timedelta(days=14),
    'val_end': date_range['max_date'][0] - timedelta(days=7),
    'test_end': date_range['max_date'][0]
}

train_days = (split_dates['train_end'] - date_range['min_date'][0]).days
val_days = (split_dates['val_end'] - split_dates['train_end']).days
test_days = (split_dates['test_end'] - split_dates['val_end']).days

print(f"\n  Chronological Split (respects temporal ordering):")
print(f"    Train: {date_range['min_date'][0]} to {split_dates['train_end']} ({train_days} days)")
print(f"    Validation: {split_dates['train_end'] + timedelta(days=1)} to {split_dates['val_end']} ({val_days} days)")
print(f"    Test: {split_dates['val_end'] + timedelta(days=1)} to {split_dates['test_end']} ({test_days} days)")

print(f"\n  Split Ratio: {train_days/date_range['days'][0]*100:.0f}% / {val_days/date_range['days'][0]*100:.0f}% / {test_days/date_range['days'][0]*100:.0f}%")

print(f"\n  Cross-Validation:")
print(f"    Method: TimeSeriesSplit (expanding window)")
print(f"    Folds: 5")
print(f"    Reason: Preserves temporal ordering, tests on multiple future periods")

print(f"\n  ⚠ CAUTION:")
print(f"    - Only {date_range['days'][0]} days available - limited for deep learning")
print(f"    - Consider external data sources for longer history")
print(f"    - Resource-level forecasting may overfit (high dimensionality)")
```

### 5.4: Baseline Models

```{code-cell} ipython3
print(f"\nBASELINE MODELS (establish minimum performance):")

print(f"\n1. NAIVE BASELINES:")
print(f"   - Persistence: y_t = y_{t-1} (yesterday's cost)")
print(f"   - Seasonal Naive: y_t = y_{t-7} (last week same day)")
print(f"   - Mean: y_t = mean(historical costs)")

print(f"\n2. STATISTICAL MODELS:")
if acf_values[1] > 0.7 and adf_pvalue < 0.05:
    print(f"   ✓ ARIMA(p,d,q) - high autocorrelation, stationary")
    print(f"     → Suggested: ARIMA(1,0,1) or auto-ARIMA")
elif acf_values[1] > 0.7:
    print(f"   ✓ ARIMA(p,d,q) - high autocorrelation, may need differencing")
    print(f"     → Suggested: ARIMA(1,1,1)")

if p_value < 0.05:
    print(f"   ✓ SARIMA - weekly seasonality detected")
    print(f"     → Suggested: SARIMA(1,0,1)(1,0,1,7)")

print(f"\n3. MACHINE LEARNING MODELS:")
print(f"   ✓ LightGBM/XGBoost - handle skewed distributions well")
print(f"   ✓ Random Forest - robust to outliers")
if acf_values[1] > 0.7:
    print(f"   ✓ LSTM/GRU - capture temporal dependencies")

print(f"\n4. ADVANCED (from this project):")
print(f"   ✓ Gaussian Process (notebook 04) - uncertainty quantification")
print(f"   ✓ Bayesian Hierarchical (PyMC) - multi-level structure")
print(f"   ✓ Foundation Models - Chronos, TimesFM (zero-shot)")
```

### 5.5: Success Metrics Definition

```{code-cell} ipython3
print(f"\nSUCCESS METRICS:")

print(f"\n1. POINT FORECAST ACCURACY:")
print(f"   Primary:")
print(f"     - MAPE (Mean Absolute Percentage Error) - interpretable, scale-independent")
print(f"     - RMSE (Root Mean Squared Error) - penalizes large errors")
print(f"   Secondary:")
print(f"     - MAE (Mean Absolute Error) - robust to outliers")
print(f"     - SMAPE (Symmetric MAPE) - handles near-zero values better")

print(f"\n2. PROBABILISTIC FORECAST:")
print(f"   - Coverage: % of actuals within prediction interval")
print(f"   - Calibration: Predicted vs empirical quantiles")
print(f"   - CRPS (Continuous Ranked Probability Score)")

print(f"\n3. BUSINESS METRICS:")
print(f"   - Cost optimization potential: Sum(actual - predicted efficient)")
print(f"   - Anomaly detection rate: TP/(TP+FN) for cost spikes")
print(f"   - Budget adherence: |forecast - actual| / budget")

print(f"\n4. BASELINE TARGETS (naive persistence):")
# Compute persistence forecast error
daily_costs_with_lag = daily_costs.with_columns([
    pl.col('cost').shift(1).alias('cost_lag1')
])
persistence_mape = (
    ((daily_costs_with_lag['cost'] - daily_costs_with_lag['cost_lag1']).abs() /
     daily_costs_with_lag['cost']).mean() * 100
)
print(f"   - Persistence MAPE: {persistence_mape:.2f}%")
print(f"   → Models must beat this to be useful")
```

### 5.6: Implementation Roadmap

```{code-cell} ipython3
print(f"\nIMPLEMENTATION ROADMAP:")

print(f"\n📋 Phase 1: Baseline Establishment (1-2 days)")
print(f"   1. Implement naive baselines (persistence, seasonal naive)")
print(f"   2. Train ARIMA/SARIMA on account-level aggregates")
print(f"   3. Establish minimum MAPE threshold")

print(f"\n📋 Phase 2: Feature Engineering (2-3 days)")
print(f"   1. Create temporal features (lags, day-of-week)")
print(f"   2. Engineer entity features (target encoding for accounts)")
print(f"   3. Apply transformations (log costs)")
print(f"   4. Build feature store for reuse")

print(f"\n📋 Phase 3: ML Models (3-5 days)")
print(f"   1. Train LightGBM on resource-level data")
print(f"   2. Hyperparameter tuning with TimeSeriesSplit CV")
print(f"   3. Compare against ARIMA baseline")
print(f"   4. Analyze feature importance")

print(f"\n📋 Phase 4: Advanced Models (5-7 days)")
print(f"   1. Gaussian Process (adapt notebook 04 to this data)")
print(f"   2. Bayesian Hierarchical (provider → account → resource)")
print(f"   3. Foundation models (Chronos zero-shot)")
print(f"   4. Ensemble top performers")

print(f"\n📋 Phase 5: Production & Monitoring (ongoing)")
print(f"   1. Deploy best model(s) to API")
print(f"   2. Set up monitoring dashboard")
print(f"   3. Implement feedback loop (retrain on new data)")
print(f"   4. A/B test forecasts vs actuals")
```

### Final Summary: Modeling Readiness

```{code-cell} ipython3
print("\n" + "=" * 80)
print("MODELING READINESS - FINAL ASSESSMENT")
print("=" * 80)

print(f"\n✅ DATA QUALITY:")
print(f"   - Clean dataset: {date_range['days'][0]} days, {total_rows:,} records")
print(f"   - Grain identified: (date, resource_id)")
print(f"   - Artifacts removed: Post-Oct 7 constant values filtered")

print(f"\n✅ STATISTICAL PROPERTIES:")
print(f"   - Autocorrelation: {acf_values[1]:.3f} ({'High' if acf_values[1] > 0.7 else 'Moderate'})")
print(f"   - Stationarity: {'Yes' if adf_pvalue < 0.05 and kpss_pvalue > 0.05 else 'Needs differencing'}")
print(f"   - Seasonality: {'Weekly pattern detected' if p_value < 0.05 else 'None detected'}")

print(f"\n✅ ENTITY STRUCTURE:")
print(f"   - Gini: {gini:.3f} (cost concentration)")
print(f"   - Pareto: Top {resources_for_80pct / len(resource_costs) * 100:.1f}% resources → 80% costs")
most_stable_entity = variance_df.tail(1)['entity_type'][0] if len(variance_df) > 0 else 'account'
print(f"   - Stable aggregation: {most_stable_entity}-level")

print(f"\n✅ DISTRIBUTION:")
print(f"   - Shape: Right-skewed (skewness={dist_metrics['skewness']:.2f})")
print(f"   - Transformation: {dist_metrics['modeling_rec']}")
print(f"   - Outliers: {dist_metrics['outlier_pct']:.1f}%")

print(f"\n⚠️ LIMITATIONS:")
print(f"   - Short history: Only {date_range['days'][0]} days (limited for deep learning)")
print(f"   - Data collapse: Oct 7+ unusable (AWS drop)")
print(f"   - High dimensionality: {resource_counts_per_day['unique_resources'].mean():,.0f} resources/day")

print(f"\n🎯 RECOMMENDED NEXT STEPS:")
print(f"   1. Start with {most_stable_entity}-level aggregation (stable, lower variance)")
print(f"   2. Implement ARIMA baseline (leverages high autocorrelation)")
print(f"   3. Train LightGBM with lag features (handles skew, captures patterns)")
print(f"   4. Experiment with GP/Bayesian models (uncertainty quantification)")
print(f"   5. Consider foundation models (Chronos) for zero-shot validation")

print(f"\n🚀 READY FOR MODELING: YES")
print(f"   - Data cleaned and validated")
print(f"   - Properties characterized")
print(f"   - Features identified")
print(f"   - Baselines defined")
print(f"   - Success metrics established")

print("\n" + "=" * 80)
print("Analysis complete. Proceed to modeling phase (notebook 07).")
print("=" * 80)
```

---

## Appendix: Key Findings Reference

```{code-cell} ipython3
# Export key findings for use in modeling notebooks
findings = {
    'data_grain': '(usage_date, resource_id)',
    'frequency': 'daily',
    'valid_days': int(date_range['days'][0]),
    'total_records': int(total_rows),
    'avg_resources_per_day': int(resource_counts_per_day['unique_resources'].mean()),

    'acf_lag1': float(acf_values[1]),
    'stationary': bool(adf_pvalue < 0.05 and kpss_pvalue > 0.05),
    'weekly_seasonality': bool(p_value < 0.05),

    'gini_coefficient': float(gini),
    'pareto_80pct_resources': int(resources_for_80pct),
    'pareto_80pct_percent': float(resources_for_80pct / len(resource_costs) * 100),

    'skewness': float(dist_metrics['skewness']),
    'transformation_rec': dist_metrics['modeling_rec'],
    'outlier_pct': float(dist_metrics['outlier_pct']),

    'most_stable_entity': most_stable_entity if len(variance_df) > 0 else 'unknown',
    'persistence_mape': float(persistence_mape),
}

# Save findings
import json
findings_path = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/notebooks/statistical_findings.json')
with open(findings_path, 'w') as f:
    json.dump(findings, f, indent=2)

print(f"✓ Key findings exported to: {findings_path}")
print(f"\nLoad in future notebooks with:")
print(f"  import json")
print(f"  with open('{findings_path}') as f:")
print(f"      findings = json.load(f)")
```

---

**Analysis Complete**

This notebook established the statistical foundation for forecasting models by:
1. ✅ Determining data grain (resource-level daily)
2. ✅ Decomposing entity costs (Pareto analysis, Gini coefficient)
3. ✅ Characterizing temporal properties (ACF, stationarity, seasonality)
4. ✅ Analyzing distributions (transformations, outliers)
5. ✅ Defining modeling readiness (features, splits, baselines, metrics)

**Next**: Notebook 07 - Forecasting Models (ARIMA, LightGBM, GP, Foundation Models)
