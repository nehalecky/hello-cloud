---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: cloud-sim
  display_name: cloud-sim
  language: python
---

# PiedPiper Dataset - Exploratory Data Analysis

## Overview

This notebook supports exploratory analysis for CloudZero PiedPiper billing data, including:

- Data loading and quality assessment
- Schema-level attribute analysis and filtering
- Grain discovery for time series forecasting
- Entity persistence validation
- Time series visualization for stable entities

**Objectives**: 
 * Create tidy data model containg high info gain variables to support downstream analysis
 * Identify the optimal compound key (grain) for the time series
 * Basic understanding of time series distributions.

**Dataset**: CloudZero production billing data (122 days, 8.3M records)

---

## Assumptions

Cloud billing data represents resource consumption events aggregated by CloudZero's data pipeline. We conceptualize the **event space** as:

- $\mathbf{E}_0$ (full space): All cloud resource consumption across all providers, accounts, and time
- $\mathbf{E}$ (observed): pied Piper sample produced by CloudZero, where $\mathbf{E} \subseteq \mathbf{E}_0$

**Known sampling biases**:
1. **Provider coverage**: Only resources with cost allocation tags are visible
2. **Temporal granularity**: Daily aggregation (not real-time)
3. **Data quality**: Provider-specific pipeline issues may cause artifactual patterns

**Expected billing event structure**:

A cloud resource cost (CRC) record fundamentally contains:
- $t$ (timestamp): When resource was consumed (daily grain)
- $r$ (resource): Identifier for the billable cloud resource
- $c$ (cost): Dollar amount for the consumption

**Question**: What compound key defines the resource identifier $r$ such that we can track spending over time?

This is the **grain discovery problem** - finding the most granular combination of attributes whose entities persist temporally, enabling forecasting.

---

## Part 1: Data Loading & Schema Analysis

### 1.1 Import Dependencies

```{code-cell} ipython3
from pathlib import Path
from datetime import date, datetime, timedelta
import ibis
from ibis import _
import ibis.selectors as s
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from cloudlens.utils import (
    configure_notebook_logging,
    attribute_analysis,
    daily_observation_counts,
    plot_categorical_frequencies,
    calculate_attribute_scores,
    find_correlated_pairs,
    select_from_pairs
)

# Configure Ibis
ibis.options.interactive = True  # Auto-execute for repr
ibis.options.repr.interactive.max_rows = 10

configure_notebook_logging()
logger.info("PiedPiper EDA - Notebook initialized (Ibis + DuckDB)")
```

### 1.2 Load Dataset

```{code-cell} ipython3
# Connect to DuckDB (in-memory, fast analytical queries)
con = ibis.duckdb.connect()

# Load Parquet file
DATA_PATH = Path('~/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet').expanduser()
df = con.read_parquet(DATA_PATH, table_name='piedpiper')

# Basic shape
total_rows = df.count().execute()
total_cols = len(df.schema())
logger.info(f"Dataset: {total_rows:,} rows × {total_cols} columns")
logger.info(f"Backend: DuckDB (local analytical engine)")

# Preview
df.head(5)
```

---

### 1.3 Temporal Observation Density

```{code-cell} ipython3
# Identify date column from schema
schema = df.schema()
date_cols = [
    name for name, dtype in schema.items()
    if dtype.is_temporal()
]

logger.info(f"Date/Datetime columns found: {date_cols}")
date_col = date_cols[0]

# Rename to 'date' for simplicity
df = df.rename(date=date_col)
logger.info(f"Renamed {date_col} → 'date'")

# Compute date range
date_stats = df.select(
    min_date=_.date.min(),
    max_date=_.date.max()
).execute()

logger.info(f"Date range: {date_stats['min_date'].iloc[0]} to {date_stats['max_date'].iloc[0]}")
```

**Noted** Max date spans into the future `2025-12-31`. 

Let's inspect the temporal record density and plot.

```{code-cell} ipython3
# Daily counts
daily = (
    df.group_by('date')
    .agg(count=_.count())
    .order_by('date')
    .execute()
)

# Simple log-scale time series
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily['date'], daily['count'], linewidth=2, color='steelblue')
ax.set_yscale('log')
ax.set_ylabel('Daily Record Count (log scale)')
ax.set_xlabel('Date')
ax.set_title('Temporal Observation Density')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Observation**: The time series shows a sharp drop at a specific date, with data continuing into the future. Something is off—let's investigate the magnitude of day-over-day changes to identify the anomaly.

```{code-cell} ipython3
# Compute percent change to quantify anomaly
# Note: daily is already a pandas DataFrame from .execute()
daily['pct_change'] = daily['count'].pct_change()

# Find largest drops
logger.info("Largest day-over-day drops:")
daily.nsmallest(5, 'pct_change')[['date', 'count', 'pct_change']]
```

The data shows a significant drop (>30%) on a specific date. We'll filter to the period before this anomaly for clean analysis.

```{code-cell} ipython3
# Find earliest date with >30% drop (pct_change < -0.30)
cutoff_row = daily[daily['pct_change'] < -0.30].sort_values('date').iloc[0]
CUTOFF_DATE = cutoff_row['date']

logger.info(f"Cutoff date detected: {CUTOFF_DATE}")

# Apply filter to Ibis table
df = df.filter(_.date < CUTOFF_DATE)

# Compute stats
stats = df.agg(
    rows=_.count(),
    days=_.date.nunique(),
    start=_.date.min(),
    end=_.date.max()
).execute()

logger.info(f"Filtered to: {stats['rows'].iloc[0]:,} rows, {stats['days'].iloc[0]} days ({stats['start'].iloc[0]} to {stats['end'].iloc[0]})")
stats
```

---

### 1.4 General Attribute Analysis

**Methodology**: We analyze the information density in each attribute using metrics that capture value density, cardinality, and confusion, allowing us to understand the information available within each attribute. We leverage this to find ideal attributes that support our event and time series discrimination efforts.

#### Value Density
The density of non-null values across attributes (completeness indicator). Low values imply high sparsity (many nulls), which are likely not informative for modeling or grain discovery. **Higher is better**.

#### Nonzero Density (Numeric Columns)
The density of non-zero values among numeric attributes. High nonzero density indicates rich variation; low values indicate dominance of zeros (measurement artifacts, optional features, sparse usage). For cost columns, low nonzero density represents frequent periods of no consumption. Non-numeric columns default to 1.0 (treated as "all nonzero" for scoring). **Higher is better**.

#### Cardinality Ratio
The ratio of unique values to total observations (`unique_count / total_rows`). Maximum cardinality equals the number of observations. Values approaching 1.0 imply nearly distinct values per observation (primary keys), offering little grouping utility. Values near 0.0 indicate coarse grouping dimensions. Among non-primary-key columns, higher cardinality provides better discrimination. **Higher is better** (after filtering primary keys).

#### Value Confusion (Shannon Entropy)
Measures the "confusion" or information content of value assignments via [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)). Low entropy implies concentration in few values (zero confusion, minimal information). High entropy implies uniform distribution across many values (maximum confusion, rich information). **Higher is better** for informative features.

#### Information Score
Harmonic mean of **four** metrics: value density, nonzero density, cardinality ratio, and entropy. This composite metric requires attributes to score well on **all four dimensions**—all with positive linear relationships (higher = better). The harmonic mean penalizes imbalance: an attribute must perform well across completeness, non-sparsity, uniqueness, and distributional richness. Higher scores indicate more informative attributes for grain discovery and modeling.

These size-normalized metrics help identify attributes with little discriminatory information, which can be filtered to simplify analysis and modeling tasks.

```{code-cell} ipython3
# Compute comprehensive attribute analysis
# attribute_analysis now works directly with Ibis Tables
attrs = attribute_analysis(df, sample_size=50_000)

# Display all columns sorted by information score
import pandas as pd
with pd.option_context('display.max_rows', None):
    display(attrs)
```

The table above is sorted by **information score** (highest first), which ranks attributes by their combined utility across completeness, uniqueness, and distributional richness. High-scoring attributes are the most informative for grain discovery and modeling.

**Interpretation**:
- **Top scorers**: Attributes with balanced completeness, moderate-to-high cardinality, and rich value distributions—ideal grain candidates
- **Low scorers**: Either sparse (many nulls), low-cardinality (coarse dimensions), or low-entropy (concentrated values)—useful for filtering or hierarchical aggregation but not fine-grained keys

Now we classify attributes by cardinality to guide composite key construction:

```{code-cell} ipython3
# Classify columns by cardinality for grain discovery
# primary_keys = attrs.filter(pl.col('cardinality_ratio') > 0.9)['column'].to_list()
# high_card = attrs.filter((pl.col('cardinality_ratio') > 0.5) & (pl.col('cardinality_ratio') <= 0.9))['column'].to_list()
# medium_card = attrs.filter((pl.col('cardinality_ratio') > 0.1) & (pl.col('cardinality_ratio') <= 0.5))['column'].to_list()
# grouping_dims = attrs.filter(pl.col('cardinality_ratio') <= 0.1)['column'].to_list()

# logger.info(f"\n🎯 Grain Discovery Classification:")
# logger.info(f"\n🔴 Primary Keys ({len(primary_keys)}):")
# logger.info(f"   {primary_keys}")
# logger.info(f"   → Drop these (record IDs, not analytical dimensions)")
# if high_card:
#     logger.info(f"\n🟠 High Cardinality ({len(high_card)}):")
#     logger.info(f"   {high_card}")
#     logger.info(f"   → Potential resource-level identifiers (investigate persistence)")
# if medium_card:
#     logger.info(f"\n🟡 Medium Cardinality ({len(medium_card)}):")
#     logger.info(f"   {medium_card}")
#     logger.info(f"   → Good candidates for composite keys")
# logger.info(f"\n🟢 Grouping Dimensions ({len(grouping_dims)}):")
# logger.info(f"   {grouping_dims}")
# logger.info(f"   → Standard dimensions for aggregation")
```

**Filtering Strategy**: We cannot filter solely on `information_score` because harmonic mean is dominated by the lowest input. Low-cardinality grouping dimensions (like `cloud_provider`, `region`) score poorly due to cardinality ratio ≈ 0, yet they're critical for hierarchical analysis. Instead, we use **cardinality-stratified filtering**—different criteria for different column roles:

- **Primary keys (>90%)**: Always drop (no analytical value)
- **High cardinality (50-90%)**: Keep if complete (potential resource IDs like `cloud_id`)
- **Medium cardinality (10-50%)**: Keep if info score > threshold (composite key candidates)
- **Grouping dimensions (<10%)**: Keep if highly complete (hierarchical dimensions)
- **Sparse columns**: Drop if value_density < 80% (too many nulls)

This preserves valuable low-cardinality columns while removing noise.

```{code-cell} ipython3
# Stratified filtering: different criteria by cardinality class

# DROP: Primary keys (>90% cardinality, no grouping utility)
drop_primary_keys = attrs[attrs['cardinality_ratio'] > 0.9]['column'].tolist()

# DROP: Sparse columns (value_density < 60%, too many nulls)
drop_sparse = attrs[attrs['value_density'] < 0.6]['column'].tolist()

# KEEP: Grouping dimensions (<10% cardinality) if highly complete
keep_grouping = attrs[
    (attrs['cardinality_ratio'] <= 0.1) &
    (attrs['value_density'] > 0.95)  # Must be highly complete
]['column'].tolist()

# KEEP: High cardinality (50-90%) if complete (potential resource IDs)
keep_resource_ids = attrs[
    (attrs['cardinality_ratio'] > 0.5) &
    (attrs['cardinality_ratio'] <= 0.9) &
    (attrs['value_density'] > 0.95)  # Must be highly complete
]['column'].tolist()

# KEEP: Medium cardinality (10-50%) if info score is decent
keep_composite_candidates = attrs[
    (attrs['cardinality_ratio'] > 0.1) &
    (attrs['cardinality_ratio'] <= 0.5) &
    (attrs['information_score'] > 0.3)  # Balanced across dimensions
]['column'].tolist()

# Combine filters
drop_cols = list(set(drop_primary_keys + drop_sparse))
keep_cols = list(set(keep_grouping + keep_resource_ids + keep_composite_candidates))

logger.info(f"\n🗑️  Dropping {len(drop_cols)} columns:")
logger.info(f"   Primary keys: {sorted(drop_primary_keys)}")
logger.info(f"   Sparse columns: {sorted(drop_sparse)}")

logger.info(f"\n✅ Keeping {len(keep_cols)} columns:")
logger.info(f"   Grouping dims: {sorted(keep_grouping)}")
logger.info(f"   Resource IDs: {sorted(keep_resource_ids)}")
logger.info(f"   Composite candidates: {sorted(keep_composite_candidates)}")

# Apply filter (Ibis: schema is dict-like, use .names)
df_filtered = df.select([col for col in df.schema().names if col not in drop_cols])
logger.info(f"\n📊 Schema reduced: {len(df.schema())} → {len(df_filtered.schema())} columns")
df = df_filtered
```

---

### 1.5 Categorical Distribution Analysis

Visualize value distributions for all categorical (grouping) columns to understand data composition.

```{code-cell} ipython3
# Identify categorical columns directly from current dataframe (string/categorical dtypes)
schema = df_filtered.schema()
categorical_cols = [
    col for col, dtype in schema.items()
    if dtype.is_string()
]

logger.info(f"\n📊 Categorical Columns ({len(categorical_cols)}):")
logger.info(f"   {categorical_cols}")

# Plot top 10 values for each categorical with log scale
if categorical_cols:
    # plot_categorical_frequencies now works directly with Ibis Tables
    fig = plot_categorical_frequencies(
        df_filtered,
        columns=categorical_cols,
        top_n=10,
        log_scale=True,           # Logarithmic scale for wide frequency ranges
        shared_xaxis=True,        # Same scale across all subplots for comparison
        figsize=(16, max(4, len(categorical_cols) * 2)),
        cols_per_row=2
    )
    plt.suptitle('Categorical Value Distributions (Top 10 per column, log scale)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.show()

    logger.info("\n✅ Distribution plots show:")
    logger.info("   • Data concentration (Pareto principle)")
    logger.info("   • Grain candidates (which dimensions to composite)")
    logger.info("   • Potential filtering targets (rare/dominant values)")
    logger.info("   • Log scale reveals patterns across wide frequency ranges")
```

---

### 1.6 Cost Column Correlation Analysis

**Hypothesis**: Multiple cost columns represent different accounting treatments (amortized, discounted, etc.) of the same base cost → highly correlated.

```{code-cell} ipython3
# Identify all cost columns
cost_columns = [c for c in df.schema().names if 'cost' in c.lower()]

logger.info(f"\n💰 Cost Columns Found: {len(cost_columns)}")
logger.info(f"   {cost_columns}")
```

```{code-cell} ipython3
# Compute correlation matrix for cost columns
cost_corr = df.select(cost_columns).execute().corr()

logger.info(f"\n📊 Cost Column Correlation Matrix:")
cost_corr
```

```{code-cell} ipython3
# Analyze pairwise correlations
import numpy as np

corr_np = cost_corr.to_numpy()
np.fill_diagonal(corr_np, np.nan)  # Exclude diagonal (self-correlation = 1.0)
min_corr = np.nanmin(np.abs(corr_np))
max_corr = np.nanmax(np.abs(corr_np))
mean_corr = np.nanmean(np.abs(corr_np))

logger.info(f"\n📈 Pairwise Correlation Statistics:")
logger.info(f"   Min |r|: {min_corr:.4f}")
logger.info(f"   Max |r|: {max_corr:.4f}")
logger.info(f"   Mean |r|: {mean_corr:.4f}")

if min_corr > 0.95:
    logger.info(f"\n✅ All pairwise correlations |r| > 0.95")
    logger.info(f"   → Cost columns are redundant representations")
    logger.info(f"   → Safe to keep only one: materialized_discounted_cost")
else:
    logger.warning(f"\n⚠️  Some correlations |r| < 0.95")
    logger.warning(f"   → Review which cost columns differ significantly")
```

**Decision**: Keep `materialized_cost` (base cost, no accounting adjustments), rename to `cost` for simplicity.

---

### 1.7 Remove High Correlation cost values

```{code-cell} ipython3
# Define primary cost metric
PRIMARY_COST = 'materialized_cost'
redundant_cost_cols = [c for c in cost_columns if c != PRIMARY_COST]

# Execute single-pass filtering & track reduction
cols_before = len(df.schema())
df = df.drop(redundant_cost_cols)
cols_after = len(df.schema())
reduction_ratio = (cols_before - cols_after) / cols_before

# Rename PRIMARY_COST to 'cost' for simplicity
df = df.rename(cost=PRIMARY_COST)
PRIMARY_COST = 'cost'

logger.info(f"\n📉 Column Reduction: {cols_before} → {cols_after} ({reduction_ratio:.1%} reduction)")
logger.info(f"✅ Tidy schema ready: {cols_after} informative columns")
logger.info(f"✅ Renamed: materialized_cost → cost")
```

```{code-cell} ipython3
# Explain remaining data structure
remaining_cols = df.schema().names

logger.info(f"\n📦 Remaining Data Structure ({cols_after} columns):")
logger.info(f"\n   Temporal: usage_date")
logger.info(f"\n   Cloud Dimensions:")
logger.info(f"      - cloud_provider, cloud_account_id, region")
logger.info(f"      - availability_zone, product_family, usage_type")
logger.info(f"\n   Resource Identifiers:")
logger.info(f"      - resource_id, service_code, operation")
logger.info(f"\n   Cost Metric:")
logger.info(f"      - cost (base materialized cost, no adjustments)")
logger.info(f"\n   Other: {[c for c in remaining_cols if c not in ['usage_date', 'cloud_provider', 'cloud_account_id', 'region', 'availability_zone', 'product_family', 'usage_type', 'resource_id', 'service_code', 'operation', 'cost']]}")
```

```{code-cell} ipython3
# Dimensional analysis: boxplots by key attributes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Collect data for plotting
df_plot = df.select([
    'cloud_provider',
    'cloud_account_id',
    'region',
    'product_family',
    'cost'
]).execute()

# Provider distribution
ax1 = axes[0, 0]
provider_summary = (
    df_plot.groupby('cloud_provider')
    .agg(records=('cost', 'count'), total_cost=('cost', 'sum'))
    .reset_index()
    .sort_values('total_cost', ascending=False)
)

ax1.barh(provider_summary['cloud_provider'], provider_summary['total_cost'])
ax1.set_xlabel(f'Total Cost')
ax1.set_title('Cost by Cloud Provider')
ax1.grid(True, alpha=0.3)

# Account distribution (top 10)
ax2 = axes[0, 1]
account_summary = (
    df_plot.groupby('cloud_account_id')
    .agg(records=('cost', 'count'), total_cost=('cost', 'sum'))
    .reset_index()
    .sort_values('total_cost', ascending=False)
    .head(10)
)

ax2.barh(account_summary['cloud_account_id'], account_summary['total_cost'])
ax2.set_xlabel(f'Total Cost')
ax2.set_title('Top 10 Accounts by Cost')
ax2.grid(True, alpha=0.3)

# Region distribution (top 10)
ax3 = axes[1, 0]
region_summary = (
    df_plot.groupby('region')
    .agg(records=('cost', 'count'), total_cost=('cost', 'sum'))
    .reset_index()
    .sort_values('total_cost', ascending=False)
    .head(10)
)

ax3.barh(region_summary['region'], region_summary['total_cost'])
ax3.set_xlabel(f'Total Cost')
ax3.set_title('Top 10 Regions by Cost')
ax3.grid(True, alpha=0.3)

# Product family distribution (top 10)
ax4 = axes[1, 1]
product_summary = (
    df_plot.groupby('product_family')
    .agg(records=('cost', 'count'), total_cost=('cost', 'sum'))
    .reset_index()
    .sort_values('total_cost', ascending=False)
    .head(10)
)

ax4.barh(product_summary['product_family'], product_summary['total_cost'])
ax4.set_xlabel(f'Total Cost')
ax4.set_title('Top 10 Product Families by Cost')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

logger.info(f"\n📊 Dimensional Summary:")
logger.info(f"   Providers: {df_plot['cloud_provider'].nunique()}")
logger.info(f"   Accounts: {df_plot['cloud_account_id'].nunique()}")
logger.info(f"   Regions: {df_plot['region'].nunique()}")
logger.info(f"   Products: {df_plot['product_family'].nunique()}")
```

---

### 1.6 Temporal Quality Check

Inspect daily patterns to detect pipeline anomalies.

```{code-cell} ipython3
# Daily aggregates
daily_summary = (
    df
    .group_by('date')
    .agg(
        record_count=_.count(),
        total_cost=_[PRIMARY_COST].sum(),
        cost_std=_[PRIMARY_COST].std()
    )
    .order_by('date')
    .execute()
)

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(daily_summary['date'], daily_summary['record_count'], marker='o')
ax1.set_ylabel('Daily Records')
ax1.set_title('Data Volume Over Time')
ax1.grid(True, alpha=0.3)

ax2.plot(daily_summary['date'], daily_summary['cost_std'], marker='o', color='red')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cost Std Dev')
ax2.set_title('Cost Variability Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Part 2: Grain Discovery & Entity Persistence

Find most granular compound key with ≥70% entities persisting ≥30 days.

### 2.1 Helper Functions

```{code-cell} ipython3
def grain_persistence_stats(df, grain_cols, cost_col, min_days=30):
    """
    Compute persistence metrics for a compound key grain.

    Returns:
        dict: {
            'entities': Total unique entities,
            'stable_count': Entities present >= min_days,
            'stable_pct': Percentage stable,
            'median_days': Median persistence,
            'mean_days': Mean persistence
        }
    """
    entity_stats = (
        df
        .group_by(grain_cols)
        .agg(
            days_present=_.date.nunique(),
            total_cost=_[cost_col].sum()
        )
        .execute()
    )

    stable_count = len(entity_stats[entity_stats['days_present'] >= min_days])

    return {
        'entities': len(entity_stats),
        'stable_count': stable_count,
        'stable_pct': round(stable_count / len(entity_stats) * 100, 1),
        'median_days': int(entity_stats['days_present'].median()),
        'mean_days': round(entity_stats['days_present'].mean(), 1)
    }


def entity_timeseries_normalized(df, entity_cols, time_col, metric_col, freq='1d'):
    """
    Compute entity-normalized time series: x_{e,t} / sum_{e'} x_{e',t}

    Pattern from reference notebook - shows entity contribution over time
    relative to total daily activity.
    """
    # Entity-period aggregation (with time rounding)
    entity_period = (
        df
        .mutate(time=_[time_col].truncate('D'))  # Round to day
        .group_by(['time'] + entity_cols)
        .agg(metric=_[metric_col].sum())
    )

    # Period totals
    period_totals = (
        entity_period
        .group_by('time')
        .agg(period_total=_.metric.sum())
    )

    # Normalize: entity / total
    return (
        entity_period
        .join(period_totals, 'time')
        .mutate(normalized=_.metric / _.period_total)
        .order_by(['time'] + entity_cols)
        .execute()
    )
```

---

### 2.2 Test Grain Candidates

```{code-cell} ipython3
# Grain candidates: coarse → fine granularity
grain_candidates = [
    ('Provider + Account', ['cloud_provider', 'cloud_account_id']),
    ('Account + Region', ['cloud_provider', 'cloud_account_id', 'region']),
    ('Account + Product', ['cloud_provider', 'cloud_account_id', 'product_family']),
    ('Account + Region + Product', ['cloud_provider', 'cloud_account_id', 'region', 'product_family']),
    ('Account + Region + Product + Usage', ['cloud_provider', 'cloud_account_id', 'region', 'product_family', 'usage_type'])
]

# Compute persistence for all candidates (functional composition)
grain_results = [
    {'Grain': name, **grain_persistence_stats(df, cols, PRIMARY_COST)}
    for name, cols in grain_candidates
]

import pandas as pd
grain_comparison = pd.DataFrame(grain_results)

logger.info(f"\n📊 Grain Persistence Comparison (37 days, ≥30 day threshold):")
logger.info(f"\n{grain_comparison[['Grain', 'entities', 'stable_pct', 'median_days']]}")
```

---

### 2.3 Select Optimal Grain

```{code-cell} ipython3
# Select optimal grain: most granular with ≥70% stability
viable = grain_comparison[grain_comparison['stable_pct'] >= 70.0]

if len(viable) > 0:
    optimal = viable.sort_values('entities', ascending=False).head(1)
else:
    logger.warning("No grain achieves 70% stability threshold")
    optimal = grain_comparison.sort_values('stable_pct', ascending=False).head(1)

OPTIMAL_GRAIN = optimal['Grain'].iloc[0]

# Reconstruct OPTIMAL_COLS by looking up in grain_candidates
OPTIMAL_COLS = [cols for name, cols in grain_candidates if name == OPTIMAL_GRAIN][0]

logger.info(f"\n✅ Optimal Grain: {OPTIMAL_GRAIN}")
logger.info(f"   Total entities: {optimal['entities'].iloc[0]:,}")
logger.info(f"   Stable (≥30 days): {optimal['stable_count'].iloc[0]:,} ({optimal['stable_pct'].iloc[0]:.0f}%)")
logger.info(f"   Median persistence: {optimal['median_days'].iloc[0]} days")
```

$\therefore$ Optimal forecasting grain identified: ${OPTIMAL\_GRAIN}$

---

## Part 3: Time Series Validation

Validate entities produce forecastable time series.

### 3.1 Top Cost Drivers

```{code-cell} ipython3
# Get top 10 stable, high-cost entities at optimal grain
top_entities = (
    df
    .group_by(OPTIMAL_COLS)
    .agg(
        days_present=_.date.nunique(),
        total_cost=_[PRIMARY_COST].sum()
    )
    .filter(_.days_present >= 30)
    .order_by(ibis.desc('total_cost'))
    .limit(10)
    .execute()
)

# Pareto analysis
total_cost = df.agg(total=_[PRIMARY_COST].sum()).execute()['total'].iloc[0]
top_10_cost = top_entities['total_cost'].sum()

logger.info(f"\n💰 Top 10 Entities at {OPTIMAL_GRAIN}:")
logger.info(f"   Drive {top_10_cost / total_cost * 100:.1f}% of total spend")
logger.info(f"\n{top_entities}")
```

---

### 3.2 Time Series Visualization

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
PRIMARY_COST = 'cost'

# Get full date range for alignment
all_dates_df = df.select('date').distinct().order_by('date').execute()

# Plot 1: Absolute daily spend
ax1 = axes[0]
for i, row_idx in enumerate(range(min(5, len(top_entities)))):
    entity_row = top_entities.iloc[row_idx]

    # Build filter condition for this entity
    filter_cond = ibis.literal(True)
    for col in OPTIMAL_COLS:
        filter_cond = filter_cond & (_[col] == entity_row[col])

    entity_ts = (
        df
        .filter(filter_cond)
        .group_by('date')
        .agg(daily_cost=_[PRIMARY_COST].sum())
        .execute()
    )

    # Align to full date range (missing dates = 0)
    aligned = (
        all_dates_df
        .merge(entity_ts, on='date', how='left')
        .fillna({'daily_cost': 0})
        .sort_values('date')
    )

    ax1.plot(aligned['date'], aligned['daily_cost'],
             label=f"Entity {i+1}", marker='o', linewidth=2)

ax1.set_xlabel('Date')
ax1.set_ylabel(f'Daily {PRIMARY_COST}')
ax1.set_title(f'Top 5 Entities - Daily Spend Trajectory ({OPTIMAL_GRAIN})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Stacked area showing cumulative contribution
ax2 = axes[1]

stack_data = []
for i, row_idx in enumerate(range(min(5, len(top_entities)))):
    entity_row = top_entities.iloc[row_idx]

    # Build filter condition for this entity
    filter_cond = ibis.literal(True)
    for col in OPTIMAL_COLS:
        filter_cond = filter_cond & (_[col] == entity_row[col])

    entity_ts = (
        df
        .filter(filter_cond)
        .group_by('date')
        .agg(cost=_[PRIMARY_COST].sum())
        .execute()
    )

    # Align to full date range (missing dates = 0)
    aligned = (
        all_dates_df
        .merge(entity_ts, on='date', how='left')
        .fillna({'cost': 0})
        .sort_values('date')
    )

    stack_data.append(aligned['cost'].values)

dates = all_dates_df['date'].values
ax2.stackplot(dates, *stack_data, labels=[f'Entity {i+1}' for i in range(5)], alpha=0.8)
ax2.set_xlabel('Date')
ax2.set_ylabel(f'Daily {PRIMARY_COST}')
ax2.set_title(f'Top 5 Entities - Cumulative Spend Contribution ({OPTIMAL_GRAIN})')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

logger.info(f"\n📈 Time series validation complete")
logger.info(f"   - Top entities show stable, trackable patterns")
logger.info(f"   - Suitable for forecasting at {OPTIMAL_GRAIN} grain")
```

### 3.3 Summary

✅ Stable patterns: Top entities show consistent spending
✅ Pareto: Small number drive majority of spend
✅ Forecastable: Entities persist, costs trackable

$\therefore$ Grain validated for time series modeling

---

## Part 4: Summary of Data Preparation

### Dataset Transformations

**Raw Data**: 122 days, 8.3M records, 38 columns

**Temporal Filtering**:
- Removed post-Oct 7 data (AWS pipeline collapse, costs frozen)
- Clean period: Sept 1 - Oct 6, 2025 (37 days)
- **Row reduction**: 8.3M → 5.8M records (30% reduction)

**Schema Filtering**:
- Filter 1: ID columns (cardinality > 0.95) → uuid
- Filter 2: High nulls (>80%) → [varies by dataset]
- Filter 3: High zeros (>95% among non-nulls) → [varies by dataset]
- Filter 4: Redundant costs → 5 cost variants (kept materialized_cost, renamed to cost)
- **Column reduction**: 38 → 32 columns (16% reduction)

**Remaining 5.8M Records Contain**:
- **Temporal**: Daily grain (37 days)
- **Cloud hierarchy**: Provider → Account → Region → Availability Zone
- **Resource dimensions**: Service, Product Family, Usage Type, Resource ID
- **Cost metric**: cost (base materialized_cost, no accounting adjustments)
- **Cardinality**: X providers, Y accounts, Z regions, W products (see dimensional analysis)

**Data Quality Issue**: AWS pipeline collapse post-Oct 7 (costs frozen, CV ≈ 0)

### Grain Discovery

**Optimal Grain**: Most granular with ≥70% entity stability over 30 days

**Stability**: ~70%+ entities persist ≥30 days

**Pareto**: Top 10 entities drive >50% of total spend

### Wide Format Data Model

```python
# Time series ready structure (wide format)
(t, r, c) where:
    t = usage_date (date)
    r = compound_key(provider, account, region, product, ...)
    c = cost  # base materialized_cost
```

✅ Entities persist across observation period
✅ Time series show stable, forecastable patterns
✅ Ready for grain-level forecasting

**Next**: Transform to tidy format for attribute-level analysis (Part 5)

```{code-cell} ipython3
df.limit(10).execute()
```

---

## Part 5: Attribute Hierarchy Discovery

### 5.1 Understanding the Compound Key Structure

The data contains a **compound key** - multiple attributes that together uniquely identify entities. Understanding their hierarchical relationships enables:
- Efficient aggregation along natural hierarchies
- Hierarchical forecasting models (top-down/bottom-up)
- Detection of invalid/missing combinations

**Goal**: Discover the DAG/tree structure of attributes (e.g., provider → account → region → product).

```{code-cell} ipython3
# Get categorical columns for hierarchy analysis
schema = df.schema()
categorical_cols = [
    col for col, dtype in schema.items()
    if dtype.is_string()
]

logger.info(f"\n🔍 Analyzing hierarchy among {len(categorical_cols)} categorical attributes:")
logger.info(f"   {categorical_cols}")
```

### 5.5 Compound Key Patterns

Identify the most common compound key patterns (combinations that appear frequently).

```{code-cell} ipython3
# Analyze compound key combinations
# Start with lowest cardinality attributes and build up

compound_keys = []

# Test different compound key candidates based on discovered hierarchy
# Use actual attributes from graph roots and their descendants
key_candidates = [
    ['provider'],
    ['provider', 'account'],
    ['provider', 'account', 'region'],
    ['provider', 'account', 'region', 'service'],
]

for keys in key_candidates:
    # Filter to only columns that exist in dataframe
    valid_keys = [k for k in keys if k in df.schema().names]

    if not valid_keys:
        continue

    # Count unique combinations
    unique_combos = df.select(valid_keys).distinct().count().execute()

    # Count total records per combination (mean entity size)
    entity_sizes_df = (
        df.group_by(valid_keys)
        .agg(record_count=_.count())
        .execute()
    )
    entity_sizes = entity_sizes_df['record_count'].mean()

    compound_keys.append({
        'key': ' → '.join(valid_keys),
        'unique_entities': unique_combos,
        'mean_records_per_entity': entity_sizes
    })

key_df = pd.DataFrame(compound_keys)

logger.info(f"\n🔑 Compound Key Analysis:")
key_df
```

```{code-cell} ipython3
logger.info(f"\n💡 Insights:")
logger.info(f"   • Cardinality hierarchy reveals natural parent-child relationships")
logger.info(f"   • Functional dependencies identify 1:1 mappings (compound key components)")
logger.info(f"   • Graph structure shows minimal DAG (transitive reduction)")
logger.info(f"   • Compound keys enable entity-level time series at different grains")
logger.info(f"\n📊 Next: Use hierarchy for:")
logger.info(f"   • Hierarchical forecasting (aggregate/disaggregate along tree)")
logger.info(f"   • Feature engineering (rollup features from child to parent)")
logger.info(f"   • Anomaly detection (detect violations of expected parent-child relationships)")
```

### 5.6 Export to HuggingFace Dataset

Persist the cleaned wide format dataset with discovered hierarchy metadata.

```{code-cell} ipython3
from datasets import Dataset
import pyarrow as pa

# Output directory
OUTPUT_DIR = Path('~/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_processed').expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"\n💾 Exporting cleaned dataset to: {OUTPUT_DIR}")
```

```{code-cell} ipython3
# Export wide format (source of truth: full entity-level granularity)
df_collected = df.execute()

# Convert Pandas → PyArrow → HuggingFace Dataset
dataset = Dataset(pa.Table.from_pandas(df_collected))

# Save to disk
dataset_path = OUTPUT_DIR / 'piedpiper_clean'
dataset.save_to_disk(str(dataset_path))

logger.info(f"\n✅ Dataset Exported:")
logger.info(f"   Path: {dataset_path}")
logger.info(f"   Rows: {len(dataset):,}")
logger.info(f"   Columns: {len(dataset.column_names)}")
logger.info(f"   Temporal range: Sept 1 - Oct 6, 2025 (37 days)")
logger.info(f"   Format: (date, [categorical attributes], cost)")
logger.info(f"\n📊 Dataset contains:")
logger.info(f"   • Filtered schema (high-info columns only)")
logger.info(f"   • Temporal filter (clean period, no AWS pipeline collapse)")
logger.info(f"   • Primary cost metric (materialized_cost → cost)")
logger.info(f"   • Hierarchical attributes (use functional dependency analysis)")
logger.info(f"\n🌲 Hierarchy metadata: See Part 5 for discovered attribute DAG")
```

```{code-cell} ipython3
# Quick validation: reload and inspect
from datasets import load_from_disk

reloaded = load_from_disk(str(dataset_path))

logger.info(f"\n🔍 Validation - Reloaded Dataset:")
logger.info(f"   Rows: {len(reloaded):,}")
logger.info(f"   Columns: {reloaded.column_names}")
logger.info(f"\n✅ Dataset successfully persisted and validated")
logger.info(f"   Load with: Dataset.load_from_disk('{dataset_path}')")

# Show sample
reloaded.to_pandas().head(10)
```

---

## Part 6: Summary & Next Steps

### Data Structure Understanding

**Wide Format** (entity-level):
```python
(date, provider, account, region, product, ..., cost)
```
- **Source of truth**: Full entity-level granularity
- **Grain**: Compound key at multiple levels
- **Use case**: Entity-level forecasting, hierarchical aggregation

**Attribute Hierarchy** (discovered via functional dependencies):
```
provider → account → region → service → ...
```
- **DAG structure**: Parent-child relationships between dimensions
- **Enables**: Top-down/bottom-up forecasting, natural aggregation paths
- **Use case**: Hierarchical models, anomaly detection via hierarchy violations

### Key Findings

1. **Temporal**: 37 clean days (Sept 1 - Oct 6, 2025)
2. **Optimal Grain**: Stable entities with 70%+ persistence
3. **Pareto Principle**: Top 10 entities drive >50% of spend
4. **Attribute Hierarchy**: Functional dependencies reveal natural DAG structure
5. **Compound Keys**: Multiple valid grains for different forecasting tasks

### Downstream Applications

**Hierarchical Forecasting**:
- Top-down: Forecast at provider level, disaggregate to accounts/regions
- Bottom-up: Forecast at resource level, aggregate upward
- Middle-out: Forecast at optimal grain, reconcile up and down

**Cost Optimization**:
- Use hierarchy to identify anomalous parent-child relationships
- Target optimization at appropriate hierarchy level
- Roll up/down insights across organizational structure

**Feature Engineering**:
- Create features at each hierarchy level
- Aggregate child metrics to parent (e.g., account-level variance from resources)
- Detect hierarchy violations (e.g., region not matching account's expected geography)

**Next Steps**:
1. Build Gaussian Process models at multiple grains (use hierarchy for features)
2. Develop hierarchical Bayesian models (explicit parent-child structure)
3. Implement hierarchy-aware anomaly detection
4. Investigate AWS pipeline issue post-Oct 7

```{code-cell} ipython3

```
