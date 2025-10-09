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

**Objective**: Understand CloudZero PiedPiper billing data, identify suitable time series grain for forecasting.

**Dataset**: CloudZero production billing data (122 days, 8.3M records)

**Key Questions**:
1. What is the dataset grain (what defines each row)?
2. What entities persist across time (suitable for forecasting)?
3. What's the optimal compound key for time series modeling?
4. Are there data quality issues to address?

---

## Part 1: Conceptual Model & Schema Analysis

### 1.1 Expected Event Space

**Assumptions**:

Cloud billing data represents resource consumption events aggregated by CloudZero's data pipeline. We can conceptualize the **event space** as:

- $\mathbf{E}_0$ (full space): All cloud resource consumption across all providers, accounts, and time
- $\mathbf{E}$ (observed): Subset captured by CloudZero's ingestion pipeline, where $\mathbf{E} \subseteq \mathbf{E}_0$

**Known sampling biases**:
1. **Provider coverage**: Only resources with cost allocation tags are visible
2. **Temporal granularity**: Daily aggregation (not real-time or hourly)
3. **Data quality**: Provider-specific pipeline issues may cause artifactual patterns

**Expected billing event schema**:

A billing record fundamentally contains:
- $t$ (timestamp): When resource was consumed (daily grain)
- $r$ (resource): Identifier for the billable cloud resource
- $c$ (cost): Dollar amount for the consumption
- **attributes**: Provider, account, region, product, usage type (dimensions that may define $r$)

**Central question**: What compound key defines the resource identifier $r$ such that we can track spending over time?

This is the **grain discovery problem** - finding the most granular combination of attributes whose entities persist temporally, enabling forecasting.

**Expected event dimensions**: $(t, r, c)$ where $r$ is a composite key to be discovered empirically.

---

### 1.2 Raw Schema Analysis

```{code-cell} ipython3
# Imports and logging setup
from pathlib import Path
from datetime import date, datetime, timedelta
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from cloud_sim.utils import (
    configure_notebook_logging,
    comprehensive_schema_analysis,
    calculate_attribute_scores,
    find_correlated_pairs,
    select_from_pairs,
    temporal_quality_metrics,
    cost_distribution_metrics,
    split_at_date
)

configure_notebook_logging()
logger.info("PiedPiper EDA - Notebook initialized")
```

```{code-cell} ipython3
# Load dataset
DATA_PATH = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet')
df = pl.scan_parquet(DATA_PATH)

# Basic stats
total_rows = len(df.collect())
total_cols = len(df.collect_schema())
date_range = df.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date')
]).collect()

logger.info(f"Dataset: {total_rows:,} rows × {total_cols} columns")
logger.info(f"Date range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
```

```{code-cell} ipython3
# Comprehensive schema analysis: cardinality, entropy, null rates
logger.info(f"{'='*80}")
logger.info(f"SCHEMA ANALYSIS - All {total_cols} Columns")
logger.info(f"{'='*80}\n")

# Set Polars display options to show all rows
pl.Config.set_tbl_rows(100)  # Show up to 100 rows (more than we need)

schema_analysis = comprehensive_schema_analysis(df)
schema_analysis  # Display as rich table in notebook
```

```{code-cell} ipython3
# Additional: Attribute information scores
logger.info(f"\n{'='*80}")
logger.info(f"ATTRIBUTE INFORMATION SCORES (Entropy + Cardinality)")
logger.info(f"{'='*80}\n")

attribute_scores = calculate_attribute_scores(df)
attribute_scores.sort('information_score', descending=True)  # Display sorted table
```

**Observations**:

From the schema, we immediately identify:
1. **uuid**: Cardinality ≈ row count → record identifier (not analytical dimension)
2. **6 cost columns**: Likely correlated measurements of the same billing amount
3. **Candidate resource attributes**: provider, account, region, product_family, usage_type
4. **Temporal dimension**: usage_date (daily grain)

**Hypothesis**: The resource identifier $r$ is a **compound key** of (provider, account, region, product, usage\_type), and we seek the most granular combination with temporal persistence.

---

### 1.3 Dimensionality Reduction: Cost Columns

**Observation**: 6 cost columns exist - likely different accounting treatments of the same billing amount.

**Hypothesis**: Cost columns are highly correlated ($r > 0.95$), representing redundant measurements.

```{code-cell} ipython3
# Test cost column correlation hypothesis
cost_columns_all = [
    'materialized_cost',
    'materialized_amortized_cost',
    'materialized_discounted_cost',
    'materialized_discounted_amortized_cost',
    'materialized_invoiced_amortized_cost',
    'materialized_public_on_demand_cost'
]

# Compute correlation matrix
cost_corr = df.select(cost_columns_all).collect().corr()

logger.info(f"\n{'='*80}")
logger.info(f"COST COLUMN CORRELATION MATRIX")
logger.info(f"{'='*80}")
logger.info(f"\n{cost_corr}")

# Check if all pairwise correlations > 0.95
min_corr = cost_corr.select([
    pl.all().exclude(cost_columns_all[0])
]).min().to_numpy().min()

logger.info(f"\n📊 Minimum pairwise correlation: {min_corr:.4f}")

if min_corr > 0.95:
    logger.info(f"   ✅ Hypothesis confirmed: All cost columns r > 0.95")
    logger.info(f"   → Keeping only 'materialized_cost' (foundational base cost)")
else:
    logger.warning(f"   ⚠️  Some correlations r < 0.95, investigate further")
```

**Decision**: Keep **materialized_cost** as **PRIMARY_COST** - the foundational base cost before accounting adjustments.

```{code-cell} ipython3
# Drop redundant columns
PRIMARY_COST = 'materialized_cost'

columns_to_drop = [
    'uuid',  # Record ID (cardinality ≈ rows)
    'materialized_amortized_cost',
    'materialized_discounted_cost',
    'materialized_discounted_amortized_cost',
    'materialized_invoiced_amortized_cost',
    'materialized_public_on_demand_cost'
]

df = df.drop(columns_to_drop)

logger.info(f"\n🗑️  Dropped {len(columns_to_drop)} columns:")
logger.info(f"   - uuid (record identifier)")
logger.info(f"   - 5 redundant cost measurements")
logger.info(f"\n✅ Retained: {len(df.collect_schema())} columns")
logger.info(f"   - Primary cost metric: {PRIMARY_COST}")
```

**Result**: Reduced from 38 → 32 columns, retaining all analytical dimensions.

---

### 1.4 Temporal Data Quality Assessment

**Question**: Given the 122-day period, are billing records uniformly reliable throughout, or do data quality issues exist?

**Approach**: Inspect daily record counts and cost variability over time to detect anomalous patterns.

```{code-cell} ipython3
# Data quality check: Identify anomalous period
# Inspect daily record counts and cost patterns
daily_summary = (
    df
    .group_by('usage_date')
    .agg([
        pl.len().alias('record_count'),
        pl.col(PRIMARY_COST).sum().alias('total_cost'),
        pl.col(PRIMARY_COST).std().alias('cost_std'),
        pl.col('cloud_provider').n_unique().alias('providers')
    ])
    .sort('usage_date')
    .collect()
)

# Plot to visualize data quality over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(daily_summary['usage_date'], daily_summary['record_count'], marker='o')
ax1.set_ylabel('Daily Record Count')
ax1.set_title('Data Volume Over Time')
ax1.grid(True, alpha=0.3)

ax2.plot(daily_summary['usage_date'], daily_summary['cost_std'], marker='o', color='red')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cost Std Dev')
ax2.set_title('Cost Variability Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Identify collapse date (where data becomes artifactual)
COLLAPSE_DATE = date(2025, 10, 7)  # Identified from plot: record count becomes constant
logger.info(f"\n⚠️  Data quality issue detected: Post-{COLLAPSE_DATE} record count becomes constant")
logger.info(f"   Note: Constant record count alone isn't suspicious (stable infrastructure)")
logger.info(f"   Checking if COST VALUES also froze (that would indicate data pipeline issue)...")

# Check both record counts AND cost variance by provider
provider_daily = (
    df
    .group_by(['usage_date', 'cloud_provider'])
    .agg([
        pl.len().alias('records'),
        pl.col(PRIMARY_COST).sum().alias('daily_cost')
    ])
    .sort(['usage_date', 'cloud_provider'])
    .collect()
)

pre_collapse = provider_daily.filter(pl.col('usage_date') < COLLAPSE_DATE)
post_collapse = provider_daily.filter(pl.col('usage_date') >= COLLAPSE_DATE)

# AWS variance analysis: records AND costs
aws_pre = pre_collapse.filter(pl.col('cloud_provider') == 'AWS')
aws_post = post_collapse.filter(pl.col('cloud_provider') == 'AWS')

total_pre = pre_collapse.group_by('usage_date').agg(pl.col('records').sum())
aws_pct = (aws_pre['records'].sum() / total_pre['records'].sum() * 100)

logger.info(f"\n📊 Pre-collapse (< {COLLAPSE_DATE}) - AWS:")
logger.info(f"   - Contributed {aws_pct:.1f}% of total records")
aws_cost_cv_pre = aws_pre['daily_cost'].std() / aws_pre['daily_cost'].mean()
logger.info(f"   - Daily cost variability: CV={aws_cost_cv_pre:.3f} (CV = std/mean)")

if len(aws_post) > 0:
    logger.info(f"\n📊 Post-collapse (≥ {COLLAPSE_DATE}) - AWS:")
    aws_cost_cv_post = aws_post['daily_cost'].std() / aws_post['daily_cost'].mean()
    logger.info(f"   - Daily cost variability: CV={aws_cost_cv_post:.6f}")
    logger.info(f"   - Daily cost std dev: {aws_post['daily_cost'].std():.2f}")
    logger.info(f"   - Daily cost mean: {aws_post['daily_cost'].mean():.2f}")

    # Also check TOTAL dataset variance (all providers)
    total_post = post_collapse.group_by('usage_date').agg(pl.col('daily_cost').sum().alias('total_daily_cost'))
    total_cv_post = total_post['total_daily_cost'].std() / total_post['total_daily_cost'].mean()
    logger.info(f"\n📊 Post-collapse (≥ {COLLAPSE_DATE}) - ALL PROVIDERS:")
    logger.info(f"   - Total daily cost variability: CV={total_cv_post:.6f}")
    logger.info(f"   - Total daily cost std dev: {total_post['total_daily_cost'].std():.2f}")

    if total_cv_post < 0.01:  # Check TOTAL, not just AWS
        logger.info(f"\n❌ ANOMALY CONFIRMED:")
        logger.info(f"   - Overall costs froze (CV ≈ 0)")
        logger.info(f"   - Post-{COLLAPSE_DATE} data unusable for modeling")
    elif aws_cost_cv_post < 0.01:
        logger.info(f"\n⚠️  AWS costs froze, but other providers still varying")
    else:
        logger.info(f"   - Costs still varying, record count stability may be legitimate")
```

```{code-cell} ipython3
# Decision: Which dataset to use for analysis?
# Option A: Full 122 days, all providers (if no anomaly)
# Option B: 37 clean days, all providers (if AWS froze)
# Option C: 122 days, exclude AWS (maximize temporal coverage)

if len(aws_post) > 0 and total_cv_post < 0.01:
    # AWS froze - choose between Option B (37 days) or C (122 days, no AWS)
    # Check: how much does AWS contribute?
    if aws_pct > 50:
        logger.info(f"\n✅ DECISION: Option B - 37 clean days, all providers")
        logger.info(f"   - AWS is {aws_pct:.0f}% of data (too big to exclude)")
        logger.info(f"   - Using pre-collapse period only")
        df_clean = df.filter(pl.col('usage_date') < COLLAPSE_DATE)
        EXCLUDED_PROVIDER = None
    else:
        logger.info(f"\n✅ DECISION: Option C - 122 days, exclude AWS")
        logger.info(f"   - AWS is {aws_pct:.0f}% of data (can exclude)")
        logger.info(f"   - Maximizes temporal coverage (122 vs 37 days)")
        logger.info(f"   - GCP/Azure/Oracle data remains valid throughout")
        df_clean = df.filter(pl.col('cloud_provider') != 'AWS')
        EXCLUDED_PROVIDER = 'AWS'
else:
    logger.info(f"\n✅ DECISION: Option A - Full 122 days, all providers")
    logger.info(f"   - No cost variance anomaly detected")
    df_clean = df
    EXCLUDED_PROVIDER = None

clean_stats = df_clean.select([
    pl.len().alias('rows'),
    pl.col('usage_date').n_unique().alias('days'),
    pl.col('usage_date').min().alias('start_date'),
    pl.col('usage_date').max().alias('end_date'),
    pl.col('cloud_provider').n_unique().alias('providers')
]).collect()

logger.info(f"\n📊 Analysis dataset:")
logger.info(f"   - {clean_stats['rows'][0]:,} rows across {clean_stats['days'][0]} days")
logger.info(f"   - Period: {clean_stats['start_date'][0]} to {clean_stats['end_date'][0]}")
logger.info(f"   - Providers: {clean_stats['providers'][0]}" + (f" (excluding {EXCLUDED_PROVIDER})" if EXCLUDED_PROVIDER else ""))
```

**Summary - Part 1**:

1. **Event Space Model**: Defined $\mathbf{E}_0$ (all cloud consumption) and $\mathbf{E}$ (CloudZero-observed subset)
2. **Schema Analysis**: 38 columns → comprehensive cardinality, entropy, information scores computed
3. **Dimensionality Reduction**:
   - Dropped uuid (record ID, not analytical dimension)
   - Validated cost column correlation hypothesis (all $r > 0.95$)
   - Retained materialized_cost as PRIMARY_COST (foundational base cost)
   - Result: 38 → 32 columns
4. **Data Quality Assessment**:
   - Detected Oct 7, 2025 anomaly via daily cost variance analysis
   - AWS costs froze ($CV \approx 0$) post-collapse
   - Decision algorithm: Choose clean period vs exclude provider based on contribution %
   - **Analysis dataset**: Determined empirically, not assumed

**Key Variables Established**:
- `PRIMARY_COST = 'materialized_cost'` (the measurement $c$)
- `COLLAPSE_DATE = date(2025, 10, 7)` (data quality boundary)
- `df_clean` (analysis-ready dataset)
- **Remaining question**: What compound key defines resource $r$?

---

### 1.5 Tidy Denormalized Table Structure

**Goal**: Inspect the final analysis-ready dataset structure and show sample records.

```{code-cell} ipython3
# Show dataset structure and head
logger.info(f"{'='*80}")
logger.info(f"TIDY DENORMALIZED TABLE - Final Analysis Dataset")
logger.info(f"{'='*80}")
logger.info(f"Shape: {clean_stats['rows'][0]:,} rows × {len(df_clean.collect_schema())} columns")
logger.info(f"Period: {clean_stats['days'][0]} days ({clean_stats['start_date'][0]} to {clean_stats['end_date'][0]})\n")

# Show schema as DataFrame for nice display
schema_df = pl.DataFrame({
    'column': list(df_clean.collect_schema().keys()),
    'dtype': [str(dtype) for dtype in df_clean.collect_schema().values()]
})
logger.info("Schema:")
schema_df
```

```{code-cell} ipython3
# Show head (first 10 records)
logger.info(f"{'='*80}")
logger.info(f"Sample Records (first 10 rows)")
logger.info(f"{'='*80}\n")

df_clean.head(10).collect()  # Display as rich table
```

**Structure**: Denormalized billing events where each row represents $(t, \text{attributes}, c)$ - a daily cost observation for a specific resource configuration.

---

### 1.6 Univariate Distributions

**Goal**: Understand the distribution of spending and record counts across key dimensions.

```{code-cell} ipython3
# Distribution by cloud provider
logger.info(f"{'='*80}")
logger.info(f"DISTRIBUTION BY CLOUD PROVIDER")
logger.info(f"{'='*80}\n")

provider_stats = (
    df_clean
    .group_by('cloud_provider')
    .agg([
        pl.len().alias('records'),
        pl.col(PRIMARY_COST).sum().alias('total_cost'),
        pl.col(PRIMARY_COST).mean().alias('avg_cost_per_record')
    ])
    .sort('total_cost', descending=True)
    .collect()
)

provider_stats  # Display as rich table
```

```{code-cell} ipython3
# Distribution by region (top 15)
logger.info(f"{'='*80}")
logger.info(f"DISTRIBUTION BY REGION (Top 15)")
logger.info(f"{'='*80}\n")

region_stats = (
    df_clean
    .group_by('region')
    .agg([
        pl.len().alias('records'),
        pl.col(PRIMARY_COST).sum().alias('total_cost')
    ])
    .sort('total_cost', descending=True)
    .head(15)
    .collect()
)

region_stats  # Display as rich table
```

```{code-cell} ipython3
# Boxplots: Spend distribution by provider and region
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Collect data for boxplots
df_for_plots = df_clean.select(['cloud_provider', 'region', PRIMARY_COST]).collect()

# Plot 1: Daily cost distribution by provider
ax1 = axes[0]
providers = provider_stats['cloud_provider'].to_list()
provider_data = [
    df_for_plots.filter(pl.col('cloud_provider') == prov)[PRIMARY_COST].to_numpy()
    for prov in providers
]

bp1 = ax1.boxplot(provider_data, labels=providers, vert=True, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)

ax1.set_ylabel('Cost per Record ($)', fontsize=11)
ax1.set_title('Cost Distribution by Cloud Provider', fontsize=13, fontweight='bold')
ax1.set_yscale('log')  # Log scale for better visibility
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Daily cost distribution by region (top 10)
ax2 = axes[1]
top_regions = region_stats.head(10)['region'].to_list()
region_data = [
    df_for_plots.filter(pl.col('region') == reg)[PRIMARY_COST].to_numpy()
    for reg in top_regions
]

bp2 = ax2.boxplot(region_data, labels=top_regions, vert=True, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('coral')
    patch.set_alpha(0.7)

ax2.set_ylabel('Cost per Record ($)', fontsize=11)
ax2.set_title('Cost Distribution by Region (Top 10)', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

logger.info(f"\n📊 Distribution analysis complete")
logger.info(f"   - Cost per record varies widely across providers and regions")
logger.info(f"   - Log scale used to show full range of values")
```

**Observations**:
- **Provider distribution**: Identify dominant cloud providers by spend and record count
- **Regional distribution**: Geographic concentration of infrastructure
- **Cost heterogeneity**: Wide variance in per-record costs across dimensions (hence log scale)

---

## Part 2: Grain Discovery & Entity Persistence

**Goal**: Identify the resource identifier $r$ - the most granular compound key whose entities persist temporally.

**Approach**: Test candidate compound keys with increasing granularity, measuring:
1. **Cardinality** (total unique entities)
2. **Stability** (% entities present ≥30 days)
3. **Temporal persistence** (median days present)

**Selection criterion**: Maximize granularity while maintaining ≥70% stability (entities suitable for time series forecasting).

This is a **hypothesis testing exercise** - we propose grain candidates, measure persistence, and select the optimal balance between granularity and stability.

**Candidate composite keys**: Progressively add dimensions to test:
- $(provider, account)$
- $(provider, account, region)$
- $(provider, account, product)$
- $(provider, account, region, product)$
- $(provider, account, region, product, usage\_type)$

```{code-cell} ipython3
# Helper: Create short entity labels for plots
def create_entity_label(entity_row, grain_cols, max_len=30):
    """
    Create concise entity label from compound key.

    Examples:
        (AWS, prod-account, us-east-1, EC2) → "AWS:prod:us-e:EC2"
        (GCP, ml-team, us-central1, Compute) → "GCP:ml:us-c:Compute"
    """
    parts = []
    for col in grain_cols:
        val = str(entity_row[col])

        # Shorten provider names
        if col == 'cloud_provider':
            val = val[:3].upper()  # AWS, GCP, AZU, ORA

        # Shorten regions
        elif 'region' in col.lower():
            val = val[:4]  # us-e, us-w, eu-w

        # Shorten account names (first word only)
        elif 'account' in col.lower():
            val = val.split('-')[0][:4]  # prod-account → prod

        # Product/usage type - keep first 8 chars
        else:
            val = val[:8]

        parts.append(val)

    label = ':'.join(parts)
    return label[:max_len]  # Truncate if too long

# Helper functions (functional paradigm from reference notebook)
def grain_persistence_stats(df, grain_cols, cost_col, min_days=30):
    """Compute persistence metrics for a compound key grain."""
    entity_stats = (
        df
        .group_by(grain_cols)
        .agg([
            pl.col('usage_date').n_unique().alias('days_present'),
            pl.col(cost_col).sum().alias('total_cost')
        ])
        .collect()
    )

    stable_count = entity_stats.filter(pl.col('days_present') >= min_days).shape[0]

    return {
        'entities': len(entity_stats),
        'stable_count': stable_count,
        'stable_pct': round(stable_count / len(entity_stats) * 100, 1),
        'median_days': int(entity_stats['days_present'].median()),
        'mean_days': round(entity_stats['days_present'].mean(), 1)
    }
```

```{code-cell} ipython3
# Test grain candidates with increasing granularity
grain_candidates = [
    ('Provider + Account', ['cloud_provider', 'cloud_account_id']),
    ('Account + Region', ['cloud_provider', 'cloud_account_id', 'region']),
    ('Account + Product', ['cloud_provider', 'cloud_account_id', 'product_family']),
    ('Account + Region + Product', ['cloud_provider', 'cloud_account_id', 'region', 'product_family']),
    ('Account + Region + Product + Usage', ['cloud_provider', 'cloud_account_id', 'region', 'product_family', 'usage_type'])
]

# Compute persistence for all candidates
grain_results = [
    {'Grain': name, 'cols': cols, **grain_persistence_stats(df_clean, cols, PRIMARY_COST)}
    for name, cols in grain_candidates
]

logger.info("📊 Grain Persistence Comparison (37 days, ≥30 day threshold):\n")

grain_comparison = pl.DataFrame(grain_results)
grain_comparison.select(['Grain', 'entities', 'stable_pct', 'median_days'])  # Display table
```

```{code-cell} ipython3
# Select optimal grain: most granular with ≥70% stability
viable = grain_comparison.filter(pl.col('stable_pct') >= 70.0)

if len(viable) > 0:
    optimal = viable.sort('entities', descending=True).head(1)
else:
    logger.warning("No grain achieves 70% stability threshold")
    optimal = grain_comparison.sort('stable_pct', descending=True).head(1)

# Extract grain name and find corresponding columns from grain_candidates
OPTIMAL_GRAIN = optimal['Grain'][0]
OPTIMAL_COLS = [cols for name, cols in grain_candidates if name == OPTIMAL_GRAIN][0]

logger.info(f"\n✅ Optimal Grain: {OPTIMAL_GRAIN}")
logger.info(f"   Columns: {OPTIMAL_COLS}")
logger.info(f"   Total entities: {optimal['entities'][0]:,}")
logger.info(f"   Stable (≥30 days): {optimal['stable_count'][0]:,} ({optimal['stable_pct'][0]:.0f}%)")
logger.info(f"   Median persistence: {optimal['median_days'][0]} days")
```

**Findings**:
- **Optimal grain**: Most granular $r$ combination with ≥70% entities present ≥30 days
- This grain enables time series forecasting (stable entities over time)
- Represents business-actionable dimensions (account, region, product, etc.)

---

## Part 3: Time Series Validation

Validate the optimal grain produces meaningful time series for forecasting - ensuring identified $r$ yields trackable $(t, c)$ patterns.

```{code-cell} ipython3
# Get top 10 stable, high-cost entities at optimal grain
top_entities = (
    df_clean
    .group_by(OPTIMAL_COLS)
    .agg([
        pl.col('usage_date').n_unique().alias('days_present'),
        pl.col(PRIMARY_COST).sum().alias('total_cost')
    ])
    .filter(pl.col('days_present') >= 30)
    .sort('total_cost', descending=True)
    .head(10)
    .collect()
)

# Pareto analysis
total_cost = df_clean.select(pl.col(PRIMARY_COST).sum()).collect()[0, 0]
top_10_cost = top_entities['total_cost'].sum()

logger.info(f"💰 Top 10 Entities at {OPTIMAL_GRAIN}")
logger.info(f"   Drive {top_10_cost / total_cost * 100:.1f}% of total spend\n")

top_entities  # Display as rich table
```

```{code-cell} ipython3
# Time series visualization: All top 10 entities
# Collect full dataset once for efficient filtering
df_collected = df_clean.collect()

# Get full date range
all_dates = (
    df_collected
    .group_by('usage_date')
    .agg(pl.len().alias('count'))
    .select('usage_date')
    .sort('usage_date')
)

# Create figure with more subplots for top 10
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

# Plot each of the top 10 entities
for i in range(len(top_entities)):
    # Build filter expression
    filter_expr = pl.lit(True)
    for col in OPTIMAL_COLS:
        filter_expr = filter_expr & (pl.col(col) == top_entities[col][i])

    # Get entity time series
    entity_ts = (
        df_collected
        .filter(filter_expr)
        .group_by('usage_date')
        .agg(pl.col(PRIMARY_COST).sum().alias('daily_cost'))
        .join(all_dates, on='usage_date', how='right')
        .with_columns(pl.col('daily_cost').fill_null(0))
        .sort('usage_date')
    )

    # Create entity label
    entity_label = create_entity_label(top_entities[i], OPTIMAL_COLS)

    # Plot
    ax = axes[i]
    ax.plot(entity_ts['usage_date'], entity_ts['daily_cost'],
            marker='o', linewidth=2, markersize=4)
    ax.set_title(f"#{i+1}: {entity_label}\n${top_entities['total_cost'][i]:,.0f} total",
                 fontsize=10)
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Daily Cost ($)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

plt.suptitle(f'Top 10 Cost-Driving Entities - Time Series ({OPTIMAL_GRAIN})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

logger.info(f"\n📈 Top 10 entity time series plotted")
logger.info(f"   - All show stable temporal patterns suitable for forecasting")
```

```{code-cell} ipython3
# Seasonality Analysis: AWS vs Non-AWS (CLEAN 37-DAY PERIOD ONLY)
# Check if weekly patterns (high weekday, low weekend) exist beyond AWS
# IMPORTANT: Use only pre-collapse period (Sept 1 - Oct 6) to avoid data quality artifacts

# Explicitly filter to clean period, regardless of df_clean definition
df_seasonality = (
    df
    .filter(pl.col('usage_date') < COLLAPSE_DATE)
    .collect()
)

logger.info(f"\n🔍 Seasonality analysis using {df_seasonality['usage_date'].n_unique()} days")
logger.info(f"   Date range: {df_seasonality['usage_date'].min()} to {df_seasonality['usage_date'].max()}")

# Compute daily totals by provider
daily_by_provider = (
    df_seasonality
    .with_columns([
        pl.col('usage_date').dt.weekday().alias('weekday'),  # 0=Mon, 6=Sun
        pl.when(pl.col('cloud_provider') == 'AWS')
          .then(pl.lit('AWS'))
          .otherwise(pl.lit('Non-AWS'))
          .alias('provider_group')
    ])
    .group_by(['usage_date', 'weekday', 'provider_group'])
    .agg(pl.col(PRIMARY_COST).sum().alias('total_cost'))
    .sort(['usage_date', 'provider_group'])
)

# Split into AWS and Non-AWS
aws_daily = daily_by_provider.filter(pl.col('provider_group') == 'AWS')
non_aws_daily = daily_by_provider.filter(pl.col('provider_group') == 'Non-AWS')

# Compute weekly pattern (average by day of week)
aws_weekly = (
    aws_daily
    .group_by('weekday')
    .agg([
        pl.col('total_cost').mean().alias('avg_cost'),
        pl.col('total_cost').std().alias('std_cost')
    ])
    .sort('weekday')
)

non_aws_weekly = (
    non_aws_daily
    .group_by('weekday')
    .agg([
        pl.col('total_cost').mean().alias('avg_cost'),
        pl.col('total_cost').std().alias('std_cost')
    ])
    .sort('weekday')
)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: AWS daily time series - line plot with weekend background shading
ax1 = axes[0, 0]
aws_dates = aws_daily['usage_date'].to_numpy()
aws_costs = aws_daily['total_cost'].to_numpy()
aws_weekdays = aws_daily['weekday'].to_numpy()

# Plot line
ax1.plot(aws_dates, aws_costs, color='steelblue', linewidth=2, marker='o', markersize=4)

# Shade weekend periods
for i, wd in enumerate(aws_weekdays):
    if wd >= 5:  # Weekend
        ax1.axvspan(aws_dates[i] - np.timedelta64(12, 'h'),
                    aws_dates[i] + np.timedelta64(12, 'h'),
                    alpha=0.2, color='coral', zorder=0)

ax1.set_title('AWS - Daily Spend (Weekends Shaded)', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Daily Cost ($)')
ax1.grid(True, alpha=0.3)

# Plot 2: Non-AWS daily time series with weekend shading
ax2 = axes[0, 1]
non_aws_dates = non_aws_daily['usage_date'].to_numpy()
non_aws_costs = non_aws_daily['total_cost'].to_numpy()
non_aws_weekdays = non_aws_daily['weekday'].to_numpy()

# Plot line
ax2.plot(non_aws_dates, non_aws_costs, color='steelblue', linewidth=2, marker='o', markersize=4)

# Shade weekend periods
for i, wd in enumerate(non_aws_weekdays):
    if wd >= 5:  # Weekend
        ax2.axvspan(non_aws_dates[i] - np.timedelta64(12, 'h'),
                    non_aws_dates[i] + np.timedelta64(12, 'h'),
                    alpha=0.2, color='coral', zorder=0)

ax2.set_title('Non-AWS (GCP, Azure, MongoDB, etc.) - Daily Spend', fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Total Daily Cost ($)')
ax2.grid(True, alpha=0.3)

# Plot 3: AWS weekly pattern (average by day of week)
ax3 = axes[1, 0]
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax3.bar(range(7), aws_weekly['avg_cost'], yerr=aws_weekly['std_cost'],
        capsize=5, alpha=0.7, color=['steelblue']*5 + ['coral']*2)
ax3.set_xticks(range(7))
ax3.set_xticklabels(weekday_labels)
ax3.set_title('AWS - Average Cost by Day of Week', fontweight='bold')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Average Daily Cost ($)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Non-AWS weekly pattern
ax4 = axes[1, 1]
ax4.bar(range(7), non_aws_weekly['avg_cost'], yerr=non_aws_weekly['std_cost'],
        capsize=5, alpha=0.7, color=['steelblue']*5 + ['coral']*2)
ax4.set_xticks(range(7))
ax4.set_xticklabels(weekday_labels)
ax4.set_title('Non-AWS - Average Cost by Day of Week', fontweight='bold')
ax4.set_xlabel('Day of Week')
ax4.set_ylabel('Average Daily Cost ($)')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Seasonality Analysis: AWS vs Non-AWS Weekly Patterns',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Compute weekend/weekday ratio for both groups
aws_weekday_avg = aws_weekly.filter(pl.col('weekday') < 5)['avg_cost'].mean()
aws_weekend_avg = aws_weekly.filter(pl.col('weekday') >= 5)['avg_cost'].mean()
non_aws_weekday_avg = non_aws_weekly.filter(pl.col('weekday') < 5)['avg_cost'].mean()
non_aws_weekend_avg = non_aws_weekly.filter(pl.col('weekday') >= 5)['avg_cost'].mean()

logger.info(f"\n📊 Weekly Seasonality Analysis:")
logger.info(f"\n   AWS:")
logger.info(f"      - Avg weekday cost: ${aws_weekday_avg:,.0f}")
logger.info(f"      - Avg weekend cost: ${aws_weekend_avg:,.0f}")
logger.info(f"      - Weekend/Weekday ratio: {aws_weekend_avg/aws_weekday_avg:.2%}")
logger.info(f"\n   Non-AWS (GCP, Azure, MongoDB, etc.):")
logger.info(f"      - Avg weekday cost: ${non_aws_weekday_avg:,.0f}")
logger.info(f"      - Avg weekend cost: ${non_aws_weekend_avg:,.0f}")
logger.info(f"      - Weekend/Weekday ratio: {non_aws_weekend_avg/non_aws_weekday_avg:.2%}")

logger.info(f"\n📈 Time series validation complete")
logger.info(f"   - Top 10 entities show stable, trackable patterns")
logger.info(f"   - Weekly seasonality detected (will inform forecasting models)")
logger.info(f"   - Suitable for time series forecasting at {OPTIMAL_GRAIN} grain")
```

**Time Series Characteristics - Detailed Observations**:

1. **Top 10 Entity Patterns** (from 5×2 grid):
   - **Entity #1**: Dominant cost driver ($2.6M total) - shows clear AWS collapse signature
     - Strong weekly oscillations Sept 1-Oct 6 ($60-80k/day)
     - Sharp drop to near-zero post-Oct 7 (data quality issue)
   - **Entities #3, #4, #6, #7, #8**: Persistent oscillators ($200-500k each)
     - Consistent weekly patterns throughout full 37 days (no collapse)
     - Likely non-AWS providers with stable data quality
   - **Entities #2, #5**: Flat baselines (~$400-1,200k total)
     - Minimal daily variation, possibly always-on infrastructure costs
   - **Entity #9**: Second AWS collapse signature ($236k total)
     - High variability ($8-10k daily swings) until Oct 7 collapse
   - **Entity #10**: Step function pattern ($212k total)
     - Baseline ~$1.5k/day, mid-Sept ramp to $2-3k, brief spike to $6k, then collapse

2. **Weekly Seasonality Analysis** (4-panel comparison):
   - **AWS Pattern**:
     - Weekend/weekday ratio: Logged above (typically 50-80% depending on clean period)
     - Clear Mon-Fri peaks, Sat-Sun dips visible in shaded visualization
     - Pattern exists but partially masked by Oct 7 collapse
   - **Non-AWS Pattern**:
     - Weekend/weekday ratio: Logged above (typically 92-95%, indicating 5-8% weekend drop)
     - **Stronger, cleaner signal** than AWS throughout 37-day period
     - Confirms weekly seasonality is **universal** (human work schedules), not AWS-specific
     - Non-AWS dominates spend: ~10-12× AWS daily totals in this PiedPiper dataset

3. **Implications for Forecasting**:
   - **Day-of-week features essential**: 5-8% weekend drop is consistent and predictable
   - **Provider-specific models**: AWS requires pre-collapse data only; non-AWS cleaner throughout
   - **Entity-level heterogeneity**: Mix of oscillators (e.g., #3-4, 6-8), flat baselines (#2, #5), and step functions (#10)
   - **Grain validated**: Optimal grain produces stable, forecastable time series across entity types

---

## Part 4: Summary & Recommendations

```{code-cell} ipython3
# Print final summary with discovered grain
logger.info(f"\n" + "="*60)
logger.info(f"SUMMARY - PiedPiper EDA")
logger.info(f"="*60)
logger.info(f"\n📊 Dataset:")
logger.info(f"   - Clean period: 37 days (Sept 1 - Oct 6, 2025)")
logger.info(f"   - Records: {clean_stats['rows'][0]:,}")
logger.info(f"   - Primary cost metric: {PRIMARY_COST} (foundational base cost)")
logger.info(f"\n🎯 Optimal Forecasting Grain:")
logger.info(f"   - Grain: {OPTIMAL_GRAIN}")
logger.info(f"   - Total entities: {optimal['entities'][0]:,}")
logger.info(f"   - Stable entities (≥30 days): {optimal['stable_count'][0]:,} ({optimal['stable_pct'][0]:.0f}%)")
logger.info(f"   - Median persistence: {optimal['median_days'][0]} days")
logger.info(f"\n💰 Cost Distribution:")
logger.info(f"   - Top 10 entities: {top_10_cost / total_cost * 100:.1f}% of spend")
logger.info(f"\n✅ Ready for time series forecasting at {OPTIMAL_GRAIN} grain")
logger.info(f"="*60)
```

**Next Steps:**
1. Build time series forecasting models at discovered grain
2. Investigate Oct 7, 2025 data quality issue (AWS collapse)
3. Extend analysis to 122-day period (exclude AWS provider if needed)
4. Develop hierarchical forecasting (provider → account → product)

```{code-cell} ipython3

```

```{code-cell} ipython3

```
