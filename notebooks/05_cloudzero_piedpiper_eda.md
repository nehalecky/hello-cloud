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

# CloudZero PiedPiper Dataset - Exploratory Data Analysis

## Background


* **Dataset**: PiedPiper optimized daily billing data
* **Coverage**: September 1 - December 31, 2025 (122 days)
* **Records**: 8,336,995 rows × 38 columns
* **Format**: SNAPPY-compressed Parquet (0.96 GB)

This analysis provides a rigorous exploration of the PiedPiper billing dataset, establishing a foundation for subsequent modeling and forecasting efforts. We proceed methodically, building knowledge progressively.

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
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import time

# Import our custom EDA utilities
from cloud_sim.utils import (
    comprehensive_schema_analysis,
    calculate_attribute_scores,
    time_normalized_size,
    entity_normalized_by_day,
    smart_sample,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    create_info_score_chart,
    create_correlation_heatmap,
)

# Configure visualization libraries
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("✓ Libraries loaded and configured")
```

```{code-cell} ipython3
# Dataset path
DATA_PATH = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet')

# Verify file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Load as LazyFrame for efficient operations
df = pl.scan_parquet(DATA_PATH)

# Collect basic statistics
total_rows = df.select(pl.len()).collect()[0, 0]
schema = df.collect_schema()

print(f"✓ Dataset loaded: {DATA_PATH.name}")
print(f"  Dimensions: {total_rows:,} rows × {len(schema)} columns")
print(f"  File size: {DATA_PATH.stat().st_size / (1024**3):.2f} GB")
```

```{code-cell} ipython3
# Performance benchmark - establish computational constraints
print("Running performance benchmarks on full dataset...")
benchmarks = []

# Test aggregation speed
start = time.time()
_ = df.group_by('cloud_provider').agg(pl.col('materialized_discounted_cost').sum()).collect()
benchmarks.append(('Groupby aggregation (full)', time.time() - start))

# Test correlation on sample
start = time.time()
cost_cols = [col for col in schema.names() if 'cost' in col.lower()]
_ = df.head(100_000).select(cost_cols).collect().corr()
benchmarks.append(('Correlation (100K sample)', time.time() - start))

# Test full scan
start = time.time()
_ = df.select(pl.len()).collect()
benchmarks.append(('Full table scan', time.time() - start))

benchmark_df = pl.DataFrame({
    'operation': [b[0] for b in benchmarks],
    'seconds': [round(b[1], 3) for b in benchmarks]
})

print("\nPerformance Results:")
print(benchmark_df)
print("\n✓ Benchmarking complete - dataset is manageable for full analysis")
```

---

## Part 1: Comprehensive Schema Analysis

Having established our computational environment, we now turn to understanding the dataset structure. We seek to identify all 38 attributes, their data types, completeness, and cardinality - providing a complete picture without truncation.

```{code-cell} ipython3
# Generate comprehensive schema analysis
schema_analysis = comprehensive_schema_analysis(df)

# Display ALL columns (no truncation)
with pl.Config(tbl_rows=-1, tbl_width_chars=200):
    display(schema_analysis)
```

### Dataset Summary Statistics

We now separate numeric and categorical columns for appropriate statistical summaries. Columns with >95% null values are excluded as they provide minimal analytical value.

**Numeric columns** receive distribution statistics (min, max, quartiles, mean, std), enabling us to understand value ranges, central tendency, and dispersion.

**Categorical columns** receive cardinality analysis, entropy scores, and top value identification, enabling us to understand the diversity and concentration of categorical values.

```{code-cell} ipython3
# Import analysis and visualization functions
from cloud_sim.utils import (
    semantic_column_analysis,
    numeric_column_summary,
    categorical_column_summary,
    plot_numeric_distributions,
    plot_categorical_frequencies
)
```

### Semantic Column Analysis

Before examining distributions, we infer semantic meaning from column names. This helps us understand what each column represents and sets expectations for data characteristics.

```{code-cell} ipython3
# Infer semantic meaning from column names
print("=" * 80)
print("SEMANTIC COLUMN ANALYSIS")
print("=" * 80)
semantic_analysis = semantic_column_analysis(df)

print(f"\n{len(semantic_analysis)} columns analyzed")

# Group by semantic category
category_summary = semantic_analysis.group_by('semantic_category').agg([
    pl.len().alias('count'),
    pl.col('column').alias('columns')
]).sort('count', descending=True)

print("\nSemantic Categories:")
print(category_summary.select(['semantic_category', 'count']))

# Display full semantic analysis
print("\nFull Semantic Analysis:")
with pl.Config(tbl_rows=-1, tbl_width_chars=250):
    display(semantic_analysis)
```

### Numeric Column Summary

Now we generate distribution statistics for numeric columns, informed by our semantic understanding:

```{code-cell} ipython3
# Generate numeric summary (excludes columns with >95% nulls)
print("=" * 80)
print("NUMERIC COLUMNS SUMMARY")
print("=" * 80)
numeric_summary = numeric_column_summary(df, null_threshold=95.0)

print(f"\n{len(numeric_summary)} numeric columns (after filtering >95% null)")
with pl.Config(tbl_rows=-1, tbl_width_chars=250, fmt_float='mixed'):
    display(numeric_summary)
```

```{code-cell} ipython3
# Generate categorical summary (excludes columns with >95% nulls)
print("=" * 80)
print("CATEGORICAL COLUMNS SUMMARY")
print("=" * 80)
categorical_summary = categorical_column_summary(df, null_threshold=95.0)

print(f"\n{len(categorical_summary)} categorical columns (after filtering >95% null)")
with pl.Config(tbl_rows=-1, tbl_width_chars=250):
    display(categorical_summary)
```

### Initial Observations

From semantic analysis and statistical summaries, we observe:

**Semantic Understanding:**
- **Financial metrics** (8 columns): Cost columns with different accounting treatments (discounted, amortized, invoiced)
  - *Critical task*: Identify ONE primary cost column, drop redundant ones
- **Cloud hierarchy** (multiple): Provider, account, product, service, region identifiers
- **Kubernetes overlay** (sparse): Container metadata with expected high nulls
- **Identifiers** (high cardinality): UUIDs, resource IDs for granular tracking

**Data Quality Concerns:**
1. **Negative cost values** (min = -524.54): Financial columns show negatives, likely refunds/credits
2. **Near-zero concentrations**: Q25 values ~10^-7 indicate many zero or near-zero records
3. **Extreme right skew**: Max ~97K vs Q75 ~2.8 suggests heavy-tailed distributions
4. **Column redundancy**: 8 cost columns likely highly correlated - need to select primary metric

**Rigorous Next Steps:**
1. **Part 3 - Information Scoring**: Quantify which columns carry meaningful signal
2. **Column Filtering**: Use correlation + information scores to identify ONE cost column, drop redundant
3. **Part 4+ - Detailed Analysis**: Only AFTER filtering, visualize and deeply analyze the informative subset

We defer all visualization until column selection is complete. Plotting 8 redundant cost columns wastes effort.

---

## Part 2: Conceptual Model & Assumptions

Having no established mental model of CloudZero's data structure demands we speculate as to the schema, hierarchy, and aggregation characteristics before deep analysis. We conceptualize a billing event space $\mathbf{B}_0$, representing the complete universe of cloud resource consumption events across all infrastructure.

### The Cloud Billing Hierarchy

Cloud billing data naturally follows a hierarchical structure:

$$\text{Cloud Provider} \rightarrow \text{Account} \rightarrow \text{Product Family} \rightarrow \text{Service} \rightarrow \text{Resource}$$

This hierarchy reflects both organizational structure (accounts) and technical architecture (products, services, resources). We expect attributes in our dataset supporting each level.

### Kubernetes Overlay

Container orchestration introduces an orthogonal dimension - infrastructure-level billing enriched with application-level context:

$$\text{Cluster} \rightarrow \text{Namespace} \rightarrow \text{Pod} \rightarrow \text{Container}$$

This overlay enables cost attribution to applications, teams, or workloads, beyond traditional infrastructure boundaries.

### Assumptions

We know that cloud billing systems aggregate raw consumption events (compute seconds, storage bytes, network transfers) into daily summaries for practical data management. Hence, we observe not raw events but **pre-aggregated daily records**.

Such aggregation introduces several considerations:

1. **Temporal resolution**: Daily granularity implies loss of intraday patterns. We cannot observe hour-of-day seasonality or sub-daily bursts.

2. **Multiple cost views**: Different "cost" fields likely represent:
   - **On-demand cost**: List pricing without discounts
   - **Discounted cost**: With commitment discounts (reserved instances, savings plans)
   - **Amortized cost**: Spreading upfront commitments over time periods

3. **Sampling effects**: The `aggregated_records` field suggests each row represents multiple underlying events, introducing potential sampling considerations.

4. **Kubernetes metadata**: We expect K8s fields to be sparse (present only for containerized workloads, typically 10-30% of infrastructure).

### Expected Event Space

We conceptualize our observations as containing these fundamental dimensions:

- $t$: temporal dimension (date of resource consumption)
- $a$: account identifier (billing entity)
- $s$: service/product (what was consumed)
- $r$: resource identifier (specific instance/volume/cluster)
- $c$: cost metric (monetary value under various accounting treatments)
- $u$: usage metric (quantity consumed)
- $k$: Kubernetes context (cluster, namespace, pod) - optional enrichment

Thus, our expected event space $\mathbf{B}$ having dimensions $(t, a, s, r, c, u, k)$, and we seek to identify attributes in the dataset that support this conceptual model.

### Unit Economics - The FinOps Lens

We notice that this model contains no direct **unit economics** metric - the core of FinOps analysis. Such metrics must be derived from the relationship between $c$ and $u$:

$$\phi = \frac{c}{u} \quad \text{(cost per unit of consumption)}$$

This represents resource efficiency - the fundamental question in cloud financial management: *Are we getting value for money?*

Now, armed with our conceptual model and expectations, we proceed forward to test how well reality aligns with theory. Patience is a virtue.

---

## Part 3: Intelligent Feature Selection

We now apply information theory and correlation analysis to identify the most valuable columns and eliminate redundancy. This streamlines all subsequent analysis.

### Step 1: Information Scoring

We score all attributes using harmonic mean of three metrics:
- **Value density**: Non-null percentage
- **Cardinality ratio**: Unique values / total records
- **Shannon entropy**: Information content

```{code-cell} ipython3
# Calculate attribute scores (samples 100K for entropy calculation)
print("Computing attribute information scores...")
attribute_scores = calculate_attribute_scores(df, sample_size=100_000)

print(f"✓ Scored {len(attribute_scores)} attributes")
print(f"  Score range: [{attribute_scores['information_score'].min():.6f}, {attribute_scores['information_score'].max():.6f}]")

# Show top performers only
print("\nTop 15 Most Informative Attributes:")
with pl.Config(tbl_rows=15):
    display(attribute_scores.head(15))
```

### Step 2: Correlation-Based Redundancy Removal

For numeric columns, we identify highly correlated pairs and keep only the one with higher information score.

```{code-cell} ipython3
# Get numeric columns with >5% information score
numeric_cols = numeric_summary.filter(
    pl.col('column').is_in(
        attribute_scores.filter(pl.col('information_score') > 0.05)['attribute'].to_list()
    )
)['column'].to_list()

print(f"Analyzing {len(numeric_cols)} numeric columns for correlation...")

# Compute correlation matrix (stratified sample for efficiency)
sample_df = smart_sample(df, n=100_000, stratify_col='cloud_provider')
corr_matrix = sample_df.select(numeric_cols).corr()

print("✓ Correlation matrix computed")

# Find highly correlated pairs (|r| > 0.90)
CORR_THRESHOLD = 0.90
columns_to_drop = set()

corr_pandas = corr_matrix.to_pandas()
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        col_i = numeric_cols[i]
        col_j = numeric_cols[j]
        corr_val = abs(corr_pandas.iloc[i, j])

        if corr_val > CORR_THRESHOLD:
            # Get information scores
            score_i = attribute_scores.filter(pl.col('attribute') == col_i)['information_score'][0]
            score_j = attribute_scores.filter(pl.col('attribute') == col_j)['information_score'][0]

            # Drop the one with lower information score
            if score_i > score_j:
                columns_to_drop.add(col_j)
                print(f"  Dropping {col_j} (r={corr_val:.3f} with {col_i}, lower info score)")
            else:
                columns_to_drop.add(col_i)
                print(f"  Dropping {col_i} (r={corr_val:.3f} with {col_j}, lower info score)")

print(f"\n✓ Identified {len(columns_to_drop)} redundant numeric columns to drop")
```

### Step 3: Create Final Column Set

```{code-cell} ipython3
# Start with all columns
all_cols = df.collect_schema().names()

# Remove low-information columns (score < 0.01)
low_info_cols = attribute_scores.filter(
    pl.col('information_score') < 0.01
)['attribute'].to_list()

# Remove high-null columns (>95% null)
high_null_cols = schema_analysis.filter(
    pl.col('null_pct') > 95.0
)['column'].to_list()

# Combine all exclusions
all_drops = set(columns_to_drop) | set(low_info_cols) | set(high_null_cols)
final_cols = [col for col in all_cols if col not in all_drops]

# Create filtered dataframe
df_filtered = df.select(final_cols)

print("="*70)
print("FEATURE SELECTION SUMMARY")
print("="*70)
print(f"  Original columns: {len(all_cols)}")
print(f"  Dropped (low information): {len(low_info_cols)}")
print(f"  Dropped (high nulls): {len(high_null_cols)}")
print(f"  Dropped (correlation redundancy): {len(columns_to_drop)}")
print(f"  FINAL COLUMN SET: {len(final_cols)}")
print("="*70)

# Show kept columns by category
print("\nRetained columns by semantic category:")
for category in ['financial', 'cloud_hierarchy', 'identifier', 'temporal', 'consumption', 'kubernetes']:
    category_cols = [col for col in final_cols
                     if col in semantic_analysis.filter(pl.col('semantic_category') == category)['column'].to_list()]
    if category_cols:
        print(f"\n  {category.upper()} ({len(category_cols)}):")
        for col in sorted(category_cols):
            score = attribute_scores.filter(pl.col('attribute') == col)['information_score']
            if len(score) > 0:
                print(f"    - {col} (info: {score[0]:.4f})")
```

```{code-cell} ipython3
# Visualize final correlation matrix (should show low redundancy)
final_numeric_cols = [col for col in final_cols if col in numeric_cols and col not in columns_to_drop]

if len(final_numeric_cols) > 1:
    final_corr = sample_df.select(final_numeric_cols).corr()

    fig = create_correlation_heatmap(
        final_corr,
        title=f'Final Numeric Features Correlation Matrix ({len(final_numeric_cols)} cols)',
        annotate=True,
        figsize=(10, 8)
    )
    plt.show()

    # Verify low redundancy
    corr_np = final_corr.to_numpy()
    np.fill_diagonal(corr_np, 0)  # Ignore diagonal
    max_corr = np.abs(corr_np).max()
    print(f"\n✓ Maximum absolute correlation in final set: {max_corr:.3f}")
    if max_corr < CORR_THRESHOLD:
        print(f"  → Successfully reduced redundancy below {CORR_THRESHOLD} threshold")
```

---

## Part 4: Quick Cardinality Overview

Now working with our streamlined column set, we quickly examine cardinality patterns to understand grouping capabilities.

```{code-cell} ipython3
# Cardinality distribution of final columns
final_schema = schema_analysis.filter(pl.col('column').is_in(final_cols))

cardinality_summary = final_schema.group_by('card_class').agg([
    pl.len().alias('count'),
    pl.col('column').alias('columns')
]).sort('count', descending=True)

print("Cardinality Distribution (Final Column Set):")
print(cardinality_summary.select(['card_class', 'count']))

# Show key columns by cardinality class
for card_class in ['Low', 'Medium', 'High']:
    examples = final_schema.filter(pl.col('card_class') == card_class).head(5)
    print(f"\n{card_class} Cardinality Examples:")
    print(examples.select(['column', 'cardinality_ratio', 'null_pct']))
```

---

## Part 5: Temporal Quality & Patterns

Quick temporal validation using our filtered dataset.

```{code-cell} ipython3
# Date coverage check
date_range = df_filtered.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date'),
    pl.col('usage_date').n_unique().alias('unique_dates')
]).collect()

min_date = date_range['min_date'][0]
max_date = date_range['max_date'][0]
expected_days = (max_date - min_date).days + 1

print("Temporal Coverage:")
print(f"  Range: {min_date} to {max_date}")
print(f"  Expected days: {expected_days}")
print(f"  Actual unique dates: {date_range['unique_dates'][0]}")
print(f"  ✓ Complete" if date_range['unique_dates'][0] == expected_days else "  ⚠ Gaps detected")
```

```{code-cell} ipython3
# Daily record volume and cost stability
daily_agg = df_filtered.group_by('usage_date').agg([
    pl.len().alias('record_count'),
    pl.col([col for col in final_cols if 'cost' in col.lower()][0]).sum().alias('daily_cost')
]).sort('usage_date').collect()

# Add day of week for pattern detection
daily_with_dow = daily_agg.with_columns([
    pl.col('usage_date').dt.strftime('%A').alias('day_name')
])

# Statistics
stats = daily_agg.select([
    pl.col('record_count').mean().alias('avg_records'),
    (pl.col('record_count').std() / pl.col('record_count').mean()).alias('record_cv'),
])

print("\nDaily Volume Statistics:")
print(f"  Average records/day: {stats['avg_records'][0]:,.0f}")
print(f"  Coefficient of variation: {stats['record_cv'][0]:.4f}")

# Cost autocorrelation (lag-1)
from scipy.stats import pearsonr
cost_series = daily_agg['daily_cost'].to_numpy()
lag1_corr, _ = pearsonr(cost_series[:-1], cost_series[1:])
print(f"\nCost Autocorrelation (lag-1): {lag1_corr:.4f}")
print("  ✓ High stability" if lag1_corr > 0.7 else "  ⚠ Moderate/low stability")
```

```{code-cell} ipython3
# Visualize temporal patterns
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Daily cost trend
plot_data = daily_agg.to_pandas()
axes[0].plot(plot_data['usage_date'], plot_data['daily_cost'],
             linewidth=2, color='steelblue', marker='o', markersize=3)
axes[0].set_ylabel('Daily Cost ($)', fontweight='bold')
axes[0].set_title('Daily Cost Trend', fontweight='bold')
axes[0].grid(alpha=0.3)

# Daily record count
axes[1].plot(plot_data['usage_date'], plot_data['record_count'],
             linewidth=2, color='darkgreen', marker='o', markersize=3)
axes[1].set_xlabel('Date', fontweight='bold')
axes[1].set_ylabel('Record Count', fontweight='bold')
axes[1].set_title('Daily Record Volume', fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.show()
```

---

## Part 6: Cost Distribution Analysis

Examine the primary cost metric distribution (already filtered for redundancy in Part 3).

```{code-cell} ipython3
# Identify primary cost column from final set
primary_cost_col = [col for col in final_cols if 'cost' in col.lower()][0]
print(f"Primary cost metric: {primary_cost_col}")

# Distribution statistics
cost_series = df_filtered.select(pl.col(primary_cost_col)).collect().to_series()

percentiles = [0, 1, 10, 25, 50, 75, 90, 99, 100]
percentile_df = pl.DataFrame({
    'percentile': [f'P{p}' for p in percentiles],
    'value': [cost_series.quantile(p/100) for p in percentiles]
})

print("\nCost Distribution Percentiles:")
display(percentile_df)

# Skewness check
mean = cost_series.mean()
std = cost_series.std()
skew = ((cost_series - mean) ** 3).mean() / (std ** 3)

print(f"\nSkewness: {skew:.4f}")
print("  → Highly right-skewed" if skew > 1 else "  → Moderate skew")
print("  → Log transformation recommended for modeling" if skew > 1 else "")
```

```{code-cell} ipython3
# Visualize distribution
sample_costs = smart_sample(df_filtered, n=50_000).select(primary_cost_col).to_series().to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
sns.histplot(sample_costs, bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_xlabel('Cost ($)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title(f'{primary_cost_col} Distribution', fontweight='bold')
axes[0].grid(alpha=0.3)

# Log scale (for skewed data)
sns.histplot(sample_costs[sample_costs > 0], bins=50, kde=True, ax=axes[1],
             color='darkgreen', log_scale=True)
axes[1].set_xlabel('Cost ($, log scale)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Distribution (Log Scale)', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Quick outlier detection (IQR method)
outliers_iqr = detect_outliers_iqr(cost_series, multiplier=1.5)
n_outliers = outliers_iqr.sum()
pct_outliers = (n_outliers / len(cost_series)) * 100

print(f"Outlier Analysis (IQR, k=1.5):")
print(f"  Outliers detected: {n_outliers:,} ({pct_outliers:.2f}%)")
print(f"  → Normal for long-tailed billing data" if pct_outliers < 5 else "  → High outlier rate - investigate")
```

---

## Summary: Streamlined Foundation

**Part 0-2: Setup & Context**
- 8.3M rows × 38 columns, 122 days of production CloudZero billing data
- Conceptual model $(t, a, s, r, c, u, k)$ established

**Part 3: Intelligent Feature Selection** ✨
- Information scoring via harmonic mean (density, cardinality, entropy)
- Correlation-based redundancy removal (|r| > 0.90 threshold)
- **Result**: 38 → ~20-25 high-value columns, one primary cost metric retained

**Part 4-6: Quality Validation**
- Complete temporal coverage, high cost autocorrelation (sticky infrastructure)
- Right-skewed cost distribution → log transformation recommended
- Low correlation in final feature set → redundancy successfully eliminated

**Next**: This streamlined dataset enables efficient deep dives into hierarchical patterns, Kubernetes workloads, and unit economics.

---

_Analysis continues with focused exploration of the filtered, high-information column set..._
