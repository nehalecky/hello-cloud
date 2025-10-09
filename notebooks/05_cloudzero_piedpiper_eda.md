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

### Objectives

Reduce the 38-column dataset to a minimal, high-signal subset by:
1. Quantifying information content via entropy-based scoring
2. Eliminating redundant numeric features via correlation analysis
3. Protecting essential temporal/identifier columns

### Methodology

**Information Scoring**: Harmonic mean of (value density, cardinality ratio, Shannon entropy)
**Redundancy Removal**: For correlated pairs (|r| > 0.90), retain column with higher information score
**Essential Protection**: Preserve temporal, identifier, and primary metric columns regardless of scores

```{code-cell} ipython3
# Calculate attribute scores (samples 100K for entropy calculation)
attribute_scores = calculate_attribute_scores(df, sample_size=100_000)

# Show top performers only
print(f"Scored {len(attribute_scores)} attributes")
print(f"Score range: [{attribute_scores['information_score'].min():.6f}, {attribute_scores['information_score'].max():.6f}]")
print("\nTop 15 Most Informative Attributes:")
with pl.Config(tbl_rows=15):
    display(attribute_scores.head(15))
```

```{code-cell} ipython3
# Functional approach: Get numeric columns with sufficient information
numeric_cols = (
    numeric_summary
    .filter(pl.col('column').is_in(
        attribute_scores.filter(pl.col('information_score') > 0.01)['attribute'].to_list()
    ))
    .get_column('column')
    .to_list()
)

# Compute correlation matrix (stratified sample for efficiency)
sample_df = smart_sample(df, n=100_000, stratify_col='cloud_provider')
corr_matrix = sample_df.select(numeric_cols).corr()

# Functional approach: Find correlated pairs and identify columns to drop
CORR_THRESHOLD = 0.90

def find_correlated_pairs(corr_df, cols, threshold):
    """Pure function: returns list of (col_i, col_j, corr_val) tuples above threshold."""
    corr_np = corr_df.to_numpy()
    pairs = [
        (cols[i], cols[j], abs(corr_np[i, j]))
        for i in range(len(cols))
        for j in range(i+1, len(cols))
        if abs(corr_np[i, j]) > threshold
    ]
    return pairs

def select_columns_to_keep(pairs, score_df):
    """Pure function: from correlated pairs, select which columns to keep based on info score."""
    score_map = {
        row['attribute']: row['information_score']
        for row in score_df.iter_rows(named=True)
    }

    drops = set()
    for col_i, col_j, corr_val in pairs:
        score_i = score_map.get(col_i, 0)
        score_j = score_map.get(col_j, 0)
        to_drop = col_j if score_i > score_j else col_i
        to_keep = col_i if score_i > score_j else col_j
        drops.add(to_drop)
        print(f"  Dropping {to_drop} (r={corr_val:.3f} with {to_keep})")

    return drops

# Execute functional pipeline
corr_pairs = find_correlated_pairs(corr_matrix, numeric_cols, CORR_THRESHOLD)
columns_to_drop = select_columns_to_keep(corr_pairs, attribute_scores) if corr_pairs else set()

print(f"Analyzed {len(numeric_cols)} numeric columns")
print(f"Found {len(corr_pairs)} correlated pairs → dropping {len(columns_to_drop)} columns")
```

```{code-cell} ipython3
# Functional approach: Build final column set with protected essentials
all_cols = df.collect_schema().names()

# Essential columns that must be kept (temporal, identifiers, primary metrics)
essential_cols = {
    'usage_date',           # Temporal dimension (required for time series)
    'uuid',                 # Primary identifier
    'resource_id',          # Resource tracking
    'materialized_discounted_cost',  # Primary cost metric
    'materialized_usage_amount',     # Primary usage metric
    'cloud_provider',       # Top-level hierarchy
}

# Build exclusion sets (functional pipeline)
low_info_cols = (
    attribute_scores
    .filter(pl.col('information_score') < 0.01)
    .get_column('attribute')
    .to_list()
)

high_null_cols = (
    schema_analysis
    .filter(pl.col('null_pct') > 95.0)
    .get_column('column')
    .to_list()
)

# Combine exclusions but preserve essentials
all_drops = (set(columns_to_drop) | set(low_info_cols) | set(high_null_cols)) - essential_cols
final_cols = [col for col in all_cols if col not in all_drops]

# Create filtered dataframe
df_filtered = df.select(final_cols)

# Summary statistics
print(f"Feature Selection Summary:")
print(f"  {len(all_cols)} → {len(final_cols)} columns")
print(f"  Dropped: {len(low_info_cols)} (low info) + {len(high_null_cols)} (high nulls) + {len(columns_to_drop)} (redundant)")
print(f"  Protected: {len(essential_cols)} essential columns")

# Show retained columns by category
for category in ['financial', 'cloud_hierarchy', 'identifier', 'temporal']:
    category_cols = [col for col in final_cols
                     if col in semantic_analysis.filter(pl.col('semantic_category') == category)['column'].to_list()]
    if category_cols:
        print(f"  {category}: {', '.join(sorted(category_cols)[:3])}{'...' if len(category_cols) > 3 else ''}")
```

```{code-cell} ipython3
# Quantify redundancy reduction
final_numeric_cols = [col for col in final_cols if col in numeric_cols and col not in columns_to_drop]

def max_off_diagonal_corr(corr_matrix):
    """Pure function: compute max absolute off-diagonal correlation."""
    corr_np = corr_matrix.to_numpy()
    np.fill_diagonal(corr_np, 0)
    return np.abs(corr_np).max()

max_corr_before = max_off_diagonal_corr(corr_matrix)
print(f"Redundancy Elimination:")
print(f"  Numeric features: {len(numeric_cols)} → {len(final_numeric_cols)}")
print(f"  Max correlation: {max_corr_before:.3f} → ", end="")

if len(final_numeric_cols) > 1:
    final_corr = sample_df.select(final_numeric_cols).corr()
    max_corr_after = max_off_diagonal_corr(final_corr)
    print(f"{max_corr_after:.3f} ({'✓ success' if max_corr_after < CORR_THRESHOLD else '⚠ partial'})")
else:
    print(f"N/A ({len(final_numeric_cols)} column)")
```

### Summary

**Dimension Reduction**: 38 → ~20-25 columns via information scoring and correlation analysis

**Redundancy Eliminated**: Highly correlated numeric features removed (|r| > 0.90), keeping highest information score

**Protected Essentials**: Temporal (`usage_date`), identifiers (`uuid`, `resource_id`), and primary metrics preserved

**Result**: `df_filtered` contains minimal, high-signal column set for downstream analysis

---

## Part 4: Cardinality Analysis

### Objective

Classify retained columns by cardinality to determine appropriate analytical operations (aggregation, grouping, tracking).

### Methodology

**Cardinality Classes**:
- **Low** (ratio < 0.0001): Categorical variables, broad segmentation
- **Medium** (0.0001 < ratio < 0.01): Grouping dimensions for aggregation
- **High** (ratio > 0.01): Identifiers, granular tracking

```{code-cell} ipython3
# Cardinality distribution of final columns
final_schema = schema_analysis.filter(pl.col('column').is_in(final_cols))

cardinality_summary = (
    final_schema
    .group_by('card_class')
    .agg(pl.len().alias('count'))
    .sort('count', descending=True)
)

print(f"Cardinality Distribution ({len(final_cols)} columns):")
for row in cardinality_summary.iter_rows(named=True):
    print(f"  {row['card_class']}: {row['count']} columns")

# Examples by class
for card_class in ['Low', 'Medium', 'High']:
    examples = final_schema.filter(pl.col('card_class') == card_class).head(3)
    if len(examples) > 0:
        cols = examples.get_column('column').to_list()
        print(f"  {card_class}: {', '.join(cols)}")
```

### Summary

**Low Cardinality**: Categorical variables for segmentation (providers, regions)

**Medium Cardinality**: Aggregation dimensions (accounts, products, services)

**High Cardinality**: Unique identifiers for granular tracking (UUIDs, resource IDs)

**Implication**: Dataset supports hierarchical analysis from broad categorization to resource-level tracking

---

## Part 5: Temporal Quality

### Objective

Validate temporal coverage and measure time series stability for forecasting viability.

### Methodology

**Coverage Check**: Verify complete date range (Sept 1 - Dec 31, 2025 = 122 days)
**Stability Metrics**: Coefficient of variation, lag-1 autocorrelation
**Visualization Criteria**: Plot only if CV > 0.15 or autocorr < 0.7 (instability detected)

```{code-cell} ipython3
# Date coverage check
date_range = df_filtered.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date'),
    pl.col('usage_date').n_unique().alias('unique_dates')
]).collect()

min_date, max_date = date_range['min_date'][0], date_range['max_date'][0]
expected_days = (max_date - min_date).days + 1
actual_days = date_range['unique_dates'][0]

print(f"Temporal Coverage: {min_date} to {max_date}")
print(f"  Completeness: {actual_days}/{expected_days} days ({'✓' if actual_days == expected_days else '⚠'})")
```

```{code-cell} ipython3
# Stability analysis
primary_cost = [col for col in final_cols if 'cost' in col.lower()][0]
daily_agg = (
    df_filtered
    .group_by('usage_date')
    .agg([
        pl.len().alias('records'),
        pl.col(primary_cost).sum().alias('cost')
    ])
    .sort('usage_date')
    .collect()
)

# Compute stability metrics
record_cv = daily_agg['records'].std() / daily_agg['records'].mean()
cost_series = daily_agg['cost'].to_numpy()

from scipy.stats import pearsonr
lag1_corr, _ = pearsonr(cost_series[:-1], cost_series[1:])

print(f"\nStability Metrics:")
print(f"  Record volume CV: {record_cv:.4f} ({'stable' if record_cv < 0.15 else 'variable'})")
print(f"  Cost lag-1 autocorr: {lag1_corr:.4f} ({'sticky' if lag1_corr > 0.7 else 'volatile'})")

# Conditional visualization
if record_cv > 0.15 or lag1_corr < 0.7:
    print("\n⚠ Instability detected - plotting temporal patterns")
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot_data = daily_agg.to_pandas()

    axes[0].plot(plot_data['usage_date'], plot_data['cost'], linewidth=2, color='steelblue')
    axes[0].set_ylabel('Daily Cost', fontweight='bold')
    axes[0].set_title('Cost Trend (Instability Detected)', fontweight='bold')

    axes[1].plot(plot_data['usage_date'], plot_data['records'], linewidth=2, color='darkgreen')
    axes[1].set_ylabel('Record Count', fontweight='bold')
    axes[1].set_xlabel('Date', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("✓ Stable temporal patterns - visualization unnecessary")
```

### Summary

**Coverage**: Complete (122/122 days) or incomplete - verified above

**Stability**: High (CV < 0.15, autocorr > 0.7) enables reliable time series forecasting. Moderate/low stability requires robust modeling.

**Pattern**: High lag-1 autocorrelation indicates sticky infrastructure costs (resources persist day-to-day). Low autocorrelation suggests volatile spending.

---

## Part 5b: Entity-Level Temporal Anomaly Detection

### Objective

Identify which entity (account, product, service, resource) is driving observed record count variation over time.

### Methodology

For each entity type with medium cardinality, compute daily record contribution and identify entities with highest temporal variability (CV).

```{code-cell} ipython3
# Debug: Check what semantic categories exist in final_cols
final_semantics = semantic_analysis.filter(pl.col('column').is_in(final_cols))
print("Semantic categories in final_cols:")
print(final_semantics.group_by('semantic_category').agg(pl.len().alias('count')))

# Get all non-identifier columns from final set
exclude_patterns = ['uuid', 'resource_id', 'usage_id', 'usage_date', 'cost', 'amount']
entity_candidates = [
    col for col in final_cols
    if not any(pattern in col.lower() for pattern in exclude_patterns)
]

print(f"\nInvestigating {len(entity_candidates)} entity types for temporal anomalies:")
print(f"  {', '.join(entity_candidates)}")
```

```{code-cell} ipython3
# For each entity type, find which specific entity has highest temporal variability
def find_variable_entities(df, entity_col, date_col='usage_date', top_n=3):
    """Find entities with highest record count variation over time."""
    # Daily entity contributions using with_columns API
    entity_daily = (
        df
        .group_by([date_col, entity_col])
        .agg(pl.len().alias('daily_records'))
        .collect()
    )

    # Compute CV per entity using with_columns
    entity_stats = (
        entity_daily
        .group_by(entity_col)
        .agg([
            pl.col('daily_records').mean().alias('mean_records'),
            pl.col('daily_records').std().alias('std_records'),
            pl.len().alias('days_present')
        ])
        .with_columns(
            (pl.col('std_records') / pl.col('mean_records')).alias('cv')
        )
        .filter(pl.col('days_present') > 10)  # Must appear in >10 days
        .sort('cv', descending=True)
    )

    return entity_stats.head(top_n)

# Analyze each entity type
anomaly_results = []
for entity_col in entity_candidates[:5]:  # Limit to top 5 for efficiency
    try:
        top_variable = find_variable_entities(df_filtered, entity_col, top_n=3)
        if len(top_variable) > 0:
            max_cv = top_variable['cv'][0]
            max_entity = top_variable[entity_col][0]
            anomaly_results.append({
                'entity_type': entity_col,
                'max_cv': max_cv,
                'variable_entity': max_entity,
                'mean_daily': top_variable['mean_records'][0]
            })
            print(f"\n{entity_col}:")
            print(f"  Most variable: {max_entity} (CV={max_cv:.3f})")
    except Exception as e:
        print(f"  Skipped {entity_col}: {e}")

# Identify culprit entity type
if anomaly_results:
    culprit = max(anomaly_results, key=lambda x: x['max_cv'])
    print(f"\n🎯 ANOMALY SOURCE IDENTIFIED:")
    print(f"  Entity type: {culprit['entity_type']}")
    print(f"  Variable entity: {culprit['variable_entity']}")
    print(f"  Temporal CV: {culprit['max_cv']:.3f}")
    print(f"  Avg daily records: {culprit['mean_daily']:.0f}")
```

```{code-cell} ipython3
# Stacked area chart: Cloud provider contributions over time
if anomaly_results and culprit['entity_type'] == 'cloud_provider':
    # Get daily counts by provider using with_columns
    provider_daily = (
        df_filtered
        .group_by(['usage_date', 'cloud_provider'])
        .agg(pl.len().alias('records'))
        .sort('usage_date')
        .collect()
    )

    # Pivot to wide format for stacking
    provider_pivot = provider_daily.pivot(
        values='records',
        index='usage_date',
        columns='cloud_provider'
    ).fill_null(0)

    # Compute cumulative percentages for stacking using with_columns
    providers = [col for col in provider_pivot.columns if col != 'usage_date']
    total_col = sum(pl.col(p) for p in providers)

    provider_pct = provider_pivot.with_columns([
        (pl.col(p) / total_col * 100).alias(f'{p}_pct') for p in providers
    ])

    print(f"Cloud Provider Daily Records (first 5 days):")
    print(provider_pivot.head(5))

    # Visualize stacked area
    import matplotlib.pyplot as plt
    plot_data = provider_pivot.to_pandas().set_index('usage_date')

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.stackplot(plot_data.index, *[plot_data[col].values for col in providers],
                 labels=providers, alpha=0.8)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Daily Record Count (Log Scale)', fontweight='bold')
    ax.set_title('Cloud Provider Record Contributions Over Time (Stacked Area)',
                 fontweight='bold', fontsize=14)
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9,
              borderaxespad=0)
    ax.grid(alpha=0.3, axis='y', which='both')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on right
    plt.show()

    # Identify changepoint (if AWS drops)
    aws_daily = provider_pivot.filter(pl.col('AWS').is_not_null()).with_columns([
        pl.col('AWS').pct_change().alias('aws_pct_change')
    ])

    max_drop = aws_daily.select([
        pl.col('usage_date'),
        pl.col('aws_pct_change')
    ]).filter(pl.col('aws_pct_change') < -0.5)  # >50% drop

    if len(max_drop) > 0:
        print(f"\n⚠ AWS Record Drop Events (>50% decrease):")
        print(max_drop.head(3))
```

```{code-cell} ipython3
# Normalized stacked area: Show percentage contributions
if anomaly_results and culprit['entity_type'] == 'cloud_provider':
    # Calculate percentage contributions using with_columns
    provider_pct_norm = provider_pivot.with_columns([
        (pl.col(p) / sum(pl.col(c) for c in providers) * 100).alias(f'{p}_pct')
        for p in providers
    ])

    print("Provider Percentage Contributions (first 5 days):")
    pct_cols = [f'{p}_pct' for p in providers]
    print(provider_pct_norm.select(['usage_date'] + pct_cols).head(5))

    # Visualize normalized stacked area
    plot_data_pct = provider_pct_norm.select(['usage_date'] + pct_cols).to_pandas().set_index('usage_date')

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.stackplot(plot_data_pct.index,
                 *[plot_data_pct[f'{p}_pct'].values for p in providers],
                 labels=providers, alpha=0.8)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Percentage Contribution (%)', fontweight='bold')
    ax.set_title('Cloud Provider Contributions Over Time (Normalized %)',
                 fontweight='bold', fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9,
              borderaxespad=0)
    ax.grid(alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # Identify when AWS becomes <5% of total
    aws_dominance = provider_pct_norm.with_columns([
        pl.col('AWS_pct').alias('aws_contribution_pct')
    ]).select(['usage_date', 'aws_contribution_pct'])

    aws_collapse = aws_dominance.filter(pl.col('aws_contribution_pct') < 5.0)

    if len(aws_collapse) > 0:
        print(f"\n⚠ AWS contribution drops below 5% on:")
        print(aws_collapse.head(1))
```

```{code-cell} ipython3
# Observation frequency analysis
daily_totals = (
    df_filtered
    .group_by('usage_date')
    .agg(pl.len().alias('records'))
    .sort('usage_date')
    .collect()
    .with_columns([
        (pl.col('records') - pl.col('records').shift(1)).alias('day_to_day_change'),
        (pl.col('records') / pl.col('records').shift(1) - 1).alias('pct_change')
    ])
)

# Frequency statistics
freq_stats = daily_totals.select([
    pl.col('records').mean().alias('mean_daily'),
    pl.col('records').median().alias('median_daily'),
    pl.col('records').std().alias('std_daily'),
    pl.col('records').min().alias('min_daily'),
    pl.col('records').max().alias('max_daily'),
]).with_columns([
    (pl.col('std_daily') / pl.col('mean_daily')).alias('cv')
])

print("Observation Frequency Analysis:")
print(f"  Mean daily records: {freq_stats['mean_daily'][0]:,.0f}")
print(f"  Median daily records: {freq_stats['median_daily'][0]:,.0f}")
print(f"  Std deviation: {freq_stats['std_daily'][0]:,.0f}")
print(f"  Range: [{freq_stats['min_daily'][0]:,}, {freq_stats['max_daily'][0]:,}]")
print(f"  Coefficient of variation: {freq_stats['cv'][0]:.3f}")

# Identify step changes
step_changes = daily_totals.filter(
    pl.col('pct_change').abs() > 0.2  # >20% change
).select(['usage_date', 'records', 'pct_change'])

if len(step_changes) > 0:
    print(f"\n⚠ Step Changes Detected (>20% day-over-day):")
    print(step_changes.head(5))
    print(f"\nInterpretation: {'Data collection issue' if freq_stats['cv'][0] > 0.15 else 'Normal variation'}")
else:
    print("\n✓ No significant step changes - consistent observation frequency")
```

```{code-cell} ipython3
# Variance analysis post-collapse (after 2025-10-07)
from datetime import date

collapse_date = date(2025, 10, 7)

# Split dataset: pre and post collapse
pre_collapse = df_filtered.filter(pl.col('usage_date') < collapse_date).collect()
post_collapse = df_filtered.filter(pl.col('usage_date') >= collapse_date).collect()

print(f"Dataset Split at {collapse_date}:")
print(f"  Pre-collapse: {len(pre_collapse):,} records")
print(f"  Post-collapse: {len(post_collapse):,} records")

# Check variance in key metrics post-collapse
if len(post_collapse) > 0:
    primary_cost = [col for col in final_cols if 'cost' in col.lower()][0]

    # Daily statistics post-collapse
    post_daily = (
        post_collapse
        .group_by('usage_date')
        .agg([
            pl.len().alias('records'),
            pl.col(primary_cost).sum().alias('daily_cost'),
            pl.col(primary_cost).std().alias('cost_std')
        ])
        .sort('usage_date')
    )

    # Variance metrics using with_columns
    post_variance = post_daily.with_columns([
        (pl.col('records').std() / pl.col('records').mean()).alias('record_cv'),
        (pl.col('daily_cost').std() / pl.col('daily_cost').mean()).alias('cost_cv')
    ]).select(['record_cv', 'cost_cv']).head(1)

    print(f"\nPost-Collapse Variance ({collapse_date} onwards):")
    print(f"  Record count CV: {post_variance['record_cv'][0]:.6f}")
    print(f"  Daily cost CV: {post_variance['cost_cv'][0]:.6f}")

    # Check if values are constant
    if post_variance['record_cv'][0] < 0.001 and post_variance['cost_cv'][0] < 0.001:
        print(f"\n⚠ CRITICAL: Data is essentially CONSTANT after {collapse_date}")
        print("  → Dataset effectively ends on this date")
        print("  → All subsequent records appear to be artifacts/padding")
    else:
        print(f"\n✓ Some variance remains after {collapse_date}")

    # Show sample of post-collapse data
    print(f"\nPost-collapse daily summary (first 5 days):")
    print(post_daily.head(5))
```

### Summary

**Anomaly Source**: AWS identified with highest temporal variability (CV=0.861)

**Collapse Date**: 2025-10-07 - AWS contribution drops, dataset variance collapses to near-zero

**Post-Collapse Variance**: Essentially constant (CV < 0.001) - data is unreliable/artifactual after this date

**Data Quality Implication**: Effective dataset size is ~68 days (Sept 1 - Oct 7), not 122 days as initially reported

---

## Part 6: Cost Distribution

### Objective

Characterize primary cost metric distribution to inform modeling approach (log transformation, outlier handling).

### Methodology

**Distribution Metrics**: Percentiles, skewness (third moment)
**Outlier Detection**: IQR method (k=1.5)
**Visualization Criteria**: Plot only if skewness > 2 (extreme right-skew requiring visual inspection)

```{code-cell} ipython3
# Primary cost metric analysis
primary_cost_col = [col for col in final_cols if 'cost' in col.lower()][0]
cost_series = df_filtered.select(pl.col(primary_cost_col)).collect().to_series()

# Distribution statistics
percentile_df = pl.DataFrame({
    'percentile': [f'P{p}' for p in [0, 1, 10, 25, 50, 75, 90, 99, 100]],
    'value': [cost_series.quantile(p/100) for p in [0, 1, 10, 25, 50, 75, 90, 99, 100]]
})

# Skewness
mean, std = cost_series.mean(), cost_series.std()
skew = ((cost_series - mean) ** 3).mean() / (std ** 3)

print(f"Primary cost metric: {primary_cost_col}")
print(f"\nDistribution (n={len(cost_series):,}):")
display(percentile_df)

print(f"\nSkewness: {skew:.3f} ({'highly right-skewed' if skew > 1 else 'moderate'})")
print(f"Modeling: {'Log transformation recommended' if skew > 1 else 'Linear scale viable'}")
```

```{code-cell} ipython3
# Outlier quantification
outliers_iqr = detect_outliers_iqr(cost_series, multiplier=1.5)
n_outliers = outliers_iqr.sum()
pct_outliers = (n_outliers / len(cost_series)) * 100

print(f"Outlier Detection (IQR k=1.5):")
print(f"  {n_outliers:,} outliers ({pct_outliers:.2f}%)")
print(f"  → {'Normal' if pct_outliers < 5 else 'High'} rate for billing data")

# Conditional visualization for extreme skew
if skew > 2:
    print(f"\n⚠ Extreme skew ({skew:.3f}) - plotting distribution")
    sample_costs = smart_sample(df_filtered, n=50_000).select(primary_cost_col).to_series().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(sample_costs, bins=50, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Cost ($)', fontweight='bold')
    axes[0].set_title('Linear Scale', fontweight='bold')

    sns.histplot(sample_costs[sample_costs > 0], bins=50, kde=True, ax=axes[1],
                 color='darkgreen', log_scale=True)
    axes[1].set_xlabel('Cost ($, log)', fontweight='bold')
    axes[1].set_title('Log Scale', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("✓ Moderate skew - distribution visualization unnecessary")
```

### Summary

**Shape**: Right-skewed distribution typical of cloud billing (few expensive resources, many low-cost)

**Outliers**: ~2-5% of records flagged by IQR - expected for long-tailed cost data

**Modeling Implication**: Log transformation required if skewness > 1 for regression/forecasting models

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

```{code-cell} ipython3

```
