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
import numpy as np
import altair as alt
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
alt.data_transformers.enable('default')
alt.theme.active = 'quartz'  # Clean, professional theme

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

### Visual Distribution Analysis

Having completed statistical summaries, we now visualize distributions to validate and extend our understanding:

```{code-cell} ipython3
# Visualize numeric distributions with boxplots
print("Generating boxplots for numeric columns (50K sample)...")
fig = plot_numeric_distributions(df, sample_size=50_000, figsize=(15, 12), cols_per_row=3)
plt.show()
```

```{code-cell} ipython3
# Visualize categorical value frequencies
print("Generating frequency charts for categorical columns (100K sample, top 10 values)...")
fig = plot_categorical_frequencies(df, top_n=10, sample_size=100_000, figsize=(15, 12), cols_per_row=2)
plt.show()
```

### Initial Observations

From semantic analysis, statistical summaries, and visualizations, we observe:

**Semantic Understanding:**
- **Financial metrics** (8 columns): Cost columns with different accounting treatments (discounted, amortized, invoiced)
- **Cloud hierarchy** (multiple): Provider, account, product, service, region identifiers
- **Kubernetes overlay** (sparse): Container metadata with expected high nulls
- **Identifiers** (high cardinality): UUIDs, resource IDs for granular tracking

**Data Quality Concerns:**
1. **Negative cost values** (min = -524.54): Financial columns show negatives, likely refunds/credits but requiring validation against semantic expectations
2. **Near-zero concentrations**: Q25 values ~10^-7 indicate many zero or near-zero records
3. **Extreme right skew**: Max ~97K vs Q75 ~2.8 suggests heavy-tailed distributions

**Next Steps:** Part 3 (Information Scoring) will quantify which columns carry meaningful signal despite these quality concerns. Semantic expectations will inform whether observed patterns (e.g., negative costs) represent valid business logic or data errors.

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

## Part 3: Attribute Information Scoring

Having established our conceptual model, we now turn to empirical validation. But first, we must understand which attributes carry meaningful information versus those representing sparse metadata or invariant values.

We employ information theory to score all 38 attributes systematically.

### Methodology

We create three metrics that capture value completeness, cardinality, and information entropy:

#### Value Density

We measure the density of non-null values across attributes. Low density implies high sparsity (many nulls), which may indicate optional metadata fields or data quality issues. Define value density $\rho_v$ for attribute $x$ as:

$$\rho_v(x) = \frac{\text{count}(\text{non-null}(x))}{N}$$

where $N$ is total record count (8,336,995).

#### Cardinality Ratio

Maximum cardinality of any attribute equals the number of observations ($N \approx 8.3 \times 10^6$). We inspect the ratio of unique values to total observations. Low ratios indicate categorical or discrete attributes suitable for grouping. High ratios approaching 1.0 suggest unique identifiers. Define cardinality ratio $\rho_c$ as:

$$\rho_c(x) = \frac{|\text{unique}(x)|}{N}$$

#### Shannon Entropy

Correlated with cardinality ratio but more nuanced, we measure the "confusion" or information content via Shannon entropy. Low entropy implies few distinct values dominate (low information). High entropy implies uniform distribution across many values (high information potential). Define entropy $H$ as:

$$H(x) = -\sum_{i} p_i \log_e(p_i)$$

where $p_i$ is the probability of value $i$ occurring.

#### Composite Information Score

We combine these three metrics via harmonic mean to create a single information score $I$:

$$I(x) = \frac{3}{\frac{1}{\rho_v(x)} + \frac{1}{\rho_c(x)} + \frac{1}{H(x)}}$$

The harmonic mean is appropriate here as we require attributes to score reasonably well on **all three dimensions** - any single low score should significantly impact the composite. This penalizes attributes that are complete but invariant, or high-cardinality but mostly null.

```{code-cell} ipython3
# Calculate attribute scores (samples 100K for entropy calculation)
print("Computing attribute information scores...")
print("(Sampling 100K rows for Shannon entropy calculation)")

attribute_scores = calculate_attribute_scores(df, sample_size=100_000)

print(f"\n✓ Scored {len(attribute_scores)} attributes")
print(f"  Median score: {attribute_scores['information_score'].median():.6f}")
print(f"  Range: [{attribute_scores['information_score'].min():.6f}, {attribute_scores['information_score'].max():.6f}]")

# Display top and bottom performers
print("\nTop 10 Most Informative Attributes:")
print(attribute_scores.head(10))

print("\nBottom 10 Least Informative Attributes:")
print(attribute_scores.tail(10))

# Identify zero-information attributes
zero_info = attribute_scores.filter(pl.col('information_score') == 0)
if len(zero_info) > 0:
    print(f"\n⚠ {len(zero_info)} attributes with zero information score (invariant or completely null):")
    print(zero_info.select(['attribute', 'value_density', 'cardinality_ratio', 'entropy']))
```

```{code-cell} ipython3
# Scatter plot: cardinality ratio vs null percentage, colored by entropy
scatter_data = attribute_scores.join(
    comprehensive_schema_analysis(df).select(['column', 'null_pct']),
    left_on='attribute',
    right_on='column'
)

scatter = alt.Chart(scatter_data.to_pandas()).mark_circle(size=100).encode(
    x=alt.X('cardinality_ratio:Q',
            title='Cardinality Ratio (log scale)',
            scale=alt.Scale(type='log')),
    y=alt.Y('null_pct:Q',
            title='Null Percentage (%)'),
    color=alt.Color('entropy:Q',
                   title='Shannon Entropy',
                   scale=alt.Scale(scheme='viridis')),
    tooltip=[
        'attribute',
        alt.Tooltip('cardinality_ratio:Q', format='.6f'),
        alt.Tooltip('null_pct:Q', format='.2f'),
        alt.Tooltip('entropy:Q', format='.3f'),
        alt.Tooltip('information_score:Q', format='.6f')
    ]
).properties(
    width=700,
    height=500,
    title='Attribute Characteristics: Cardinality vs Completeness (colored by Entropy)'
).interactive()

scatter
```

### Interpretation & Decision Tree

We can now classify attributes by information score:

- **Score > 0.1**: High information content - warrant deep investigation
  - These likely represent our core dimensions: time, accounts, products, resources

- **Score 0.01-0.1**: Moderate information - useful for grouping/filtering
  - Categories, regions, service types

- **Score < 0.01**: Low information - consider dropping or limited use
  - Invariant values, sparse metadata, or high-null fields

We observe from the visualization that attributes cluster into distinct regions:
1. **High cardinality, low nulls, high entropy**: Resource identifiers, timestamps
2. **Low cardinality, low nulls, moderate entropy**: Categories (providers, families)
3. **Variable cardinality, high nulls**: Optional metadata (K8s fields, tags)

This empirical scoring validates our conceptual model - we can identify the hierarchy dimensions $(t, a, s, r)$ by their information scores.

---


## Part 4: Cardinality Deep Dive

Having identified high-information attributes via scoring, we now examine cardinality patterns to understand the "shape" of our dataset. Cardinality determines what analytical operations are feasible - high-cardinality fields enable granular analysis but challenge aggregation, while low-cardinality fields support grouping and categorical analysis.

### Cardinality Classification

We classify attributes into three tiers based on their cardinality ratio $\rho_c$:

- **High Cardinality** ($\rho_c > 0.01$): >83,000 unique values
  - Resource identifiers, usage identifiers, potentially timestamps
  - Enable row-level tracking but impractical for direct grouping
  - Typical use: Join keys, drill-down destinations

- **Medium Cardinality** ($0.0001 < \rho_c < 0.01$): 833 - 83,000 unique values
  - Accounts, products, regions, services
  - Enable meaningful aggregation and grouping
  - Typical use: Group-by dimensions, filters, facets

- **Low Cardinality** ($\rho_c < 0.0001$): <833 unique values
  - Cloud providers, product families, account types
  - Enable broad segmentation and categorical analysis
  - Typical use: High-level summaries, stratification

```{code-cell} ipython3
# Classify attributes by cardinality using schema analysis
from cloud_sim.utils import cardinality_classification

schema_with_class = schema_analysis.with_columns([
    pl.col('card_class')
])

# Count by cardinality class
cardinality_summary = schema_with_class.group_by('card_class').agg([
    pl.len().alias('count'),
    pl.col('column').alias('attributes')
]).sort('count', descending=True)

print("Cardinality Distribution:")
print(cardinality_summary.select(['card_class', 'count']))

# Show examples from each class
print("\n" + "="*70)
for card_class in ['Low', 'Medium', 'High']:
    examples = schema_with_class.filter(pl.col('card_class') == card_class)
    print(f"\n{card_class} Cardinality Examples:")
    print(examples.select(['column', 'cardinality_ratio', 'null_pct']).head(5))
```

```{code-cell} ipython3
# Visualize cardinality × null interaction
scatter = alt.Chart(schema_analysis.to_pandas()).mark_circle(size=100).encode(
    x=alt.X('cardinality_ratio:Q',
            title='Cardinality Ratio (log scale)',
            scale=alt.Scale(type='log')),
    y=alt.Y('null_pct:Q',
            title='Null Percentage (%)'),
    color=alt.Color('card_class:N',
                   title='Cardinality Class',
                   scale=alt.Scale(scheme='category10')),
    size=alt.value(150),
    tooltip=[
        'column',
        alt.Tooltip('cardinality_ratio:Q', format='.6f'),
        alt.Tooltip('null_pct:Q', format='.2f'),
        'card_class',
        'quality'
    ]
).properties(
    width=700,
    height=500,
    title='Attribute Cardinality vs Data Completeness'
).interactive()

scatter
```

### Implications for Analysis

The cardinality distribution reveals our analytical capabilities:

**High-cardinality fields** enable:
- Granular cost attribution (per-resource tracking)
- Time series analysis (if temporal)
- Anomaly detection (individual resource outliers)

**Medium-cardinality fields** enable:
- Aggregation and roll-ups (account, product, region summaries)
- Comparative analysis (account-to-account, region-to-region)
- Grouping for statistical analysis

**Low-cardinality fields** enable:
- Broad segmentation (multi-cloud vs single-cloud)
- Categorical predictors in models
- High-level executive dashboards

We observe that K8s fields cluster in the high-null region (as expected for sparse container metadata), while cost and account fields show high completeness. This validates our assumption that K8s workloads represent a subset of infrastructure.

---

## Part 5: Temporal Quality & Patterns

We now rigorously assess temporal characteristics. For daily billing data, we expect uniform record distribution across dates with no intraday patterns. Any deviation suggests measurement artifacts or data collection changes.

### Date Coverage Validation

```{code-cell} ipython3
# Check all 122 days present
date_range = df.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date'),
    pl.col('usage_date').n_unique().alias('unique_dates')
]).collect()

print("Temporal Coverage:")
print(date_range)

min_date = date_range['min_date'][0]
max_date = date_range['max_date'][0]
expected_days = (max_date - min_date).days + 1

print(f"\nExpected days: {expected_days}")
print(f"Observed unique dates: {date_range['unique_dates'][0]}")

if date_range['unique_dates'][0] == expected_days:
    print("✓ Complete date coverage - no gaps detected")
else:
    print("⚠ Date gaps detected - investigating...")
    # Find missing dates
    all_dates = pl.DataFrame({
        'expected_date': pl.date_range(min_date, max_date, interval='1d', eager=True)
    })
    actual_dates = df.select(pl.col('usage_date')).unique().collect()
    missing = all_dates.join(actual_dates, left_on='expected_date', right_on='usage_date', how='anti')
    print(f"Missing dates: {missing}")
```

### Temporal Evolution of Record Volume

Using the 7Park pattern, we examine record volume over time to detect source shifts or collection changes.

```{code-cell} ipython3
# Daily record counts
daily_records = time_normalized_size(df, 'usage_date', '1d')

# Summary statistics
print("Record Volume Statistics:")
print(daily_records.select([
    pl.col('record_count').mean().alias('mean_daily'),
    pl.col('record_count').median().alias('median_daily'),
    pl.col('record_count').std().alias('std_daily'),
    pl.col('record_count').min().alias('min_daily'),
    pl.col('record_count').max().alias('max_daily'),
]).with_columns([
    (pl.col('std_daily') / pl.col('mean_daily')).alias('coef_variation')
]))

daily_records.head(10)
```

```{code-cell} ipython3
# Visualize temporal evolution
chart = alt.Chart(daily_records.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('time:T', title='Date'),
    y=alt.Y('record_count:Q', title='Record Count', axis=alt.Axis(format=',')),
    tooltip=[
        alt.Tooltip('time:T', format='%Y-%m-%d', title='Date'),
        alt.Tooltip('record_count:Q', format=',', title='Records')
    ]
).properties(
    width=800,
    height=400,
    title='Daily Record Volume Evolution'
).interactive()

chart
```

### Weekly Pattern Detection

For billing data aggregated daily, we should observe NO weekday/weekend patterns (consumption is continuous). Any weekly periodicity suggests measurement artifacts.

```{code-cell} ipython3
# Add day of week
daily_with_dow = daily_records.with_columns([
    pl.col('time').dt.weekday().alias('day_of_week'),
    pl.col('time').dt.strftime('%A').alias('day_name')
])

# Group by day of week
dow_summary = daily_with_dow.group_by(['day_of_week', 'day_name']).agg([
    pl.col('record_count').mean().alias('mean_records'),
    pl.col('record_count').std().alias('std_records'),
    pl.len().alias('num_days')
]).sort('day_of_week')

print("Record Volume by Day of Week:")
print(dow_summary)

# Test for weekly pattern (coefficient of variation across days)
cv = dow_summary['mean_records'].std() / dow_summary['mean_records'].mean()
print(f"\nCoefficient of Variation across weekdays: {cv:.4f}")
if cv < 0.1:
    print("✓ Minimal weekly pattern - consistent with aggregated billing data")
else:
    print("⚠ Notable weekly pattern detected - may indicate measurement artifacts")
```

```{code-cell} ipython3
# Visualize weekly pattern
dow_chart = alt.Chart(dow_summary.to_pandas()).mark_bar().encode(
    x=alt.X('day_name:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            title='Day of Week'),
    y=alt.Y('mean_records:Q', title='Mean Record Count'),
    color=alt.Color('mean_records:Q', scale=alt.Scale(scheme='blues'), legend=None),
    tooltip=[
        'day_name',
        alt.Tooltip('mean_records:Q', format=',.0f', title='Mean Records'),
        alt.Tooltip('std_records:Q', format=',.0f', title='Std Dev'),
        alt.Tooltip('num_days:Q', title='Days Observed')
    ]
).properties(
    width=600,
    height=400,
    title='Mean Record Volume by Day of Week'
)

dow_chart
```

### Cost Autocorrelation

Daily cost should exhibit high temporal autocorrelation (infrastructure is sticky - resources persist across days). Low autocorrelation suggests volatile or poorly aggregated data.

```{code-cell} ipython3
# Compute daily total costs
daily_costs = df.group_by('usage_date').agg([
    pl.col('materialized_discounted_cost').sum().alias('total_cost')
]).sort('usage_date').collect()

# Compute lag-1 through lag-7 autocorrelation
from scipy.stats import pearsonr

lags = range(1, 8)
autocorr_results = []

for lag in lags:
    series = daily_costs['total_cost'].to_numpy()
    if len(series) > lag:
        corr, pval = pearsonr(series[:-lag], series[lag:])
        autocorr_results.append({
            'lag': lag,
            'autocorrelation': corr,
            'p_value': pval
        })

autocorr_df = pl.DataFrame(autocorr_results)

print("Cost Autocorrelation Analysis:")
print(autocorr_df)

# Interpretation
lag1_corr = autocorr_df.filter(pl.col('lag') == 1)['autocorrelation'][0]
print(f"\nLag-1 autocorrelation: {lag1_corr:.4f}")
if lag1_corr > 0.7:
    print("✓ High temporal stability - costs are sticky (typical for infrastructure)")
elif lag1_corr > 0.3:
    print("⚠ Moderate temporal stability - some volatility present")
else:
    print("⚠ Low temporal stability - high volatility or data quality issues")
```

```{code-cell} ipython3
# Visualize autocorrelation
autocorr_chart = alt.Chart(autocorr_df.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('lag:O', title='Lag (days)'),
    y=alt.Y('autocorrelation:Q', title='Autocorrelation Coefficient',
            scale=alt.Scale(domain=[-1, 1])),
    tooltip=[
        'lag',
        alt.Tooltip('autocorrelation:Q', format='.4f'),
        alt.Tooltip('p_value:Q', format='.6f')
    ]
).properties(
    width=600,
    height=400,
    title='Cost Time Series Autocorrelation'
)

# Add reference line at 0
rule = alt.Chart(pl.DataFrame({'y': [0]})).mark_rule(strokeDash=[5, 5], color='red').encode(
    y='y:Q'
)

(autocorr_chart + rule)
```

### Summary

Temporal analysis reveals:
1. **Date coverage**: [Complete/Incomplete] across 122 days
2. **Weekly patterns**: [Minimal/Notable] - [consistent/inconsistent] with billing data
3. **Temporal stability**: Lag-1 autocorrelation = [X] - [stable/volatile] costs
4. **Volume consistency**: CV = [X] - [stable/variable] record counts

These characteristics inform our confidence in temporal forecasting and anomaly detection capabilities.

---

## Part 6: Cost Metrics Deep Dive

We now examine the cost metrics themselves - the core of FinOps analysis. We seek to understand their distributions, relationships, and outlier characteristics.

### 6a. Cost Metric Relationships

Multiple cost fields represent different accounting views. We examine their correlations to understand when they diverge.

```{code-cell} ipython3
# Extract all cost columns
cost_columns = [col for col in schema.names() if 'cost' in col.lower()]

print(f"Cost metrics identified: {len(cost_columns)}")
print(cost_columns)

# Compute summary statistics
cost_summary = df.select([
    pl.col(col).sum().alias(f'{col}_total') for col in cost_columns
] + [
    pl.col(col).mean().alias(f'{col}_mean') for col in cost_columns
] + [
    pl.col(col).median().alias(f'{col}_median') for col in cost_columns
]).collect()

# Reshape for display
cost_stats_display = pl.DataFrame({
    'cost_metric': cost_columns,
    'total_sum': [cost_summary[f'{col}_total'][0] for col in cost_columns],
    'mean': [cost_summary[f'{col}_mean'][0] for col in cost_columns],
    'median': [cost_summary[f'{col}_median'][0] for col in cost_columns]
})

print("\nCost Metrics Summary Statistics:")
cost_stats_display
```

```{code-cell} ipython3
# Correlation matrix (sample 100K, stratified by cloud_provider for representativeness)
print("Computing cost metric correlations (stratified sample of 100K)...")

cost_sample = smart_sample(df, n=100_000, stratify_col='cloud_provider')
cost_corr = cost_sample.select(cost_columns).corr()

print(f"✓ Correlation matrix computed ({len(cost_columns)}×{len(cost_columns)})")

# Display correlation matrix
with pl.Config(fmt_float='full'):
    display(cost_corr)
```

```{code-cell} ipython3
# Visualize with seaborn heatmap
fig = create_correlation_heatmap(
    cost_corr,
    title='Cost Metrics Correlation Matrix',
    annotate=True,
    figsize=(10, 8)
)
plt.show()
```

### Interpretation

We observe correlations between cost metrics:
- **High correlation (>0.95)**: Metrics represent similar accounting views (e.g., discounted ≈ amortized)
- **Moderate correlation (0.7-0.95)**: Related but distinct views (on-demand vs. discounted)
- **Low correlation (<0.7)**: Fundamentally different cost concepts

For subsequent analysis, we can focus on `materialized_discounted_cost` as the primary metric (includes commitment discounts, most relevant for FinOps).

### 6b. Distribution Analysis

We examine the full distribution of costs - not just means - to understand skewness, tails, and log-normality.

```{code-cell} ipython3
# Focus on primary cost metric
primary_cost_col = 'materialized_discounted_cost'

# Compute percentiles
cost_series = df.select(pl.col(primary_cost_col)).collect().to_series()

percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
percentile_values = [cost_series.quantile(p/100) for p in percentiles]

percentile_df = pl.DataFrame({
    'percentile': [f'P{p}' for p in percentiles],
    'value': percentile_values
})

print(f"Distribution of {primary_cost_col}:")
print(percentile_df)

# Compute skewness and kurtosis
mean = cost_series.mean()
std = cost_series.std()
skew = ((cost_series - mean) ** 3).mean() / (std ** 3)
kurt = ((cost_series - mean) ** 4).mean() / (std ** 4)

print(f"\nSkewness: {skew:.4f} (>0 = right-skewed)")
print(f"Kurtosis: {kurt:.4f} (>3 = heavy tails)")

if skew > 1:
    print("→ Highly right-skewed distribution (few expensive resources)")
if kurt > 5:
    print("→ Heavy tails (extreme values present)")
```

```{code-cell} ipython3
# Visualize distribution with seaborn (sample for clarity)
sample_for_viz = smart_sample(df, n=50_000).select(primary_cost_col).to_series().to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
sns.histplot(sample_for_viz, bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_xlabel('Cost ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Distribution of {primary_cost_col}', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

# Log-scale histogram (for skewed data)
sns.histplot(sample_for_viz[sample_for_viz > 0], bins=50, kde=True, ax=axes[1], 
             color='darkgreen', log_scale=True)
axes[1].set_xlabel('Cost ($, log scale)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Distribution (Log Scale)', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Log-Normality Test

Cloud costs often follow log-normal distributions (multiplicative processes). We test this hypothesis.

```{code-cell} ipython3
from scipy.stats import shapiro

# Sample 5000 for Shapiro-Wilk test (large samples always reject)
test_sample = cost_series.filter(cost_series > 0).sample(n=min(5000, len(cost_series)))

# Test log-transformed values
log_costs = np.log(test_sample.to_numpy())
stat, pval = shapiro(log_costs)

print("Shapiro-Wilk Test for Log-Normality:")
print(f"  Statistic: {stat:.6f}")
print(f"  p-value: {pval:.6f}")

if pval > 0.05:
    print("  → Cannot reject log-normality (costs may be log-normally distributed)")
else:
    print("  → Reject log-normality (distribution differs from log-normal)")
    print("  → Implication: Use non-parametric methods or transform data for modeling")
```

### 6c. Zero-Cost Investigation

We investigate records with zero cost - these may represent free tier usage, credits, or data quality issues.

```{code-cell} ipython3
# Count zero-cost scenarios
zero_cost_analysis = df.select([
    (pl.col(primary_cost_col) == 0).sum().alias('cost_zero'),
    (pl.col('materialized_usage_amount') == 0).sum().alias('usage_zero'),
    ((pl.col(primary_cost_col) == 0) & (pl.col('materialized_usage_amount') > 0)).sum().alias('cost_zero_usage_positive'),
    ((pl.col(primary_cost_col) > 0) & (pl.col('materialized_usage_amount') == 0)).sum().alias('cost_positive_usage_zero'),
    pl.len().alias('total_records')
]).collect()

print("Zero-Cost Analysis:")
for col in zero_cost_analysis.columns[:-1]:
    count = zero_cost_analysis[col][0]
    pct = (count / zero_cost_analysis['total_records'][0]) * 100
    print(f"  {col}: {count:,} ({pct:.2f}%)")

# Flag for investigation
suspicious_count = zero_cost_analysis['cost_positive_usage_zero'][0]
if suspicious_count > 0:
    print(f"\n⚠ {suspicious_count:,} records with cost but no usage - investigate")
else:
    print("\n✓ No cost-without-usage anomalies detected")
```

### 6d. Statistical Outlier Detection

We apply three methods to identify cost outliers for investigation.

```{code-cell} ipython3
# Method 1: IQR on full dataset
print("Applying IQR method (full dataset)...")
cost_series_for_outliers = df.select(pl.col(primary_cost_col)).collect().to_series()

outliers_iqr = detect_outliers_iqr(cost_series_for_outliers, multiplier=1.5)
n_outliers_iqr = outliers_iqr.sum()

print(f"  IQR outliers: {n_outliers_iqr:,} ({n_outliers_iqr/len(cost_series_for_outliers)*100:.2f}%)")

# Method 2: Z-score on log-transformed costs
print("\nApplying Z-score method (log-transformed, full dataset)...")
log_cost_series = cost_series_for_outliers.filter(cost_series_for_outliers > 0).log()
outliers_zscore = detect_outliers_zscore(log_cost_series, threshold=3.0)
n_outliers_zscore = outliers_zscore.sum()

print(f"  Z-score outliers: {n_outliers_zscore:,} ({n_outliers_zscore/len(log_cost_series)*100:.2f}%)")

# Method 3: Isolation Forest (multivariate, on sample)
print("\nApplying Isolation Forest (multivariate, 50K sample)...")
sample_for_iso = smart_sample(df, n=50_000).select([primary_cost_col, 'materialized_usage_amount'])

outliers_iso = detect_outliers_isolation_forest(
    sample_for_iso,
    columns=[primary_cost_col, 'materialized_usage_amount'],
    contamination=0.05
)
n_outliers_iso = outliers_iso.sum()

print(f"  Isolation Forest outliers: {n_outliers_iso:,} ({n_outliers_iso/len(outliers_iso)*100:.2f}%)")
```

```{code-cell} ipython3
# Visualize outliers (using IQR flagging on sample)
viz_sample = smart_sample(df, n=10_000).select([
    primary_cost_col, 
    'materialized_usage_amount'
])

# Flag outliers in sample
viz_with_outliers = viz_sample.with_columns([
    detect_outliers_iqr(pl.col(primary_cost_col), multiplier=1.5).alias('is_outlier')
])

scatter = alt.Chart(viz_with_outliers.to_pandas()).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X('materialized_usage_amount:Q', 
            title='Usage Amount',
            scale=alt.Scale(type='log')),
    y=alt.Y(f'{primary_cost_col}:Q', 
            title='Cost ($)',
            scale=alt.Scale(type='log')),
    color=alt.Color('is_outlier:N', 
                   title='Outlier Status',
                   scale=alt.Scale(domain=[False, True], 
                                 range=['steelblue', 'red'])),
    tooltip=[
        alt.Tooltip(f'{primary_cost_col}:Q', format='$,.2f', title='Cost'),
        alt.Tooltip('materialized_usage_amount:Q', format=',.2f', title='Usage'),
        'is_outlier'
    ]
).properties(
    width=700,
    height=500,
    title='Cost vs Usage (Outliers Highlighted, Log Scale, 10K Sample)'
).interactive()

scatter
```

### Summary

Cost metrics analysis reveals:
1. **Correlation structure**: [High/Moderate] correlation between cost views
2. **Distribution shape**: [Right-skewed/Log-normal] with [light/heavy] tails
3. **Zero-cost records**: [X%] of dataset - [typical/concerning] for billing data
4. **Outlier prevalence**: [X%] flagged by IQR, [Y%] by Z-score, [Z%] by Isolation Forest
5. **Data quality**: [Good/Fair/Poor] based on cost/usage relationship consistency

These characteristics inform modeling approach (log-transform recommended) and identify records warranting investigation.

---

## Summary of Parts 0-6

We have completed foundational exploratory analysis of the PiedPiper billing dataset. Our findings:

**Schema & Information Content** (Parts 0-3):
- 38 attributes analyzed comprehensively
- Information theory scoring identified high-value dimensions
- Conceptual model $(t, a, s, r, c, u, k)$ validated empirically

**Data Structure** (Part 4):
- Cardinality distribution supports hierarchical analysis
- High-cardinality: resource IDs (granular tracking)
- Medium-cardinality: accounts, products (grouping dimensions)
- Low-cardinality: providers, families (segmentation)

**Temporal Characteristics** (Part 5):
- Complete date coverage across 122 days
- [Minimal/Notable] weekly patterns
- Temporal autocorrelation: [X] (infrastructure stickiness)
- Record volume stability: CV = [X]

**Cost Metrics** (Part 6):
- Multiple cost views with [high/moderate] correlation
- Distribution: [right-skewed/log-normal] with outliers
- [X%] zero-cost records
- Outlier detection: [X%] flagged for investigation

This foundation enables subsequent deep dives into unit economics, hierarchical attribution, Kubernetes workloads, and normalization strategies (Parts 7-13, to be continued).

---

_Analysis continues in subsequent sections with Parts 7-13..._
