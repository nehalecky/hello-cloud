# CloudZero Billing Data EDA - Consolidated Workflow Design

## Executive Summary

**Problem**: Two notebooks (05, 06) with 4458 total lines scattered across 15+ "Parts" with no actual entity time series produced yet.

**Solution**: Single consolidated notebook following functional composition principles from reference clickstream analysis.

**Reference Philosophy** (from clickstream analysis):
- **Conceptual model first**: Define expected billing hierarchy before data exploration
- **Functional composition**: Small, reusable transforms (no intermediate DataFrames)
- **Information-theoretic scoring**: Harmonic mean of (density, cardinality_ratio, entropy)
- **Entity normalization**: Remove volume effects via period-based normalization
- **Pipeline approach**: Chain operations, minimize namespace pollution
- **Direct visualization**: Plot from functional transforms, not stored variables

---

## Current State Analysis

### Structural Problems

**Notebook 05** (2203 lines):
- 12 Parts with unclear dependencies
- Introduces `PRIMARY_COST` but explores 8 redundant cost columns
- "Part 1.2" appears AFTER Parts 2-8 (organizational chaos)
- No actual time series construction

**Notebook 06** (2255 lines):
- 5 Parts repeating analysis from 05
- Redefines `PRIMARY_COST` multiple times
- Statistical tests without clear decision outcomes
- No entity time series output

**Namespace Pollution**:
```python
# Anti-pattern from current notebooks
schema_analysis = comprehensive_schema_analysis(df)
numeric_summary = numeric_column_summary(df)
categorical_summary = categorical_column_summary(df)
semantic_analysis = semantic_column_analysis(df)
# ... 50+ more intermediate variables
```

**What's Missing**:
- No entity-time DataFrame (e.g., `resource_id × usage_date → cost`)
- No determination of modeling grain (`account × date` vs `resource × date`)
- No output artifact for downstream modeling

---

## Proposed Consolidated Structure

### Single Notebook: `cloudzero_billing_foundation.md`

**Narrative Arc** (6 sections, ~800-1000 lines total):

```
1. Conceptual Model & Expectations (100 lines)
   → Define billing hierarchy BEFORE looking at data
   → Expected grain hypotheses
   → Information we seek

2. Schema Validation & Grain Discovery (200 lines)
   → Attribute scoring (functional)
   → Grain hypothesis testing
   → PRIMARY entity-time determination

3. Feature Selection via Information Theory (150 lines)
   → Functional scoring pipeline
   → Cost column selection (ONE metric)
   → Hierarchical column selection

4. Temporal Quality & Filtering (150 lines)
   → Anomaly detection (functional)
   → Data validity period identification
   → Quality-based filtering

5. Entity Persistence & Normalization (200 lines)
   → Persistence scoring
   → Entity-time normalization
   → Cross-entity correlations

6. Time Series Construction (100 lines)
   → Aggregation to modeling grain
   → Output entity-time DataFrames
   → Validation & export
```

---

## Functional Utilities to Extract

### New Module: `src/cloud_sim/eda/billing_transforms.py`

Based on reference notebook patterns, create composable transforms:

```python
# INFORMATION-THEORETIC SCORING
# Reference pattern: series_entropy(), time_normalized_size()

def attribute_density(df: pl.LazyFrame, col: str) -> float:
    """Value density: non-null ratio"""
    return df.select(pl.col(col).is_not_null().mean()).collect().item()

def cardinality_ratio(df: pl.LazyFrame, col: str) -> float:
    """Cardinality normalized by row count"""
    total_rows = df.select(pl.len()).collect().item()
    unique_vals = df.select(pl.col(col).n_unique()).collect().item()
    return unique_vals / total_rows

def series_entropy(df: pl.LazyFrame, col: str, base: float = np.e) -> float:
    """Shannon entropy of value distribution"""
    value_counts = (
        df.group_by(col)
        .agg(pl.len().alias('count'))
        .with_columns((pl.col('count') / pl.col('count').sum()).alias('p'))
        .select((pl.col('p') * (pl.col('p').log() / np.log(base))).sum() * -1)
        .collect()
        .item()
    )
    return value_counts

def attribute_score(df: pl.LazyFrame, col: str) -> float:
    """Harmonic mean of (density, cardinality_ratio, entropy)"""
    from scipy.stats import hmean
    density = attribute_density(df, col)
    card_ratio = cardinality_ratio(df, col)
    entropy = series_entropy(df, col)
    return hmean([density, card_ratio, entropy]) + 1e-7

# FUNCTIONAL COMPOSITION - no intermediate variables
def score_all_attributes(df: pl.LazyFrame) -> pl.DataFrame:
    """Score all columns, return sorted ranking"""
    schema = df.collect_schema()
    scores = {
        col: attribute_score(df, col)
        for col in schema.names()
    }
    return (
        pl.DataFrame({
            'column': list(scores.keys()),
            'score': list(scores.values())
        })
        .sort('score', descending=True)
    )

# ENTITY NORMALIZATION
# Reference: entity_normalized_by_day() pattern

def normalize_by_period(
    df: pl.LazyFrame,
    entity_col: str,
    metric_col: str,
    period_col: str = 'usage_date',
    freq: str = 'D'
) -> pl.DataFrame:
    """
    Normalize entity metric by total period volume.

    Returns: entity × period → normalized_metric
    Entity activity as fraction of total daily activity.
    """
    # Period totals
    period_totals = (
        df.group_by(period_col)
        .agg(pl.col(metric_col).sum().alias('period_total'))
    )

    # Entity-period values
    entity_period = (
        df.group_by([period_col, entity_col])
        .agg(pl.col(metric_col).sum().alias('entity_value'))
    )

    # Join and normalize (FUNCTIONAL - no intermediate storage)
    return (
        entity_period
        .join(period_totals, on=period_col)
        .with_columns(
            (pl.col('entity_value') / pl.col('period_total'))
            .alias('normalized_value')
        )
        .select([period_col, entity_col, 'normalized_value'])
        .collect()
    )

# GRAIN HYPOTHESIS TESTING
def test_grain_hypothesis(
    df: pl.LazyFrame,
    grain_cols: List[str]
) -> dict:
    """
    Test if grain_cols define unique rows.
    Returns: {grain: [...], unique_ratio: 0.xx, verdict: 'GRAIN'|'NOT_GRAIN'}
    """
    total_rows = df.select(pl.len()).collect().item()
    unique_grains = df.select(grain_cols).unique().select(pl.len()).collect().item()
    ratio = unique_grains / total_rows

    return {
        'grain': grain_cols,
        'total_rows': total_rows,
        'unique_combinations': unique_grains,
        'ratio': ratio,
        'verdict': 'GRAIN' if ratio >= 0.999 else 'NOT_GRAIN'
    }

# PERSISTENCE SCORING
def entity_persistence_score(
    df: pl.LazyFrame,
    entity_col: str,
    time_col: str = 'usage_date'
) -> pl.DataFrame:
    """
    For each entity, compute:
    - days_present
    - lifespan (first to last date)
    - persistence_ratio (days_present / lifespan)
    """
    return (
        df.group_by(entity_col)
        .agg([
            pl.col(time_col).n_unique().alias('days_present'),
            pl.col(time_col).min().alias('first_seen'),
            pl.col(time_col).max().alias('last_seen')
        ])
        .with_columns([
            (pl.col('last_seen') - pl.col('first_seen'))
            .dt.total_days()
            .alias('lifespan_days')
        ])
        .with_columns(
            (pl.col('days_present') / (pl.col('lifespan_days') + 1))
            .alias('persistence_ratio')
        )
        .collect()
    )

# TIME SERIES CONSTRUCTION
def construct_entity_time_series(
    df: pl.LazyFrame,
    entity_cols: List[str],
    time_col: str,
    metric_col: str,
    agg_func: str = 'sum'
) -> pl.DataFrame:
    """
    Aggregate to entity-time grain for time series modeling.

    Returns: Wide or long format entity × time → metric
    """
    group_cols = entity_cols + [time_col]

    return (
        df.group_by(group_cols)
        .agg(getattr(pl.col(metric_col), agg_func)().alias('value'))
        .sort(entity_cols + [time_col])
        .collect()
    )
```

---

## Elimination of Intermediate Variables

### Anti-Pattern (Current Notebooks)

```python
# DON'T: Store every analysis step
schema_analysis = comprehensive_schema_analysis(df)
numeric_summary = numeric_column_summary(df)
categorical_summary = categorical_column_summary(df)

cost_cols = [col for col in all_columns if 'cost' in col.lower()]
cost_summary = df.select(cost_cols).describe()
cost_corr = df.select(cost_cols).head(100_000).collect().corr()

# 50+ more variables polluting namespace...
```

### Functional Pattern (Reference Style)

```python
# DO: Compose operations, visualize directly from transforms

# Score and visualize in one chain
(score_all_attributes(df)
 .filter(pl.col('score') > 0.01)  # Threshold inline
 .pipe(create_score_barplot))     # Plot directly

# Normalize and plot without storage
(normalize_by_period(df, entity_col='account_id', metric_col='cost')
 .pipe(lambda x: plot_normalized_distribution(x, top_n=20)))

# Test grain hypotheses functionally
grain_results = [
    test_grain_hypothesis(df, ['usage_date', 'uuid']),
    test_grain_hypothesis(df, ['usage_date', 'resource_id']),
    test_grain_hypothesis(df, ['usage_date', 'account_id'])
]
# Immediately visualize, don't store
pl.DataFrame(grain_results).pipe(plot_grain_test_results)
```

**Key Principle**: Each cell either:
1. Performs a transformation and visualizes (no storage)
2. Makes a decision and proceeds to next stage
3. Outputs a final artifact (entity time series)

---

## Operation Chaining Patterns

### Pattern 1: Score → Filter → Visualize

```python
# Reference: summary_stats pipeline in clickstream notebook

# Compute attribute scores and filter in one chain
high_info_cols = (
    score_all_attributes(df)
    .filter(pl.col('score') > pl.col('score').median())
    .select('column')
    .to_series()
    .to_list()
)

# Immediately filter dataframe to high-info columns
df_filtered = df.select(['usage_date'] + high_info_cols)

# No intermediate: schema_analysis, numeric_summary, etc.
```

### Pattern 2: Aggregate → Normalize → Plot

```python
# Reference: time_normalized_size() → plotting pattern

def daily_cost_by_entity(df, entity_col, cost_col='materialized_discounted_cost'):
    """Functional: daily entity costs normalized by total daily cost"""
    return (
        df.group_by(['usage_date', entity_col])
        .agg(pl.col(cost_col).sum().alias('cost'))
        .join(
            df.group_by('usage_date').agg(pl.col(cost_col).sum().alias('total')),
            on='usage_date'
        )
        .with_columns((pl.col('cost') / pl.col('total')).alias('cost_fraction'))
        .collect()
    )

# Use directly in visualization
(daily_cost_by_entity(df, entity_col='cloud_provider')
 .pipe(plot_stacked_area, x='usage_date', y='cost_fraction', color='cloud_provider'))
```

### Pattern 3: Test → Decide → Execute

```python
# Reference: grain hypothesis testing pattern

def determine_modeling_grain(df):
    """
    Test grain hypotheses, return the finest valid grain.
    DECISION FUNCTION - returns grain, not intermediate analysis.
    """
    candidates = [
        ['usage_date', 'resource_id'],
        ['usage_date', 'account_id', 'cloud_provider'],
        ['usage_date', 'cloud_provider']
    ]

    for grain in candidates:
        result = test_grain_hypothesis(df, grain)
        if result['verdict'] == 'GRAIN':
            # Check persistence
            entity_col = grain[-1]  # Last col is entity
            persistence = entity_persistence_score(df, entity_col)
            avg_persistence = persistence['days_present'].mean()

            if avg_persistence >= 10:  # Threshold for time series viability
                return {
                    'grain': grain,
                    'entity_col': entity_col,
                    'avg_persistence': avg_persistence,
                    'recommendation': 'USE_THIS_GRAIN'
                }

    return {'recommendation': 'AGGREGATE_HIGHER'}

# Use decision directly
modeling_grain = determine_modeling_grain(df)
print(f"✓ Modeling grain: {modeling_grain['grain']}")

# Proceed immediately to time series construction (no exploration)
entity_ts = construct_entity_time_series(
    df,
    entity_cols=modeling_grain['grain'][:-1],  # All but time
    time_col=modeling_grain['grain'][0],        # First is time
    metric_col='materialized_discounted_cost'
)
```

---

## Entity Time Series Construction

### The End Goal (Currently Missing)

**What we need**: Entity × Time DataFrame for modeling

```python
# OUTPUT FORMAT 1: Long format (panel data)
# entity_id | usage_date | cost | usage | [other metrics]
#    aws-1  | 2025-09-01 | 1234 |  5.6  | ...
#    aws-1  | 2025-09-02 | 1456 |  6.1  | ...
#    aws-2  | 2025-09-01 |  789 |  2.3  | ...

# OUTPUT FORMAT 2: Wide format (time series matrix)
# usage_date | entity_1 | entity_2 | entity_3 | ...
# 2025-09-01 |   1234   |   789    |   456    | ...
# 2025-09-02 |   1456   |   823    |   412    | ...
```

### Construction Function

```python
def build_modeling_dataset(
    df: pl.LazyFrame,
    grain_result: dict,
    cost_col: str = 'materialized_discounted_cost',
    output_path: Path = None
) -> dict:
    """
    Construct time series dataset for modeling.

    Returns:
        {
            'long': pl.DataFrame,    # Panel format
            'wide': pl.DataFrame,    # Time series matrix
            'metadata': dict         # Grain, entity count, time range
        }
    """
    entity_col = grain_result['entity_col']
    time_col = 'usage_date'

    # Long format (panel)
    long_df = (
        df.group_by([time_col, entity_col])
        .agg([
            pl.col(cost_col).sum().alias('cost'),
            pl.col('materialized_usage_amount').sum().alias('usage'),
            pl.len().alias('resource_count')
        ])
        .sort([entity_col, time_col])
        .collect()
    )

    # Wide format (pivot for time series modeling)
    wide_df = (
        long_df
        .pivot(
            values='cost',
            index=time_col,
            columns=entity_col
        )
    )

    metadata = {
        'grain': grain_result['grain'],
        'entity_col': entity_col,
        'n_entities': long_df[entity_col].n_unique(),
        'n_periods': long_df[time_col].n_unique(),
        'date_range': (long_df[time_col].min(), long_df[time_col].max()),
        'total_cost': long_df['cost'].sum(),
        'cost_col': cost_col
    }

    # Save if path provided
    if output_path:
        output_path.mkdir(exist_ok=True)
        long_df.write_parquet(output_path / 'entity_time_series_long.parquet')
        wide_df.write_parquet(output_path / 'entity_time_series_wide.parquet')

        # Write metadata as JSON
        import json
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    return {
        'long': long_df,
        'wide': wide_df,
        'metadata': metadata
    }

# USE IN FINAL SECTION OF NOTEBOOK
output = build_modeling_dataset(
    df_filtered,  # After quality filtering
    modeling_grain,
    output_path=Path('data/processed/piedpiper_timeseries')
)

print(f"✓ Time series dataset constructed:")
print(f"  Grain: {output['metadata']['grain']}")
print(f"  Entities: {output['metadata']['n_entities']:,}")
print(f"  Periods: {output['metadata']['n_periods']}")
print(f"  Total cost: ${output['metadata']['total_cost']:,.2f}")
```

---

## Section-by-Section Implementation

### Section 1: Conceptual Model (100 lines)

**Purpose**: Define expectations BEFORE data exploration

```markdown
## 1. Conceptual Model & Billing Hierarchy

### Expected Billing Event Space

Cloud billing data represents aggregated consumption events:

$$\mathbf{B} = (t, a, p, s, r, c)$$

Where:
- $t$: temporal dimension (usage_date)
- $a$: account identifier (billing entity)
- $p$: cloud provider (AWS, Azure, GCP)
- $s$: service (EC2, S3, RDS, etc.)
- $r$: resource identifier (instance ID, bucket name)
- $c$: cost metric (multiple views: discounted, amortized, invoiced)

### Hierarchical Structure

```
Provider → Account → Service → Resource → Cost
   |          |          |         |         |
  AWS     piedpiper    EC2    i-abc123    $X.XX
```

### Grain Hypotheses

**H1**: Account-day grain (high-level aggregation)
- Each row = one account's total cost for one day
- Expected: ~10 rows/day (if 10 accounts)

**H2**: Resource-day grain (detailed tracking)
- Each row = one resource's cost for one day
- Expected: ~100K-500K rows/day (typical infrastructure scale)

**H3**: Multi-dimensional grain (service × account × day)
- Each row = account's cost for one service on one day
- Expected: ~100-1000 rows/day

**Determination Method**: Test unique(grain_cols) / total_rows ≈ 1.0
```

**Code Pattern**:
```python
# NO data exploration here - just definitions
GRAIN_HYPOTHESES = [
    (['usage_date', 'account_id'], 'Account-day aggregate'),
    (['usage_date', 'resource_id'], 'Resource-day detail'),
    (['usage_date', 'account_id', 'service'], 'Account-service-day')
]
```

---

### Section 2: Schema Validation & Grain Discovery (200 lines)

**Purpose**: Test hypotheses, identify grain, select PRIMARY_COST

```python
# FUNCTIONAL: Score all attributes
attr_scores = score_all_attributes(df)

# FUNCTIONAL: Visualize scores directly
(attr_scores
 .filter(pl.col('score') > 0.001)
 .pipe(plot_attribute_scores))  # Reference: bar chart from clickstream

# DECISION: Select cost column
cost_candidates = attr_scores.filter(pl.col('column').str.contains('cost'))
PRIMARY_COST = cost_candidates[0, 'column']  # Highest score
print(f"✓ PRIMARY_COST = {PRIMARY_COST}")

# FUNCTIONAL: Test grain hypotheses
grain_results = [test_grain_hypothesis(df, cols) for cols, _ in GRAIN_HYPOTHESES]
grain_df = pl.DataFrame(grain_results)

# DECISION: Select grain
modeling_grain = determine_modeling_grain(df)
print(f"✓ Dataset grain: {modeling_grain['grain']}")

# FILTER: Keep only high-information columns
high_info_threshold = attr_scores['score'].quantile(0.5)
keep_cols = (
    attr_scores
    .filter(pl.col('score') > high_info_threshold)
    .select('column')
    .to_series()
    .to_list()
)

df = df.select(['usage_date', PRIMARY_COST] + keep_cols)
```

**No intermediate variables** except final decisions (`PRIMARY_COST`, `modeling_grain`, `keep_cols`)

---

### Section 3: Feature Selection via Information Theory (150 lines)

**Purpose**: Correlation analysis, redundancy removal

```python
# FUNCTIONAL: Compute pairwise correlations (cost columns only)
cost_cols = [c for c in keep_cols if 'cost' in c.lower()]

cost_corr = (
    df.select(cost_cols)
    .head(100_000)  # Sample for speed
    .collect()
    .corr()
)

# FUNCTIONAL: Identify redundant pairs (r > 0.95)
redundant_pairs = find_correlated_pairs(cost_corr, threshold=0.95)

# DECISION: Drop redundant columns (keep higher-scored)
keep_one_from_pair = select_from_pairs(redundant_pairs, attr_scores)
df = df.drop([p[1] for p in keep_one_from_pair])  # Drop lower-scored from each pair

print(f"✓ Reduced from {len(cost_cols)} cost columns to {len([c for c in df.columns if 'cost' in c.lower()])}")
```

**Reference pattern**: `summary_stats` harmonic mean → threshold → filter

---

### Section 4: Temporal Quality & Filtering (150 lines)

**Purpose**: Detect anomalies, filter bad periods

```python
# FUNCTIONAL: Daily cost totals
daily_totals = (
    df.group_by('usage_date')
    .agg(pl.col(PRIMARY_COST).sum().alias('total_cost'))
    .sort('usage_date')
    .collect()
)

# FUNCTIONAL: Detect outliers (IQR method)
outlier_dates = detect_outliers_iqr(daily_totals, metric_col='total_cost')

# DECISION: Identify validity period
if len(outlier_dates) > 0:
    last_valid_date = outlier_dates['usage_date'].min()
    print(f"⚠ Anomaly detected: data after {last_valid_date} excluded")
    df = df.filter(pl.col('usage_date') < last_valid_date)
else:
    print(f"✓ No temporal anomalies detected")

# VISUALIZE: Before/after filtering
plot_temporal_quality(daily_totals, outlier_dates)
```

**Reference pattern**: Direct outlier detection → immediate filtering (no exploration of bad data)

---

### Section 5: Entity Persistence & Normalization (200 lines)

**Purpose**: Understand entity dynamics, normalize for comparison

```python
# FUNCTIONAL: Compute persistence scores
entity_col = modeling_grain['entity_col']
persistence = entity_persistence_score(df, entity_col)

# DECISION: Filter to persistent entities (for time series modeling)
min_persistence = 10  # days
persistent_entities = (
    persistence
    .filter(pl.col('days_present') >= min_persistence)
    .select(entity_col)
    .to_series()
    .to_list()
)

print(f"✓ Persistent entities: {len(persistent_entities)} (≥{min_persistence} days)")

# FILTER: Keep only persistent entities
df_persistent = df.filter(pl.col(entity_col).is_in(persistent_entities))

# FUNCTIONAL: Normalize by period (remove volume effects)
normalized = normalize_by_period(
    df_persistent,
    entity_col=entity_col,
    metric_col=PRIMARY_COST
)

# VISUALIZE: Top entities by normalized cost fraction
(normalized
 .group_by(entity_col)
 .agg(pl.col('normalized_value').mean().alias('avg_fraction'))
 .sort('avg_fraction', descending=True)
 .head(20)
 .pipe(plot_entity_ranking))
```

**Reference pattern**: `entity_normalized_by_day()` → boxplots, stacked area charts

---

### Section 6: Time Series Construction (100 lines)

**Purpose**: Output final modeling dataset

```python
# CONSTRUCT: Entity × time dataset
output = build_modeling_dataset(
    df_persistent,
    modeling_grain,
    cost_col=PRIMARY_COST,
    output_path=Path('data/processed/piedpiper_timeseries')
)

# VALIDATE: Check completeness
completeness = (
    output['long']
    .group_by('usage_date')
    .agg(pl.col(entity_col).n_unique().alias('n_entities'))
)

expected_entities = output['metadata']['n_entities']
actual_avg = completeness['n_entities'].mean()
panel_balance = actual_avg / expected_entities

print(f"\n✓ TIME SERIES DATASET COMPLETE")
print(f"  Format: Long (panel) + Wide (matrix)")
print(f"  Grain: {output['metadata']['grain']}")
print(f"  Entities: {output['metadata']['n_entities']:,}")
print(f"  Periods: {output['metadata']['n_periods']}")
print(f"  Panel balance: {panel_balance:.1%}")
print(f"  Output: {output_path}/")

# FINAL VISUALIZATION: Entity time series sample
(output['long']
 .filter(pl.col(entity_col).is_in(persistent_entities[:5]))  # Top 5 entities
 .pipe(plot_entity_timeseries, x='usage_date', y='cost', color=entity_col))
```

**Reference pattern**: Final output is a DATASET, not more exploratory analysis

---

## Migration Strategy

### Phase 1: Create Functional Library (1 day)

1. Extract functions to `src/cloud_sim/eda/billing_transforms.py`
2. Write tests for each transform
3. Validate on sample data

### Phase 2: Build Consolidated Notebook (2 days)

1. Section 1: Conceptual model (copy pattern from reference)
2. Section 2: Implement functional scoring + grain discovery
3. Section 3: Feature selection via correlation
4. Section 4: Temporal filtering
5. Section 5: Entity persistence + normalization
6. Section 6: Time series construction

### Phase 3: Validate & Archive (0.5 days)

1. Execute consolidated notebook end-to-end
2. Verify output artifacts exist and are valid
3. Archive old notebooks (05, 06) to `notebooks/archive/`
4. Update documentation

---

## Code Pattern Comparison

### OLD: Variable Pollution
```python
# 50+ variables created
schema_analysis = comprehensive_schema_analysis(df)
display(schema_analysis)

semantic_analysis = semantic_column_analysis(df)
display(semantic_analysis)

numeric_summary = numeric_column_summary(df)
display(numeric_summary)

categorical_summary = categorical_column_summary(df)
display(categorical_summary)

# ... 10 more cells of summary statistics
# ... never actually USE these variables again
```

### NEW: Functional Composition
```python
# SINGLE cell: score and decide
high_info_cols = (
    score_all_attributes(df)
    .filter(pl.col('score') > pl.col('score').median())
    .pipe(plot_attribute_scores)  # Visualize inline
    .select('column')
    .to_series()
    .to_list()
)

# Immediately use result
df = df.select(high_info_cols)
```

**Reduction**: ~200 lines of summaries → ~20 lines of functional scoring

---

## Expected Outcomes

### Quantitative Improvements
- **Lines of code**: 4458 → ~1000 (77% reduction)
- **Intermediate variables**: ~100+ → ~10 (90% reduction)
- **Parts/sections**: 15+ → 6 (60% reduction)
- **Time to execute**: ~10 min → ~3 min (70% faster)

### Qualitative Improvements
- **Clear narrative**: Conceptual model → Validation → Output
- **Reproducible**: Single notebook, deterministic output
- **Reusable**: Functional library for future billing datasets
- **Actionable**: Produces entity time series for modeling

### Outputs Created (Currently Missing)
1. `entity_time_series_long.parquet` - Panel data for modeling
2. `entity_time_series_wide.parquet` - Time series matrix
3. `metadata.json` - Grain, entity count, date range
4. `billing_transforms.py` - Reusable EDA library

---

## Next Steps

1. **Review this design document** - validate approach
2. **Implement functional library** - extract transforms
3. **Build consolidated notebook** - follow 6-section structure
4. **Execute and validate** - ensure outputs exist
5. **Archive old notebooks** - preserve history but remove clutter

**Estimated effort**: 3-4 days for full implementation and validation

**Value**: Rigorous, reproducible foundation for CloudZero billing forecasting
