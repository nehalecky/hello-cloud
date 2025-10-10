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

# CloudZero PiedPiper Dataset - Rigorous Analysis

## Background

**Dataset**: PiedPiper optimized daily billing data
**Coverage**: September 1 - December 31, 2025 (122 days)
**Records**: 8,336,995 rows × 38 columns
**Format**: SNAPPY-compressed Parquet (0.96 GB)

## Assumptions

Such raw observations can be represented by billing event space $\mathbf{B}_0$, referring to the complete universe of cloud resource consumption events. We conceptualize these fundamental dimensions:

- **$t$**: temporal dimension (date of resource consumption)
- **$e$**: entity identifier (account, resource, service consuming infrastructure)
- **$c$**: cost metric (monetary value under various accounting treatments)
- **$u$**: usage metric (quantity consumed)
- **$h$**: hierarchy context (cloud provider, account, product, service)

Thus, our expected event space $\mathbf{B}$ having dimensions $(t, e, c, u, h)$, and we seek to identify attributes that support this billing model.

**Critical Question**: What is the **grain** of this dataset? At what level of $(t, e)$ does each row represent a unique observation?

Understanding grain determines what forecasting is possible:
- **Daily-Resource grain** → Forecast individual resource costs
- **Daily-Account grain** → Forecast account-level spending
- **Daily-Provider grain** → Forecast only macro cloud totals

We proceed methodically to discover the true grain and build time series for modeling.

---

## Part 0: Setup & Configuration

```{code-cell} ipython3
# Hot reload pattern - library changes auto-reload without kernel restart
%load_ext autoreload
%autoreload 2

# Core libraries
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from pathlib import Path
from scipy import stats

# Import utilities
from cloudlens.utils import (
    temporal_quality_metrics,
    cost_distribution_metrics,
    detect_entity_anomalies,
    normalize_by_period,
)

# Configure visualization
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

# GLOBAL: Primary cost metric
PRIMARY_COST = 'materialized_discounted_cost'

print("✓ Libraries loaded")
print(f"✓ PRIMARY_COST = '{PRIMARY_COST}'")
```

```{code-cell} ipython3
# Load dataset
DATA_PATH = Path('../data/piedpiper_optimized_daily.parquet')

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pl.scan_parquet(DATA_PATH)

# Basic dimensions
total_rows = df.select(pl.len()).collect()[0, 0]
schema = df.collect_schema()

print(f"✓ Loaded: {total_rows:,} rows × {len(schema)} columns")
print(f"  File: {DATA_PATH.stat().st_size / (1024**3):.2f} GB")
```

---

## Part 1: How would you validate the data and look for anomalies?

Validation refers to comparing the dataset's shape, size, and distributions to our expectations. We validate by:

1. **Direct inspection** - View schema and sample records
2. **Attribute profiling** - Cardinality, null density, information content
3. **Temporal analysis** - Coverage, stability, anomalies

This is iterative — we build knowledge that informs next steps. Patience is a virtue.

### 1.1: Attribute Information Scoring

Create general attribute metrics capturing value density, cardinality, and entropy. Combine via harmonic mean to identify informative attributes.

```{code-cell} ipython3
def attribute_score(df: pl.LazyFrame, sample_size: int = 50_000) -> pl.DataFrame:
    """
    Score attributes by information content: harmonic mean of value density,
    cardinality ratio, and Shannon entropy.

    Memory-efficient: Works on sampled data to avoid full collection.

    Returns DataFrame with columns: attribute, value_density, card_ratio,
    entropy, information_score (sorted descending).
    """
    from cloudlens.utils import calculate_attribute_scores

    print(f"Computing attribute scores on {sample_size:,} row sample...")
    return calculate_attribute_scores(df, sample_size=sample_size)

# Score all attributes (memory-safe via sampling)
scores = attribute_score(df, sample_size=50_000)

print(f"\nAttribute Scoring Complete:")
print(f"  Columns analyzed: {len(scores)}")
print(f"  Score range: [{scores['information_score'].min():.6f}, {scores['information_score'].max():.6f}]")
print("\nTop 15 Most Informative:")
display(scores.head(15))
```

```{code-cell} ipython3
# Visualize information scores (log scale)
from cloudlens.utils import create_info_score_chart

fig = create_info_score_chart(scores)
plt.show()
```

### Summary: Attribute Selection

**Low-information attributes** (score < 0.01): Dropped for analysis
**High-information attributes** (score > 0.1): Core billing dimensions
**Essential attributes** (temporal, identifiers): Preserved regardless of score

---

## Part 2: What is the grain of this dataset?

The grain determines what each row represents. We test hypotheses:

**Hypothesis 1**: Row-level unique (`usage_date × uuid`)
**Hypothesis 2**: Resource-day grain (`usage_date × resource_id`)
**Hypothesis 3**: Account-day grain (`usage_date × account_id × cloud_provider`)

```{code-cell} ipython3
def test_grain_hypothesis(df: pl.LazyFrame, grain_cols: list[str]) -> dict:
    """
    Test if grain_cols define unique rows (memory-efficient).

    Returns: {
        'grain': grain_cols,
        'total_rows': int,
        'unique_grains': int,
        'uniqueness_ratio': float,
        'is_grain': bool (ratio >= 0.999)
    }
    """
    # Memory-safe: use SQL-style distinct count
    stats = df.select([
        pl.len().alias('total'),
        pl.col(grain_cols[0]).n_unique().alias('check')  # Quick sanity check
    ]).collect()

    total_rows = stats['total'][0]

    # Only collect unique if reasonable size
    if total_rows > 1_000_000:
        print(f"  ⚠ Large dataset ({total_rows:,} rows) - using approximate uniqueness")
        unique_grains = df.select(grain_cols).unique().select(pl.len()).collect()[0, 0]
    else:
        unique_grains = df.select(grain_cols).unique().select(pl.len()).collect()[0, 0]

    ratio = unique_grains / total_rows

    return {
        'grain': ' × '.join(grain_cols),
        'total_rows': total_rows,
        'unique_grains': unique_grains,
        'uniqueness_ratio': ratio,
        'is_grain': ratio >= 0.999
    }

# Test grain hypotheses
grain_tests = [
    ['usage_date', 'uuid'],
    ['usage_date', 'resource_id'],
    ['usage_date', 'cloud_provider', 'account_id'],
]

print("GRAIN HYPOTHESIS TESTING")
print("=" * 80)

for grain_cols in grain_tests:
    # Check columns exist
    available = df.collect_schema().names()
    if not all(col in available for col in grain_cols):
        print(f"\nSkipped: {' × '.join(grain_cols)} (columns missing)")
        continue

    result = test_grain_hypothesis(df, grain_cols)

    print(f"\nGrain: {result['grain']}")
    print(f"  Total rows: {result['total_rows']:,}")
    print(f"  Unique combinations: {result['unique_grains']:,}")
    print(f"  Ratio: {result['uniqueness_ratio']:.6f}")
    print(f"  {'✓ IS DATASET GRAIN' if result['is_grain'] else '✗ Not grain (duplicates exist)'}")
```

### Summary: Grain Discovery

**Dataset grain**: _[Determined from analysis above]_

This grain determines forecasting granularity:
- If resource-level → Can forecast individual resource costs
- If account-level → Can forecast account spending trends
- If provider-level → Can only forecast macro totals

---

## Part 3: Entity Persistence - Which entities form viable time series?

For time series modeling, entities must persist across multiple days. We analyze:
- **Resource persistence**: How many days does each resource appear?
- **Account persistence**: Lifespan of billing accounts
- **Provider persistence**: Cloud provider consistency

```{code-cell} ipython3
def entity_persistence_score(
    df: pl.LazyFrame,
    entity_col: str,
    date_col: str = 'usage_date',
    min_days: int = 10
) -> pl.DataFrame:
    """
    Analyze entity persistence for time series viability.

    Returns DataFrame with:
        - entity: entity identifier
        - days_present: number of unique dates observed
        - first_seen, last_seen: date range
        - lifespan_days: calendar days from first to last
        - total_cost: sum of PRIMARY_COST

    Sorted by days_present descending.
    """
    persistence = (
        df.group_by(entity_col)
        .agg([
            pl.col(date_col).n_unique().alias('days_present'),
            pl.col(date_col).min().alias('first_seen'),
            pl.col(date_col).max().alias('last_seen'),
            pl.col(PRIMARY_COST).sum().alias('total_cost')
        ])
        .collect()
        .with_columns([
            (pl.col('last_seen') - pl.col('first_seen')).dt.total_days().alias('lifespan_days')
        ])
        .filter(pl.col('days_present') >= min_days)
        .sort('days_present', descending=True)
    )

    return persistence

# Test different entity types
entity_candidates = ['resource_id', 'account_id', 'cloud_provider']

print("ENTITY PERSISTENCE ANALYSIS")
print("=" * 80)

for entity_col in entity_candidates:
    schema_cols = df.collect_schema().names()
    if entity_col not in schema_cols:
        print(f"\nSkipped: {entity_col} (not in schema)")
        continue

    persistence = entity_persistence_score(df, entity_col, min_days=10)

    print(f"\n{entity_col.upper()}:")
    print(f"  Unique entities (10+ days): {len(persistence):,}")
    print(f"  Avg days present: {persistence['days_present'].mean():.1f}")
    print(f"  Max persistence: {persistence['days_present'].max()} days")

    # Time series viability classification
    avg_days = persistence['days_present'].mean()
    if avg_days >= 20:
        print(f"  ✓ EXCELLENT for time series (long histories)")
    elif avg_days >= 10:
        print(f"  ✓ GOOD for time series")
    else:
        print(f"  ⚠ MARGINAL (short-lived entities)")

    # Show top 5
    print(f"\n  Top 5 by persistence:")
    display(persistence.head(5))
```

### Summary: Time Series Viability

**Entity with best persistence**: _[Identified from analysis]_
**Average days present**: _[Value]_
**Implication**: Can build _[resource/account/provider]_-level time series for forecasting

---

## Part 4: Temporal Quality & Anomaly Detection

Validate temporal coverage and detect instabilities that affect time series modeling.

```{code-cell} ipython3
# Temporal quality using library utility
quality = temporal_quality_metrics(
    df,
    date_col='usage_date',
    metric_col=PRIMARY_COST
)

print("TEMPORAL QUALITY METRICS")
print("=" * 80)
print(f"Date range: {quality['date_range'][0]} to {quality['date_range'][1]}")
print(f"Coverage: {quality['coverage_days']}/{quality['expected_days']} days ({quality['completeness_pct']}%)")
print(f"Record volume CV: {quality['record_volume_cv']:.4f}")
print(f"Cost lag-1 autocorr: {quality['metric_lag1_autocorr']:.4f}")
print(f"Stability: {quality['stability_class']}")
```

```{code-cell} ipython3
# Entity-level anomaly detection
anomaly_entities = detect_entity_anomalies(
    df,
    entity_col='cloud_provider',
    date_col='usage_date',
    min_days=10,
    top_n=3
)

print("\nENTITY ANOMALIES (High temporal variance):")
display(anomaly_entities)

# If anomalies detected, investigate
if len(anomaly_entities) > 0:
    culprit = anomaly_entities[0, 'entity']
    cv = anomaly_entities[0, 'cv']
    print(f"\n⚠ {culprit} has high variance (CV={cv:.3f}) - investigate further")
```

### Summary: Data Quality

**Temporal coverage**: Complete ✓ or Incomplete ⚠
**Stability**: Stable ✓ or Volatile ⚠
**Anomalies detected**: _[Yes/No and description]_
**Action**: _[Filter dates, exclude entities, or proceed as-is]_

---

## Part 5: Build Entity Time Series for Modeling

Having identified the grain and validated entity persistence, we construct time series datasets.

```{code-cell} ipython3
def build_entity_time_series(
    df: pl.LazyFrame,
    entity_col: str,
    date_col: str = 'usage_date',
    metric_col: str = PRIMARY_COST,
    min_days: int = 10
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build entity time series in two formats:

    Returns:
        (long_format, wide_format)

    long_format: Panel data (entity, date, cost, usage, ...)
    wide_format: Matrix (date as rows, entities as columns)
    """
    # Get persistent entities
    persistent = entity_persistence_score(df, entity_col, date_col, min_days)
    entity_list = persistent.get_column(entity_col).to_list()

    # Build panel (long format)
    panel = (
        df.filter(pl.col(entity_col).is_in(entity_list))
        .group_by([date_col, entity_col])
        .agg([
            pl.col(metric_col).sum().alias('cost'),
            pl.col(metric_col).mean().alias('avg_cost'),
            pl.len().alias('records'),
        ])
        .sort([date_col, entity_col])
        .collect()
    )

    # Pivot to wide format (time series matrix)
    wide = (
        panel
        .pivot(
            values='cost',
            index=date_col,
            columns=entity_col
        )
        .fill_null(0)  # Assume zero cost on missing days
    )

    return panel, wide

# Build time series for best entity type
# (Replace 'resource_id' with actual best entity from Part 3)
entity_type = 'resource_id'  # or 'account_id' or 'cloud_provider'

panel_df, matrix_df = build_entity_time_series(
    df,
    entity_col=entity_type,
    min_days=10
)

print(f"ENTITY TIME SERIES CONSTRUCTED")
print(f"=" * 80)
print(f"Entity type: {entity_type}")
print(f"Panel format: {len(panel_df):,} rows (entity × date combinations)")
print(f"Matrix format: {matrix_df.shape[0]} dates × {matrix_df.shape[1]-1} entities")
print(f"\nPanel sample:")
display(panel_df.head(10))
```

```{code-cell} ipython3
# Visualize top 5 entity time series
top_entities = (
    panel_df
    .group_by(entity_type)
    .agg(pl.col('cost').sum())
    .sort('cost', descending=True)
    .head(5)
    .get_column(entity_type)
    .to_list()
)

plot_data = (
    panel_df
    .filter(pl.col(entity_type).is_in(top_entities))
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(16, 8))

for entity in top_entities:
    entity_data = plot_data[plot_data[entity_type] == entity]
    ax.plot(entity_data['usage_date'], entity_data['cost'],
           marker='o', linewidth=2, label=str(entity)[:30], alpha=0.7)

ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('Daily Cost ($)', fontweight='bold', fontsize=12)
ax.set_title(f'Top 5 {entity_type}s: Daily Cost Trends',
            fontweight='bold', fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend(title=entity_type, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✓ Time series visualization complete")
```

### Summary: Modeling-Ready Datasets

**Output artifacts**:
1. `panel_df`: Long format (entity × date × metrics) for modeling
2. `matrix_df`: Wide format (dates × entities) for matrix operations

**Use cases**:
- Panel: Hierarchical models, mixed effects, entity-specific forecasts
- Matrix: VAR models, dimensionality reduction, correlation analysis

---

## Part 6: Export for Modeling

```{code-cell} ipython3
# Save modeling datasets
OUTPUT_DIR = Path('data/modeling')
OUTPUT_DIR.mkdir(exist_ok=True)

# Export panel (long format)
panel_path = OUTPUT_DIR / 'entity_time_series_panel.parquet'
panel_df.write_parquet(panel_path)

# Export matrix (wide format)
matrix_path = OUTPUT_DIR / 'entity_time_series_matrix.parquet'
matrix_df.write_parquet(matrix_path)

# Metadata
metadata = {
    'entity_type': entity_type,
    'grain': f'usage_date × {entity_type}',
    'date_range': {
        'start': str(panel_df['usage_date'].min()),
        'end': str(panel_df['usage_date'].max()),
        'days': int(panel_df['usage_date'].n_unique())
    },
    'entities': {
        'total': int(panel_df[entity_type].n_unique()),
        'persistent_10d': len(top_entities)
    },
    'temporal_quality': {
        'completeness_pct': quality['completeness_pct'],
        'stability_class': quality['stability_class'],
        'autocorr': quality['metric_lag1_autocorr']
    }
}

import json
metadata_path = OUTPUT_DIR / 'metadata.json'
metadata_path.write_text(json.dumps(metadata, indent=2))

print("✓ Datasets exported:")
print(f"  {panel_path}")
print(f"  {matrix_path}")
print(f"  {metadata_path}")
print(f"\n✓ Ready for time series modeling")
```

---

## Conclusions

**Grain discovered**: `usage_date × {entity_type}` - each row represents one entity's activity on one day

**Entity persistence**: {entity_type} entities persist an average of X days, enabling time series forecasting

**Temporal quality**: {quality['stability_class']} stability with {quality['completeness_pct']}% coverage

**Modeling datasets**: Panel and matrix formats exported with metadata

**Next steps**:
1. Hierarchical time series forecasting (account → resource)
2. Anomaly detection on entity-level trends
3. Cost optimization via resource lifecycle analysis

```{code-cell} ipython3

```
