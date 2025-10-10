"""
Entity-level cost analysis for CloudZero PiedPiper dataset.

Uses utilities from cloud_sim.utils.cost_analysis to:
1. Analyze cost distribution by entity type
2. Normalize costs over time to remove volume effects
3. Identify cost concentration (Pareto analysis)
4. Assess temporal stability by entity
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import date
from cloud_sim.utils.cost_analysis import (
    normalize_by_period,
    detect_entity_anomalies,
    cost_distribution_metrics,
    temporal_quality_metrics,
)

# Dataset path
DATA_PATH = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet')

# Load and filter to clean data
df = pl.scan_parquet(DATA_PATH)
COLLAPSE_DATE = date(2025, 10, 7)
df_clean = df.filter(pl.col('usage_date') < COLLAPSE_DATE)

print("=" * 80)
print("ENTITY COST ANALYSIS - CloudZero PiedPiper Dataset")
print("=" * 80)

# Identify primary cost column
COST_COL = 'materialized_discounted_cost'

# Basic statistics
total_rows = df_clean.select(pl.len()).collect()[0, 0]
total_cost = df_clean.select(pl.col(COST_COL).sum()).collect()[0, 0]
date_range = df_clean.select([
    pl.col('usage_date').min(),
    pl.col('usage_date').max(),
    pl.col('usage_date').n_unique().alias('days')
]).collect()

print(f"\nDataset Summary:")
print(f"  Period: {date_range[0, 0]} to {date_range[0, 1]} ({date_range[0, 2]} days)")
print(f"  Total rows: {total_rows:,}")
print(f"  Total cost: ${total_cost:,.2f}")
print(f"  Avg daily cost: ${total_cost / date_range[0, 2]:,.2f}")

print("\n" + "=" * 80)
print("1. COST DISTRIBUTION BY ENTITY TYPE")
print("=" * 80)

# Define entity hierarchies
entity_types = {
    'cloud_provider': 'Top-level provider (AWS, Azure, GCP)',
    'resource_id': 'Individual resources',
}

for entity_col, description in entity_types.items():
    print(f"\n{entity_col.upper()} ({description}):")
    print("-" * 60)

    # Aggregate costs by entity
    entity_costs = (
        df_clean
        .group_by(entity_col)
        .agg([
            pl.col(COST_COL).sum().alias('total_cost'),
            pl.len().alias('record_count'),
            pl.col(COST_COL).mean().alias('avg_cost_per_record'),
            pl.col('usage_date').n_unique().alias('days_present')
        ])
        .sort('total_cost', descending=True)
        .collect()
    )

    # Add percentages and cumulative
    entity_total = entity_costs['total_cost'].sum()
    entity_costs = entity_costs.with_columns([
        (pl.col('total_cost') / entity_total * 100).alias('pct_total'),
        (pl.col('total_cost').cum_sum() / entity_total * 100).alias('cumulative_pct')
    ])

    # Summary statistics
    n_entities = len(entity_costs)
    print(f"  Unique entities: {n_entities:,}")
    print(f"  Total cost: ${entity_total:,.2f}")

    # Top 10
    print(f"\n  Top 10 by cost:")
    top10 = entity_costs.head(10)
    for idx, row in enumerate(top10.iter_rows(named=True), 1):
        print(f"    {idx:2d}. {str(row[entity_col])[:40]:40s} ${row['total_cost']:>12,.2f} ({row['pct_total']:5.2f}%)")

    # Pareto analysis (80/20 rule)
    entities_for_80pct = len(entity_costs.filter(pl.col('cumulative_pct') <= 80))
    pct_for_80 = (entities_for_80pct / n_entities) * 100

    print(f"\n  Pareto Analysis (80/20 rule):")
    print(f"    {entities_for_80pct:,} entities ({pct_for_80:.1f}%) account for 80% of costs")

    if pct_for_80 < 20:
        print(f"    → STRONG concentration (< 20% → 80% of costs)")
    elif pct_for_80 < 40:
        print(f"    → MODERATE concentration")
    else:
        print(f"    → WEAK concentration (costs evenly distributed)")

    # Gini coefficient (inequality measure)
    sorted_costs = entity_costs.get_column('total_cost').sort().to_numpy()
    n = len(sorted_costs)
    cumsum = np.cumsum(sorted_costs)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_costs)) / (n * cumsum[-1]) - (n + 1) / n

    print(f"\n  Gini Coefficient: {gini:.3f}")
    if gini > 0.7:
        print(f"    → HIGH inequality (0.7-1.0)")
    elif gini > 0.4:
        print(f"    → MODERATE inequality (0.4-0.7)")
    else:
        print(f"    → LOW inequality (0-0.4)")

print("\n" + "=" * 80)
print("2. TEMPORAL STABILITY BY ENTITY")
print("=" * 80)

# Analyze temporal variability for top entities
for entity_col in ['cloud_provider']:
    print(f"\n{entity_col.upper()} Temporal Analysis:")
    print("-" * 60)

    # Use detect_entity_anomalies to find most variable entities
    anomalies = detect_entity_anomalies(
        df_clean,
        entity_col=entity_col,
        date_col='usage_date',
        min_days=5,
        top_n=5
    )

    print(f"\n  Top 5 most variable entities (by CV):")
    for idx, row in enumerate(anomalies.iter_rows(named=True), 1):
        print(f"    {idx}. {row['entity']}: CV={row['cv']:.3f}, "
              f"avg={row['mean_daily_records']:,.0f} records/day, "
              f"days={row['days_present']}")

print("\n" + "=" * 80)
print("3. NORMALIZED COST CONTRIBUTIONS OVER TIME")
print("=" * 80)

# Normalize costs by provider to remove volume effects
print("\nCloud Provider Normalized Contributions:")
print("-" * 60)

provider_normalized = normalize_by_period(
    df_clean,
    entity_col='cloud_provider',
    metric_col=COST_COL,
    time_col='usage_date',
    freq='1d'
)

# Show sample
print("\nSample (first 5 days):")
print(provider_normalized.head(15))

# Compute stability of normalized contributions
print("\nTemporal stability of normalized contributions:")
for provider in provider_normalized['cloud_provider'].unique().to_list():
    provider_data = provider_normalized.filter(pl.col('cloud_provider') == provider)

    normalized_series = provider_data.get_column('metric_normalized').to_numpy()
    mean_contribution = normalized_series.mean()
    std_contribution = normalized_series.std()
    cv = std_contribution / mean_contribution if mean_contribution > 0 else 0

    print(f"  {provider}: mean={mean_contribution:.3f}, std={std_contribution:.3f}, CV={cv:.3f}")

print("\n" + "=" * 80)
print("4. COST DISTRIBUTION CHARACTERISTICS")
print("=" * 80)

# Use cost_distribution_metrics
dist = cost_distribution_metrics(df_clean, COST_COL)

print(f"\nPrimary cost metric: {COST_COL}")
print(f"\nDistribution Percentiles:")
for percentile, value in dist['percentiles'].items():
    print(f"  P{percentile:3d}: ${value:>12,.6f}")

print(f"\nDistribution Characteristics:")
print(f"  Skewness: {dist['skewness']}")
if dist['skewness'] > 2:
    print(f"    → HIGHLY right-skewed (long tail of expensive resources)")
elif dist['skewness'] > 1:
    print(f"    → MODERATELY right-skewed")
else:
    print(f"    → RELATIVELY symmetric")

print(f"\n  Outliers (IQR method, k=1.5):")
print(f"    Count: {dist['outlier_count_iqr']:,} ({dist['outlier_pct']:.2f}%)")
print(f"    Modeling recommendation: {dist['modeling_rec']}")

print("\n" + "=" * 80)
print("5. TIME SERIES PROPERTIES")
print("=" * 80)

# Use temporal_quality_metrics
quality = temporal_quality_metrics(
    df_clean,
    date_col='usage_date',
    metric_col=COST_COL
)

print(f"\nTemporal Coverage:")
print(f"  Date range: {quality['date_range'][0]} to {quality['date_range'][1]}")
print(f"  Completeness: {quality['coverage_days']}/{quality['expected_days']} days ({quality['completeness_pct']}%)")

print(f"\nStability Metrics:")
print(f"  Record volume CV: {quality['record_volume_cv']:.4f}")
if quality['record_volume_cv'] < 0.15:
    print(f"    → STABLE record volume")
else:
    print(f"    → VARIABLE record volume")

print(f"\n  Cost lag-1 autocorrelation: {quality['metric_lag1_autocorr']:.4f}")
if quality['metric_lag1_autocorr'] > 0.7:
    print(f"    → HIGH autocorrelation (sticky infrastructure costs)")
elif quality['metric_lag1_autocorr'] > 0.5:
    print(f"    → MODERATE autocorrelation")
else:
    print(f"    → LOW autocorrelation (volatile spending)")

print(f"\n  Overall classification: {quality['stability_class'].upper()}")

# Compute lag-7 and lag-30 autocorrelation manually
daily_costs = (
    df_clean
    .group_by('usage_date')
    .agg(pl.col(COST_COL).sum().alias('daily_cost'))
    .sort('usage_date')
    .collect()
    .get_column('daily_cost')
    .to_numpy()
)

from scipy.stats import pearsonr

if len(daily_costs) > 7:
    lag7_corr, _ = pearsonr(daily_costs[:-7], daily_costs[7:])
    print(f"\n  Cost lag-7 autocorrelation: {lag7_corr:.4f}")

if len(daily_costs) > 30:
    lag30_corr, _ = pearsonr(daily_costs[:-30], daily_costs[30:])
    print(f"  Cost lag-30 autocorrelation: {lag30_corr:.4f}")

# Check for seasonality (day of week pattern)
daily_with_dow = (
    df_clean
    .group_by('usage_date')
    .agg(pl.col(COST_COL).sum().alias('daily_cost'))
    .collect()
    .with_columns([
        pl.col('usage_date').dt.weekday().alias('day_of_week')
    ])
)

dow_stats = (
    daily_with_dow
    .group_by('day_of_week')
    .agg([
        pl.col('daily_cost').mean().alias('mean_cost'),
        pl.col('daily_cost').std().alias('std_cost')
    ])
    .sort('day_of_week')
)

print(f"\nDay-of-week seasonality:")
dow_cv = (dow_stats['mean_cost'].std() / dow_stats['mean_cost'].mean())
print(f"  Coefficient of variation across weekdays: {dow_cv:.4f}")
if dow_cv > 0.1:
    print(f"    → SEASONALITY detected (>10% variation by day of week)")
else:
    print(f"    → NO significant day-of-week pattern")

print("\n" + "=" * 80)
print("SUMMARY: KEY FINDINGS")
print("=" * 80)

print("\n1. FREQUENCY DETERMINATION:")
print("   - Grain: ONE ROW per (resource_id, usage_date)")
print(f"   - ~{total_rows / date_range[0, 2]:,.0f} unique resources tracked daily")
print("   - This is RESOURCE-LEVEL daily data, not aggregated")

print("\n2. COST CONCENTRATION:")
provider_top1_pct = entity_costs.filter(pl.col('cloud_provider') == entity_costs['cloud_provider'][0])['pct_total'][0] if 'cloud_provider' in entity_types else 0
print(f"   - Top provider: {provider_top1_pct:.1f}% of total cost")
print(f"   - Gini coefficient: {gini:.3f} (inequality measure)")

print("\n3. TIME SERIES CHARACTERISTICS:")
print(f"   - Lag-1 autocorr: {quality['metric_lag1_autocorr']:.3f} (day-to-day persistence)")
print(f"   - Stability: {quality['stability_class']}")
print(f"   - Skewness: {dist['skewness']:.2f} (distribution shape)")

print("\n4. MODELING IMPLICATIONS:")
print(f"   - Recommended transformation: {dist['modeling_rec']}")
print(f"   - Temporal modeling viable: {'YES' if quality['metric_lag1_autocorr'] > 0.5 else 'NO'}")
print(f"   - Concentration suggests: {'Entity-specific models needed' if gini > 0.7 else 'Aggregate models sufficient'}")

print("\n" + "=" * 80)
