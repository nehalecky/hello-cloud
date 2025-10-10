"""
Analyze CloudZero PiedPiper dataset to determine actual temporal frequency.

Goal: Understand what creates 157K rows/day when data is supposedly "daily".
"""

import polars as pl
from pathlib import Path
from datetime import date

# Dataset path
DATA_PATH = Path('/Users/nehalecky/Projects/cloudzero/cloud-resource-simulator/data/piedpiper_optimized_daily.parquet')

# Load as LazyFrame
df = pl.scan_parquet(DATA_PATH)

# Filter to clean data (Sept 1 - Oct 6, 2025) based on notebook findings
COLLAPSE_DATE = date(2025, 10, 7)
df_clean = df.filter(pl.col('usage_date') < COLLAPSE_DATE)

print("=" * 80)
print("FREQUENCY ANALYSIS - CloudZero PiedPiper Dataset")
print("=" * 80)

# Basic statistics
total_rows = df_clean.select(pl.len()).collect()[0, 0]
date_stats = df_clean.select([
    pl.col('usage_date').min().alias('min_date'),
    pl.col('usage_date').max().alias('max_date'),
    pl.col('usage_date').n_unique().alias('unique_days')
]).collect()

min_date = date_stats['min_date'][0]
max_date = date_stats['max_date'][0]
unique_days = date_stats['unique_days'][0]

print(f"\nDataset Summary (Clean Data):")
print(f"  Total rows: {total_rows:,}")
print(f"  Date range: {min_date} to {max_date}")
print(f"  Unique days: {unique_days}")
print(f"  Rows per day: {total_rows / unique_days:,.0f}")

# CRITICAL QUESTION: What defines uniqueness of each row?
# Hypothesis: Each row is NOT one-per-day, but one-per-(resource, day) or similar

print("\n" + "=" * 80)
print("GRAIN ANALYSIS - What makes each row unique?")
print("=" * 80)

# Get schema
schema = df_clean.collect_schema()
print(f"\nTotal columns: {len(schema)}")

# Test different grain hypotheses
print("\nTesting grain hypotheses:")

# Hypothesis 1: usage_date only (should be ~37 rows if daily)
daily_grain = df_clean.group_by('usage_date').agg(pl.len().alias('count')).collect()
print(f"\n1. Grain = (usage_date):")
print(f"   Unique combinations: {len(daily_grain)}")
print(f"   Matches total rows? {len(daily_grain) == total_rows}")
print(f"   → NOT the grain (too few unique values)")

# Hypothesis 2: (usage_date, resource_id)
resource_daily_grain = df_clean.group_by(['usage_date', 'resource_id']).agg(pl.len().alias('count')).collect()
print(f"\n2. Grain = (usage_date, resource_id):")
print(f"   Unique combinations: {len(resource_daily_grain):,}")
print(f"   Matches total rows? {len(resource_daily_grain) == total_rows}")
if len(resource_daily_grain) == total_rows:
    print(f"   ✓ THIS IS THE GRAIN!")
else:
    print(f"   Difference: {abs(len(resource_daily_grain) - total_rows):,} rows")

# Hypothesis 3: uuid (should be unique if it's a primary key)
uuid_grain = df_clean.select(pl.col('uuid').n_unique()).collect()[0, 0]
print(f"\n3. Grain = (uuid):")
print(f"   Unique values: {uuid_grain:,}")
print(f"   Matches total rows? {uuid_grain == total_rows}")
if uuid_grain == total_rows:
    print(f"   ✓ UUID is a unique primary key!")

# Hypothesis 4: (usage_date, cloud_provider, resource_id)
provider_resource_grain = df_clean.group_by(['usage_date', 'cloud_provider', 'resource_id']).agg(pl.len().alias('count')).collect()
print(f"\n4. Grain = (usage_date, cloud_provider, resource_id):")
print(f"   Unique combinations: {len(provider_resource_grain):,}")
print(f"   Matches total rows? {len(provider_resource_grain) == total_rows}")

# Check for duplicates in any grain
duplicates = resource_daily_grain.filter(pl.col('count') > 1)
if len(duplicates) > 0:
    print(f"\n⚠ DUPLICATES FOUND in (usage_date, resource_id) grain:")
    print(f"   {len(duplicates):,} combinations have multiple rows")
    print(f"   Example:")
    print(duplicates.head(5))

print("\n" + "=" * 80)
print("ENTITY CARDINALITY ANALYSIS")
print("=" * 80)

# Analyze cardinality of key dimensions
entity_cols = ['cloud_provider', 'resource_id', 'uuid']

for col in entity_cols:
    if col in schema.names():
        unique_count = df_clean.select(pl.col(col).n_unique()).collect()[0, 0]
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count:,}")
        print(f"  Cardinality ratio: {unique_count / total_rows:.6f}")

        # Show top 5 values by frequency
        top_values = (
            df_clean
            .group_by(col)
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .head(5)
            .collect()
        )
        print(f"  Top 5 values:")
        for row in top_values.iter_rows(named=True):
            print(f"    {row[col]}: {row['count']:,} rows")

print("\n" + "=" * 80)
print("RESOURCE-LEVEL TEMPORAL ANALYSIS")
print("=" * 80)

# How many unique resources per day?
resources_per_day = (
    df_clean
    .group_by('usage_date')
    .agg([
        pl.col('resource_id').n_unique().alias('unique_resources'),
        pl.len().alias('total_rows')
    ])
    .sort('usage_date')
    .collect()
)

print(f"\nResources per day (first 5 days):")
print(resources_per_day.head(5))

print(f"\nResources per day statistics:")
stats = resources_per_day.select([
    pl.col('unique_resources').mean().alias('mean'),
    pl.col('unique_resources').median().alias('median'),
    pl.col('unique_resources').std().alias('std'),
    pl.col('unique_resources').min().alias('min'),
    pl.col('unique_resources').max().alias('max')
]).to_dicts()[0]

for metric, value in stats.items():
    print(f"  {metric}: {value:,.0f}")

print("\n" + "=" * 80)
print("CONCLUSION: ACTUAL FREQUENCY")
print("=" * 80)

# Final determination
if uuid_grain == total_rows and len(resource_daily_grain) == total_rows:
    print("\n✓ GRAIN IDENTIFIED:")
    print("  Each row represents: ONE RESOURCE on ONE DAY")
    print("  Uniqueness key: (usage_date, resource_id)")
    print("  Alternative key: uuid (synthetic unique identifier)")
    print(f"\n  Interpretation:")
    print(f"    - {total_rows:,} total rows = {unique_days} days × ~{total_rows / unique_days:,.0f} resources/day")
    print(f"    - This is RESOURCE-LEVEL daily data, not account-level or aggregated")
    print(f"    - Each resource that existed on each day gets one row")
elif uuid_grain == total_rows:
    print("\n✓ GRAIN IDENTIFIED:")
    print("  Each row is uniquely identified by UUID")
    print("  BUT grain may be more complex than (usage_date, resource_id)")
    print("  → Further investigation needed")
else:
    print("\n⚠ GRAIN UNCLEAR - multiple rows share same UUID or keys")
    print("  → Data quality issue or complex grain structure")

print("\n" + "=" * 80)
