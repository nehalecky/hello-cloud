---
title: "API Reference"
---

# API Reference

Comprehensive API documentation generated from Python docstrings, covering all public interfaces for data generation, time series operations, and model building.

## Data Generation

Generate realistic synthetic cloud workload data based on [empirical research](../research/cloud-resource-patterns-research.md) showing 12-15% average CPU utilization and 25-35% cloud waste. Use these classes to create time series that match real-world patterns for testing forecasting models, building demos, or training ML systems.

**Key capabilities:**

- **20+ workload archetypes** - Web apps, databases, ML training, batch jobs, and more
- **Research-grounded defaults** - All parameters based on [published cloud infrastructure studies](../research/cloud-resource-patterns-research.md#references)
- **Temporal patterns** - Business hours, weekly cycles, seasonal trends
- **Anomaly injection** - Realistic spikes, drops, and operational incidents

**Example use case:** Generate 30 days of synthetic CPU/memory metrics for a web application to test your forecasting pipeline without requiring production data.

::: hellocloud.generation.WorkloadPatternGenerator
    options:
      show_root_heading: true
      show_root_full_path: true
      show_source: false
      members:
        - generate_time_series

::: hellocloud.generation.WorkloadType
    options:
      show_root_heading: true
      show_root_full_path: true

## TimeSeries API

Hierarchical time series operations built on PySpark for analyzing cloud billing data, metrics, and KPIs. Designed for datasets with entity hierarchies (accounts > regions > services) where you need to filter, aggregate, and analyze across multiple granularities.

**Key capabilities:**

- **Hierarchical operations** - Filter, sample, and aggregate across entity hierarchies
- **PySpark-native** - Distributed processing for large-scale datasets
- **EDA-optimized** - Built-in summary statistics, time range detection, and data quality checks
- **Flexible loaders** - Pre-configured loaders for common cloud data sources (PiedPiper, CloudZero)

**Example use case:** Load 2 years of multi-account AWS billing data, filter to specific services, aggregate to monthly grain, and compute summary statistics for cost optimization analysis.

::: hellocloud.timeseries.TimeSeries
    options:
      show_root_heading: true
      show_root_full_path: true
      show_source: false

::: hellocloud.io.PiedPiperLoader
    options:
      show_root_heading: true
      show_root_full_path: true
      show_source: false
