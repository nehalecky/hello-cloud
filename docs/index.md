---
title: " "
---

<div align="center">
  <!-- Theme-adaptive logo: dark grey text for light mode, light grey text for dark mode -->
  <picture>
    <source srcset="assets/logo-full-dark.png" media="(prefers-color-scheme: dark)">
    <img src="assets/logo-full-light.png" alt="Hello Cloud" style="width: 80%; max-width: 550px;">
  </picture>

  <p style="font-size: 1.2em; margin-top: 1.5em;"><strong>Time series forecasting and anomaly detection for cloud resources.</strong></p>
</div>

<div style="height: 2em;"></div>

## Overview

Hello Cloud is a Python library for modeling cloud resource utilization patterns, forecasting future usage, and detecting anomalies in operational metrics. Built on empirical research showing 25-35% cloud waste and surprisingly low average utilization (12-15% CPU, 18-25% memory).

**Core Capabilities:**

- **Workload Characterization** - Generate realistic synthetic cloud metrics based on empirical patterns
- **Time Series Forecasting** - Gaussian Processes, ARIMA, and foundation models (Chronos, TimesFM)
- **Hierarchical Analysis** - Multi-level cost analysis across providers, accounts, and resources
- **Anomaly Detection** - Statistical and ML-based approaches for operational metrics
- **PySpark Integration** - Distributed processing for local development and production scale

## Architecture

**Technology Stack:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Processing** | PySpark 4.0 | Distributed DataFrames (local & scale) |
| **Statistical Models** | GPyTorch | Gaussian Process time series forecasting |
| **Probabilistic Models** | PyMC | Bayesian hierarchical inference |
| **Foundation Models** | Chronos, TimesFM | Pre-trained forecasters (optional) |
| **Notebooks** | Jupyter + MyST | Interactive tutorials with hot reload |
| **Documentation** | MkDocs Material | Live docs with API reference |

**Design Principles:**

- **Empirically Grounded** - All synthetic data based on published cloud utilization research
- **Production Ready** - 92% test coverage on GP library, comprehensive CI/CD
- **Research Informed** - Implements patterns from 35+ academic papers
- **Developer Friendly** - Hot reload notebooks, comprehensive docs, type hints

## Quick Examples

### Generate Synthetic Cloud Metrics

```python
from hellocloud.data_generation import WorkloadPatternGenerator, WorkloadType

# Generate realistic web app metrics
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    interval_minutes=60
)

# Built-in patterns: WEB_APP, BATCH_JOB, MICROSERVICE, DATABASE,
# ML_TRAINING, STREAMING, SEASONAL_BATCH, DEV_ENVIRONMENT, and more
```

### Time Series Forecasting with Gaussian Processes

```python
from hellocloud.ml_models.gaussian_process import SparseGPModel
from pyspark.sql import functions as F

# Load your time series data
df = spark.read.parquet('cloud_metrics.parquet')

# Prepare data
train_df = df.filter(F.col('date') < '2024-01-01')
test_df = df.filter(F.col('date') >= '2024-01-01')

# Train GP model
model = SparseGPModel(
    num_inducing=100,
    fast_period=24.0,  # Daily pattern
    slow_period=168.0   # Weekly pattern
)
model.fit(train_df)

# Forecast
predictions = model.predict(test_df)
```

### Hierarchical Cost Analysis

```python
from hellocloud.timeseries import TimeSeries, PiedPiperLoader

# Load hierarchical billing data
loader = PiedPiperLoader(data_path='billing_data.parquet')
ts = loader.load()

# Filter to specific account
aws_ts = ts.filter(provider='aws', account='123456789')

# Aggregate to daily spend
daily = aws_ts.aggregate('daily', time_col='date', value_col='cost')

# Summary statistics
stats = daily.summary_stats(value_col='cost')
print(stats.toPandas())
```

## Documentation Sections

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="border-left: 4px solid var(--md-primary-fg-color); padding-left: 1em;">

**[📚 Notebooks](notebooks/index.md)**

Interactive tutorials with code and outputs. Try them in Google Colab or run locally. Covers workload patterns, time series analysis, Gaussian processes, and forecasting.

</div>

<div style="border-left: 4px solid var(--md-primary-fg-color); padding-left: 1em;">

**[💡 Concepts](concepts/index.md)**

Deep dives into research and architecture. Empirical cloud utilization patterns, Gaussian process design, foundation model evaluations, and time series anomaly detection.

</div>

<div style="border-left: 4px solid var(--md-primary-fg-color); padding-left: 1em;">

**[📖 API Reference](reference/index.md)**

Auto-generated documentation from source code. Complete function signatures, parameters, return types, and usage examples for all public APIs.

</div>

</div>

## Research Context

- CPU Utilization: 12-15% average
- Memory Utilization: 18-25% average
- Resource Waste: 25-35% of cloud spending
- Temporal Autocorrelation: 0.7-0.8

See [Cloud Resource Patterns Research](concepts/research/cloud-resource-patterns-research.md).
