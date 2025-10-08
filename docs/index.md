# Technical Documentation

## Architecture Overview

The Cloud Resource Utilization Simulator implements a three-tier architecture for probabilistic modeling of cloud workload patterns.

### System Components

```
┌─────────────────────────────────────────────────┐
│             Application Layer                    │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │  Streamlit   │  │  Jupyter Notebooks   │    │
│  │  Dashboard   │  │  (MyST Markdown)      │    │
│  └──────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│             ML/Forecasting Layer                 │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │   Chronos    │  │  PyMC Hierarchical   │    │
│  │   TimesFM    │  │  Bayesian Models     │    │
│  │   TiReX      │  │                      │    │
│  └──────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│          Data Generation Layer                   │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │  Workload    │  │  Cloud Metrics       │    │
│  │  Patterns    │  │  Simulator           │    │
│  └──────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Core Modules

### cloud_sim.data_generation

The data generation module provides realistic synthetic cloud resource data based on empirical research.

#### WorkloadPatternGenerator

Generates time series data for different workload types with realistic utilization patterns.

```python
from cloud_sim.data_generation import WorkloadPatternGenerator, WorkloadType

generator = WorkloadPatternGenerator(seed=42)

# Available workload types
workload_types = [
    WorkloadType.WEB_APP,           # Web applications
    WorkloadType.BATCH_PROCESSING,  # Batch jobs
    WorkloadType.ML_TRAINING,       # Machine learning
    WorkloadType.DATABASE_OLTP,     # Transactional DB
    WorkloadType.DATABASE_OLAP,     # Analytical DB
    WorkloadType.STREAMING,         # Stream processing
    WorkloadType.DEV_ENVIRONMENT,   # Development
    WorkloadType.MICROSERVICES,     # Microservices
    WorkloadType.SERVERLESS,        # Lambda/Functions
    WorkloadType.CONTAINER,         # Containerized
    WorkloadType.HPC,               # High-performance
    WorkloadType.CDN                # Content delivery
]

# Generate with specific characteristics
df = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    interval_minutes=60,
    include_anomalies=True,
    anomaly_rate=0.02
)
```

**Generated Metrics:**
- `cpu_utilization`: CPU usage percentage (0-100)
- `memory_utilization`: Memory usage percentage (0-100)
- `network_in_mbps`: Network ingress (Mbps)
- `network_out_mbps`: Network egress (Mbps)
- `disk_iops`: Disk I/O operations per second
- `efficiency_score`: Calculated efficiency metric (0-1)
- `is_idle`: Boolean flag for idle detection
- `is_overprovisioned`: Boolean flag for overprovisioning
- `waste_percentage`: Estimated resource waste (0-100)

#### CloudMetricsSimulator

Lower-level simulator for custom resource generation scenarios.

```python
from cloud_sim.data_generation import CloudMetricsSimulator

simulator = CloudMetricsSimulator(
    num_resources=100,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    sampling_interval_minutes=5,
    cloud_providers=["AWS", "Azure", "GCP"],
    resource_types=["compute", "storage", "network"]
)

df = simulator.generate_dataset(
    include_anomalies=True,
    anomaly_rate=0.05,
    include_unit_economics=True
)
```

### cloud_sim.ml_models

Machine learning and forecasting capabilities built on foundation models and Bayesian inference.

#### PyMC Hierarchical Models

```python
from cloud_sim.ml_models import CloudResourceHierarchicalModel

# Three-level hierarchy: Industry → Application → Resource
model = CloudResourceHierarchicalModel()

# Fit to observed data
model.fit(
    industry_data=industry_df,
    application_data=app_df,
    resource_data=resource_df
)

# Generate posterior predictions
predictions = model.predict(
    industry="retail",
    application="web_app",
    n_samples=1000
)

# Access uncertainty estimates
uncertainty = model.get_uncertainty_intervals(
    predictions,
    credible_interval=0.95
)
```

#### Foundation Model Integration

```python
from cloud_sim.ml_models import FoundationForecaster

# Initialize with ensemble of models
forecaster = FoundationForecaster(
    models=["chronos", "timesfm", "tirex"],
    ensemble_method="weighted_average"
)

# Zero-shot forecasting
forecast = forecaster.predict(
    historical_data=df,
    horizon=168,  # 1 week ahead
    frequency="H"  # Hourly
)

# Get prediction intervals
intervals = forecaster.prediction_intervals(
    confidence_levels=[0.5, 0.9, 0.95]
)
```

#### Application Taxonomy

Comprehensive taxonomy of cloud workload patterns with empirically-derived characteristics.

```python
from cloud_sim.ml_models import ApplicationTaxonomy

taxonomy = ApplicationTaxonomy()

# Get archetype characteristics
archetype = taxonomy.get_archetype("ecommerce_platform")
print(f"CPU P50: {archetype.resource_pattern.cpu_p50}%")
print(f"CPU P95: {archetype.resource_pattern.cpu_p95}%")
print(f"Memory correlation with CPU: {archetype.correlation_matrix['cpu']['memory']}")

# Generate synthetic data for archetype
synthetic_data = taxonomy.generate_for_archetype(
    archetype_name="ecommerce_platform",
    duration_hours=720,  # 1 month
    include_seasonality=True
)
```

## Bayesian Modeling Framework

### Hierarchical Structure

The framework implements a three-level Bayesian hierarchy:

1. **Industry Level** (Hyperpriors)
   - Global parameters shared across all industries
   - Captures fundamental resource utilization patterns
   - Prior: `Normal(μ=15, σ=5)` for CPU utilization

2. **Application Archetype Level**
   - Industry-specific parameters for each application type
   - Captures workload-specific patterns
   - Prior: `Normal(μ=industry_mean, σ=industry_std * 0.5)`

3. **Resource Level**
   - Individual resource observations
   - Captures instance-specific variations
   - Likelihood: `Beta(α=archetype_α, β=archetype_β)`

### Correlation Modeling

Multivariate correlations between resource metrics using LKJ priors:

```python
import pymc as pm

with pm.Model() as correlation_model:
    # LKJ prior for correlation matrix
    corr = pm.LKJCorr('corr', n=4, eta=2)  # 4 metrics

    # Cholesky decomposition for efficiency
    chol = pm.expand_packed_triangular(4, corr)

    # Multivariate normal with correlations
    mv = pm.MvNormal('metrics',
                     mu=[15, 25, 10, 5],  # CPU, Memory, Network, Disk
                     chol=chol,
                     observed=data)
```

## Advanced Features

### Anomaly Detection

Statistical anomaly detection using isolation forests and Bayesian change point detection:

```python
from cloud_sim.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(
    method="isolation_forest",
    contamination=0.05
)

# Train on normal data
detector.fit(normal_data)

# Detect anomalies
anomalies = detector.predict(new_data)
anomaly_scores = detector.decision_function(new_data)
```

### Cost Optimization Recommendations

Optimization engine that suggests resource adjustments:

```python
from cloud_sim.optimization import CostOptimizer

optimizer = CostOptimizer()

# Analyze resource usage
analysis = optimizer.analyze(
    usage_data=df,
    cost_model="aws_on_demand",
    optimization_goal="minimize_waste"
)

# Get recommendations
recommendations = optimizer.recommend(
    target_utilization=0.7,
    safety_margin=0.2
)

print(f"Potential savings: ${recommendations.savings_estimate:,.2f}")
print(f"Rightsizing opportunities: {recommendations.rightsizing_count}")
```

## Data Formats

### HuggingFace Dataset Integration

Export synthetic data to HuggingFace datasets format:

```python
from cloud_sim.data_generation import HFDatasetBuilder

builder = HFDatasetBuilder()

# Create dataset with metadata
dataset = builder.create_dataset(
    data=synthetic_df,
    name="cloud-resource-sim-1M",
    description="1M synthetic cloud resource observations",
    features={
        "timestamp": "timestamp",
        "cpu_utilization": "float32",
        "memory_utilization": "float32",
        "workload_type": "category"
    }
)

# Push to HuggingFace Hub
dataset.push_to_hub("username/cloud-resource-sim-1M")
```

### Polars DataFrame Operations

All data operations use Polars for performance:

```python
import polars as pl

# Efficient aggregations
hourly_stats = (
    df.lazy()
    .group_by_dynamic("timestamp", every="1h")
    .agg([
        pl.col("cpu_utilization").mean().alias("cpu_mean"),
        pl.col("cpu_utilization").std().alias("cpu_std"),
        pl.col("memory_utilization").mean().alias("mem_mean"),
        pl.col("waste_percentage").mean().alias("waste_mean")
    ])
    .collect()
)

# Window functions for time series
df_with_lag = df.with_columns([
    pl.col("cpu_utilization").shift(1).alias("cpu_lag1"),
    pl.col("cpu_utilization").rolling_mean(24).alias("cpu_24h_ma")
])
```

## Performance Considerations

### Memory Optimization

- Use Polars lazy evaluation for large datasets
- Batch processing for time series generation
- Incremental model updates for Bayesian inference

### Computation Optimization

- Vectorized operations with NumPy/JAX
- Parallel sampling with PyMC
- GPU acceleration for foundation models (when available)

## API Reference

See module docstrings for detailed API documentation:

```python
from cloud_sim import data_generation
help(data_generation.WorkloadPatternGenerator)

from cloud_sim import ml_models
help(ml_models.CloudResourceHierarchicalModel)
```

## Research References

The implementation is based on extensive empirical research:

- [Cloud Resource Patterns Research](research/cloud-resource-patterns-research.md) - 35+ citations on utilization patterns
- [Resource Correlation Analysis](research/cloud-resource-correlations-report.md) - Multivariate correlation structures
- [Research Overview](research/) - Summary and key findings

## Development Guidelines

### Testing

```bash
# Run test suite
uv run pytest

# With coverage
uv run pytest --cov=cloud_sim --cov-report=html

# Specific test file
uv run pytest tests/test_data_generation.py
```

### Code Quality

```bash
# Format code
uv run black src/
uv run ruff format src/

# Lint
uv run ruff check src/

# Type checking
uv run mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.