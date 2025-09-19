# CloudZero AI Simulation Platform

## Overview

This project explores the cloud cost optimization problem space through advanced synthetic data generation and machine learning approaches. Based on empirical research revealing shocking inefficiencies in cloud resource utilization.

## Core Problem

The cloud computing industry faces a massive inefficiency crisis:
- **$226 billion** wasted annually (30-32% of total spend)
- **13% average CPU utilization** across infrastructure
- **20% average memory utilization**
- **70% waste in development environments**

This project builds tools to understand, model, and address these challenges.

## Key Innovations

### 1. Realistic Cloud Resource Simulation
- Based on empirical research from major cloud providers
- 20+ application archetypes with distinct resource patterns
- Multivariate correlation modeling using PyMC
- Temporal patterns (daily, weekly, seasonal)

### 2. Foundation Model Integration
- Amazon Chronos for zero-shot time series forecasting
- TimesFM and TiReX models for advanced predictions
- Ensemble approaches for robust forecasting
- Uncertainty quantification in all predictions

### 3. Hierarchical Bayesian Modeling
- Industry → Archetype → Resource hierarchy
- Learns from real data and generates synthetic patterns
- Quantifies uncertainty in all parameters
- Bayesian updating as more data becomes available

## Quick Start

### Installation with uv

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/cloudzero-ai-simulation.git
cd cloudzero-ai-simulation

# Create environment and install
uv venv
source .venv/bin/activate
uv sync --all-extras  # Installs everything from pyproject.toml
```

### Basic Usage

```python
from cloudzero_sim.data_generation import WorkloadPatternGenerator, WorkloadType
from datetime import datetime, timedelta

# Generate realistic workload patterns
generator = WorkloadPatternGenerator()
df = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    interval_minutes=60
)

# Validate against research
print(f"Average CPU: {df['cpu_utilization'].mean():.1f}%")  # ~15%
print(f"Average Memory: {df['memory_utilization'].mean():.1f}%")  # ~35%
print(f"Waste: {df['waste_percentage'].mean():.1f}%")  # ~35%
```

### Advanced Forecasting

```python
from cloudzero_sim.ml_models import CloudCostForecaster

# Initialize forecaster with ensemble
forecaster = CloudCostForecaster(ensemble=True)

# Generate cost forecast
results = forecaster.forecast_unit_economics(
    df=df,
    horizon=48,  # 48 hours ahead
    metrics=["hourly_cost", "cpu_utilization", "efficiency_score"]
)
```

## Documentation Structure

- [Architecture](architecture.md) - System design and components
- [Data Generation](data_generation.md) - Synthetic data creation methodology
- [ML Models](ml_models.md) - Forecasting and optimization approaches
- [Research](research/README.md) - Empirical foundations
- [API Reference](api_reference.md) - Complete API documentation

## Research Foundation

This project synthesizes findings from:
- Academic research (35+ papers)
- Industry reports (Gartner, FinOps Foundation)
- Cloud provider documentation (AWS, Azure, GCP)
- Real-world case studies (Netflix, Uber, Shopify)

Key findings that inform our models:
- **CPU utilization**: 13% average (shocking but documented)
- **Memory utilization**: 20% average
- **Waste percentage**: 30-32% across industries
- **Development environments**: 70% waste (worst category)
- **Batch processing**: 60% waste
- **ML/GPU workloads**: 45% waste from idle periods

## The "Workload Genome" Vision

Like the Human Genome Project mapped human DNA, this project aims to map the "DNA" of cloud workloads:

1. **Comprehensive taxonomy** of application patterns
2. **Standardized dataset** for benchmarking
3. **Open-source contribution** to accelerate innovation
4. **Industry collaboration** to refine patterns

## Contributing

We welcome contributions in:
- Additional workload patterns
- Empirical correlation data
- Optimization algorithms
- Visualization improvements
- Documentation enhancements

## License

MIT - Open for research and commercial use