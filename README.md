# Cloud Resource Utilization Simulator

## An Exploratory Framework for Bayesian Modeling of Cloud Workload Patterns

This research project explores cloud resource utilization patterns through probabilistic modeling and synthetic data generation. Using hierarchical Bayesian approaches with PyMC, we investigate the documented inefficiencies in cloud infrastructure utilization and develop frameworks for understanding optimization opportunities.

## Research Objectives

This exploratory project investigates:
- Probabilistic modeling of cloud resource utilization patterns
- Correlation structures between compute, memory, and I/O metrics
- Hierarchical Bayesian approaches to workload characterization
- Time series forecasting using foundation models
- Synthetic data generation for optimization algorithm development

## Technical Approach

### Probabilistic Modeling with PyMC
- Hierarchical Bayesian models capturing industry, application, and resource-level patterns
- Multivariate distributions with empirically-informed correlation structures
- Uncertainty quantification throughout the modeling pipeline

### Synthetic Data Generation
- Research-based workload patterns (informed by published utilization studies)
- Multiple application archetypes with distinct resource signatures
- Temporal patterns including daily, weekly, and seasonal variations

### Time Series Analysis
- Integration with foundation models (Amazon Chronos, Google TimesFM)
- Ensemble forecasting approaches
- Zero-shot prediction capabilities

## Quick Start

### Installation with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/cloud-resource-simulator.git
cd cloud-resource-simulator

# Create virtual environment and sync dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --all-extras  # Install all dependencies from pyproject.toml
```

### Basic Usage

```python
from cloud_sim.data_generation import WorkloadPatternGenerator, WorkloadType
from datetime import datetime, timedelta

# Generate synthetic workload data
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

# Analyze utilization patterns
print(f"Mean CPU Utilization: {data['cpu_utilization'].mean():.1f}%")
print(f"Mean Memory Utilization: {data['memory_utilization'].mean():.1f}%")
```

## Empirical Basis

The simulation parameters are informed by published research on cloud utilization:

| Metric | Literature Values | Model Range |
|--------|------------------|-------------|
| CPU Utilization | 12-15% | 13-15% |
| Memory Utilization | 18-25% | 18-22% |
| Resource Waste | 25-35% | 28-35% |
| Development Environment Efficiency | 25-35% | 28-32% |
| Batch Workload Efficiency | 35-45% | 38-42% |

## The Workload Genome Initiative

This project contributes to developing a comprehensive taxonomy of cloud workload patterns. By establishing standardized workload characterizations and synthetic datasets, we aim to:

1. Enable reproducible benchmarking of optimization algorithms
2. Provide training data for machine learning models
3. Facilitate research collaboration through shared datasets
4. Advance the field of cloud resource optimization

## Project Structure

```
cloud-resource-simulator/
├── pyproject.toml              # Package configuration and dependencies
├── src/cloud_sim/              # Core simulation modules
├── tests/                      # Test suite
├── docs/                       # Extended documentation and research
├── notebooks/myst/             # Reproducible analysis notebooks
└── data/                       # Synthetic datasets
```

## Key Dependencies

- **Polars**: DataFrame operations and data processing
- **PyMC**: Probabilistic programming and Bayesian inference
- **HuggingFace Ecosystem**: Datasets and model integration
- **Chronos/TimesFM**: Foundation models for time series
- **Jupytext**: Notebook version control with MyST markdown
- **uv**: Python package management

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Data Generation Methods](docs/data_generation.md)
- [ML Models and Forecasting](docs/ml_models.md)
- [Research References](docs/research/)

## Contributing

This exploratory research project welcomes contributions in:
- Empirical correlation data from production systems
- Additional workload characterization patterns
- Bayesian model refinements
- Forecasting algorithm improvements

## License

MIT License - Open for research and commercial use