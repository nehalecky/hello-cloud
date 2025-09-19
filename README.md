# Cloud Resource Utilization Simulator

A Bayesian modeling framework for cloud workload patterns and synthetic data generation using PyMC.

## Overview

This research project explores cloud resource utilization patterns through hierarchical Bayesian models, generating realistic synthetic data for optimization algorithm development. The framework is based on [empirical research](docs/research/) showing 12-15% average CPU utilization and 25-35% resource waste across cloud infrastructure.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/cloud-resource-simulator.git
cd cloud-resource-simulator

# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --all-extras
```

### Basic Usage

```python
from cloud_sim.data_generation import WorkloadPatternGenerator, WorkloadType
from datetime import datetime, timedelta

# Generate synthetic workload data
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

# Analyze patterns
print(f"CPU Utilization: {data['cpu_utilization'].mean():.1f}%")
print(f"Memory Utilization: {data['memory_utilization'].mean():.1f}%")
```

## Key Features

- **Hierarchical Bayesian Models** - Industry → Application → Resource modeling with PyMC
- **20+ Workload Archetypes** - Web apps, batch processing, ML training, databases, etc.
- **Foundation Model Integration** - Amazon Chronos, Google TimesFM for forecasting
- **Realistic Correlations** - Multivariate resource metric relationships
- **Temporal Patterns** - Daily, weekly, and seasonal variations

## Research Foundation

The simulation parameters are grounded in comprehensive empirical research:

- **[Cloud Resource Patterns Research](docs/research/cloud-resource-patterns-research.md)** - Analysis of utilization statistics across industries
- **[Resource Correlation Analysis](docs/research/cloud-resource-correlations-report.md)** - Multivariate correlation structures
- **[Research Overview](docs/research/)** - Summary of key findings and references

Key findings informing our models:
- CPU utilization: 12-15% average
- Memory utilization: 18-25% average
- Resource waste: 25-35% of cloud spend
- Development environments: 70% waste
- Strong temporal autocorrelation: 0.7-0.8

## Running Notebooks

```bash
# Install all dependencies including Altair visualizations
uv sync --all-extras

# Convert MyST markdown to Jupyter notebook
jupytext --to notebook notebooks/myst/02_workload_signatures_guide.md

# Execute notebook directly
jupytext --execute notebooks/myst/02_workload_signatures_guide.md

# Start Jupyter Lab for interactive exploration
uv run jupyter lab

# Convert all MyST notebooks to .ipynb
jupytext --to notebook notebooks/myst/*.md
```

## Documentation

- **[Technical Documentation](docs/index.md)** - Comprehensive architecture and API reference
- **[Research Papers](docs/research/)** - Empirical foundations and methodology
- **[Example Notebooks](notebooks/myst/)** - Analysis and visualization examples
  - `01_data_exploration.md` - Basic data generation and validation
  - `02_workload_signatures_guide.md` - Understanding why workloads have distinct patterns

## The Workload Genome Initiative

Contributing to a standardized taxonomy of cloud workload patterns for reproducible benchmarking and collaborative research.

## License

[MIT License](LICENSE) - Open for research and commercial use

## Contributing

Contributions welcome in:
- Empirical correlation data
- Additional workload patterns
- Bayesian model refinements
- Optimization algorithms

See [technical documentation](docs/index.md) for development guidelines.