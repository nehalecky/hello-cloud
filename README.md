# Cloud Resource Utilization Simulator

A comprehensive framework for cloud workload pattern analysis and synthetic data generation using multiple modeling approaches.

## Overview

This research project explores cloud resource utilization patterns through multiple modeling techniques, generating realistic synthetic data for optimization algorithm development. The framework combines Gaussian Processes, Bayesian hierarchical models, and foundation model integration, all grounded in [empirical research](docs/research/) showing 12-15% average CPU utilization and 25-35% resource waste across cloud infrastructure.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/nehalecky/cloud-resource-simulator.git
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

- **20+ Workload Archetypes** - Web apps, batch processing, ML training, databases, etc.
- **Gaussian Process Models** - Time series forecasting with uncertainty quantification (GPyTorch)
- **Bayesian Hierarchical Models** - Industry → Application → Resource modeling (PyMC)
- **Foundation Model Integration** - Amazon Chronos, Google TimesFM for forecasting (stubs)
- **Realistic Correlations** - Multivariate resource metric relationships
- **Temporal Patterns** - Daily, weekly, and seasonal variations
- **Research-Grounded** - All parameters derived from empirical studies

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

## Interactive Analysis with Jupyter

### Quick Start
```bash
# Install all dependencies
uv sync --all-extras

# One-time setup: Install kernel for your uv environment
uv run ipython kernel install --user --name=cloud-sim

# Start Jupyter Lab
uv run jupyter lab

# In Jupyter Lab, select "cloud-sim" kernel for notebooks
```

### Notebook Testing
```bash
# Fast syntax/import tests (0.4s)
uv run pytest tests/test_notebooks.py -m smoke -v

# Full runbook execution tests (slower)
uv run pytest tests/test_notebooks.py -k "execution_success" -v
```

### Notebook Architecture
All notebooks use **MyST format**:
- `notebooks/*.md` - Clean MyST markdown (git-friendly, documentation-ready)
- `.ipynb` files auto-generated (not tracked in git)
- Jupytext handles conversion automatically

## Documentation

### Core Documentation
- **[Technical Documentation](docs/index.md)** - Comprehensive architecture and API reference
- **[Research Papers](docs/research/)** - Empirical foundations and methodology
- **[Example Notebooks](notebooks/)** - Analysis and visualization examples
  - `01_data_exploration.md` - Basic data generation and validation
  - `02_workload_signatures_guide.md` - Understanding why workloads have distinct patterns
  - `04_gaussian_process_modeling.md` - GP library usage and anomaly detection

### Technical Design Documents
- **[Design Documents Overview](docs/design/README.md)** - Index of technical design documents

## The Workload Genome Initiative

Contributing to a standardized taxonomy of cloud workload patterns for reproducible benchmarking and collaborative research.

## License

[MIT License](LICENSE) - Open for research and commercial use

## Contributing

Contributions welcome in:
- Empirical correlation data
- Additional workload patterns
- ML model improvements (GP, PyMC, foundation models)
- Optimization algorithms
- Testing and validation

See [technical documentation](docs/index.md) for development guidelines.