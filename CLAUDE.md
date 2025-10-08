# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies with uv (NOT pip)
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

### Testing
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src/cloud_sim --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_data_generation.py -v

# Run specific test function
uv run pytest tests/test_data_generation.py::TestWorkloadPatternGenerator::test_generate_time_series -v

# Run tests with coverage threshold (CI requirement: 70%)
uv run pytest tests/ -v --cov=src/cloud_sim --cov-fail-under=70
```

### Code Quality
```bash
# Format code with black
uv run black src/ tests/

# Run linter
uv run ruff check src/ tests/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/

# Type checking (optional, many imports ignored)
uv run mypy src/
```

### Jupyter Notebooks

**👉 See [notebooks/WORKFLOW.md](notebooks/WORKFLOW.md) for the complete hot reload workflow!**

```bash
# One-time setup: Install kernel pointing to your uv environment
uv run ipython kernel install --user --name=cloud-sim

# Start Jupyter Lab with workflow helper
./scripts/notebook-workflow.sh lab

# Or start directly
uv run jupyter lab notebooks/

# Hot Reload Pattern (avoids kernel restarts)
# Add this to the top of your notebook:
%load_ext autoreload
%autoreload 2
# Now library code changes auto-reload without kernel restart!

# Test notebooks as runbooks (fast smoke tests)
uv run pytest tests/test_notebooks.py -m smoke -v

# Test full notebook execution (slow but thorough)
uv run pytest tests/test_notebooks.py -k "execution_success" -v

# Workflow helper commands
./scripts/notebook-workflow.sh open 04_gaussian_process_modeling  # Open notebook
./scripts/notebook-workflow.sh convert 04_gaussian_process_modeling  # Convert to .ipynb
./scripts/notebook-workflow.sh execute 04_gaussian_process_modeling  # Execute notebook
./scripts/notebook-workflow.sh clean  # Remove generated .ipynb files
./scripts/notebook-workflow.sh publish  # Execute all for publishing
```

### Notebook Architecture
All notebooks use **MyST format**:
- **Source**: `notebooks/*.md` - Clean MyST markdown (git-friendly, version controlled)
- **Generated**: `notebooks/_build/*.ipynb` - Converted notebooks (gitignored, recreated from .md)
- **Jupytext**: Handles `.md` ↔ `.ipynb` conversion automatically via `jupytext.toml`
- **Kernel**: All notebooks use `cloud-sim` kernel (configured in YAML frontmatter)

## Architecture Overview

### Core Design Principles
- **Multi-Model Approach**: Gaussian Processes, Bayesian hierarchical models, and foundation model integration
- **Empirical Foundation**: Based on research showing 12-15% average CPU utilization and 25-35% cloud waste
- **Polars-First**: Uses Polars exclusively for data processing (no pandas)
- **Production-Ready**: 92% test coverage on GP library, comprehensive testing framework

### Module Structure
```
src/cloud_sim/
├── data_generation/       # Synthetic data generation based on empirical patterns
│   ├── workload_patterns.py     # 20+ workload archetypes with realistic utilization
│   ├── cloud_metrics_simulator.py # Multivariate metric correlation simulation
│   └── hf_dataset_builder.py    # HuggingFace dataset integration
├── ml_models/            # Machine learning and forecasting
│   ├── gaussian_process/        # GP models for time series (GPyTorch)
│   ├── pymc_cloud_model.py      # Hierarchical Bayesian models
│   ├── foundation/              # Foundation model stubs (Chronos, TimesFM)
│   └── application_taxonomy.py  # Workload genome taxonomy
├── etl/                  # Data loaders (CloudZero stub, Alibaba trace)
└── (api, dashboard, utils planned for future)
```

### Key Classes and Their Purpose

**WorkloadPatternGenerator** (`data_generation/workload_patterns.py`):
- Generates realistic time series data for 12+ workload types
- Implements research-based utilization patterns (shockingly low: 12-15% CPU average)
- Includes temporal patterns, bursts, and anomalies

**SparseGPModel** (`ml_models/gaussian_process/models.py`):
- GPyTorch-based sparse variational GP for time series forecasting
- Composite periodic kernel for multi-scale patterns
- 92% test coverage, production-ready
- See: `docs/modeling/gaussian-process-design.md`

**CloudResourceHierarchicalModel** (`ml_models/pymc_cloud_model.py`):
- PyMC-based hierarchical Bayesian model
- Models Industry → Application → Resource hierarchy
- Captures uncertainty and correlations between metrics

**CloudMetricsDatasetBuilder** (`data_generation/hf_dataset_builder.py`):
- Creates HuggingFace datasets for model training
- Handles synthetic and real data ingestion
- Supports multiple output formats

## Research Context

The simulator is grounded in empirical research showing:
- **CPU Utilization**: 12-15% average across cloud infrastructure
- **Memory Utilization**: 18-25% average
- **Resource Waste**: 25-35% of cloud spending
- **Development Environments**: 70% waste (often forgotten/idle)
- **Temporal Autocorrelation**: 0.7-0.8 (strong patterns)

These findings inform all synthetic data generation parameters.

## Development Patterns

### Data Processing
Always use Polars, never pandas:
```python
import polars as pl  # ✓ Correct
# import pandas as pd  # ✗ Never use pandas in this codebase
```

### Error Handling
Use loguru for logging:
```python
from loguru import logger
logger.info("Processing workload type: {}", workload_type)
```

### Type Hints and Validation
Use Pydantic for data validation:
```python
from pydantic import BaseModel, Field, field_validator
```

### Testing Approach
- All new features require tests with >70% coverage
- Use pytest fixtures defined in `tests/conftest.py`
- Mock external services (HuggingFace, cloud APIs)

## Important Dependencies

### Core Libraries
- **polars**: Data processing (NOT pandas)
- **pymc**: Bayesian modeling
- **chronos-forecasting**: Amazon's foundation model
- **transformers**: HuggingFace integration
- **streamlit**: Dashboard (if implementing)
- **fastapi**: API endpoints (if implementing)

### Development Tools
- **uv**: Package manager (NOT pip directly)
- **pytest**: Testing framework
- **black**: Code formatter
- **ruff**: Linter (replaces flake8, isort, etc.)
- **jupytext**: MyST notebook support

## Common Tasks

### Adding a New Workload Type
1. Add enum value to `WorkloadType` in `workload_patterns.py`
2. Define characteristics in `WORKLOAD_PROFILES` dict
3. Add corresponding tests in `test_data_generation.py`
4. Update documentation in `docs/index.md`

### Creating Synthetic Datasets
```python
from cloud_sim.data_generation import WorkloadPatternGenerator, WorkloadType
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)
```

### Running Hierarchical Models
```python
from cloud_sim.ml_models import CloudResourceHierarchicalModel
model = CloudResourceHierarchicalModel()
results = model.fit(data)
```

## CI/CD Configuration
- GitHub Actions workflow in `.github/workflows/ci.yml`
- Tests run on Python 3.11 and 3.12
- Coverage requirement: 70% minimum
- Uses uv for dependency management