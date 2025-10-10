# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies with uv (NOT pip)
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate

# Install documentation dependencies
uv sync --group docs
```

### Testing
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src/hellocloud --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_data_generation.py -v

# Run specific test function
uv run pytest tests/test_data_generation.py::TestWorkloadPatternGenerator::test_generate_time_series -v

# Run tests with coverage threshold (CI requirement: 70%)
uv run pytest tests/ -v --cov=src/hellocloud --cov-fail-under=70
```

### Code Quality

**📚 See [docs/development/CODE_QUALITY.md](docs/development/CODE_QUALITY.md) for comprehensive workflow documentation**

```bash
# Recommended: Format and fix all issues at once
just fix

# Or run individually
just format     # Format code with black
just lint       # Check with ruff
just lint-fix   # Auto-fix ruff issues

# Pre-commit hooks (automatic on git commit)
just pre-commit-install  # One-time setup
just pre-commit          # Run manually on all files

# Type checking (optional, many imports ignored)
uv run mypy src/
```

### Documentation
```bash
# Quick commands with just (recommended)
just docs-api      # Generate API reference
just docs-preview  # Preview with auto-refresh
just docs-build    # Build static site
just docs          # API + build

# Or use commands directly
uv run quartodoc build --config docs/_quarto.yml  # Generate API reference
quarto preview docs/                              # Preview (auto-refreshes)
quarto render docs/                               # Build static site
```

### Jupyter Notebooks

**👉 See [notebooks/WORKFLOW.md](notebooks/WORKFLOW.md) for the complete hot reload workflow!**

```bash
# Quick commands with just (recommended)
just lab              # Start Jupyter Lab
just notebook         # Start classic Jupyter Notebook

# Or use commands directly
uv run jupyter lab notebooks/
uv run jupyter notebook notebooks/

# Kernel Setup (automatic!)
# The "Python 3 (ipykernel)" kernel is automatically created in .venv/
# when ipykernel is installed. Select it in Jupyter Lab's kernel picker.
# It's project-local and portable - no manual installation needed!

# Hot Reload Pattern (avoids kernel restarts)
# Add this to the top of your notebook:
%load_ext autoreload
%autoreload 2
# Now library code changes auto-reload without kernel restart!

# Test notebooks as runbooks (fast smoke tests)
uv run pytest tests/test_notebooks.py -m smoke -v

# Test full notebook execution (slow but thorough)
uv run pytest tests/test_notebooks.py -k "execution_success" -v
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
- **Ibis+DuckDB Stack**: Portable DataFrame API with DuckDB backend for local analytics, PySpark for scale
- **Production-Ready**: 92% test coverage on GP library, comprehensive testing framework

### Module Structure
```
src/hellocloud/
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

### Polars → Ibis Migration Patterns

The project migrated from Polars to Ibis+DuckDB for better backend portability and analytical performance. Key conversion patterns:

#### Execution
```python
# Polars
df.collect()  # Execute lazy operations

# Ibis
df.execute()  # Returns pandas DataFrame
```

#### Schema Access
```python
# Polars
schema = df.collect_schema()
col_names = df.columns

# Ibis
schema = df.schema()  # Returns ibis.Schema
col_names = df.schema().names  # List of column names
```

#### Column References
```python
# Polars
pl.col('cost')
pl.len()

# Ibis
_.cost  or  _['cost']  # Column deference
_.count()  # Row count aggregation
```

#### Aggregations
```python
# Polars
df.agg([
    pl.col('cost').sum().alias('total'),
    pl.col('date').n_unique().alias('days')
])

# Ibis
df.agg(
    total=_.cost.sum(),
    days=_.date.nunique()
)  # No list wrapping, keyword args for aliases
```

#### Filtering
```python
# Polars
df.filter(pl.col('cost') > 100)

# Ibis
df.filter(_.cost > 100)
```

#### Sorting & Renaming
```python
# Polars
df.sort('date')
df.rename({'old_name': 'new_name'})

# Ibis
df.order_by('date')
df.rename(new_name='old_name')  # Reversed argument order!
```

#### Dtype Checks
```python
# Polars
isinstance(dtype, (pl.Utf8, pl.Categorical))
isinstance(dtype, (pl.Date, pl.Datetime))

# Ibis
dtype.is_string()
dtype.is_temporal()
```

#### Count Operations
```python
# Polars (returns DataFrame)
count_df = df.count()  # Returns single-row DataFrame
count_val = count_df[0, 0]  # Extract scalar

# Ibis (returns scalar directly)
count_val = df.count().execute()  # Already a scalar int
```

### Data Processing
Use Ibis for portable DataFrame operations with DuckDB backend:
```python
import ibis
from ibis import _
import pandas as pd  # Used for results after .execute()

# Connect to DuckDB (in-memory analytics)
con = ibis.duckdb.connect()

# Read data
df = con.read_parquet('data/file.parquet', table_name='data')

# Query with Ibis (lazy evaluation)
result = (
    df.filter(_.cost > 0)
    .group_by('region')
    .agg(total=_.cost.sum())
    .order_by(ibis.desc('total'))
    .execute()  # Returns pandas DataFrame
)

# Backend portability: Same code works with PySpark
# con = ibis.pyspark.connect(spark_session)
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
- **ibis-framework[duckdb]**: Portable DataFrame API for analytics (local: DuckDB, scale: PySpark)
- **duckdb**: In-memory OLAP database for fast analytical queries
- **pandas**: Used for results after Ibis `.execute()` and visualization
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
from hellocloud.data_generation import WorkloadPatternGenerator, WorkloadType
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)
```

### Running Hierarchical Models
```python
from hellocloud.ml_models import CloudResourceHierarchicalModel
model = CloudResourceHierarchicalModel()
results = model.fit(data)
```

## CI/CD Configuration
- GitHub Actions workflow in `.github/workflows/ci.yml`
- Tests run on Python 3.11 and 3.12
- Coverage requirement: 70% minimum
- Uses uv for dependency management
