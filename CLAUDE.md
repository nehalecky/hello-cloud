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

# Optional: Foundation models (NOT on Apple Silicon)
# Only install on x86_64 Linux/Intel Mac
uv sync --extra foundation
```

**Optional Dependencies:**

- **Foundation Models** (`foundation`): Time series forecasting models (TimesFM)
  - ⚠️ **NOT compatible with Apple Silicon** (ARM architecture)
  - Requires x86_64 Linux or Intel Mac
  - Tests automatically skip when not installed

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
- **PySpark 4.0 Stack**: Distributed DataFrame processing for local development and production scale
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

### PySpark Transform Patterns

The project uses PySpark 4.0 for all DataFrame operations with composable `.transform()` pattern.

#### Spark Session
```python
from hellocloud.spark import get_spark_session

# Get or create Spark session (singleton)
spark = get_spark_session(app_name="my-analysis")

# Local mode defaults (configured automatically):
# - local[*] master (all cores)
# - 4GB driver memory
# - 8 shuffle partitions (not 200!)
```

#### Loading Data
```python
# Read Parquet
df = spark.read.parquet('data/file.parquet')
df = df.cache()  # Cache for repeated access

# Basic operations
total_rows = df.count()
df.show(10)
df.toPandas()  # Convert to pandas for display
```

#### Column Operations
```python
from pyspark.sql import functions as F

# Column references
F.col('cost')
F.sum('cost')
F.countDistinct('date')

# Filtering
df.filter(F.col('cost') > 100)

# Aggregations
result = df.groupBy('region').agg(
    F.sum('cost').alias('total'),
    F.countDistinct('date').alias('days')
)
```

#### Transform Pattern
```python
from hellocloud.transforms import pct_change, summary_stats

# Composable transforms using .transform()
result = df.transform(
    pct_change(value_col='cost', order_col='date', group_cols=['resource_id'])
)

# Summary statistics by group
stats = df.transform(
    summary_stats(value_col='cost', group_col='region')
)
```

#### Important: Decimal (Fractional) Values
Our PySpark transforms return **decimal values** (0.10 = 10%), like pandas:

```python
# pct_change returns fractional form: 0.10 for 10% increase
daily_change = df.transform(pct_change('cost', 'date'))

# Filter for >30% drops (use -0.30, not -30.0!)
large_drops = daily_change.filter(F.col('pct_change') < -0.30)
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
- **pyspark**: Distributed DataFrame processing engine (requires Java 21)
- **pandas**: Used for results after PySpark `.toPandas()` and visualization
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

### Java Requirements
- **Java 21 (OpenJDK)**: Required for PySpark 4.0
- Installed via Homebrew: `brew install openjdk@21`
- JAVA_HOME configured in ~/.zshrc (see dotfiles)
- Automatically used by Spark session

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
