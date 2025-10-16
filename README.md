# Hello Cloud

[![CI](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nehalecky/hello-cloud/branch/master/graph/badge.svg)](https://codecov.io/gh/nehalecky/hello-cloud)
[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://nehalecky.github.io/hello-cloud)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)

Hands-on exploration of cloud resource usage and cost optimization.

Workload characterization • Cost analysis • Time series forecasting • Anomaly detection

**PySpark 4.0** (distributed processing) • **GPyTorch** (time series modeling) • **PyMC** (Bayesian inference)

**Documentation:** https://nehalecky.github.io/hello-cloud

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nehalecky/hello-cloud.git
cd hello-cloud

# Install dependencies
uv sync --all-extras
```

## Usage

### Basic Data Analysis

```python
import ibis
from ibis import _
from hellocloud.utils import attribute_analysis, grain_discovery

# Connect to billing data
con = ibis.duckdb.connect()
df = con.read_parquet('billing_data.parquet', table_name='billing')

# Analyze attribute patterns
attrs = attribute_analysis(df, sample_size=50_000)
print(attrs[['column', 'cardinality', 'information_score']])

# Discover optimal forecasting grain
optimal_grain = grain_discovery(
    df,
    grain_cols=['provider', 'account', 'region', 'service'],
    cost_col='cost',
    min_days=30
)
```

### Time Series Forecasting

```python
# Entity-level time series
entity_ts = (
    df
    .filter((_.provider == 'aws') & (_.account == '123456'))
    .group_by('date')
    .agg(daily_cost=_.cost.sum())
    .order_by('date')
    .execute()
)

# Forecast with GP model (requires GPU extras)
from hellocloud.ml_models.gaussian_process import SparseGPModel
model = SparseGPModel()
predictions = model.forecast(entity_ts, horizon=30)
```

## Stack

- **PySpark 4.0**: Distributed DataFrame processing (local & scale)
- **pandas**: Results and visualization
- **GPyTorch**: Time series modeling (optional, GPU)
- **PyMC**: Bayesian hierarchical models (optional)
- **HuggingFace datasets**: Data storage

## Interactive Notebooks (Google Colab)

Try our tutorials directly in your browser:

| Notebook | Description | Colab |
|----------|-------------|-------|
| **Quickstart** | TimeSeries API in 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/06_quickstart_timeseries_loader.ipynb) |
| **Workload Signatures** | Understanding cloud workload patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/02_guide_workload_signatures_guide.ipynb) |
| **IOPS Analysis** | Time series EDA with anomaly detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/03_EDA_iops_web_server.ipynb) |
| **Gaussian Processes** | GP modeling tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/04_modeling_gaussian_process.ipynb) |
| **PiedPiper EDA** | Hierarchical billing data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/05_EDA_piedpiper_data.ipynb) |

**All notebooks include automatic library installation** - just click and run!

## Documentation

See [`docs/`](docs/) for:
- API reference
- Tutorial notebooks
- Development guides

## Project Structure

```
hello-cloud/
├── src/hellocloud/         # Source code
│   ├── data_generation/    # Synthetic workload pattern generation
│   ├── utils/              # EDA and analysis utilities
│   └── ml_models/          # Time series models (GP, PyMC)
├── notebooks/              # Analysis notebooks (MyST format)
├── tests/                  # Test suite
└── docs/                   # Documentation (MkDocs)
```

## Development

```bash
# Run tests
uv run pytest tests/ -v --cov=src/hellocloud

# Format code
uv run black src/ tests/

# Lint
uv run ruff check --fix src/ tests/

# Build documentation
just docs
```

## License

MIT
