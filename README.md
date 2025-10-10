# Hello Cloud

Hands-on exploration of cloud resource usage and cost optimization.

Workload characterization • Cost analysis • Time series forecasting • Anomaly detection

Ibis+DuckDB (local) • PySpark (scale)

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

- **Ibis** + **DuckDB**: Data processing (local analytics)
- **pandas**: Results and visualization
- **GPyTorch**: Time series modeling (optional, GPU)
- **PyMC**: Bayesian hierarchical models (optional)
- **HuggingFace datasets**: Data storage

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
└── docs/                   # Documentation (Quarto)
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
