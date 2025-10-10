# CloudLens

Analytics platform for cloud resource usage and cost optimization.

Covers workload characterization, cost analysis, time series modeling, forecasting, and anomaly detection.

Built on Ibis+DuckDB (local) with PySpark compatibility (scale).

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nehalecky/cloudlens.git
cd cloudlens

# Install dependencies
uv sync --all-extras
```

## Usage

### Basic Data Analysis

```python
import ibis
from ibis import _
from cloudlens.utils import attribute_analysis, grain_discovery

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
from cloudlens.ml_models.gaussian_process import SparseGPModel
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
cloudlens/
├── src/cloudlens/          # Source code
│   ├── utils/              # EDA and analysis utilities
│   ├── etl/                # Data loaders (CloudZero, Alibaba traces)
│   └── ml_models/          # Time series models (GP, PyMC)
├── notebooks/              # Analysis notebooks (MyST format)
├── tests/                  # Test suite
└── docs/                   # Documentation (Quarto)
```

## Development

```bash
# Run tests
uv run pytest tests/ -v --cov=src/cloudlens

# Format code
uv run black src/ tests/

# Lint
uv run ruff check --fix src/ tests/

# Execute notebooks
cd notebooks && uv run jupytext --execute 05_cloudzero_piedpiper_eda.md
```

## License

MIT
