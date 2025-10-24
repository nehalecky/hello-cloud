<div align="center">
  <img src="docs/_assets/images/logo-full-light.png" alt="Hello Cloud" width="500">

  <p><strong>Hands-on exploration of cloud resource usage and cost optimization.</strong></p>

  [![CI](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml)
  [![codecov](https://codecov.io/gh/nehalecky/hello-cloud/branch/master/graph/badge.svg)](https://codecov.io/gh/nehalecky/hello-cloud)
  [![Documentation](https://img.shields.io/badge/docs-live-blue)](https://nehalecky.github.io/hello-cloud)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
</div>

---

**Workload characterization** • **Cost analysis** • **Time series forecasting** • **Anomaly detection**

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

### Loading Time Series Data

```python
from hellocloud.io import PiedPiperLoader
from hellocloud.timeseries import TimeSeries
from hellocloud.spark import get_spark_session

# Get Spark session
spark = get_spark_session(app_name="analysis")

# Load billing data
raw_df = spark.read.parquet('billing_data.parquet')

# Load into TimeSeries with EDA-informed defaults
ts = PiedPiperLoader.load(raw_df)

# Filter to specific provider/account
filtered = ts.filter(provider='aws', account='123456')

# Aggregate to daily totals
daily = filtered.aggregate(by=['date'])
```

### Time Series Forecasting

```python
# Forecast with Gaussian Process model
from hellocloud.ml_models.gaussian_process import SparseGPModel

model = SparseGPModel()
predictions = model.forecast(daily, horizon=30)
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
| **Quickstart** | TimeSeries API in 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/06_quickstart_timeseries_loader.ipynb) |
| **Workload Signatures** | Understanding cloud workload patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/02_guide_workload_signatures_guide.ipynb) |
| **IOPS Analysis** | Time series EDA with anomaly detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/03_EDA_iops_web_server.ipynb) |
| **Gaussian Processes** | GP modeling tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/04_modeling_gaussian_process.ipynb) |
| **PiedPiper EDA** | Hierarchical billing data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/05_EDA_piedpiper_data.ipynb) |

**All notebooks include automatic library installation** - just click and run!

## Documentation

See [`docs/`](docs/) for:
- API reference
- Tutorial notebooks
- Development guides

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Quick start guide
- Development workflow
- Code quality standards
- Pull request process

For detailed guidelines, visit the [Contributing Guide](https://nehalecky.github.io/hello-cloud/contributing/).

## Project Structure

```
hello-cloud/
├── src/hellocloud/         # Source code
│   ├── io/                 # Data loaders (PiedPiperLoader, etc.)
│   ├── timeseries/         # TimeSeries core classes
│   ├── data_generation/    # Synthetic workload patterns
│   ├── ml_models/          # GP, PyMC, foundation models
│   ├── analysis/           # EDA utilities
│   └── transforms/         # PySpark transforms
├── examples/               # Tutorial notebooks (Python percent)
├── tests/                  # Test suite (92% coverage on GP)
└── docs/                   # Documentation (MkDocs Material)
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
