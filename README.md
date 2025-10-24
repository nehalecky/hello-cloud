<div align="center">
  <img src="https://raw.githubusercontent.com/nehalecky/hello-cloud/master/docs/_assets/images/logo-full-light.png" alt="Hello Cloud" width="500">

  <p><strong>Research-driven cloud resource use and cost modeling</strong></p>

  [![CI](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/nehalecky/hello-cloud/actions/workflows/ci.yml)
  [![codecov](https://codecov.io/gh/nehalecky/hello-cloud/branch/master/graph/badge.svg)](https://codecov.io/gh/nehalecky/hello-cloud)
  [![Documentation](https://img.shields.io/badge/docs-live-blue)](https://nehalecky.github.io/hello-cloud)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
</div>

---

<!--content-start-->

## Overview

**hello cloud** ☁️ is a Python library for understanding, analyzing, and modeling cloud resource utilization and cost patterns. The project combines empirical research, conceptual modeling, and practical tools to support data-driven cloud resource optimization.

**Outputs:**
- Literature-informed time series models (Gaussian Processes, hierarchical Bayesian)
- Python library (`hellocloud`) for forecasting and analysis
- Interactive notebooks documenting research workflows
- Empirical findings from 35+ academic papers

**Approach:** Empirical foundations → Conceptual models → Practical tools

## Research Foundation

All synthetic data patterns and model parameters are grounded in published research on real cloud infrastructure behavior:

- **CPU Utilization**: 12-15% average across cloud infrastructure
- **Memory Utilization**: 18-25% average
- **Resource Waste**: 25-35% of cloud spending
- **Temporal Autocorrelation**: 0.7-0.8 (strong patterns)
- **Literature Basis**: 35+ peer-reviewed papers

See [Cloud Resource Patterns Research](https://nehalecky.github.io/hello-cloud/research/cloud-resource-patterns-research/) for full citations and analysis.

## Core Capabilities

- **Workload Characterization** - Generate realistic synthetic cloud metrics based on empirical patterns
- **Time Series Forecasting** - Gaussian Processes, ARIMA, and foundation models (Chronos, TimesFM)
- **Hierarchical Analysis** - Multi-level cost analysis across providers, accounts, and resources
- **Anomaly Detection** - Statistical and ML-based approaches for operational metrics
- **PySpark Integration** - Distributed processing for local development and production scale

## Technology Stack & Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Processing** | PySpark 4.0 | Distributed DataFrames (local & scale) |
| **Statistical Models** | GPyTorch | Gaussian Process time series forecasting |
| **Probabilistic Models** | PyMC | Bayesian hierarchical inference |
| **Foundation Models** | Chronos, TimesFM | Pre-trained forecasters (optional) |
| **Notebooks** | Jupyter + Python % | Interactive research workflows |
| **Documentation** | MkDocs Material | Live docs with API reference |

**Design Principles:**

- **Empirically Grounded** - All synthetic data based on published cloud utilization research
- **Production Ready** - 92% test coverage on GP library, comprehensive CI/CD
- **Research Informed** - Implements patterns from 35+ academic papers
- **Developer Friendly** - Hot reload notebooks, comprehensive docs, type hints

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

## Usage Examples

### Generate Synthetic Cloud Metrics

```python
from hellocloud.data_generation import WorkloadPatternGenerator, WorkloadType
from datetime import datetime, timedelta

# Generate realistic web app metrics
generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    interval_minutes=60
)

# Built-in patterns: WEB_APP, BATCH_JOB, MICROSERVICE, DATABASE,
# ML_TRAINING, STREAMING, SEASONAL_BATCH, DEV_ENVIRONMENT, and more
```

### Time Series Forecasting with Gaussian Processes

```python
from hellocloud.ml_models.gaussian_process import SparseGPModel
from hellocloud.spark import get_spark_session
from pyspark.sql import functions as F

# Get Spark session
spark = get_spark_session(app_name="forecasting")

# Load time series data
df = spark.read.parquet('cloud_metrics.parquet')

# Prepare data
train_df = df.filter(F.col('date') < '2024-01-01')

# Train GP model with multi-scale periodic kernel
model = SparseGPModel(
    num_inducing=100,
    fast_period=24.0,   # Daily pattern
    slow_period=168.0   # Weekly pattern
)
model.fit(train_df)

# Forecast
predictions = model.predict(test_df, horizon=30)
```

### Hierarchical Cost Analysis

```python
from hellocloud.io import PiedPiperLoader
from hellocloud.timeseries import TimeSeries

# Load hierarchical billing data with EDA-informed defaults
spark = get_spark_session(app_name="analysis")
raw_df = spark.read.parquet('billing_data.parquet')
ts = PiedPiperLoader.load(raw_df)

# Filter to specific provider/account
aws_ts = ts.filter(provider='aws', account='123456789')

# Aggregate to daily spend
daily = aws_ts.aggregate(by=['date'])

# Summary statistics
stats = daily.summary_stats()
print(stats.toPandas())
```

## Interactive Notebooks (Google Colab)

Try our research workflows directly in your browser:

| Notebook | Description | Colab |
|----------|-------------|-------|
| **Quickstart** | TimeSeries API in 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/06_quickstart_timeseries_loader.ipynb) |
| **Workload Signatures** | Understanding cloud workload patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/02_guide_workload_signatures_guide.ipynb) |
| **IOPS Analysis** | Time series EDA with anomaly detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/03_EDA_iops_web_server.ipynb) |
| **Gaussian Processes** | GP modeling tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/04_modeling_gaussian_process.ipynb) |
| **PiedPiper EDA** | Hierarchical billing data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/05_EDA_piedpiper_data.ipynb) |

**All notebooks include automatic library installation** - just click and run!

## Documentation

**[📖 Full Documentation](https://nehalecky.github.io/hello-cloud)**

- **[Getting Started](https://nehalecky.github.io/hello-cloud/getting-started/)** - Installation and quick start
- **[Tutorials](https://nehalecky.github.io/hello-cloud/tutorials/)** - Step-by-step guides
- **[Research](https://nehalecky.github.io/hello-cloud/research/)** - Empirical foundations and literature reviews
- **[API Reference](https://nehalecky.github.io/hello-cloud/reference/)** - Complete API documentation

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

# Format and lint
just fix

# Build documentation
just docs-serve
```

## License

MIT

<!--content-end-->
