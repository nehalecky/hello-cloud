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

**`hellocloud`** 👋☁️ is a project and Python library for cloud (data center) cost and resource use analysis, forecasting, and anomaly detection. It includes literature review, conceptual foundations, exporatory data analysis and a bit of modeling using time series foundation models (TSFMs), all supported by a dedicated and a dedicated library. 👋 `hellocloud`!.

## Installation

```bash
# Install uv on POSIX like system
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/nehalecky/hello-cloud.git
cd hello-cloud

# Install dependencies
uv sync --all-extras
```

## Interactive Notebooks (Google Colab)

| Notebook | Description | Colab |
|----------|-------------|-------|
| **Quickstart** | TimeSeries API in 15 minutes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/06_quickstart_timeseries_loader.ipynb) |
| **Workload Signatures** | Understanding cloud workload patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/02_guide_workload_signatures_guide.ipynb) |
| **IOPS Analysis** | Time series EDA with anomaly detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/03_EDA_iops_web_server.ipynb) |
| **Gaussian Processes** | GP modeling tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/04_modeling_gaussian_process.ipynb) |
| **PiedPiper EDA** | Hierarchical billing data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/examples/published/05_EDA_piedpiper_data.ipynb) |

## Documentation

**[📖 nehalecky.github.io/hello-cloud](https://nehalecky.github.io/hello-cloud)**

## Development


```bash
# Install dependencies (editable install from current directory)
uv sync --all-extras

# Run tests
just test

# Format and lint
just fix

# Build documentation
just docs-serve
```

## License

MIT, have at it. ;)

<!--content-end-->
