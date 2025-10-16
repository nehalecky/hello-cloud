# Hello Cloud

Time series forecasting and anomaly detection for cloud resources.

## Overview

Hello Cloud is a Python library for modeling cloud resource utilization patterns, forecasting future usage, and detecting anomalies in operational metrics.

**Key Features:**
- Empirically grounded (12-15% average CPU utilization)
- Multiple models (Gaussian Processes, ARIMA, foundation models)
- Production-ready (92% test coverage on GP library)

## Getting Started

### Installation

```bash
pip install git+https://github.com/nehalecky/hello-cloud.git
```

### Quick Start

```python
from hellocloud.generation import WorkloadPatternGenerator, WorkloadType

generator = WorkloadPatternGenerator()
data = generator.generate_time_series(
    workload_type=WorkloadType.WEB_APP,
    interval_minutes=60
)
```

## Documentation

**[Notebooks](notebooks/index.md)** - Interactive tutorials (executed with outputs)

**[Concepts](concepts/index.md)** - Research reports and design docs

**[API Reference](reference/index.md)** - Auto-generated from docstrings

## Research Context

- CPU Utilization: 12-15% average
- Memory Utilization: 18-25% average
- Resource Waste: 25-35% of cloud spending
- Temporal Autocorrelation: 0.7-0.8

See [Cloud Resource Patterns Research](concepts/research/cloud-resource-patterns-research.md).
