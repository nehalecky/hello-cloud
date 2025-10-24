# Tutorials

Learn hello cloud through hands-on, interactive Jupyter notebooks. Each tutorial builds on empirical cloud research and demonstrates practical workflows.

## Tutorial Categories

### 📚 Guided Tutorials

Step-by-step introductions to core concepts with explanations and best practices.

**[Workload Signatures Guide](../notebooks/published/02_guide_workload_signatures_guide.ipynb)**

Learn to generate realistic cloud workload patterns based on empirical research. Covers:

- Understanding the 12+ canonical workload types
- Generating synthetic time series with temporal patterns
- Modeling CPU, memory, and I/O correlations
- Adding realistic anomalies and bursts

**[Gaussian Process Modeling](../notebooks/published/04_modeling_gaussian_process.ipynb)**

Deep dive into GP-based time series forecasting with GPyTorch. Covers:

- Sparse variational GPs for scalability
- Composite periodic kernels for multi-scale patterns
- Training strategies and hyperparameter tuning
- Production-ready inference patterns

### 🔬 Exploratory Data Analysis

Practical examples analyzing real and synthetic cloud metrics.

**[IOPS Analysis](../notebooks/published/03_EDA_iops_web_server.ipynb)**

Analyze I/O patterns for web server workloads:

- Loading and exploring IOPS time series
- Detecting periodic patterns and anomalies
- Comparing statistical and GP forecasting approaches
- Visualizing forecast uncertainty

**[PiedPiper Data Analysis](../notebooks/published/05_EDA_piedpiper_data.ipynb)**

Work with hierarchical cloud cost data:

- Hierarchical time series operations (filter, sample, aggregate)
- Multi-entity analysis across accounts and services
- Cost attribution and trend analysis
- Publication-quality visualizations

### 📈 Forecasting Comparison

**[Forecasting Methods Comparison](../notebooks/published/07_forecasting_comparison.ipynb)**

Comprehensive comparison of forecasting approaches:

- Baseline models (Naive, Seasonal Naive, Moving Average)
- Statistical methods (ARIMA, Prophet)
- Gaussian Processes
- Foundation models (Chronos, TimesFM)
- Performance metrics and trade-offs

## Using the Tutorials

### Format

All tutorials are delivered as **Jupyter notebooks** using MyST Markdown format. This provides:

- **Executable code cells** with real outputs
- **Rich visualizations** (Matplotlib, Seaborn)
- **Inline equations** and mathematical notation
- **Collapsible sections** for optional details

### Running Locally

```bash
# Install with notebook support
uv pip install hellocloud[dev]

# Start Jupyter Lab
just lab
# or
uv run jupyter lab notebooks/

# Enable hot reload for library code
%load_ext autoreload
%autoreload 2
```

### Online (Read-Only)

All notebooks render directly in the documentation with full outputs. Perfect for:

- Quick reference
- Understanding concepts before coding
- Sharing with stakeholders

## Tutorial Progression

We recommend this learning path:

1. **[TimeSeries Quickstart](../getting-started/index.md)** (15 minutes) - Get up and running
2. **[Workload Signatures](../notebooks/published/02_guide_workload_signatures_guide.ipynb)** - Understand data generation
3. **[IOPS Analysis](../notebooks/published/03_EDA_iops_web_server.ipynb)** - Practical EDA workflow
4. **[Forecasting Comparison](../notebooks/published/07_forecasting_comparison.ipynb)** - Choosing the right model
5. **[Gaussian Process Modeling](../notebooks/published/04_modeling_gaussian_process.ipynb)** - Deep dive into GPs

## Prerequisites

All tutorials assume:

- **Python 3.11+** installed
- **Basic Python** knowledge (functions, classes, imports)
- **Pandas familiarity** (DataFrames, basic operations)
- **Matplotlib basics** (for understanding visualizations)

No prior machine learning experience required! We explain concepts as we go.

## Contributing Tutorials

Have an interesting use case? Want to share a workflow?

See **[Contributing Guide](../contributing/index.md)** for tutorial contribution guidelines.

---

**Ready to start?** Begin with the [TimeSeries Quickstart](../getting-started/index.md)!
