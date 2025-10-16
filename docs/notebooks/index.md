# Notebooks

Interactive analysis notebooks. All notebooks are executed and published to `./published/` with outputs.

## Tutorials

- [Workload Signatures](../published/02_guide_workload_signatures_guide.ipynb) - Understanding cloud resource patterns
- [IOPS Analysis](../published/03_EDA_iops_web_server.ipynb) - TSB-UAD dataset exploration
- [Gaussian Process Modeling](../published/04_modeling_gaussian_process.ipynb) - Production GP forecasting
- [PiedPiper Data](../published/05_EDA_piedpiper_data.ipynb) - Hierarchical time series
- [TimeSeries API](../published/06_quickstart_timeseries_loader.ipynb) - New TimeSeries loader
- [Forecasting Comparison](../published/07_forecasting_comparison.ipynb) - Baseline vs ARIMA vs TimesFM

## Running Locally

```bash
uv run jupyter lab notebooks/
```

All notebooks have Colab badges for cloud execution.
