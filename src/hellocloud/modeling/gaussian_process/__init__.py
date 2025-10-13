"""
Gaussian Process models for cloud resource time series analysis.

This module provides GPyTorch-based Gaussian Process models for:
- Time series forecasting with uncertainty quantification
- Anomaly detection via prediction intervals
- Pattern learning with composite periodic kernels

Based on empirical research showing cloud metrics exhibit:
- Multi-scale periodic patterns (daily + hourly cycles)
- Heavy-tailed distributions requiring robust likelihoods
- Sparse data requiring scalable variational inference

**Note**: Requires optional 'gpu' dependencies (torch, gpytorch).
Install with: `uv sync --group gpu`

References:
- GPyTorch: https://gpytorch.ai/
- Sparse GPs (SVGP): Hensman et al. (2013)
- Student-t Processes: Shah et al. (2014)
"""

try:
    from .evaluation import compute_anomaly_metrics, compute_metrics, compute_prediction_intervals
    from .kernels import CompositePeriodicKernel
    from .models import SparseGPModel, initialize_inducing_points
    from .training import load_model, save_model, train_gp_model

    __all__ = [
        "CompositePeriodicKernel",
        "SparseGPModel",
        "initialize_inducing_points",
        "train_gp_model",
        "save_model",
        "load_model",
        "compute_metrics",
        "compute_anomaly_metrics",
        "compute_prediction_intervals",
    ]
except ImportError:
    # GPU dependencies (torch, gpytorch) not installed
    __all__ = []

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "Gaussian Process models require optional 'gpu' dependencies. "
            "Install with: uv sync --group gpu"
        ) from e

    # Create stub functions that raise informative errors
    CompositePeriodicKernel = _raise_import_error
    SparseGPModel = _raise_import_error
    initialize_inducing_points = _raise_import_error
    train_gp_model = _raise_import_error
    save_model = _raise_import_error
    load_model = _raise_import_error
    compute_metrics = _raise_import_error
    compute_anomaly_metrics = _raise_import_error
    compute_prediction_intervals = _raise_import_error
