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

References:
- GPyTorch: https://gpytorch.ai/
- Sparse GPs (SVGP): Hensman et al. (2013)
- Student-t Processes: Shah et al. (2014)
"""

from .kernels import CompositePeriodicKernel
from .models import SparseGPModel, initialize_inducing_points
from .training import train_gp_model, save_model, load_model
from .evaluation import compute_metrics, compute_anomaly_metrics, compute_prediction_intervals

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
