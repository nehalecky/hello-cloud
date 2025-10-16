"""Experiment tracking and configuration."""

from pathlib import Path

import mlflow

# MLflow tracking URI (local for now)
MLFLOW_TRACKING_URI = Path(__file__).parent.parent.parent / "mlruns"
MLFLOW_TRACKING_URI.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")

# Export key functions
from .iops_experiment import IOPSExperiment  # noqa: E402

__all__ = ["IOPSExperiment", "MLFLOW_TRACKING_URI"]
