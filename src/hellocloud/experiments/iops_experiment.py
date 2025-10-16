"""IOPS forecasting experiment tracking with MLflow."""

from dataclasses import asdict, dataclass
from typing import Any

import mlflow
import numpy as np
from loguru import logger
from numpy.typing import NDArray


@dataclass
class IOPSExperimentConfig:
    """Standard configuration for IOPS experiments.

    All parameters logged to MLflow for reproducibility.
    """

    # Data parameters
    subsample_factor: int = 20
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Seasonality (from EDA)
    slow_period_original: int = 1250  # at 1-min sampling
    fast_period_original: int = 250

    # Reproducibility
    random_seed: int = 42

    @property
    def slow_period_subsampled(self) -> int:
        return int(self.slow_period_original / self.subsample_factor)

    @property
    def fast_period_subsampled(self) -> int:
        return int(self.fast_period_original / self.subsample_factor)


class IOPSExperiment:
    """Manages IOPS forecasting experiments with MLflow tracking."""

    def __init__(self, experiment_name: str = "iops-forecasting"):
        """Initialize experiment tracker.

        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")

    def start_run(self, run_name: str, config: IOPSExperimentConfig) -> Any:
        """Start MLflow run with config logging.

        Args:
            run_name: Descriptive run name (e.g., "ARIMA-auto-select")
            config: Experiment configuration

        Returns:
            MLflow active run context manager
        """
        run = mlflow.start_run(run_name=run_name)

        # Log all config parameters
        mlflow.log_params(asdict(config))
        mlflow.log_params(
            {
                "slow_period_subsampled": config.slow_period_subsampled,
                "fast_period_subsampled": config.fast_period_subsampled,
            }
        )

        logger.info(f"Started run: {run_name}")
        logger.info(f"  Subsample factor: {config.subsample_factor}")
        logger.info(
            f"  Train/Val/Test: {config.train_fraction}/{config.val_fraction}/{config.test_fraction}"
        )

        return run

    @staticmethod
    def log_model_params(model_type: str, **params):
        """Log model-specific parameters.

        Args:
            model_type: Model family (e.g., "ARIMA", "Naive", "GP")
            **params: Model parameters (e.g., order=(1,1,1))
        """
        mlflow.log_param("model_type", model_type)
        for key, value in params.items():
            mlflow.log_param(f"model_{key}", value)

        logger.info(f"Model: {model_type}")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None):
        """Log evaluation metrics.

        Args:
            metrics: Dict of metric_name -> value
            step: Optional step number (for multi-horizon forecasts)
        """
        mlflow.log_metrics(metrics, step=step)

        logger.info("Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

    @staticmethod
    def log_forecast_artifact(y_true: NDArray, y_pred: NDArray, filename: str = "forecast.npz"):
        """Save forecast arrays as artifact.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            filename: Artifact filename
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            np.savez(filepath, y_true=y_true, y_pred=y_pred)
            mlflow.log_artifact(str(filepath))

        logger.info(f"Logged forecast artifact: {filename}")

    @staticmethod
    def end_run():
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("Run completed")
