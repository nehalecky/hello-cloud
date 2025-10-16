"""Time series forecasting framework for cloud workload prediction.

This module provides a comprehensive forecasting framework including:
- Baseline forecasters (naive, seasonal naive, moving average)
- Statistical models (ARIMA, exponential smoothing)
- Machine learning models (Gaussian Process wrappers)
- Foundation models (TimesFM integration)
- Evaluation metrics and backtesting utilities
"""

from .arima import ARIMAForecaster
from .backtesting import BacktestResults, WalkForwardBacktest
from .baselines import (
    BaseForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from .evaluation import (
    compute_all_metrics,
    coverage,
    interval_sharpness,
    mae,
    mape,
    mase,
    quantile_loss,
    rmse,
    smape,
)

# Optional: Foundation models (may not be installed)
try:
    from .foundation import TimesFMForecaster

    HAS_FOUNDATION = True
except ImportError:
    HAS_FOUNDATION = False

__all__ = [
    # Forecasters
    "BaseForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "ARIMAForecaster",
    # Evaluation metrics
    "mae",
    "rmse",
    "mape",
    "smape",
    "mase",
    "quantile_loss",
    "coverage",
    "interval_sharpness",
    "compute_all_metrics",
    # Backtesting
    "WalkForwardBacktest",
    "BacktestResults",
]

# Add foundation models to __all__ if available
if HAS_FOUNDATION:
    __all__.extend(["TimesFMForecaster"])
