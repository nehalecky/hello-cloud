"""
Baseline forecasting models for time series prediction.

This module provides simple baseline forecasters that serve as minimum performance
benchmarks. Sophisticated models (GP, TimesFM, etc.) must outperform these baselines
to demonstrate value.

Baseline models included:
- NaiveForecaster: Last value propagation (random walk baseline)
- SeasonalNaiveForecaster: Seasonal repetition (for periodic series)
- MovingAverageForecaster: Smoothed constant (for stable series)

All forecasters follow a consistent API:
1. fit(y_train): Learn from training data
2. forecast(horizon): Generate predictions for future timesteps

Example:
    ```python
    import numpy as np
    from hellocloud.modeling.forecasting.baselines import (
        NaiveForecaster,
        SeasonalNaiveForecaster,
        MovingAverageForecaster
    )

    # Training data
    y_train = np.random.randn(1000)

    # Naive baseline (last value)
    naive = NaiveForecaster()
    naive.fit(y_train)
    naive_preds = naive.forecast(horizon=50)

    # Seasonal baseline (period=250 for IOPS dataset)
    seasonal = SeasonalNaiveForecaster(period=250)
    seasonal.fit(y_train)
    seasonal_preds = seasonal.forecast(horizon=50)

    # Moving average baseline
    ma = MovingAverageForecaster(window=50)
    ma.fit(y_train)
    ma_preds = ma.forecast(horizon=50)
    ```
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    Defines the common interface that all forecasting models must implement.
    This ensures consistent API across naive baselines, statistical models,
    and sophisticated ML models.

    All subclasses must implement:
    - fit(y_train): Learn from training data
    - forecast(horizon): Generate predictions
    """

    @abstractmethod
    def fit(self, y_train: NDArray[np.floating]) -> "BaseForecaster":
        """
        Fit the forecaster to training data.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance (for method chaining)

        Raises:
            ValueError: If y_train is empty or invalid
        """
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> NDArray[np.floating]:
        """
        Generate forecasts for future timesteps.

        Args:
            horizon: Number of steps ahead to forecast (must be > 0)

        Returns:
            Forecasts of shape (horizon,)

        Raises:
            ValueError: If horizon <= 0
            RuntimeError: If called before fit()
        """
        pass


class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster that propagates the last observed value.

    This is the simplest baseline, also known as the "persistence" or
    "random walk" forecaster. It repeats the last value for all future timesteps.

    Formula:
        ŷ(t+h) = y(t) for all h in [1, horizon]

    Use cases:
        - Baseline for random walk time series
        - Non-stationary series with strong persistence
        - Quick sanity check for any forecasting problem

    Performance characteristics:
        - Optimal for random walks (unpredictable changes)
        - Poor for trending or seasonal series
        - Often hard to beat for financial data (efficient markets)

    Example:
        ```python
        # Stock price prediction (random walk)
        prices = np.array([100, 102, 101, 103, 102])

        naive = NaiveForecaster()
        naive.fit(prices)
        preds = naive.forecast(horizon=3)
        # Output: [102, 102, 102] (repeats last value)
        ```

    Attributes:
        last_value_: Last value from training data (set after fit())
        is_fitted_: Whether the model has been fitted
    """

    def __init__(self):
        """Initialize the naive forecaster."""
        self.last_value_: float | None = None
        self.is_fitted_ = False

    def fit(self, y_train: NDArray[np.floating]) -> "NaiveForecaster":
        """
        Fit the naive forecaster by storing the last observed value.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance

        Raises:
            ValueError: If y_train is empty
        """
        if len(y_train) == 0:
            raise ValueError("y_train cannot be empty")

        self.last_value_ = float(y_train[-1])
        self.is_fitted_ = True
        return self

    def forecast(self, horizon: int) -> NDArray[np.floating]:
        """
        Generate forecasts by repeating the last observed value.

        Args:
            horizon: Number of steps ahead to forecast

        Returns:
            Forecasts of shape (horizon,), all equal to last_value_

        Raises:
            ValueError: If horizon <= 0
            RuntimeError: If called before fit()
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast()")

        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        return np.full(horizon, self.last_value_, dtype=np.float64)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive forecaster that repeats the last seasonal cycle.

    This baseline is appropriate for time series with strong seasonality.
    It repeats the pattern from the most recent seasonal cycle.

    Formula:
        ŷ(t+h) = y(t - period + ((h-1) mod period))

    Use cases:
        - Series with strong daily/weekly/yearly patterns
        - Cloud metrics with periodic workload patterns
        - IOPS dataset (period=250 or 1250)

    Performance characteristics:
        - Excellent for pure seasonal series (no trend)
        - Poor if seasonality changes over time
        - Requires at least one full seasonal cycle of data

    Example:
        ```python
        # Daily website traffic with weekly pattern (period=7)
        traffic = np.array([100, 120, 130, 125, 135, 90, 80,  # Week 1
                           105, 125, 135, 130, 140, 95, 85])  # Week 2

        seasonal = SeasonalNaiveForecaster(period=7)
        seasonal.fit(traffic)
        preds = seasonal.forecast(horizon=7)
        # Output: [105, 125, 135, 130, 140, 95, 85] (repeats last week)
        ```

    Attributes:
        period: Seasonal period (e.g., 250 for fast IOPS cycle)
        seasonal_values_: Last full seasonal cycle (set after fit())
        is_fitted_: Whether the model has been fitted
    """

    def __init__(self, period: int):
        """
        Initialize the seasonal naive forecaster.

        Args:
            period: Length of the seasonal cycle (must be > 0)

        Raises:
            ValueError: If period <= 0
        """
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")

        self.period = period
        self.seasonal_values_: NDArray[np.floating] | None = None
        self.is_fitted_ = False

    def fit(self, y_train: NDArray[np.floating]) -> "SeasonalNaiveForecaster":
        """
        Fit the seasonal forecaster by storing the last seasonal cycle.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance

        Raises:
            ValueError: If y_train has fewer than period samples
        """
        if len(y_train) < self.period:
            raise ValueError(
                f"y_train must have at least {self.period} samples " f"(got {len(y_train)})"
            )

        # Store the last full seasonal cycle
        self.seasonal_values_ = y_train[-self.period :].copy()
        self.is_fitted_ = True
        return self

    def forecast(self, horizon: int) -> NDArray[np.floating]:
        """
        Generate forecasts by repeating the last seasonal cycle.

        Args:
            horizon: Number of steps ahead to forecast

        Returns:
            Forecasts of shape (horizon,), repeating seasonal pattern

        Raises:
            ValueError: If horizon <= 0
            RuntimeError: If called before fit()
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast()")

        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        # Repeat the seasonal pattern as many times as needed
        n_full_cycles = horizon // self.period
        remainder = horizon % self.period

        # Tile full cycles and add remainder
        forecasts = np.tile(self.seasonal_values_, n_full_cycles)
        if remainder > 0:
            forecasts = np.concatenate([forecasts, self.seasonal_values_[:remainder]])

        return forecasts


class MovingAverageForecaster(BaseForecaster):
    """
    Moving average forecaster that uses the mean of recent observations.

    This baseline smooths out noise by averaging the last `window` values,
    then propagates this average as a constant forecast.

    Formula:
        ŷ(t+h) = mean(y(t-window+1), ..., y(t)) for all h in [1, horizon]

    Use cases:
        - Stable series with noise but no trend or seasonality
        - Smoothing short-term fluctuations
        - Series where recent average is a good predictor

    Performance characteristics:
        - Better than naive for noisy but stable series
        - Poor for trending or seasonal series
        - Sensitive to window size choice

    Example:
        ```python
        # Temperature readings with noise
        temps = np.array([20.1, 19.8, 20.3, 19.9, 20.2, 20.0, 19.7])

        ma = MovingAverageForecaster(window=3)
        ma.fit(temps)
        preds = ma.forecast(horizon=3)
        # Output: [19.97, 19.97, 19.97] (mean of last 3 values)
        ```

    Attributes:
        window: Number of recent observations to average
        mean_value_: Mean of last window observations (set after fit())
        is_fitted_: Whether the model has been fitted
    """

    def __init__(self, window: int):
        """
        Initialize the moving average forecaster.

        Args:
            window: Number of recent observations to average (must be > 0)

        Raises:
            ValueError: If window <= 0
        """
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")

        self.window = window
        self.mean_value_: float | None = None
        self.is_fitted_ = False

    def fit(self, y_train: NDArray[np.floating]) -> "MovingAverageForecaster":
        """
        Fit the moving average forecaster by computing mean of last window values.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance

        Raises:
            ValueError: If y_train has fewer than window samples
        """
        if len(y_train) < self.window:
            raise ValueError(
                f"y_train must have at least {self.window} samples " f"(got {len(y_train)})"
            )

        # Compute mean of last window values
        self.mean_value_ = float(np.mean(y_train[-self.window :]))
        self.is_fitted_ = True
        return self

    def forecast(self, horizon: int) -> NDArray[np.floating]:
        """
        Generate forecasts by repeating the moving average.

        Args:
            horizon: Number of steps ahead to forecast

        Returns:
            Forecasts of shape (horizon,), all equal to mean_value_

        Raises:
            ValueError: If horizon <= 0
            RuntimeError: If called before fit()
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast()")

        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        return np.full(horizon, self.mean_value_, dtype=np.float64)
