"""Baseline forecasting models for time series comparison."""

import numpy as np
from numpy.typing import NDArray


class NaiveForecaster:
    """
    Naive forecaster that repeats the last observed value.

    This is the simplest baseline - useful for comparison benchmarks.
    """

    def __init__(self):
        self.last_value: float | None = None

    def fit(self, y: NDArray[np.float64]) -> "NaiveForecaster":
        """
        Fit the model by storing the last observed value.

        Parameters
        ----------
        y : array-like
            Historical time series values

        Returns
        -------
        self
            Fitted forecaster
        """
        self.last_value = float(y[-1])
        return self

    def forecast(self, horizon: int) -> NDArray[np.float64]:
        """
        Generate forecast by repeating the last value.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast

        Returns
        -------
        forecast : ndarray
            Array of forecasted values
        """
        if self.last_value is None:
            raise ValueError("Must call fit() before forecast()")
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecaster:
    """
    Seasonal naive forecaster that repeats the seasonal pattern.

    For seasonal data, repeats values from the same season in history.
    """

    def __init__(self, period: int):
        """
        Initialize seasonal naive forecaster.

        Parameters
        ----------
        period : int
            Length of seasonal cycle (e.g., 12 for monthly data with yearly pattern)
        """
        self.period = period
        self.history: NDArray[np.float64] | None = None

    def fit(self, y: NDArray[np.float64]) -> "SeasonalNaiveForecaster":
        """Fit by storing the full history."""
        self.history = np.array(y)
        return self

    def forecast(self, horizon: int) -> NDArray[np.float64]:
        """Generate forecast by repeating seasonal pattern."""
        if self.history is None:
            raise ValueError("Must call fit() before forecast()")

        # Repeat the last period values
        pattern = self.history[-self.period :]
        forecast_array = np.tile(pattern, (horizon // self.period) + 1)
        return forecast_array[:horizon]


class MovingAverageForecaster:
    """
    Moving average forecaster using rolling window.

    Forecasts by averaging the last k observations.
    """

    def __init__(self, window: int = 3):
        """
        Initialize moving average forecaster.

        Parameters
        ----------
        window : int
            Number of recent observations to average
        """
        self.window = window
        self.last_values: NDArray[np.float64] | None = None

    def fit(self, y: NDArray[np.float64]) -> "MovingAverageForecaster":
        """Fit by storing the last window observations."""
        self.last_values = np.array(y[-self.window :])
        return self

    def forecast(self, horizon: int) -> NDArray[np.float64]:
        """Generate forecast using moving average."""
        if self.last_values is None:
            raise ValueError("Must call fit() before forecast()")

        # For simplicity, use average of last window for all forecasts
        avg = np.mean(self.last_values)
        return np.full(horizon, avg)
