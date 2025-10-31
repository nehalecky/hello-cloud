"""Evaluation metrics for time series forecasting."""

import numpy as np
from numpy.typing import NDArray


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values

    Returns
    -------
    error : float
        Mean absolute error
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values

    Returns
    -------
    error : float
        Root mean squared error
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values

    Returns
    -------
    error : float
        Mean absolute percentage error (0 to 1 scale)
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def mase(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_train: NDArray[np.float64],
    season_length: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scales MAE by the MAE of a naive forecast on training data.

    Parameters
    ----------
    y_true : array-like
        Ground truth test values
    y_pred : array-like
        Predicted values
    y_train : array-like
        Training data for scaling
    season_length : int
        Seasonal period (1 for non-seasonal)

    Returns
    -------
    error : float
        Mean absolute scaled error
    """
    # MAE of forecast
    forecast_mae = mae(y_true, y_pred)

    # MAE of naive forecast on training data
    naive_errors = np.abs(y_train[season_length:] - y_train[:-season_length])
    naive_mae = np.mean(naive_errors)

    # Scale
    return float(forecast_mae / naive_mae) if naive_mae > 0 else float("inf")
