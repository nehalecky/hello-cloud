"""Tests for forecasting evaluation metrics."""

import numpy as np

from hellocloud.modeling.forecasting.metrics import mae, mape, rmse


def test_mae_perfect_forecast():
    """Test MAE with perfect forecast (zero error)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    assert mae(y_true, y_pred) == 0.0


def test_mae_calculation():
    """Test MAE calculation with known values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5])

    # MAE = mean(|0.5, 0.5, 0.5, 0.5|) = 0.5
    assert mae(y_true, y_pred) == 0.5


def test_rmse_calculation():
    """Test RMSE calculation with known values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 4.0, 5.0])

    # RMSE = sqrt(mean([0, 0, 1, 1])) = sqrt(0.5) ≈ 0.707
    result = rmse(y_true, y_pred)
    assert np.isclose(result, np.sqrt(0.5))


def test_mape_calculation():
    """Test MAPE calculation (percentage error)."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])

    # MAPE = mean(|10/100, 20/200, 30/300|) = mean([0.1, 0.1, 0.1]) = 0.1
    assert np.isclose(mape(y_true, y_pred), 0.1)
