"""Tests for baseline forecasting models."""

import numpy as np

from hellocloud.modeling.forecasting import NaiveForecaster


def test_naive_forecaster_repeats_last_value():
    """Test that NaiveForecaster repeats the last observed value."""
    # Arrange
    history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    forecaster = NaiveForecaster()

    # Act
    forecaster.fit(history)
    forecast = forecaster.forecast(horizon=3)

    # Assert
    assert len(forecast) == 3
    assert np.allclose(forecast, [5.0, 5.0, 5.0])


def test_naive_forecaster_handles_single_value():
    """Test that NaiveForecaster works with single observation."""
    history = np.array([42.0])
    forecaster = NaiveForecaster()

    forecaster.fit(history)
    forecast = forecaster.forecast(horizon=5)

    assert len(forecast) == 5
    assert np.allclose(forecast, [42.0] * 5)
