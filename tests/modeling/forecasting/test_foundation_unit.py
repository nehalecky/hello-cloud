"""
Unit tests for foundation model forecasters (TimesFM) with mocking.

These tests verify the TimesFMForecaster wrapper without actually loading
the heavy TimesFM model. We mock the timesfm module to avoid:
- Downloading large model files (200M parameters)
- High memory usage (32GB+ RAM)
- Slow model loading times

This allows fast, lightweight testing of the wrapper logic.
"""

# Check if TimesFM is available
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

HAS_TIMESFM = importlib.util.find_spec("timesfm") is not None


@pytest.fixture
def mock_timesfm_model():
    """Mock TimesFM model that simulates the real API."""
    mock_model = MagicMock()

    # Mock forecast() method
    # TimesFM.forecast returns (point_forecast, quantile_forecast)
    # point_forecast: (n_series, horizon)
    # quantile_forecast: (n_series, horizon, n_quantiles)
    def mock_forecast(inputs, freq, horizon):
        n_series = len(inputs)
        n_quantiles = 3  # [0.1, 0.5, 0.9]

        # Generate dummy forecasts
        point_forecast = np.random.randn(n_series, horizon) + 10.0
        quantile_forecast = np.zeros((n_series, horizon, n_quantiles))

        # Fill quantiles (sorted: 10th < 50th < 90th)
        for i in range(n_series):
            for h in range(horizon):
                base = point_forecast[i, h]
                quantile_forecast[i, h, 0] = base - 2.0  # 10th percentile
                quantile_forecast[i, h, 1] = base  # 50th percentile (median)
                quantile_forecast[i, h, 2] = base + 2.0  # 90th percentile

        return point_forecast, quantile_forecast

    mock_model.forecast = mock_forecast
    return mock_model


@pytest.fixture
def synthetic_series():
    """Generate synthetic time series for testing."""
    np.random.seed(42)
    t = np.arange(200)
    y = 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(200)
    return y


@pytest.mark.skipif(not HAS_TIMESFM, reason="TimesFM not installed")
class TestTimesFMForecasterMocked:
    """Test TimesFMForecaster with mocked TimesFM model."""

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_init_loads_model(self, mock_from_pretrained, mock_timesfm_model):
        """Test that __init__ calls from_pretrained()."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(
            model_name="google/timesfm-2.5-200m-pytorch",
            context_len=512,
            horizon_len=128,
            backend="torch",
        )

        # Verify from_pretrained was called
        mock_from_pretrained.assert_called_once_with(
            model_name="google/timesfm-2.5-200m-pytorch",
            backend="torch",
        )

        # Verify attributes
        assert forecaster.model_ is mock_timesfm_model
        assert forecaster.context_len == 512
        assert forecaster.horizon_len == 128
        assert not forecaster.is_fitted_

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_fit_stores_context(self, mock_from_pretrained, mock_timesfm_model, synthetic_series):
        """Test that fit() stores context window correctly."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(context_len=100)
        forecaster.fit(synthetic_series)

        assert forecaster.is_fitted_
        assert len(forecaster.y_context_) == 100
        np.testing.assert_array_equal(forecaster.y_context_, synthetic_series[-100:])

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_forecast_point_predictions(
        self, mock_from_pretrained, mock_timesfm_model, synthetic_series
    ):
        """Test forecast() returns point predictions."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        forecast = forecaster.forecast(horizon=20)

        # Verify forecast shape and type
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (20,)
        assert np.all(np.isfinite(forecast))

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_forecast_with_quantiles(
        self, mock_from_pretrained, mock_timesfm_model, synthetic_series
    ):
        """Test forecast() returns quantile predictions."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        forecast, lower, upper = forecaster.forecast(horizon=20, return_quantiles=True)

        # Verify shapes
        assert forecast.shape == (20,)
        assert lower.shape == (20,)
        assert upper.shape == (20,)

        # Verify ordering: lower <= median <= upper
        assert np.all(lower <= forecast)
        assert np.all(forecast <= upper)

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_forecast_batch(self, mock_from_pretrained, mock_timesfm_model, synthetic_series):
        """Test forecast_batch() for multiple horizons."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        horizons = [5, 10, 20]
        forecasts = forecaster.forecast_batch(horizons)

        assert len(forecasts) == 3
        assert forecasts[0].shape == (5,)
        assert forecasts[1].shape == (10,)
        assert forecasts[2].shape == (20,)

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_error_handling_invalid_horizon(
        self, mock_from_pretrained, mock_timesfm_model, synthetic_series
    ):
        """Test error handling for invalid horizons."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        # horizon <= 0
        with pytest.raises(ValueError, match="horizon must be > 0"):
            forecaster.forecast(horizon=0)

        # horizon > horizon_len
        with pytest.raises(ValueError, match="exceeds horizon_len"):
            forecaster.forecast(horizon=100)

    @patch("hellocloud.modeling.forecasting.foundation.timesfm.TimesFM.from_pretrained")
    def test_error_handling_fit_before_forecast(self, mock_from_pretrained, mock_timesfm_model):
        """Test error when forecast() called before fit()."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_from_pretrained.return_value = mock_timesfm_model

        forecaster = TimesFMForecaster()

        with pytest.raises(RuntimeError, match="Must call fit\\(\\) before forecast"):
            forecaster.forecast(horizon=10)


class TestTimesFMForecasterImportHandling:
    """Test import error handling when TimesFM not installed."""

    def test_import_without_timesfm(self):
        """Test that import works even if timesfm not installed."""
        # This test verifies the try/except block in foundation.py
        # If TimesFM is installed, we skip this test (it won't fail)
        if HAS_TIMESFM:
            pytest.skip("TimesFM is installed, cannot test import error handling")

        # If TimesFM not installed, importing should work but HAS_TIMESFM=False
        from hellocloud.modeling.forecasting import foundation

        assert not foundation.HAS_TIMESFM

    def test_error_on_init_without_timesfm(self):
        """Test that TimesFMForecaster.__init__() raises ImportError if timesfm missing."""
        if HAS_TIMESFM:
            pytest.skip("TimesFM is installed, cannot test import error")

        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        with pytest.raises(ImportError, match="timesfm package is not installed"):
            TimesFMForecaster()


@pytest.mark.skipif(not HAS_TIMESFM, reason="TimesFM not installed")
class TestTimesFMForecasterValidation:
    """Test parameter validation with real imports but mocked model loading."""

    @patch("timesfm.TimesFM.from_pretrained")
    def test_validation_context_len(self, mock_from_pretrained):
        """Test context_len validation."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        # Mock the from_pretrained to avoid actual model loading
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Valid context_len
        forecaster = TimesFMForecaster(context_len=512)
        assert forecaster.context_len == 512

        # Invalid context_len
        with pytest.raises(ValueError, match="context_len must be > 0"):
            TimesFMForecaster(context_len=0)

        with pytest.raises(ValueError, match="context_len must be > 0"):
            TimesFMForecaster(context_len=-1)

    @patch("timesfm.TimesFM.from_pretrained")
    def test_validation_horizon_len(self, mock_from_pretrained):
        """Test horizon_len validation."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Valid horizon_len
        forecaster = TimesFMForecaster(horizon_len=128)
        assert forecaster.horizon_len == 128

        # Invalid horizon_len
        with pytest.raises(ValueError, match="horizon_len must be > 0"):
            TimesFMForecaster(horizon_len=0)

    @patch("timesfm.TimesFM.from_pretrained")
    def test_validation_backend(self, mock_from_pretrained):
        """Test backend validation."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Valid backends
        for backend in ["torch", "jax"]:
            forecaster = TimesFMForecaster(backend=backend)
            assert forecaster.backend == backend

        # Invalid backend
        with pytest.raises(ValueError, match="backend must be 'torch' or 'jax'"):
            TimesFMForecaster(backend="tensorflow")

    @patch("timesfm.TimesFM.from_pretrained")
    def test_validation_quantiles(self, mock_from_pretrained):
        """Test quantiles validation."""
        from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Valid quantiles
        forecaster = TimesFMForecaster(quantiles=[0.1, 0.5, 0.9])
        assert forecaster.quantiles == [0.1, 0.5, 0.9]

        # Invalid quantiles (out of [0, 1] range)
        with pytest.raises(ValueError, match="quantiles must be in"):
            TimesFMForecaster(quantiles=[0.5, 1.5])

        with pytest.raises(ValueError, match="quantiles must be in"):
            TimesFMForecaster(quantiles=[-0.1, 0.5])
