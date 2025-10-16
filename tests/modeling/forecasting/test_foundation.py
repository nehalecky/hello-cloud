"""
Tests for foundation model forecasters (TimesFM).

These tests verify the TimesFMForecaster wrapper implementation, including:
- Model loading and initialization
- Context window handling (truncation, warnings)
- Zero-shot forecasting (point and quantile forecasts)
- Error handling (missing dependencies, invalid parameters)
- Integration with BaseForecaster API

Note: TimesFM requires significant memory (32GB+ RAM) and may be slow to load.
Tests are designed to be fast using small contexts and horizons.
"""

import numpy as np
import pytest

# Check if TimesFM is available
try:
    from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False


# Skip all tests if TimesFM not installed
pytestmark = pytest.mark.skipif(
    not HAS_TIMESFM,
    reason="TimesFM not installed. Install with: uv sync --group foundation --group gpu",
)


@pytest.fixture
def synthetic_series():
    """Generate synthetic time series for testing."""
    np.random.seed(42)
    t = np.arange(200)
    # Trend + seasonality + noise
    y = 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(200)
    return y


@pytest.fixture
def short_series():
    """Generate short time series for testing context truncation."""
    np.random.seed(42)
    return np.random.randn(30)


class TestTimesFMForecasterInit:
    """Test TimesFMForecaster initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        forecaster = TimesFMForecaster()

        assert forecaster.model_name == "google/timesfm-2.5-200m-pytorch"
        assert forecaster.context_len == 512
        assert forecaster.horizon_len == 128
        assert forecaster.backend == "torch"
        assert forecaster.quantiles == [0.1, 0.5, 0.9]
        assert forecaster.model_ is not None  # Model should be loaded
        assert not forecaster.is_fitted_

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        forecaster = TimesFMForecaster(
            context_len=256,
            horizon_len=64,
            backend="torch",
            quantiles=[0.05, 0.5, 0.95],
        )

        assert forecaster.context_len == 256
        assert forecaster.horizon_len == 64
        assert forecaster.quantiles == [0.05, 0.5, 0.95]

    def test_init_invalid_context_len(self):
        """Test initialization with invalid context_len."""
        with pytest.raises(ValueError, match="context_len must be > 0"):
            TimesFMForecaster(context_len=0)

        with pytest.raises(ValueError, match="context_len must be > 0"):
            TimesFMForecaster(context_len=-1)

    def test_init_invalid_horizon_len(self):
        """Test initialization with invalid horizon_len."""
        with pytest.raises(ValueError, match="horizon_len must be > 0"):
            TimesFMForecaster(horizon_len=0)

        with pytest.raises(ValueError, match="horizon_len must be > 0"):
            TimesFMForecaster(horizon_len=-1)

    def test_init_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError, match="backend must be 'torch' or 'jax'"):
            TimesFMForecaster(backend="invalid")

    def test_init_invalid_quantiles(self):
        """Test initialization with invalid quantiles."""
        with pytest.raises(ValueError, match="quantiles must be in"):
            TimesFMForecaster(quantiles=[0.5, 1.5])  # 1.5 > 1.0

        with pytest.raises(ValueError, match="quantiles must be in"):
            TimesFMForecaster(quantiles=[-0.1, 0.5])  # -0.1 < 0.0


class TestTimesFMForecasterFit:
    """Test TimesFMForecaster.fit() method."""

    def test_fit_stores_context(self, synthetic_series):
        """Test that fit() stores the context window."""
        forecaster = TimesFMForecaster(context_len=100)
        forecaster.fit(synthetic_series)

        assert forecaster.is_fitted_
        assert forecaster.y_context_ is not None
        assert len(forecaster.y_context_) == 100  # Truncated to context_len

    def test_fit_with_short_series(self, short_series):
        """Test fit() with series shorter than context_len."""
        forecaster = TimesFMForecaster(context_len=512)
        forecaster.fit(short_series)

        assert forecaster.is_fitted_
        assert len(forecaster.y_context_) == len(short_series)  # No truncation

    def test_fit_truncates_long_series(self, synthetic_series):
        """Test that fit() truncates series longer than context_len."""
        forecaster = TimesFMForecaster(context_len=50)

        # This should log a warning about truncation
        forecaster.fit(synthetic_series)

        assert forecaster.is_fitted_
        assert len(forecaster.y_context_) == 50
        # Check that we kept the last 50 points
        np.testing.assert_array_equal(forecaster.y_context_, synthetic_series[-50:])

    def test_fit_empty_series_raises(self):
        """Test that fit() raises ValueError for empty series."""
        forecaster = TimesFMForecaster()

        with pytest.raises(ValueError, match="y_train cannot be empty"):
            forecaster.fit(np.array([]))

    def test_fit_nan_series_raises(self, synthetic_series):
        """Test that fit() raises ValueError for series with NaN."""
        forecaster = TimesFMForecaster()
        synthetic_series[10] = np.nan

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            forecaster.fit(synthetic_series)

    def test_fit_inf_series_raises(self, synthetic_series):
        """Test that fit() raises ValueError for series with Inf."""
        forecaster = TimesFMForecaster()
        synthetic_series[10] = np.inf

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            forecaster.fit(synthetic_series)

    def test_fit_returns_self(self, synthetic_series):
        """Test that fit() returns self for method chaining."""
        forecaster = TimesFMForecaster()
        result = forecaster.fit(synthetic_series)

        assert result is forecaster


class TestTimesFMForecasterForecast:
    """Test TimesFMForecaster.forecast() method."""

    def test_forecast_point_predictions(self, synthetic_series):
        """Test forecast() returns point predictions (median)."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        forecast = forecaster.forecast(horizon=20)

        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (20,)
        assert np.all(np.isfinite(forecast))

    def test_forecast_with_quantiles(self, synthetic_series):
        """Test forecast() returns quantile predictions."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        forecast, lower, upper = forecaster.forecast(horizon=20, return_quantiles=True)

        assert isinstance(forecast, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert forecast.shape == (20,)
        assert lower.shape == (20,)
        assert upper.shape == (20,)

        # Lower bound should be <= median <= upper bound
        assert np.all(lower <= forecast)
        assert np.all(forecast <= upper)

    def test_forecast_multiple_horizons(self, synthetic_series):
        """Test forecasting at different horizons."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        # Test different horizons
        for horizon in [5, 10, 20, 50]:
            forecast = forecaster.forecast(horizon=horizon)
            assert forecast.shape == (horizon,)

    def test_forecast_before_fit_raises(self):
        """Test that forecast() raises RuntimeError before fit()."""
        forecaster = TimesFMForecaster()

        with pytest.raises(RuntimeError, match="Must call fit\\(\\) before forecast"):
            forecaster.forecast(horizon=10)

    def test_forecast_invalid_horizon_raises(self, synthetic_series):
        """Test that forecast() raises ValueError for invalid horizon."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        with pytest.raises(ValueError, match="horizon must be > 0"):
            forecaster.forecast(horizon=0)

        with pytest.raises(ValueError, match="horizon must be > 0"):
            forecaster.forecast(horizon=-1)

    def test_forecast_exceeds_horizon_len_raises(self, synthetic_series):
        """Test that forecast() raises ValueError when horizon > horizon_len."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        with pytest.raises(ValueError, match="exceeds horizon_len"):
            forecaster.forecast(horizon=100)

    def test_forecast_consistency(self, synthetic_series):
        """Test that multiple forecast calls are consistent (deterministic)."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        forecast1 = forecaster.forecast(horizon=20)
        forecast2 = forecaster.forecast(horizon=20)

        # TimesFM should be deterministic (no randomness in inference)
        np.testing.assert_array_equal(forecast1, forecast2)


class TestTimesFMForecasterBatchForecast:
    """Test TimesFMForecaster.forecast_batch() method."""

    def test_forecast_batch_multiple_horizons(self, synthetic_series):
        """Test batch forecasting at multiple horizons."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        horizons = [5, 10, 20]
        forecasts = forecaster.forecast_batch(horizons)

        assert len(forecasts) == 3
        assert forecasts[0].shape == (5,)
        assert forecasts[1].shape == (10,)
        assert forecasts[2].shape == (20,)

    def test_forecast_batch_with_quantiles(self, synthetic_series):
        """Test batch forecasting with quantiles."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        horizons = [5, 10]
        results = forecaster.forecast_batch(horizons, return_quantiles=True)

        assert len(results) == 2

        # Each result is a tuple (forecast, lower, upper)
        for i, (forecast, lower, upper) in enumerate(results):
            expected_shape = (horizons[i],)
            assert forecast.shape == expected_shape
            assert lower.shape == expected_shape
            assert upper.shape == expected_shape

    def test_forecast_batch_before_fit_raises(self):
        """Test that forecast_batch() raises RuntimeError before fit()."""
        forecaster = TimesFMForecaster()

        with pytest.raises(RuntimeError, match="Must call fit\\(\\) before"):
            forecaster.forecast_batch([10, 20])

    def test_forecast_batch_invalid_horizon_raises(self, synthetic_series):
        """Test that forecast_batch() raises ValueError for invalid horizons."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)
        forecaster.fit(synthetic_series)

        with pytest.raises(ValueError, match="must be > 0"):
            forecaster.forecast_batch([10, 0, 20])

        with pytest.raises(ValueError, match="exceeds horizon_len"):
            forecaster.forecast_batch([10, 100])


class TestTimesFMForecasterIntegration:
    """Integration tests for TimesFMForecaster."""

    def test_baseforecaster_interface(self, synthetic_series):
        """Test that TimesFMForecaster follows BaseForecaster interface."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)

        # Must have fit() and forecast() methods
        assert hasattr(forecaster, "fit")
        assert hasattr(forecaster, "forecast")
        assert callable(forecaster.fit)
        assert callable(forecaster.forecast)

        # fit() should return self
        result = forecaster.fit(synthetic_series)
        assert result is forecaster

        # forecast() should return ndarray
        forecast = forecaster.forecast(horizon=10)
        assert isinstance(forecast, np.ndarray)

    def test_method_chaining(self, synthetic_series):
        """Test that fit() enables method chaining."""
        forecaster = TimesFMForecaster(context_len=100, horizon_len=50)

        # Should be able to chain fit() and forecast()
        forecast = forecaster.fit(synthetic_series).forecast(horizon=10)

        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (10,)

    def test_realistic_iops_scenario(self):
        """Test realistic IOPS forecasting scenario."""
        # Simulate IOPS data: ~1000 samples after subsampling
        np.random.seed(42)
        t = np.arange(1000)
        iops = (
            5000  # Baseline IOPS
            + 1000 * np.sin(2 * np.pi * t / 250)  # Fast cycle (4 forecasts/cycle)
            + 500 * np.sin(2 * np.pi * t / 1250)  # Slow cycle
            + np.random.randn(1000) * 200  # Noise
        )

        # Use TimesFM for zero-shot forecasting
        forecaster = TimesFMForecaster(
            context_len=512,  # Use last 512 points
            horizon_len=128,  # Forecast up to 128 steps
        )

        # Fit (just stores context)
        forecaster.fit(iops)

        # Forecast next 50 steps with quantiles
        forecast, lower, upper = forecaster.forecast(horizon=50, return_quantiles=True)

        # Verify forecast properties
        assert forecast.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)

        # Forecasts should be in reasonable range
        assert 0 < forecast.mean() < 10000  # Reasonable IOPS range
        assert np.all(lower <= forecast)
        assert np.all(forecast <= upper)

        # Uncertainty should increase with horizon (common pattern)
        # Calculate interval width at start vs end
        interval_width = upper - lower
        # Note: This may not always hold for foundation models, so just check it's positive
        assert np.all(interval_width > 0)
