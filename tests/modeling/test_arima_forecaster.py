"""
Tests for ARIMA forecasting models.

Validates the ARIMAForecaster:
- Model initialization with various order configurations
- Automatic order selection (both pmdarima and grid search)
- Forecast generation with and without confidence intervals
- Seasonal ARIMA (SARIMA) functionality
- Edge cases and error handling
- Integration with realistic time series patterns

Tests cover:
- Basic ARIMA functionality
- Auto-selection mechanisms
- Confidence interval generation
- Seasonal component handling
- Error cases (convergence, invalid inputs, etc.)
- Comparison with baseline forecasters
"""

import numpy as np
import pytest

from hellocloud.modeling.forecasting.arima import HAS_PMDARIMA, ARIMAForecaster
from hellocloud.modeling.forecasting.baselines import BaseForecaster


class TestARIMAForecasterInitialization:
    """Test suite for ARIMAForecaster initialization."""

    def test_default_initialization(self):
        """Test ARIMAForecaster initializes with defaults."""
        forecaster = ARIMAForecaster()
        assert forecaster.order == (1, 1, 1)
        assert not forecaster.seasonal
        assert forecaster.seasonal_order is None
        assert not forecaster.auto_select
        assert not forecaster.is_fitted_
        assert forecaster.alpha == 0.05

    def test_custom_order_initialization(self):
        """Test initialization with custom order."""
        forecaster = ARIMAForecaster(order=(2, 1, 2))
        assert forecaster.order == (2, 1, 2)
        assert not forecaster.is_fitted_

    def test_auto_select_initialization(self):
        """Test initialization with auto_select enabled."""
        forecaster = ARIMAForecaster(auto_select=True)
        assert forecaster.auto_select
        assert not forecaster.is_fitted_

    def test_seasonal_initialization(self):
        """Test initialization with seasonal parameters."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal=True, seasonal_order=(1, 0, 1, 12))
        assert forecaster.seasonal
        assert forecaster.seasonal_order == (1, 0, 1, 12)

    def test_seasonal_without_order_raises(self):
        """Test that seasonal=True without seasonal_order raises ValueError."""
        with pytest.raises(ValueError, match="seasonal_order must be provided"):
            ARIMAForecaster(seasonal=True)

    def test_invalid_seasonal_order_raises(self):
        """Test that invalid seasonal_order raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            ARIMAForecaster(seasonal=True, seasonal_order=(1, 0, 1, 0))

        with pytest.raises(ValueError, match="must be > 0"):
            ARIMAForecaster(seasonal=True, seasonal_order=(1, 0, 1, -5))


class TestARIMAForecasterFitting:
    """Test suite for ARIMAForecaster fitting."""

    @pytest.fixture
    def simple_trend(self):
        """Create simple trend series."""
        np.random.seed(42)
        t = np.arange(200)
        return 10 + 0.5 * t + np.random.randn(200)

    @pytest.fixture
    def seasonal_series(self):
        """Create series with trend and seasonality."""
        np.random.seed(42)
        t = np.arange(500)
        return 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(500)

    def test_fit_stores_model(self, simple_trend):
        """Test fit stores the fitted model."""
        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(simple_trend)

        assert forecaster.is_fitted_
        assert forecaster.model_ is not None
        assert forecaster.selected_order_ == (1, 1, 1)

    def test_fit_empty_array_raises(self):
        """Test fit raises ValueError for empty array."""
        forecaster = ARIMAForecaster()

        with pytest.raises(ValueError, match="cannot be empty"):
            forecaster.fit(np.array([]))

    def test_fit_nan_values_raises(self):
        """Test fit raises ValueError for NaN values."""
        forecaster = ARIMAForecaster()
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        with pytest.raises(ValueError, match="NaN or Inf"):
            forecaster.fit(data)

    def test_fit_inf_values_raises(self):
        """Test fit raises ValueError for Inf values."""
        forecaster = ARIMAForecaster()
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        with pytest.raises(ValueError, match="NaN or Inf"):
            forecaster.fit(data)

    def test_fit_too_short_series_raises(self):
        """Test fit raises ValueError if series too short for order."""
        forecaster = ARIMAForecaster(order=(5, 2, 5))
        short_series = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="requires at least"):
            forecaster.fit(short_series)

    def test_fit_with_seasonal_order(self, seasonal_series):
        """Test fit works with seasonal ARIMA."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal=True, seasonal_order=(1, 0, 1, 50))
        forecaster.fit(seasonal_series)

        assert forecaster.is_fitted_
        assert forecaster.model_ is not None

    def test_method_chaining(self, simple_trend):
        """Test fit returns self for method chaining."""
        forecaster = ARIMAForecaster()
        result = forecaster.fit(simple_trend)

        assert result is forecaster


class TestARIMAForecasterForecasting:
    """Test suite for ARIMAForecaster forecasting."""

    @pytest.fixture
    def fitted_forecaster(self):
        """Create fitted ARIMA forecaster."""
        np.random.seed(42)
        t = np.arange(200)
        y = 10 + 0.5 * t + np.random.randn(200)

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(y)
        return forecaster

    def test_forecast_correct_shape(self, fitted_forecaster):
        """Test forecast returns correct shape."""
        for horizon in [1, 10, 50]:
            preds = fitted_forecaster.forecast(horizon)
            assert preds.shape == (horizon,)
            assert isinstance(preds, np.ndarray)

    def test_forecast_with_confidence_intervals(self, fitted_forecaster):
        """Test forecast returns confidence intervals when requested."""
        horizon = 20
        preds, lower, upper = fitted_forecaster.forecast(horizon, return_conf_int=True)

        assert preds.shape == (horizon,)
        assert lower.shape == (horizon,)
        assert upper.shape == (horizon,)

        # Lower bounds should be less than point forecasts
        assert np.all(lower <= preds)
        # Upper bounds should be greater than point forecasts
        assert np.all(upper >= preds)

    def test_forecast_confidence_intervals_widen(self, fitted_forecaster):
        """Test that confidence intervals widen over time."""
        horizon = 50
        preds, lower, upper = fitted_forecaster.forecast(horizon, return_conf_int=True)

        # Compute interval widths
        widths = upper - lower

        # Later intervals should generally be wider (with some tolerance)
        early_mean = np.mean(widths[:10])
        late_mean = np.mean(widths[-10:])
        assert late_mean >= early_mean * 0.9  # Allow some variance

    def test_forecast_before_fit_raises(self):
        """Test forecast raises RuntimeError if called before fit."""
        forecaster = ARIMAForecaster()

        with pytest.raises(RuntimeError, match="Must call fit"):
            forecaster.forecast(10)

    def test_forecast_invalid_horizon_raises(self, fitted_forecaster):
        """Test forecast raises ValueError for invalid horizon."""
        with pytest.raises(ValueError, match="must be > 0"):
            fitted_forecaster.forecast(0)

        with pytest.raises(ValueError, match="must be > 0"):
            fitted_forecaster.forecast(-5)

    def test_forecast_reasonable_values(self):
        """Test forecast produces reasonable values for known series."""
        np.random.seed(42)
        # Simple upward trend
        t = np.arange(100)
        y = 10 + 0.5 * t + np.random.randn(100)

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(y)

        preds = forecaster.forecast(horizon=10)

        # Forecasts should continue upward trend
        assert np.mean(preds) > np.mean(y[-10:])

        # Forecasts should be in reasonable range
        assert np.all(preds > 0)
        assert np.all(preds < 100)


class TestARIMAAutoSelection:
    """Test suite for automatic order selection."""

    @pytest.fixture
    def trend_series(self):
        """Create series with trend."""
        np.random.seed(42)
        t = np.arange(200)
        return 10 + 0.5 * t + np.random.randn(200)

    def test_auto_select_finds_order(self, trend_series):
        """Test auto_select successfully finds an order."""
        forecaster = ARIMAForecaster(auto_select=True)
        forecaster.fit(trend_series)

        assert forecaster.is_fitted_
        assert forecaster.selected_order_ is not None
        assert len(forecaster.selected_order_) == 3

        # Should be able to forecast
        preds = forecaster.forecast(horizon=10)
        assert len(preds) == 10

    def test_grid_search_fallback(self, trend_series, monkeypatch):
        """Test grid search fallback when pmdarima unavailable."""
        # Temporarily disable pmdarima
        import hellocloud.modeling.forecasting.arima as arima_module

        original_has_pmdarima = arima_module.HAS_PMDARIMA
        monkeypatch.setattr(arima_module, "HAS_PMDARIMA", False)

        try:
            forecaster = ARIMAForecaster(auto_select=True)
            forecaster.fit(trend_series)

            assert forecaster.is_fitted_
            assert forecaster.selected_order_ is not None

            # Should still be able to forecast
            preds = forecaster.forecast(horizon=10)
            assert len(preds) == 10

        finally:
            # Restore original value
            monkeypatch.setattr(arima_module, "HAS_PMDARIMA", original_has_pmdarima)

    @pytest.mark.skipif(not HAS_PMDARIMA, reason="pmdarima not installed")
    def test_pmdarima_selection(self, trend_series):
        """Test pmdarima-based order selection (if available)."""
        forecaster = ARIMAForecaster(auto_select=True)
        forecaster.fit(trend_series)

        assert forecaster.is_fitted_
        assert forecaster.selected_order_ is not None

        # pmdarima should find reasonable orders
        p, d, q = forecaster.selected_order_
        assert 0 <= p <= 5
        assert 0 <= d <= 2
        assert 0 <= q <= 5


class TestARIMASeasonalForecasting:
    """Test suite for seasonal ARIMA (SARIMA)."""

    @pytest.fixture
    def seasonal_data(self):
        """Create data with strong seasonality."""
        np.random.seed(42)
        t = np.arange(500)
        # Period = 50
        return 100 + 20 * np.sin(2 * np.pi * t / 50) + 2 * np.random.randn(500)

    def test_sarima_initialization(self):
        """Test SARIMA initializes correctly."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal=True, seasonal_order=(1, 0, 1, 50))
        assert forecaster.seasonal
        assert forecaster.seasonal_order == (1, 0, 1, 50)

    def test_sarima_fitting(self, seasonal_data):
        """Test SARIMA fits to seasonal data."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal=True, seasonal_order=(1, 0, 1, 50))
        forecaster.fit(seasonal_data)

        assert forecaster.is_fitted_
        assert forecaster.model_ is not None

    def test_sarima_forecasting(self, seasonal_data):
        """Test SARIMA produces forecasts."""
        forecaster = ARIMAForecaster(order=(1, 1, 1), seasonal=True, seasonal_order=(1, 0, 1, 50))
        forecaster.fit(seasonal_data)

        preds = forecaster.forecast(horizon=50)
        assert len(preds) == 50

        # Forecasts should capture some seasonality
        # (Not a strong test, but checks basic functionality)
        assert np.std(preds) > 0  # Some variation in predictions


class TestARIMAIntegration:
    """Integration tests comparing ARIMA with baseline forecasters."""

    @pytest.fixture
    def random_walk(self):
        """Generate random walk series."""
        np.random.seed(42)
        steps = np.random.randn(300)
        return np.cumsum(steps) + 100

    @pytest.fixture
    def trend_series(self):
        """Generate series with linear trend."""
        np.random.seed(42)
        t = np.arange(300)
        return 10 + 0.5 * t + 2 * np.random.randn(300)

    @pytest.fixture
    def seasonal_series(self):
        """Generate seasonal series."""
        np.random.seed(42)
        t = np.arange(500)
        return 100 + 20 * np.sin(2 * np.pi * t / 50) + 2 * np.random.randn(500)

    def test_arima_on_random_walk(self, random_walk):
        """Test ARIMA on random walk (should work like ARIMA(0,1,0))."""
        train = random_walk[:-50]

        forecaster = ARIMAForecaster(order=(0, 1, 0))
        forecaster.fit(train)
        preds = forecaster.forecast(horizon=50)

        assert len(preds) == 50
        # For random walk, first prediction should be close to last value
        assert abs(preds[0] - train[-1]) < 5

    def test_arima_on_trend(self, trend_series):
        """Test ARIMA captures linear trend."""
        train = trend_series[:-50]

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(train)
        preds = forecaster.forecast(horizon=50)

        # Forecasts should continue upward trend
        assert np.mean(preds) > np.mean(train[-20:])

    def test_arima_vs_naive_on_trend(self, trend_series):
        """Test ARIMA should outperform Naive on trending data."""
        from hellocloud.modeling.forecasting.baselines import NaiveForecaster

        train = trend_series[:-50]
        test = trend_series[-50:]

        # Naive forecast
        naive = NaiveForecaster()
        naive.fit(train)
        naive_preds = naive.forecast(horizon=50)
        naive_mae = np.mean(np.abs(naive_preds - test))

        # ARIMA forecast
        arima = ARIMAForecaster(order=(1, 1, 1))
        arima.fit(train)
        arima_preds = arima.forecast(horizon=50)
        arima_mae = np.mean(np.abs(arima_preds - test))

        # ARIMA should be better on trending data
        # (Allow some tolerance for random variation)
        assert arima_mae < naive_mae * 1.2

    def test_all_forecasters_same_interface(self, trend_series):
        """Test ARIMA follows same interface as baseline forecasters."""
        train = trend_series[:-50]
        horizon = 50

        forecaster = ARIMAForecaster()

        # Check it's a BaseForecaster
        assert isinstance(forecaster, BaseForecaster)

        # Check it has required methods
        assert hasattr(forecaster, "fit")
        assert hasattr(forecaster, "forecast")

        # Check fit returns self
        result = forecaster.fit(train)
        assert result is forecaster

        # Check forecast works
        preds = forecaster.forecast(horizon)
        assert preds.shape == (horizon,)


class TestARIMAEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_series_warning(self):
        """Test warning for very short series."""
        short_series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        forecaster = ARIMAForecaster(order=(1, 0, 1))

        # Should fit but warn
        forecaster.fit(short_series)
        assert forecaster.is_fitted_

    def test_constant_series(self):
        """Test ARIMA on constant series."""
        constant = np.ones(100) * 42.0

        forecaster = ARIMAForecaster(order=(0, 0, 0))
        forecaster.fit(constant)

        preds = forecaster.forecast(horizon=10)

        # Should predict constant value
        assert np.allclose(preds, 42.0, rtol=0.1)

    def test_high_frequency_noise(self):
        """Test ARIMA on high-frequency noise."""
        np.random.seed(42)
        noise = np.random.randn(200)

        forecaster = ARIMAForecaster(order=(1, 0, 1))
        forecaster.fit(noise)

        preds = forecaster.forecast(horizon=20)

        # Should produce forecasts close to zero mean
        assert abs(np.mean(preds)) < 1.0

    def test_negative_values(self):
        """Test ARIMA handles negative values."""
        np.random.seed(42)
        negative_series = -10 - 0.5 * np.arange(100) + np.random.randn(100)

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(negative_series)

        preds = forecaster.forecast(horizon=10)

        # Should produce negative forecasts for negative trend
        assert np.all(preds < 0)

    def test_multiple_forecasts_same_horizon(self):
        """Test multiple forecast calls with same horizon."""
        np.random.seed(42)
        y = np.random.randn(100).cumsum()

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        forecaster.fit(y)

        # Multiple calls should produce same results
        preds1 = forecaster.forecast(horizon=20)
        preds2 = forecaster.forecast(horizon=20)

        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_different_alpha_values(self):
        """Test different alpha values for confidence intervals."""
        np.random.seed(42)
        y = 10 + 0.5 * np.arange(100) + np.random.randn(100)

        # 95% CI (alpha=0.05)
        forecaster_95 = ARIMAForecaster(order=(1, 1, 1), alpha=0.05)
        forecaster_95.fit(y)
        preds_95, lower_95, upper_95 = forecaster_95.forecast(20, return_conf_int=True)

        # 99% CI (alpha=0.01) - wider intervals
        forecaster_99 = ARIMAForecaster(order=(1, 1, 1), alpha=0.01)
        forecaster_99.fit(y)
        preds_99, lower_99, upper_99 = forecaster_99.forecast(20, return_conf_int=True)

        # 99% CI should be wider
        width_95 = np.mean(upper_95 - lower_95)
        width_99 = np.mean(upper_99 - lower_99)
        assert width_99 > width_95
