"""
Tests for baseline forecasting models.

Validates the three baseline forecasters:
- NaiveForecaster: Last value propagation
- SeasonalNaiveForecaster: Seasonal pattern repetition
- MovingAverageForecaster: Smoothed constant forecast

Tests cover:
- Model initialization and fit
- Forecast generation with correct shape
- Edge cases and error handling
- Seasonal cycle repetition correctness
"""

import numpy as np
import pytest

from hellocloud.modeling.forecasting.baselines import (
    BaseForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


class TestNaiveForecaster:
    """Test suite for NaiveForecaster."""

    @pytest.fixture
    def simple_series(self):
        """Create simple test series."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_initialization(self):
        """Test NaiveForecaster initializes correctly."""
        forecaster = NaiveForecaster()
        assert not forecaster.is_fitted_
        assert forecaster.last_value_ is None

    def test_fit_stores_last_value(self, simple_series):
        """Test fit stores the last observed value."""
        forecaster = NaiveForecaster()
        forecaster.fit(simple_series)

        assert forecaster.is_fitted_
        assert forecaster.last_value_ == 5.0

    def test_forecast_repeats_last_value(self, simple_series):
        """Test forecast repeats last value for entire horizon."""
        forecaster = NaiveForecaster()
        forecaster.fit(simple_series)

        horizon = 10
        preds = forecaster.forecast(horizon)

        assert len(preds) == horizon
        assert np.all(preds == 5.0)

    def test_forecast_correct_shape(self, simple_series):
        """Test forecast returns correct shape."""
        forecaster = NaiveForecaster()
        forecaster.fit(simple_series)

        for horizon in [1, 5, 50, 100]:
            preds = forecaster.forecast(horizon)
            assert preds.shape == (horizon,)

    def test_fit_empty_array_raises(self):
        """Test fit raises ValueError for empty array."""
        forecaster = NaiveForecaster()

        with pytest.raises(ValueError, match="cannot be empty"):
            forecaster.fit(np.array([]))

    def test_forecast_before_fit_raises(self):
        """Test forecast raises RuntimeError if called before fit."""
        forecaster = NaiveForecaster()

        with pytest.raises(RuntimeError, match="Must call fit"):
            forecaster.forecast(10)

    def test_forecast_invalid_horizon_raises(self, simple_series):
        """Test forecast raises ValueError for invalid horizon."""
        forecaster = NaiveForecaster()
        forecaster.fit(simple_series)

        with pytest.raises(ValueError, match="must be > 0"):
            forecaster.forecast(0)

        with pytest.raises(ValueError, match="must be > 0"):
            forecaster.forecast(-5)

    def test_method_chaining(self, simple_series):
        """Test fit returns self for method chaining."""
        forecaster = NaiveForecaster()
        result = forecaster.fit(simple_series)

        assert result is forecaster

    def test_negative_values(self):
        """Test forecaster works with negative values."""
        series = np.array([-5.0, -3.0, -1.0, -2.0])
        forecaster = NaiveForecaster()
        forecaster.fit(series)

        preds = forecaster.forecast(3)
        assert np.all(preds == -2.0)


class TestSeasonalNaiveForecaster:
    """Test suite for SeasonalNaiveForecaster."""

    @pytest.fixture
    def seasonal_series(self):
        """Create series with clear seasonal pattern (period=7)."""
        # Two full weeks
        return np.array(
            [100, 120, 130, 125, 135, 90, 80, 105, 125, 135, 130, 140, 95, 85]  # Week 1
        )  # Week 2

    def test_initialization_valid_period(self):
        """Test SeasonalNaiveForecaster initializes with valid period."""
        forecaster = SeasonalNaiveForecaster(period=7)
        assert forecaster.period == 7
        assert not forecaster.is_fitted_

    def test_initialization_invalid_period_raises(self):
        """Test initialization raises ValueError for invalid period."""
        with pytest.raises(ValueError, match="must be > 0"):
            SeasonalNaiveForecaster(period=0)

        with pytest.raises(ValueError, match="must be > 0"):
            SeasonalNaiveForecaster(period=-5)

    def test_fit_stores_last_cycle(self, seasonal_series):
        """Test fit stores the last full seasonal cycle."""
        forecaster = SeasonalNaiveForecaster(period=7)
        forecaster.fit(seasonal_series)

        assert forecaster.is_fitted_
        expected_cycle = np.array([105, 125, 135, 130, 140, 95, 85])
        np.testing.assert_array_equal(forecaster.seasonal_values_, expected_cycle)

    def test_forecast_repeats_seasonal_pattern(self, seasonal_series):
        """Test forecast correctly repeats seasonal pattern."""
        forecaster = SeasonalNaiveForecaster(period=7)
        forecaster.fit(seasonal_series)

        # Forecast one full cycle
        preds = forecaster.forecast(horizon=7)
        expected = np.array([105, 125, 135, 130, 140, 95, 85])
        np.testing.assert_array_equal(preds, expected)

    def test_forecast_multiple_cycles(self, seasonal_series):
        """Test forecast works for multiple seasonal cycles."""
        forecaster = SeasonalNaiveForecaster(period=7)
        forecaster.fit(seasonal_series)

        # Forecast 2.5 cycles (17 steps)
        preds = forecaster.forecast(horizon=17)

        # Should repeat pattern 2 full times + 3 values
        expected_cycle = np.array([105, 125, 135, 130, 140, 95, 85])
        expected = np.concatenate([expected_cycle, expected_cycle, expected_cycle[:3]])

        np.testing.assert_array_equal(preds, expected)

    def test_forecast_partial_cycle(self, seasonal_series):
        """Test forecast works for partial seasonal cycle."""
        forecaster = SeasonalNaiveForecaster(period=7)
        forecaster.fit(seasonal_series)

        # Forecast 3 steps (less than one cycle)
        preds = forecaster.forecast(horizon=3)
        expected = np.array([105, 125, 135])
        np.testing.assert_array_equal(preds, expected)

    def test_fit_insufficient_data_raises(self):
        """Test fit raises ValueError if data shorter than period."""
        forecaster = SeasonalNaiveForecaster(period=10)
        short_series = np.array([1, 2, 3, 4, 5])  # Only 5 samples

        with pytest.raises(ValueError, match="at least 10 samples"):
            forecaster.fit(short_series)

    def test_forecast_before_fit_raises(self):
        """Test forecast raises RuntimeError if called before fit."""
        forecaster = SeasonalNaiveForecaster(period=7)

        with pytest.raises(RuntimeError, match="Must call fit"):
            forecaster.forecast(10)

    def test_forecast_invalid_horizon_raises(self, seasonal_series):
        """Test forecast raises ValueError for invalid horizon."""
        forecaster = SeasonalNaiveForecaster(period=7)
        forecaster.fit(seasonal_series)

        with pytest.raises(ValueError, match="must be > 0"):
            forecaster.forecast(0)

    def test_iops_periodicities(self):
        """Test with IOPS dataset periodicities (250 and 1250)."""
        # Simulate IOPS data with fast period (250)
        np.random.seed(42)
        n_samples = 1500
        t = np.arange(n_samples)
        # Create synthetic IOPS pattern
        iops = 1000 + 500 * np.sin(2 * np.pi * t / 250) + 100 * np.random.randn(n_samples)

        forecaster = SeasonalNaiveForecaster(period=250)
        forecaster.fit(iops)

        # Forecast one cycle
        preds = forecaster.forecast(horizon=250)

        assert len(preds) == 250
        # Predictions should match last 250 values
        np.testing.assert_array_equal(preds, iops[-250:])

    def test_method_chaining(self, seasonal_series):
        """Test fit returns self for method chaining."""
        forecaster = SeasonalNaiveForecaster(period=7)
        result = forecaster.fit(seasonal_series)

        assert result is forecaster


class TestMovingAverageForecaster:
    """Test suite for MovingAverageForecaster."""

    @pytest.fixture
    def noisy_series(self):
        """Create series with noise around a constant."""
        np.random.seed(42)
        return 20.0 + 0.5 * np.random.randn(100)

    def test_initialization_valid_window(self):
        """Test MovingAverageForecaster initializes with valid window."""
        forecaster = MovingAverageForecaster(window=10)
        assert forecaster.window == 10
        assert not forecaster.is_fitted_

    def test_initialization_invalid_window_raises(self):
        """Test initialization raises ValueError for invalid window."""
        with pytest.raises(ValueError, match="must be > 0"):
            MovingAverageForecaster(window=0)

        with pytest.raises(ValueError, match="must be > 0"):
            MovingAverageForecaster(window=-5)

    def test_fit_computes_mean(self):
        """Test fit computes mean of last window values."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        forecaster = MovingAverageForecaster(window=3)
        forecaster.fit(series)

        assert forecaster.is_fitted_
        # Mean of [3, 4, 5]
        expected_mean = 4.0
        assert forecaster.mean_value_ == pytest.approx(expected_mean)

    def test_forecast_repeats_mean(self, noisy_series):
        """Test forecast repeats the moving average."""
        forecaster = MovingAverageForecaster(window=10)
        forecaster.fit(noisy_series)

        horizon = 20
        preds = forecaster.forecast(horizon)

        assert len(preds) == horizon
        # All predictions should equal the mean
        assert np.all(preds == preds[0])
        assert preds[0] == pytest.approx(forecaster.mean_value_)

    def test_forecast_correct_shape(self, noisy_series):
        """Test forecast returns correct shape."""
        forecaster = MovingAverageForecaster(window=10)
        forecaster.fit(noisy_series)

        for horizon in [1, 5, 50, 100]:
            preds = forecaster.forecast(horizon)
            assert preds.shape == (horizon,)

    def test_fit_insufficient_data_raises(self):
        """Test fit raises ValueError if data shorter than window."""
        forecaster = MovingAverageForecaster(window=10)
        short_series = np.array([1, 2, 3, 4, 5])  # Only 5 samples

        with pytest.raises(ValueError, match="at least 10 samples"):
            forecaster.fit(short_series)

    def test_forecast_before_fit_raises(self):
        """Test forecast raises RuntimeError if called before fit."""
        forecaster = MovingAverageForecaster(window=5)

        with pytest.raises(RuntimeError, match="Must call fit"):
            forecaster.forecast(10)

    def test_forecast_invalid_horizon_raises(self, noisy_series):
        """Test forecast raises ValueError for invalid horizon."""
        forecaster = MovingAverageForecaster(window=5)
        forecaster.fit(noisy_series)

        with pytest.raises(ValueError, match="must be > 0"):
            forecaster.forecast(0)

    def test_window_size_effect(self):
        """Test that different window sizes produce different means."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])

        # Small window (last 2 values)
        small_window = MovingAverageForecaster(window=2)
        small_window.fit(series)
        # Mean of [5, 10] = 7.5
        assert small_window.mean_value_ == pytest.approx(7.5)

        # Large window (last 5 values)
        large_window = MovingAverageForecaster(window=5)
        large_window.fit(series)
        # Mean of [2, 3, 4, 5, 10] = 4.8
        assert large_window.mean_value_ == pytest.approx(4.8)

    def test_method_chaining(self, noisy_series):
        """Test fit returns self for method chaining."""
        forecaster = MovingAverageForecaster(window=10)
        result = forecaster.fit(noisy_series)

        assert result is forecaster


class TestBaseForecasterInterface:
    """Test suite for BaseForecaster abstract interface."""

    def test_base_forecaster_is_abstract(self):
        """Test BaseForecaster cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseForecaster()

    def test_all_forecasters_implement_interface(self):
        """Test all concrete forecasters implement BaseForecaster interface."""
        forecasters = [
            NaiveForecaster(),
            SeasonalNaiveForecaster(period=7),
            MovingAverageForecaster(window=5),
        ]

        for forecaster in forecasters:
            assert isinstance(forecaster, BaseForecaster)
            assert hasattr(forecaster, "fit")
            assert hasattr(forecaster, "forecast")


class TestIntegration:
    """Integration tests comparing forecasters on realistic scenarios."""

    @pytest.fixture
    def random_walk(self):
        """Generate random walk series (best for Naive)."""
        np.random.seed(42)
        steps = np.random.randn(500)
        return np.cumsum(steps) + 100

    @pytest.fixture
    def seasonal_data(self):
        """Generate seasonal data (best for Seasonal Naive)."""
        np.random.seed(42)
        t = np.arange(1000)
        # Strong seasonality with period=50
        return 100 + 20 * np.sin(2 * np.pi * t / 50) + 2 * np.random.randn(1000)

    @pytest.fixture
    def stable_noisy(self):
        """Generate stable series with noise (best for MA)."""
        np.random.seed(42)
        return 50 + 5 * np.random.randn(500)

    def test_naive_on_random_walk(self, random_walk):
        """Test Naive forecaster on random walk."""
        train = random_walk[:-50]

        forecaster = NaiveForecaster()
        forecaster.fit(train)
        preds = forecaster.forecast(horizon=50)

        # All predictions should equal last training value
        assert np.all(preds == train[-1])

    def test_seasonal_on_periodic_data(self, seasonal_data):
        """Test Seasonal Naive forecaster on periodic data."""
        train = seasonal_data[:-50]

        forecaster = SeasonalNaiveForecaster(period=50)
        forecaster.fit(train)
        preds = forecaster.forecast(horizon=50)

        # Predictions should match last cycle
        expected = train[-50:]
        np.testing.assert_array_equal(preds, expected)

    def test_moving_average_on_stable_series(self, stable_noisy):
        """Test Moving Average forecaster on stable noisy series."""
        train = stable_noisy[:-50]

        forecaster = MovingAverageForecaster(window=50)
        forecaster.fit(train)
        preds = forecaster.forecast(horizon=50)

        # All predictions should be close to true mean (50)
        assert np.mean(preds) == pytest.approx(np.mean(train[-50:]), abs=1.0)

    def test_all_forecasters_same_horizon(self, random_walk):
        """Test all forecasters produce correct shape for same data."""
        train = random_walk[:-50]
        horizon = 50

        forecasters = [
            NaiveForecaster(),
            SeasonalNaiveForecaster(period=10),
            MovingAverageForecaster(window=10),
        ]

        for forecaster in forecasters:
            forecaster.fit(train)
            preds = forecaster.forecast(horizon)
            assert preds.shape == (horizon,)
