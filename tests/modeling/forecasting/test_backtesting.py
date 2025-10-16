"""
Tests for walk-forward backtesting framework.

Tests cover:
- WalkForwardBacktest initialization and validation
- run_backtest with expanding/sliding windows
- Retraining control (retrain=True/False)
- Edge cases (insufficient data, forecaster failures)
- BacktestResults dataclass structure
- Integration with different forecasters (Naive, Seasonal, MA, ARIMA)
"""

import numpy as np
import pytest

from hellocloud.modeling.forecasting.backtesting import BacktestResults, WalkForwardBacktest
from hellocloud.modeling.forecasting.baselines import (
    BaseForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


class TestWalkForwardBacktestInit:
    """Tests for WalkForwardBacktest initialization."""

    def test_valid_initialization(self):
        """Should initialize with valid parameters."""
        backtest = WalkForwardBacktest(
            initial_train_size=100, horizon=10, step_size=5, retrain=True, expanding_window=True
        )

        assert backtest.initial_train_size == 100
        assert backtest.horizon == 10
        assert backtest.step_size == 5
        assert backtest.retrain is True
        assert backtest.expanding_window is True

    def test_default_parameters(self):
        """Should use default values for optional parameters."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10)

        assert backtest.step_size == 1
        assert backtest.retrain is True
        assert backtest.expanding_window is True

    def test_invalid_initial_train_size(self):
        """Should raise ValueError for initial_train_size <= 0."""
        with pytest.raises(ValueError, match="initial_train_size must be > 0"):
            WalkForwardBacktest(initial_train_size=0, horizon=10)

        with pytest.raises(ValueError, match="initial_train_size must be > 0"):
            WalkForwardBacktest(initial_train_size=-100, horizon=10)

    def test_invalid_horizon(self):
        """Should raise ValueError for horizon <= 0."""
        with pytest.raises(ValueError, match="horizon must be > 0"):
            WalkForwardBacktest(initial_train_size=100, horizon=0)

        with pytest.raises(ValueError, match="horizon must be > 0"):
            WalkForwardBacktest(initial_train_size=100, horizon=-10)

    def test_invalid_step_size(self):
        """Should raise ValueError for step_size <= 0."""
        with pytest.raises(ValueError, match="step_size must be > 0"):
            WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=0)

        with pytest.raises(ValueError, match="step_size must be > 0"):
            WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=-5)


class TestWalkForwardBacktestRun:
    """Tests for run_backtest method."""

    def test_insufficient_data(self):
        """Should raise ValueError if data too short for even one fold."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10)
        y = np.random.randn(50)  # Too short: need 100 + 10 = 110

        with pytest.raises(ValueError, match="Time series too short"):
            backtest.run_backtest(NaiveForecaster(), y)

    def test_naive_forecaster_single_fold(self):
        """Should run successfully with naive forecaster for one fold."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=50)
        y = np.random.randn(110)  # Exactly enough for one fold

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Validate results structure
        assert isinstance(results, BacktestResults)
        assert results.n_folds == 1
        assert len(results.forecasts) == 10  # horizon
        assert len(results.actuals) == 10
        assert len(results.fold_metrics) == 1
        assert len(results.train_sizes) == 1
        assert len(results.train_times) == 1

        # Validate metrics exist
        assert "mae" in results.overall_metrics
        assert "rmse" in results.overall_metrics
        assert "mape" in results.overall_metrics
        assert "smape" in results.overall_metrics
        assert "mase" in results.overall_metrics

        # Validate fold metrics
        assert "mae" in results.fold_metrics[0]
        assert "train_size" in results.fold_metrics[0]
        assert "train_time" in results.fold_metrics[0]

        # Validate naive forecaster behavior (repeats last value)
        expected_forecast = np.full(10, y[99])  # Last value of training data
        np.testing.assert_array_equal(results.forecasts, expected_forecast)

    def test_expanding_window_multiple_folds(self):
        """Should use expanding window (training size grows)."""
        backtest = WalkForwardBacktest(
            initial_train_size=100,
            horizon=10,
            step_size=10,  # Non-overlapping folds
            expanding_window=True,
        )
        y = np.random.randn(150)  # Enough for 5 folds

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        assert results.n_folds == 5
        # Training sizes should grow: 100, 110, 120, 130, 140
        expected_sizes = [100, 110, 120, 130, 140]
        np.testing.assert_array_equal(results.train_sizes, expected_sizes)

    def test_sliding_window_multiple_folds(self):
        """Should use sliding window (training size fixed)."""
        backtest = WalkForwardBacktest(
            initial_train_size=100,
            horizon=10,
            step_size=10,  # Non-overlapping folds
            expanding_window=False,
        )
        y = np.random.randn(150)  # Enough for 5 folds

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        assert results.n_folds == 5
        # Training sizes should be constant: 100, 100, 100, 100, 100
        expected_sizes = [100, 100, 100, 100, 100]
        np.testing.assert_array_equal(results.train_sizes, expected_sizes)

    def test_retrain_true(self):
        """Should retrain model at each fold when retrain=True."""
        backtest = WalkForwardBacktest(
            initial_train_size=100, horizon=10, step_size=10, retrain=True
        )
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Each fold should have its own training time (may be very small but >= 0)
        assert len(results.train_times) == results.n_folds
        assert all(t >= 0 for t in results.train_times)

    def test_retrain_false(self):
        """Should use initial model for all folds when retrain=False."""
        backtest = WalkForwardBacktest(
            initial_train_size=100, horizon=10, step_size=10, retrain=False
        )
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Only first fold trains, others reuse that time
        assert len(results.train_times) == results.n_folds
        # All folds after first should have same time as first
        assert all(t == results.train_times[0] for t in results.train_times)

    def test_overlapping_folds(self):
        """Should create overlapping folds with step_size < horizon."""
        backtest = WalkForwardBacktest(
            initial_train_size=100,
            horizon=10,
            step_size=5,  # Overlapping
        )
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # With step_size=5, we get more folds than with step_size=10
        # Available for testing: 150 - 100 = 50
        # Number of folds: 50 // 5 = 10
        assert results.n_folds == 9  # (150 - 100 - 10) // 5 + 1

    def test_seasonal_naive_forecaster(self):
        """Should work with seasonal naive forecaster."""
        period = 20
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)

        # Generate seasonal data
        t = np.arange(150)
        y = np.sin(2 * np.pi * t / period) + np.random.randn(150) * 0.1

        forecaster = SeasonalNaiveForecaster(period=period)
        results = backtest.run_backtest(forecaster, y)

        assert results.n_folds > 0
        assert len(results.forecasts) > 0
        # Seasonal naive should outperform random walk on seasonal data
        assert "mae" in results.overall_metrics

    def test_moving_average_forecaster(self):
        """Should work with moving average forecaster."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = MovingAverageForecaster(window=20)
        results = backtest.run_backtest(forecaster, y)

        assert results.n_folds > 0
        assert len(results.forecasts) > 0
        assert "mae" in results.overall_metrics

    def test_fit_params_passed_to_forecaster(self):
        """Should pass fit_params to forecaster.fit()."""

        class MockForecaster(BaseForecaster):
            def __init__(self):
                self.fit_params_received = None
                self.is_fitted_ = False

            def fit(self, y_train, **fit_params):
                self.fit_params_received = fit_params
                self.is_fitted_ = True
                return self

            def forecast(self, horizon):
                return np.zeros(horizon)

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = MockForecaster()
        custom_params = {"param1": "value1", "param2": 42}
        results = backtest.run_backtest(forecaster, y, **custom_params)

        # Verify fit_params were passed
        assert forecaster.fit_params_received == custom_params
        assert results.n_folds > 0

    def test_forecaster_fit_failure_skips_fold(self):
        """Should skip fold and log warning if forecaster.fit() fails."""

        class FailingForecaster(BaseForecaster):
            def __init__(self):
                self.fit_call_count = 0

            def fit(self, y_train):
                self.fit_call_count += 1
                if self.fit_call_count == 1:
                    # First fold succeeds
                    return self
                else:
                    # Subsequent folds fail
                    raise ValueError("Intentional fit failure")

            def forecast(self, horizon):
                return np.zeros(horizon)

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = FailingForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Should only have 1 successful fold (first one)
        assert results.n_folds == 1

    def test_forecaster_forecast_failure_skips_fold(self):
        """Should skip fold and log warning if forecaster.forecast() fails."""

        class FailingForecastForecaster(BaseForecaster):
            def __init__(self):
                self.forecast_call_count = 0

            def fit(self, y_train):
                return self

            def forecast(self, horizon):
                self.forecast_call_count += 1
                if self.forecast_call_count == 1:
                    # First fold succeeds
                    return np.zeros(horizon)
                else:
                    # Subsequent folds fail
                    raise ValueError("Intentional forecast failure")

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = FailingForecastForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Should only have 1 successful fold
        assert results.n_folds == 1

    def test_all_folds_fail(self):
        """Should raise RuntimeError if all folds fail."""

        class AlwaysFailingForecaster(BaseForecaster):
            def fit(self, y_train):
                raise ValueError("Always fails")

            def forecast(self, horizon):
                return np.zeros(horizon)

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = AlwaysFailingForecaster()
        with pytest.raises(RuntimeError, match="No successful folds completed"):
            backtest.run_backtest(forecaster, y)


class TestBacktestResults:
    """Tests for BacktestResults dataclass."""

    def test_results_structure(self):
        """Should have all expected attributes."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Check all attributes exist
        assert hasattr(results, "forecasts")
        assert hasattr(results, "actuals")
        assert hasattr(results, "fold_metrics")
        assert hasattr(results, "overall_metrics")
        assert hasattr(results, "train_sizes")
        assert hasattr(results, "train_times")
        assert hasattr(results, "avg_train_time")
        assert hasattr(results, "n_folds")

    def test_forecasts_and_actuals_alignment(self):
        """Forecasts and actuals should have same length."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        assert len(results.forecasts) == len(results.actuals)

    def test_overall_metrics_aggregated(self):
        """Overall metrics should be computed from all forecasts."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Manually compute overall MAE
        overall_mae_manual = np.mean(np.abs(results.actuals - results.forecasts))

        assert np.isclose(results.overall_metrics["mae"], overall_mae_manual)

    def test_avg_train_time_computed(self):
        """Average training time should be mean of fold training times."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        y = np.random.randn(150)

        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        avg_time_manual = np.mean(results.train_times)
        assert np.isclose(results.avg_train_time, avg_time_manual)


class TestBacktestEstimateNFolds:
    """Tests for _estimate_n_folds helper method."""

    def test_estimate_n_folds(self):
        """Should correctly estimate number of folds."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)

        # 150 samples: 150 - 100 = 50 available, 50 // 10 = 5 folds
        assert backtest._estimate_n_folds(150) == 5

        # 200 samples: 200 - 100 = 100 available, 100 // 10 = 10 folds
        assert backtest._estimate_n_folds(200) == 10

    def test_estimate_n_folds_zero(self):
        """Should return 0 if insufficient data."""
        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)

        # 100 samples: 100 - 100 = 0 available
        assert backtest._estimate_n_folds(100) == 0


class TestBacktestIntegration:
    """Integration tests with real forecasting scenarios."""

    def test_deterministic_data_perfect_forecast(self):
        """Should achieve near-zero error on deterministic data."""
        # Create perfectly predictable data (constant)
        y = np.full(150, 42.0)

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)
        forecaster = NaiveForecaster()
        results = backtest.run_backtest(forecaster, y)

        # Naive forecaster should be perfect on constant data
        assert results.overall_metrics["mae"] == 0.0
        assert results.overall_metrics["rmse"] == 0.0

    def test_seasonal_data_seasonal_naive(self):
        """Seasonal naive should perform well on seasonal data."""
        period = 20
        n_samples = 300

        # Create strong seasonal pattern
        t = np.arange(n_samples)
        y = 10 + 5 * np.sin(2 * np.pi * t / period)

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)

        # Seasonal naive should nearly perfect on pure seasonal data
        seasonal_forecaster = SeasonalNaiveForecaster(period=period)
        seasonal_results = backtest.run_backtest(seasonal_forecaster, y)

        # Seasonal naive should have very low error on pure seasonal data
        assert seasonal_results.overall_metrics["mae"] < 0.5  # Very small error

    def test_comparison_naive_vs_seasonal(self):
        """Should be able to compare different forecasters."""
        period = 20
        n_samples = 300

        # Create seasonal data with noise
        t = np.arange(n_samples)
        y = 10 + 5 * np.sin(2 * np.pi * t / period) + np.random.randn(n_samples) * 0.5

        backtest = WalkForwardBacktest(initial_train_size=100, horizon=10, step_size=10)

        # Compare naive vs. seasonal naive
        naive_forecaster = NaiveForecaster()
        naive_results = backtest.run_backtest(naive_forecaster, y)

        seasonal_forecaster = SeasonalNaiveForecaster(period=period)
        seasonal_results = backtest.run_backtest(seasonal_forecaster, y)

        # Seasonal naive should outperform naive on seasonal data
        assert seasonal_results.overall_metrics["mae"] < naive_results.overall_metrics["mae"]
