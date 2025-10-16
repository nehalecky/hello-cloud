"""
Tests for time series forecasting evaluation metrics.

Tests cover:
- Point forecast metrics (MAE, RMSE, MAPE, SMAPE, MASE)
- Probabilistic forecast metrics (quantile loss, coverage, sharpness)
- Edge cases (empty arrays, division by zero, mismatched shapes)
- Known inputs with expected outputs
"""

import numpy as np
import pytest

from hellocloud.modeling.forecasting.evaluation import (
    compute_all_metrics,
    coverage,
    interval_sharpness,
    mae,
    mape,
    mase,
    quantile_loss,
    rmse,
    smape,
)


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mae(y_true, y_pred) == 0.0

    def test_constant_error(self):
        """MAE should equal the constant error."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])  # All off by +10
        assert mae(y_true, y_pred) == 10.0

    def test_mixed_errors(self):
        """MAE should average absolute errors."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])  # Errors: -10, +10, -10
        assert mae(y_true, y_pred) == 10.0

    def test_negative_values(self):
        """MAE should handle negative values correctly."""
        y_true = np.array([-100.0, -200.0, -300.0])
        y_pred = np.array([-110.0, -190.0, -310.0])
        assert mae(y_true, y_pred) == 10.0

    def test_returns_float(self):
        """MAE should return Python float, not numpy scalar."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = mae(y_true, y_pred)
        assert isinstance(result, float)

    def test_empty_arrays(self):
        """MAE should raise ValueError for empty arrays."""
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            mae(np.array([]), np.array([]))

    def test_mismatched_shapes(self):
        """MAE should raise ValueError for mismatched shapes."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="must have same shape"):
            mae(y_true, y_pred)


class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_predictions(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_constant_error(self):
        """RMSE should equal the constant error."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 210.0, 310.0])  # All off by +10
        assert rmse(y_true, y_pred) == 10.0

    def test_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 400.0])  # One large error (+100)

        mae_val = mae(y_true, y_pred)  # 33.33
        rmse_val = rmse(y_true, y_pred)  # 57.74

        assert rmse_val > mae_val  # RMSE penalizes large errors
        assert np.isclose(rmse_val, 57.735, atol=0.01)

    def test_returns_float(self):
        """RMSE should return Python float, not numpy scalar."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = rmse(y_true, y_pred)
        assert isinstance(result, float)


class TestMAPE:
    """Tests for Mean Absolute Percentage Error."""

    def test_perfect_predictions(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        assert mape(y_true, y_pred) == 0.0

    def test_constant_percentage_error(self):
        """MAPE should compute percentage correctly."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 220.0, 330.0])  # All +10%
        assert np.isclose(mape(y_true, y_pred), 10.0, atol=0.01)

    def test_mixed_percentage_errors(self):
        """MAPE should average percentage errors."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])  # +10%, -5%, +3.33%
        expected = (10.0 + 5.0 + (10.0 / 300.0) * 100) / 3
        assert np.isclose(mape(y_true, y_pred), expected, atol=0.01)

    def test_handles_near_zero_actuals(self):
        """MAPE should skip near-zero actuals to avoid division by zero."""
        y_true = np.array([1e-15, 100.0, 200.0])  # First value near zero
        y_pred = np.array([1.0, 110.0, 220.0])
        # Should only compute on last two values
        result = mape(y_true, y_pred)
        assert np.isfinite(result)
        assert result > 0

    def test_all_near_zero_actuals(self):
        """MAPE should return nan if all actuals are near zero."""
        y_true = np.array([1e-15, 1e-15, 1e-15])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mape(y_true, y_pred)
        assert np.isnan(result)

    def test_returns_percentage(self):
        """MAPE should return percentage (0-100+)."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])  # +10%
        result = mape(y_true, y_pred)
        assert 0 <= result <= 100  # Should be ~10%


class TestSMAPE:
    """Tests for Symmetric Mean Absolute Percentage Error."""

    def test_perfect_predictions(self):
        """SMAPE should be 0 for perfect predictions."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        assert smape(y_true, y_pred) == 0.0

    def test_symmetric_property(self):
        """SMAPE should be symmetric: SMAPE(a, b) ≈ SMAPE(b, a)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        smape_1 = smape(y_true, y_pred)
        smape_2 = smape(y_pred, y_true)

        assert np.isclose(smape_1, smape_2, atol=0.01)

    def test_bounded_range(self):
        """SMAPE should be bounded [0, 200]."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([0.0, 0.0, 0.0])  # Extreme predictions

        result = smape(y_true, y_pred)
        assert 0 <= result <= 200

    def test_handles_near_zero_values(self):
        """SMAPE should skip cases where both actual and predicted are near zero."""
        y_true = np.array([1e-15, 100.0, 200.0])
        y_pred = np.array([1e-15, 110.0, 220.0])

        result = smape(y_true, y_pred)
        assert np.isfinite(result)
        assert result > 0

    def test_more_stable_than_mape(self):
        """SMAPE should be more stable than MAPE for values crossing zero."""
        y_true = np.array([0.1, 100.0, 200.0])  # Small value
        y_pred = np.array([1.0, 110.0, 220.0])

        mape_val = mape(y_true, y_pred)
        smape_val = smape(y_true, y_pred)

        # MAPE should be inflated by the small denominator
        # SMAPE should be more reasonable
        assert smape_val < mape_val


class TestMASE:
    """Tests for Mean Absolute Scaled Error."""

    def test_perfect_predictions(self):
        """MASE should be 0 for perfect predictions."""
        y_train = np.random.randn(1000)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert mase(y_true, y_pred, y_train) == 0.0

    def test_beats_naive_baseline(self):
        """MASE < 1 means forecast beats naive baseline."""
        # Create training data with some structure
        np.random.seed(42)  # Ensure reproducibility
        y_train = np.sin(np.linspace(0, 10, 1000))

        # Create test data with good predictions
        y_true = np.sin(np.linspace(10, 11, 100))
        y_pred = y_true + np.random.randn(100) * 0.001  # Very small noise

        result = mase(y_true, y_pred, y_train)
        assert result < 1.0  # Beats naive baseline

    def test_worse_than_naive_baseline(self):
        """MASE > 1 means forecast worse than naive baseline."""
        # Create training data
        y_train = np.sin(np.linspace(0, 10, 1000))

        # Create test data with bad predictions
        y_true = np.sin(np.linspace(10, 11, 100))
        y_pred = y_true + np.random.randn(100) * 10.0  # Large noise

        result = mase(y_true, y_pred, y_train)
        assert result > 1.0  # Worse than naive baseline

    def test_seasonal_period(self):
        """MASE should support seasonal periods."""
        # Create training data with period=7 seasonality and some variation
        base_pattern = np.array([1, 2, 3, 4, 5, 6, 7])
        # Add small random variation to avoid perfect persistence
        y_train = np.tile(base_pattern, 50) + np.random.randn(350) * 0.5

        # Test data continues pattern with small errors
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Use seasonal period=7
        result = mase(y_true, y_pred, y_train, period=7)
        assert np.isfinite(result)  # Should return valid number
        assert result < 5.0  # Should be reasonable (not necessarily < 1.0)

    def test_requires_sufficient_training_data(self):
        """MASE should raise ValueError if y_train too short."""
        y_train = np.array([1.0, 2.0])  # Only 2 samples
        y_true = np.array([3.0, 4.0])
        y_pred = np.array([3.1, 4.1])

        with pytest.raises(ValueError, match="y_train must have >"):
            mase(y_true, y_pred, y_train, period=3)

    def test_handles_perfect_persistence(self):
        """MASE should return nan if training data is perfectly persistent."""
        # Training data with zero variance (perfect persistence)
        y_train = np.full(1000, 100.0)
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([100.0, 100.0, 100.0])

        result = mase(y_true, y_pred, y_train)
        assert np.isnan(result)


class TestQuantileLoss:
    """Tests for Quantile Loss (Pinball Loss)."""

    def test_perfect_predictions(self):
        """Quantile loss should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert quantile_loss(y_true, q_pred, tau=0.5) == 0.0

    def test_median_is_symmetric(self):
        """For tau=0.5 (median), loss should be symmetric."""
        y_true = np.array([100.0])
        q_pred_under = np.array([90.0])  # Under-prediction
        q_pred_over = np.array([110.0])  # Over-prediction

        loss_under = quantile_loss(y_true, q_pred_under, tau=0.5)
        loss_over = quantile_loss(y_true, q_pred_over, tau=0.5)

        assert np.isclose(loss_under, loss_over)

    def test_low_quantile_penalizes_over_prediction(self):
        """For tau=0.1 (10th percentile), over-predictions should have high loss."""
        y_true = np.array([100.0])
        q_pred_under = np.array([90.0])  # Under-prediction (OK for 10th)
        q_pred_over = np.array([110.0])  # Over-prediction (bad for 10th)

        loss_under = quantile_loss(y_true, q_pred_under, tau=0.1)
        loss_over = quantile_loss(y_true, q_pred_over, tau=0.1)

        # For low quantile (0.1), over-prediction has higher loss
        # We want the 10th percentile to be conservative, so predicting too high is bad
        assert loss_over > loss_under  # Over-prediction worse for low quantiles

    def test_high_quantile_penalizes_under_prediction(self):
        """For tau=0.9 (90th percentile), under-predictions should have high loss."""
        y_true = np.array([100.0])
        q_pred_under = np.array([90.0])  # Under-prediction (bad for 90th)
        q_pred_over = np.array([110.0])  # Over-prediction (OK for 90th)

        loss_under = quantile_loss(y_true, q_pred_under, tau=0.9)
        loss_over = quantile_loss(y_true, q_pred_over, tau=0.9)

        assert loss_under > loss_over  # Under-prediction worse for high quantiles

    def test_tau_validation(self):
        """Quantile loss should validate tau in [0, 1]."""
        y_true = np.array([1.0, 2.0, 3.0])
        q_pred = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="tau must be in"):
            quantile_loss(y_true, q_pred, tau=-0.1)

        with pytest.raises(ValueError, match="tau must be in"):
            quantile_loss(y_true, q_pred, tau=1.5)


class TestCoverage:
    """Tests for Prediction Interval Coverage."""

    def test_perfect_coverage(self):
        """Coverage should be 100% when all actuals within bounds."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        lower = np.array([90.0, 190.0, 290.0, 390.0, 490.0])
        upper = np.array([110.0, 210.0, 310.0, 410.0, 510.0])

        assert coverage(y_true, lower, upper) == 100.0

    def test_zero_coverage(self):
        """Coverage should be 0% when no actuals within bounds."""
        y_true = np.array([100.0, 200.0, 300.0])
        lower = np.array([110.0, 210.0, 310.0])  # All lower bounds above actuals
        upper = np.array([120.0, 220.0, 320.0])

        assert coverage(y_true, lower, upper) == 0.0

    def test_partial_coverage(self):
        """Coverage should compute percentage correctly."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        lower = np.array([90.0, 190.0, 290.0, 410.0, 490.0])  # 4th out of bounds
        upper = np.array([110.0, 210.0, 310.0, 420.0, 510.0])

        # 4 out of 5 within bounds = 80%
        assert coverage(y_true, lower, upper) == 80.0

    def test_on_boundary(self):
        """Coverage should include values on boundaries."""
        y_true = np.array([100.0, 200.0, 300.0])
        lower = np.array([100.0, 200.0, 300.0])  # Exact lower bounds
        upper = np.array([100.0, 200.0, 300.0])  # Exact upper bounds

        assert coverage(y_true, lower, upper) == 100.0

    def test_empty_arrays(self):
        """Coverage should raise ValueError for empty arrays."""
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            coverage(np.array([]), np.array([]), np.array([]))

    def test_mismatched_shapes(self):
        """Coverage should raise ValueError for mismatched shapes."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Array shapes must match"):
            coverage(y_true, lower, upper)


class TestIntervalSharpness:
    """Tests for Interval Sharpness."""

    def test_constant_width(self):
        """Sharpness should equal constant width."""
        lower = np.array([90.0, 190.0, 290.0])
        upper = np.array([110.0, 210.0, 310.0])  # All widths = 20

        assert interval_sharpness(lower, upper) == 20.0

    def test_varying_widths(self):
        """Sharpness should average varying widths."""
        lower = np.array([90.0, 180.0, 270.0])
        upper = np.array([110.0, 220.0, 330.0])  # Widths: 20, 40, 60

        expected = (20.0 + 40.0 + 60.0) / 3
        assert interval_sharpness(lower, upper) == expected

    def test_zero_width(self):
        """Sharpness should be 0 for point predictions."""
        lower = np.array([100.0, 200.0, 300.0])
        upper = np.array([100.0, 200.0, 300.0])  # Same as lower

        assert interval_sharpness(lower, upper) == 0.0

    def test_empty_arrays(self):
        """Sharpness should raise ValueError for empty arrays."""
        with pytest.raises(ValueError, match="lower cannot be empty"):
            interval_sharpness(np.array([]), np.array([]))

    def test_mismatched_shapes(self):
        """Sharpness should raise ValueError for mismatched shapes."""
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="Array shapes must match"):
            interval_sharpness(lower, upper)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_includes_all_point_metrics(self):
        """Should compute MAE, RMSE, MAPE, SMAPE."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "smape" in metrics

    def test_includes_mase_when_y_train_provided(self):
        """Should include MASE when y_train provided."""
        y_train = np.random.randn(1000)
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = compute_all_metrics(y_true, y_pred, y_train)

        assert "mase" in metrics

    def test_excludes_mase_when_no_y_train(self):
        """Should exclude MASE when y_train not provided."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert "mase" not in metrics

    def test_all_values_are_floats(self):
        """All metric values should be Python floats."""
        y_train = np.random.randn(1000)
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = compute_all_metrics(y_true, y_pred, y_train)

        for value in metrics.values():
            assert isinstance(value, float)

    def test_seasonal_period_parameter(self):
        """Should pass period parameter to MASE."""
        # Create training data with some variation to avoid perfect persistence
        base_pattern = np.array([1, 2, 3, 4, 5, 6, 7])
        y_train = np.tile(base_pattern, 50) + np.random.randn(350) * 0.5

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metrics = compute_all_metrics(y_true, y_pred, y_train, period=7)

        assert "mase" in metrics
        assert np.isfinite(metrics["mase"])
