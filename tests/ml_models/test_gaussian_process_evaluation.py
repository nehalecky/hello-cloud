"""
Tests for Gaussian Process evaluation metrics module.

Validates metric computation including:
- Prediction accuracy metrics (RMSE, MAE, R²)
- Calibration metrics (coverage, sharpness)
- Anomaly detection metrics (precision, recall, F1)
- Prediction interval computation
"""

import pytest
import numpy as np
from hellocloud.ml_models.gaussian_process.evaluation import (
    compute_metrics,
    compute_anomaly_metrics,
    compute_prediction_intervals,
)


class TestComputeMetrics:
    """Test suite for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions (y_true == y_pred)."""
        n = 100
        y_true = np.random.randn(n)
        y_pred = y_true.copy()

        # Create tight intervals around true values
        std = np.ones(n) * 0.01
        lower_95 = y_pred - 2 * std
        upper_95 = y_pred + 2 * std
        lower_99 = y_pred - 3 * std
        upper_99 = y_pred + 3 * std

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_99=lower_99,
            upper_99=upper_99
        )

        # With perfect predictions
        assert metrics['RMSE'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['MAE'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['R²'] == pytest.approx(1.0, abs=1e-6)

        # All points should be within intervals
        assert metrics['Coverage 95%'] == pytest.approx(1.0)
        assert metrics['Coverage 99%'] == pytest.approx(1.0)

    def test_metrics_with_errors(self):
        """Test metrics computation with prediction errors."""
        n = 100
        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.5  # Add noise

        std = np.ones(n) * 2.0
        lower_95 = y_pred - 2 * std
        upper_95 = y_pred + 2 * std
        lower_99 = y_pred - 3 * std
        upper_99 = y_pred + 3 * std

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_99=lower_99,
            upper_99=upper_99
        )

        # Should have non-zero error
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0

        # R² should be less than 1
        assert metrics['R²'] < 1.0

        # Coverage should be reasonable
        assert 0 <= metrics['Coverage 95%'] <= 1.0
        assert 0 <= metrics['Coverage 99%'] <= 1.0

    def test_calibration_perfect_95_coverage(self):
        """Test calibration with exactly 95% coverage."""
        n = 1000
        y_true = np.random.randn(n)
        y_pred = y_true + np.random.randn(n) * 0.1

        # Create intervals such that exactly 95% of points are covered
        std = np.ones(n)
        quantile_95 = 1.96
        lower_95 = y_pred - quantile_95 * std
        upper_95 = y_pred + quantile_95 * std

        # Wide intervals for 99%
        lower_99 = y_pred - 3 * std
        upper_99 = y_pred + 3 * std

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_99=lower_99,
            upper_99=upper_99
        )

        # Coverage should be approximately 95% (with some random variation)
        assert 0.90 <= metrics['Coverage 95%'] <= 1.0

    def test_sharpness_calculation(self):
        """Test sharpness (interval width) calculation."""
        n = 100
        y_true = np.random.randn(n)
        y_pred = y_true

        # Narrow intervals
        std_narrow = np.ones(n) * 0.5
        lower_95_narrow = y_pred - 2 * std_narrow
        upper_95_narrow = y_pred + 2 * std_narrow
        lower_99_narrow = y_pred - 3 * std_narrow
        upper_99_narrow = y_pred + 3 * std_narrow

        metrics_narrow = compute_metrics(
            y_true, y_pred,
            lower_95_narrow, upper_95_narrow,
            lower_99_narrow, upper_99_narrow
        )

        # Wide intervals
        std_wide = np.ones(n) * 2.0
        lower_95_wide = y_pred - 2 * std_wide
        upper_95_wide = y_pred + 2 * std_wide
        lower_99_wide = y_pred - 3 * std_wide
        upper_99_wide = y_pred + 3 * std_wide

        metrics_wide = compute_metrics(
            y_true, y_pred,
            lower_95_wide, upper_95_wide,
            lower_99_wide, upper_99_wide
        )

        # Wide intervals should have higher sharpness
        assert metrics_wide['Sharpness 95%'] > metrics_narrow['Sharpness 95%']
        assert metrics_wide['Sharpness 99%'] > metrics_narrow['Sharpness 99%']

    def test_model_name_labeling(self):
        """Test that model name is correctly stored."""
        n = 50
        y_true = np.random.randn(n)
        y_pred = y_true
        intervals = np.stack([y_pred - 1, y_pred + 1, y_pred - 2, y_pred + 2])

        metrics = compute_metrics(
            y_true, y_pred,
            intervals[0], intervals[1], intervals[2], intervals[3],
            model_name="Test Model"
        )

        assert metrics['Model'] == "Test Model"

    def test_metrics_output_types(self):
        """Test that all metrics are float type."""
        n = 50
        y_true = np.random.randn(n)
        y_pred = y_true
        intervals = np.stack([y_pred - 1, y_pred + 1, y_pred - 2, y_pred + 2])

        metrics = compute_metrics(
            y_true, y_pred,
            intervals[0], intervals[1], intervals[2], intervals[3]
        )

        # Check all numeric values are floats
        for key, value in metrics.items():
            if key != 'Model':
                assert isinstance(value, float), f"{key} should be float"


class TestComputeAnomalyMetrics:
    """Test suite for compute_anomaly_metrics function."""

    def test_perfect_anomaly_detection(self):
        """Test metrics with perfect anomaly detection."""
        n = 100
        y_true_anomaly = np.zeros(n, dtype=bool)
        y_true_anomaly[10:20] = True  # 10 anomalies

        y_pred_anomaly = y_true_anomaly.copy()  # Perfect predictions

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true_anomaly,
            y_pred_anomaly=y_pred_anomaly
        )

        # Perfect detection
        assert metrics['Precision'] == pytest.approx(1.0)
        assert metrics['Recall'] == pytest.approx(1.0)
        assert metrics['F1-Score'] == pytest.approx(1.0)

    def test_no_anomalies_detected(self):
        """Test metrics when no anomalies are detected."""
        n = 100
        y_true_anomaly = np.zeros(n, dtype=bool)
        y_true_anomaly[10:20] = True

        y_pred_anomaly = np.zeros(n, dtype=bool)  # Predict no anomalies

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true_anomaly,
            y_pred_anomaly=y_pred_anomaly
        )

        # No detections → precision=0 (with zero_division), recall=0
        assert metrics['Precision'] == 0.0
        assert metrics['Recall'] == 0.0
        assert metrics['F1-Score'] == 0.0

    def test_all_anomalies_flagged(self):
        """Test metrics when all points flagged as anomalies."""
        n = 100
        y_true_anomaly = np.zeros(n, dtype=bool)
        y_true_anomaly[10:20] = True  # 10 true anomalies

        y_pred_anomaly = np.ones(n, dtype=bool)  # Predict all as anomalies

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true_anomaly,
            y_pred_anomaly=y_pred_anomaly
        )

        # Recall should be 1.0 (all true anomalies caught)
        assert metrics['Recall'] == pytest.approx(1.0)

        # Precision should be 10/100 = 0.1
        assert metrics['Precision'] == pytest.approx(0.1)

    def test_partial_detection(self):
        """Test metrics with partial anomaly detection."""
        n = 100
        y_true_anomaly = np.zeros(n, dtype=bool)
        y_true_anomaly[[10, 20, 30, 40]] = True  # 4 anomalies

        y_pred_anomaly = np.zeros(n, dtype=bool)
        y_pred_anomaly[[10, 20, 50]] = True  # Detect 2/4, plus 1 false positive

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true_anomaly,
            y_pred_anomaly=y_pred_anomaly
        )

        # Precision: 2/3 (2 true positives out of 3 detections)
        assert metrics['Precision'] == pytest.approx(2.0 / 3.0)

        # Recall: 2/4 (2 true positives out of 4 actual anomalies)
        assert metrics['Recall'] == pytest.approx(0.5)

        # F1: harmonic mean
        expected_f1 = 2 * (2/3 * 0.5) / (2/3 + 0.5)
        assert metrics['F1-Score'] == pytest.approx(expected_f1)

    def test_threshold_labeling(self):
        """Test threshold name labeling."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true,
            y_pred_anomaly=y_pred,
            model_name="Test Model",
            threshold_name="Custom Threshold"
        )

        assert metrics['Model'] == "Test Model"
        assert metrics['Threshold'] == "Custom Threshold"

    def test_with_integer_labels(self):
        """Test that function works with integer labels (0/1)."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])

        metrics = compute_anomaly_metrics(
            y_true_anomaly=y_true,
            y_pred_anomaly=y_pred
        )

        # Should work with integers
        assert 0 <= metrics['Precision'] <= 1
        assert 0 <= metrics['Recall'] <= 1
        assert 0 <= metrics['F1-Score'] <= 1


class TestComputePredictionIntervals:
    """Test suite for compute_prediction_intervals function."""

    def test_gaussian_intervals(self):
        """Test Gaussian prediction interval computation."""
        n = 100
        mean = np.random.randn(n)
        std = np.ones(n)

        intervals = compute_prediction_intervals(
            mean=mean,
            std=std,
            confidence_levels=[0.95, 0.99],
            distribution="gaussian"
        )

        # Check 95% interval
        lower_95, upper_95 = intervals[0.95]
        assert lower_95.shape == (n,)
        assert upper_95.shape == (n,)
        assert np.all(lower_95 < upper_95)

        # 99% interval should be wider than 95%
        lower_99, upper_99 = intervals[0.99]
        assert np.all(lower_99 < lower_95)
        assert np.all(upper_99 > upper_95)

    def test_student_t_intervals(self):
        """Test Student-t prediction interval computation."""
        n = 100
        mean = np.random.randn(n)
        std = np.ones(n)

        intervals = compute_prediction_intervals(
            mean=mean,
            std=std,
            confidence_levels=[0.95, 0.99],
            distribution="student_t",
            nu=4.0
        )

        lower_95, upper_95 = intervals[0.95]

        # Basic checks
        assert lower_95.shape == (n,)
        assert upper_95.shape == (n,)
        assert np.all(lower_95 < upper_95)

    def test_student_t_wider_than_gaussian(self):
        """Test that Student-t intervals are wider than Gaussian (heavier tails)."""
        n = 50
        mean = np.zeros(n)
        std = np.ones(n)

        # Gaussian intervals
        intervals_gauss = compute_prediction_intervals(
            mean, std,
            confidence_levels=[0.95],
            distribution="gaussian"
        )

        # Student-t intervals (ν=4, fairly heavy tails)
        intervals_t = compute_prediction_intervals(
            mean, std,
            confidence_levels=[0.95],
            distribution="student_t",
            nu=4.0
        )

        lower_gauss, upper_gauss = intervals_gauss[0.95]
        lower_t, upper_t = intervals_t[0.95]

        # Student-t should be wider
        width_gauss = upper_gauss - lower_gauss
        width_t = upper_t - lower_t

        assert np.all(width_t > width_gauss)

    def test_multiple_confidence_levels(self):
        """Test computation with multiple confidence levels."""
        mean = np.array([0.0, 1.0, 2.0])
        std = np.array([1.0, 1.0, 1.0])

        intervals = compute_prediction_intervals(
            mean, std,
            confidence_levels=[0.80, 0.90, 0.95, 0.99],
            distribution="gaussian"
        )

        # Check all levels exist
        assert 0.80 in intervals
        assert 0.90 in intervals
        assert 0.95 in intervals
        assert 0.99 in intervals

        # Higher confidence → wider intervals
        width_80 = np.mean(intervals[0.80][1] - intervals[0.80][0])
        width_90 = np.mean(intervals[0.90][1] - intervals[0.90][0])
        width_95 = np.mean(intervals[0.95][1] - intervals[0.95][0])
        width_99 = np.mean(intervals[0.99][1] - intervals[0.99][0])

        assert width_80 < width_90 < width_95 < width_99

    def test_interval_symmetry(self):
        """Test that intervals are symmetric around mean."""
        mean = np.array([5.0, 10.0, -3.0])
        std = np.array([2.0, 1.0, 0.5])

        intervals = compute_prediction_intervals(
            mean, std,
            confidence_levels=[0.95],
            distribution="gaussian"
        )

        lower, upper = intervals[0.95]

        # Distance from mean should be equal
        dist_lower = mean - lower
        dist_upper = upper - mean

        np.testing.assert_array_almost_equal(dist_lower, dist_upper)

    def test_invalid_distribution_raises_error(self):
        """Test that invalid distribution raises ValueError."""
        mean = np.array([0.0])
        std = np.array([1.0])

        with pytest.raises(ValueError, match="Unknown distribution"):
            compute_prediction_intervals(
                mean, std,
                distribution="invalid_distribution"
            )

    def test_varying_std(self):
        """Test intervals with heteroscedastic uncertainty (varying std)."""
        n = 50
        mean = np.zeros(n)
        std = np.linspace(0.5, 2.0, n)  # Varying uncertainty

        intervals = compute_prediction_intervals(
            mean, std,
            confidence_levels=[0.95],
            distribution="gaussian"
        )

        lower, upper = intervals[0.95]
        widths = upper - lower

        # Interval width should increase with std
        assert widths[0] < widths[-1]
        assert np.all(np.diff(widths) > 0)  # Monotonically increasing
