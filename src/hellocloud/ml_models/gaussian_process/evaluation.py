"""
Evaluation metrics for Gaussian Process models.

Provides comprehensive evaluation metrics for:
- Prediction accuracy (RMSE, MAE, R²)
- Uncertainty calibration (coverage, sharpness)
- Anomaly detection (precision, recall, F1, AUC-ROC)

Based on proper scoring rules for probabilistic forecasts.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_95: np.ndarray,
    upper_95: np.ndarray,
    lower_99: np.ndarray,
    upper_99: np.ndarray,
    model_name: str = "GP Model",
) -> Dict[str, float]:
    """
    Compute comprehensive model evaluation metrics.

    Includes both point prediction accuracy and uncertainty quantification quality.

    Args:
        y_true: True values (n,)
        y_pred: Predicted mean values (n,)
        lower_95: Lower bound of 95% prediction interval (n,)
        upper_95: Upper bound of 95% prediction interval (n,)
        lower_99: Lower bound of 99% prediction interval (n,)
        upper_99: Upper bound of 99% prediction interval (n,)
        model_name: Name for result labeling (default: "GP Model")

    Returns:
        Dictionary with metrics:
        - 'Model': Model name
        - 'RMSE': Root mean squared error
        - 'MAE': Mean absolute error
        - 'R²': R-squared score
        - 'Coverage 95%': Fraction of points in 95% interval
        - 'Coverage 99%': Fraction of points in 99% interval
        - 'Sharpness 95%': Average width of 95% interval
        - 'Sharpness 99%': Average width of 99% interval

    Example:
        ```python
        metrics = compute_metrics(
            y_true=y_test,
            y_pred=mean_predictions,
            lower_95=lower_95_interval,
            upper_95=upper_95_interval,
            lower_99=lower_99_interval,
            upper_99=upper_99_interval,
            model_name="Robust GP"
        )
        ```

    Notes:
        - **Calibration**: Coverage should match nominal level (95%, 99%)
        - **Sharpness**: Narrower intervals are better IF well-calibrated
        - **Point accuracy**: RMSE/MAE measure prediction error, R² measures variance explained
    """
    # Prediction accuracy
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calibration (prediction interval coverage)
    coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))
    coverage_99 = np.mean((y_true >= lower_99) & (y_true <= upper_99))

    # Sharpness (interval width - narrower is better if well-calibrated)
    sharpness_95 = np.mean(upper_95 - lower_95)
    sharpness_99 = np.mean(upper_99 - lower_99)

    return {
        'Model': model_name,
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R²': float(r2),
        'Coverage 95%': float(coverage_95),
        'Coverage 99%': float(coverage_99),
        'Sharpness 95%': float(sharpness_95),
        'Sharpness 99%': float(sharpness_99),
    }


def compute_anomaly_metrics(
    y_true_anomaly: np.ndarray,
    y_pred_anomaly: np.ndarray,
    model_name: str = "GP Model",
    threshold_name: str = "95% Interval",
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Args:
        y_true_anomaly: True anomaly labels (n,) - boolean or 0/1
        y_pred_anomaly: Predicted anomaly labels (n,) - boolean or 0/1
        model_name: Model name for result labeling
        threshold_name: Threshold description (e.g., "95% Interval", "99% Interval")

    Returns:
        Dictionary with metrics:
        - 'Model': Model name
        - 'Threshold': Threshold description
        - 'Precision': Precision score
        - 'Recall': Recall score
        - 'F1-Score': F1 score

    Example:
        ```python
        # Detect anomalies as points outside 95% interval
        anomalies_pred = (y_test < lower_95) | (y_test > upper_95)

        metrics = compute_anomaly_metrics(
            y_true_anomaly=anomaly_labels,
            y_pred_anomaly=anomalies_pred,
            model_name="Robust GP",
            threshold_name="95% Interval"
        )
        ```

    Notes:
        - **Precision**: Fraction of detected anomalies that are true anomalies
        - **Recall**: Fraction of true anomalies that were detected
        - **F1-Score**: Harmonic mean of precision and recall
        - Use `zero_division=0` to handle edge cases gracefully
    """
    precision = precision_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    recall = recall_score(y_true_anomaly, y_pred_anomaly, zero_division=0)
    f1 = f1_score(y_true_anomaly, y_pred_anomaly, zero_division=0)

    return {
        'Model': model_name,
        'Threshold': threshold_name,
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
    }


def compute_prediction_intervals(
    mean: np.ndarray,
    std: np.ndarray,
    confidence_levels: list = [0.95, 0.99],
    distribution: str = "gaussian",
    nu: float = 4.0,
) -> Dict[str, tuple]:
    """
    Compute prediction intervals from GP posterior.

    Args:
        mean: Predicted mean values (n,)
        std: Predicted standard deviations (n,)
        confidence_levels: List of confidence levels (default: [0.95, 0.99])
        distribution: Distribution type - "gaussian" or "student_t" (default: "gaussian")
        nu: Degrees of freedom for Student-t (default: 4.0, only used if distribution="student_t")

    Returns:
        Dictionary mapping confidence level to (lower, upper) tuple:
        {
            0.95: (lower_95, upper_95),
            0.99: (lower_99, upper_99)
        }

    Example:
        ```python
        # Gaussian intervals
        intervals = compute_prediction_intervals(
            mean=predictions_mean,
            std=predictions_std,
            confidence_levels=[0.95, 0.99],
            distribution="gaussian"
        )
        lower_95, upper_95 = intervals[0.95]

        # Student-t intervals (heavier tails)
        intervals_robust = compute_prediction_intervals(
            mean=predictions_mean,
            std=predictions_std,
            confidence_levels=[0.95, 0.99],
            distribution="student_t",
            nu=4.0
        )
        ```

    Notes:
        - **Gaussian**: Uses normal quantiles (z-scores)
        - **Student-t**: Uses t-distribution quantiles (heavier tails for robustness)
        - Higher confidence → wider intervals
    """
    from scipy.stats import norm, t as student_t

    intervals = {}

    for conf_level in confidence_levels:
        # Compute quantile for two-sided interval
        alpha = 1 - conf_level
        quantile_prob = 1 - alpha / 2  # e.g., 0.975 for 95% interval

        if distribution == "gaussian":
            quantile = norm.ppf(quantile_prob)
        elif distribution == "student_t":
            quantile = student_t.ppf(quantile_prob, df=nu)
        else:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Choose from: 'gaussian', 'student_t'"
            )

        # Compute interval bounds
        lower = mean - quantile * std
        upper = mean + quantile * std

        intervals[conf_level] = (lower, upper)

    return intervals
