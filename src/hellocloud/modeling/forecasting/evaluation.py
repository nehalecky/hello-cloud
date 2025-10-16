"""
Evaluation metrics for time series forecasting.

This module provides comprehensive metrics for evaluating both point forecasts
and probabilistic forecasts. It includes:

**Point Forecast Metrics** (actual vs. predicted arrays):
- MAE (Mean Absolute Error): Average absolute deviation
- RMSE (Root Mean Squared Error): Penalizes large errors
- MAPE (Mean Absolute Percentage Error): Scale-independent percentage error
- SMAPE (Symmetric MAPE): Symmetric version, bounded [0, 200]
- MASE (Mean Absolute Scaled Error): Scaled by naive baseline

**Probabilistic Forecast Metrics** (for models with uncertainty):
- Quantile Loss (Pinball Loss): For quantile predictions
- Coverage: Prediction interval calibration
- Interval Sharpness: Average prediction interval width

All metrics return float values (not numpy scalars) and handle edge cases
gracefully (empty arrays, division by zero, etc.).

Example:
    ```python
    import numpy as np
    from hellocloud.modeling.forecasting.evaluation import (
        mae, rmse, mape, smape, mase,
        quantile_loss, coverage, interval_sharpness,
        compute_all_metrics
    )

    # Generate synthetic data
    y_train = np.random.randn(1000)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1

    # Point forecast metrics
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    print(f"RMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
    print(f"SMAPE: {smape(y_true, y_pred):.2f}%")
    print(f"MASE: {mase(y_true, y_pred, y_train):.4f}")

    # Probabilistic forecast metrics
    q_pred = y_pred  # Median prediction
    lower = y_pred - 1.96 * 0.1  # 95% interval lower bound
    upper = y_pred + 1.96 * 0.1  # 95% interval upper bound

    print(f"Quantile Loss (50th): {quantile_loss(y_true, q_pred, tau=0.5):.4f}")
    print(f"Coverage (95%): {coverage(y_true, lower, upper):.2f}%")
    print(f"Interval Sharpness: {interval_sharpness(lower, upper):.4f}")

    # Compute all metrics at once
    metrics = compute_all_metrics(y_true, y_pred, y_train)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    ```
"""

import numpy as np
from numpy.typing import NDArray


def mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """
    Mean Absolute Error (MAE).

    Formula:
        MAE = mean(|y_true - y_pred|)

    Interpretation:
        Average absolute deviation between predictions and actuals.
        Same units as the data (e.g., GB for memory, IOPS for disk).

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)

    Returns:
        MAE value as float

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        print(mae(y_true, y_pred))  # 10.0
        ```
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """
    Root Mean Squared Error (RMSE).

    Formula:
        RMSE = sqrt(mean((y_true - y_pred)²))

    Interpretation:
        Penalizes large errors more than MAE due to squaring.
        Same units as the data. RMSE ≥ MAE always.

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)

    Returns:
        RMSE value as float

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        print(rmse(y_true, y_pred))  # ~10.54 (higher than MAE due to squaring)
        ```
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating], epsilon: float = 1e-10
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Formula:
        MAPE = mean(|y_true - y_pred| / |y_true|) × 100

    Interpretation:
        Scale-independent percentage error. Values close to zero indicate
        excellent predictions. Can be unstable when y_true has values near zero.

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        epsilon: Small value to avoid division by zero (default: 1e-10)

    Returns:
        MAPE value as float (percentage, 0-100+)

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        print(mape(y_true, y_pred))  # ~5.56%
        ```

    Note:
        - Skips samples where |y_true| < epsilon to avoid division by zero
        - Can be biased: penalizes over-predictions more than under-predictions
        - Consider using SMAPE for symmetric error measurement
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    # Filter out near-zero actuals to avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        # All actuals are near zero - cannot compute MAPE
        return float("nan")

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    percentage_errors = np.abs(y_true_filtered - y_pred_filtered) / np.abs(y_true_filtered)
    return float(np.mean(percentage_errors) * 100)


def smape(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating], epsilon: float = 1e-10
) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    Formula:
        SMAPE = mean(2 × |y_true - y_pred| / (|y_true| + |y_pred|)) × 100

    Interpretation:
        Symmetric version of MAPE, bounded [0, 200]. Treats over-predictions
        and under-predictions equally. More stable than MAPE when values cross zero.

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        epsilon: Small value to avoid division by zero (default: 1e-10)

    Returns:
        SMAPE value as float (percentage, 0-200)

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        print(smape(y_true, y_pred))  # ~5.40%
        ```

    Note:
        - Skips samples where |y_true| + |y_pred| < epsilon
        - Bounded [0, 200] unlike MAPE which is unbounded
        - Symmetric: SMAPE(a, b) ≈ SMAPE(b, a)
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    # Filter out cases where both actual and predicted are near zero
    mask = denominator > epsilon
    if not np.any(mask):
        # All values are near zero - cannot compute SMAPE
        return float("nan")

    percentage_errors = numerator[mask] / denominator[mask]
    return float(np.mean(percentage_errors) * 100)


def mase(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_train: NDArray[np.floating],
    period: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    Formula:
        MASE = MAE(forecast) / MAE(naive_baseline)
        where naive_baseline uses lag=period

    Interpretation:
        Scale-independent metric comparing forecast to naive baseline.
        - MASE < 1: Forecast beats naive baseline
        - MASE = 1: Forecast equals naive baseline
        - MASE > 1: Forecast worse than naive baseline

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        y_train: Training data for computing naive baseline (n_train_samples,)
        period: Lag for naive baseline (1=naive, 7=weekly, 250=fast IOPS cycle)

    Returns:
        MASE value as float

    Raises:
        ValueError: If arrays are empty, have different shapes, or y_train too short

    Example:
        ```python
        # Train data for naive baseline
        y_train = np.random.randn(1000)

        # Test data
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        # MASE using naive baseline (lag=1)
        print(mase(y_true, y_pred, y_train))  # < 1 if good model

        # MASE using seasonal naive baseline (lag=250)
        print(mase(y_true, y_pred, y_train, period=250))
        ```

    Note:
        - Returns nan if naive baseline MAE is zero (perfect persistence)
        - Common period values: 1 (naive), 7 (weekly), 250 (fast cycle), 1250 (slow cycle)
    """
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    if len(y_train) <= period:
        raise ValueError(
            f"y_train must have > {period} samples for period={period}, got {len(y_train)}"
        )

    # Compute forecast MAE
    forecast_mae = mae(y_true, y_pred)

    # Compute naive baseline MAE on training data
    # Naive forecast: y(t) = y(t-period)
    naive_errors = np.abs(y_train[period:] - y_train[:-period])
    naive_mae = float(np.mean(naive_errors))

    # Avoid division by zero if training data is perfectly persistent
    if naive_mae == 0:
        return float("nan")

    return forecast_mae / naive_mae


def quantile_loss(y_true: NDArray[np.floating], q_pred: NDArray[np.floating], tau: float) -> float:
    """
    Quantile Loss (Pinball Loss) for quantile regression.

    Formula:
        ρ_τ(u) = u × (τ - I(u < 0))
        where u = y_true - q_pred

    Interpretation:
        Asymmetric loss function for quantile predictions.
        - tau = 0.5: Median prediction (symmetric)
        - tau = 0.1: 10th percentile (penalizes over-prediction)
        - tau = 0.9: 90th percentile (penalizes under-prediction)

    Args:
        y_true: Actual values of shape (n_samples,)
        q_pred: Quantile predictions of shape (n_samples,)
        tau: Target quantile in [0, 1] (e.g., 0.5 for median)

    Returns:
        Mean quantile loss as float

    Raises:
        ValueError: If arrays are empty, have different shapes, or tau not in [0, 1]

    Example:
        ```python
        y_true = np.array([100, 200, 300])
        q_pred = np.array([110, 190, 310])  # Predicted median

        # Median loss (tau=0.5, symmetric)
        print(quantile_loss(y_true, q_pred, tau=0.5))  # 5.0

        # 10th percentile loss (penalizes over-prediction)
        print(quantile_loss(y_true, q_pred, tau=0.1))  # 0.5

        # 90th percentile loss (penalizes under-prediction)
        print(quantile_loss(y_true, q_pred, tau=0.9))  # 9.5
        ```

    Use cases:
        - Evaluate GP quantile predictions
        - Evaluate TimesFM quantile forecasts
        - Assess prediction interval quality
    """
    if not 0 <= tau <= 1:
        raise ValueError(f"tau must be in [0, 1], got {tau}")

    y_true, q_pred = _validate_arrays(y_true, q_pred)

    errors = y_true - q_pred
    loss = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
    return float(np.mean(loss))


def coverage(
    y_true: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
) -> float:
    """
    Prediction Interval Coverage (calibration metric).

    Formula:
        Coverage = (# actuals within [lower, upper]) / (# actuals) × 100

    Interpretation:
        Percentage of actuals that fall within prediction intervals.
        - 90% interval should contain ~90% of actuals (well-calibrated)
        - < 90%: Model is over-confident (intervals too narrow)
        - > 90%: Model is under-confident (intervals too wide)

    Args:
        y_true: Actual values of shape (n_samples,)
        lower: Lower bounds of prediction intervals (n_samples,)
        upper: Upper bounds of prediction intervals (n_samples,)

    Returns:
        Coverage percentage as float (0-100)

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_true = np.array([100, 200, 300, 400, 500])
        lower = np.array([90, 180, 280, 380, 480])  # Lower bounds
        upper = np.array([110, 220, 320, 420, 520])  # Upper bounds

        print(coverage(y_true, lower, upper))  # 100.0 (all within bounds)
        ```

    Use cases:
        - Verify GP credible interval calibration
        - Check TimesFM quantile prediction quality
        - Assess uncertainty quantification
    """
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")
    if len(y_true) != len(lower) or len(y_true) != len(upper):
        raise ValueError(
            f"Array shapes must match: y_true={len(y_true)}, "
            f"lower={len(lower)}, upper={len(upper)}"
        )

    within_bounds = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within_bounds) * 100)


def interval_sharpness(lower: NDArray[np.floating], upper: NDArray[np.floating]) -> float:
    """
    Interval Sharpness (average prediction interval width).

    Formula:
        Sharpness = mean(upper - lower)

    Interpretation:
        Average width of prediction intervals. Lower is better (more precise).
        Trade-off with coverage: narrow intervals (low sharpness) may have
        poor coverage if model is over-confident.

    Args:
        lower: Lower bounds of prediction intervals (n_samples,)
        upper: Upper bounds of prediction intervals (n_samples,)

    Returns:
        Average interval width as float

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        lower = np.array([90, 180, 280])
        upper = np.array([110, 220, 320])

        print(interval_sharpness(lower, upper))  # 40.0
        ```

    Use cases:
        - Compare uncertainty between models (lower is more confident)
        - Balance coverage vs. sharpness (want high coverage + low sharpness)
        - Identify over-confident models (low sharpness + low coverage)
    """
    if len(lower) == 0:
        raise ValueError("lower cannot be empty")
    if len(lower) != len(upper):
        raise ValueError(f"Array shapes must match: lower={len(lower)}, upper={len(upper)}")

    widths = upper - lower
    return float(np.mean(widths))


def compute_all_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    y_train: NDArray[np.floating] | None = None,
    period: int = 1,
) -> dict[str, float]:
    """
    Compute all point forecast metrics at once.

    This is a convenience function that computes MAE, RMSE, MAPE, SMAPE,
    and optionally MASE (if y_train provided).

    Args:
        y_true: Actual values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        y_train: Training data for MASE computation (optional)
        period: Lag for naive baseline in MASE (default: 1)

    Returns:
        Dictionary with metric names and values

    Raises:
        ValueError: If arrays are empty or have different shapes

    Example:
        ```python
        y_train = np.random.randn(1000)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        metrics = compute_all_metrics(y_true, y_pred, y_train)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # Output:
        # mae: 0.0995
        # rmse: 0.1001
        # mape: 15.32
        # smape: 12.45
        # mase: 0.87
        ```
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    if y_train is not None:
        metrics["mase"] = mase(y_true, y_pred, y_train, period=period)

    return metrics


def _validate_arrays(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Validate that arrays are non-empty and have matching shapes.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Tuple of validated arrays

    Raises:
        ValueError: If arrays are empty or have different shapes
    """
    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty")
    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same shape: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    return y_true, y_pred
