"""
Walk-forward backtesting framework for time series forecasting evaluation.

This module provides comprehensive backtesting utilities for evaluating forecasting
models using walk-forward validation. Walk-forward validation simulates the production
forecasting workflow where models are trained on historical data and make predictions
for future timesteps, then the window moves forward in time.

Key Features:
- Expanding window: Accumulate training data over time (default)
- Sliding window: Fixed-size training window (for non-stationary series)
- Retraining control: Optionally retrain at each step vs. use initial model
- Comprehensive metrics: Fold-level and overall performance metrics
- Progress logging: Track long-running backtests with loguru
- Flexible forecaster API: Works with any BaseForecaster implementation

Typical Use Cases:
- Evaluate model degradation over time (performance by fold)
- Compare forecasters on realistic production scenarios
- Tune retraining frequency (accuracy vs. computation trade-off)
- Assess model stability across different time periods

Example:
    ```python
    from hellocloud.modeling.forecasting import (
        ARIMAForecaster,
        WalkForwardBacktest
    )
    import numpy as np

    # Generate synthetic data
    y_full = np.random.randn(10000)

    # Setup backtest
    forecaster = ARIMAForecaster(order=(1, 1, 1))
    backtest = WalkForwardBacktest(
        initial_train_size=5000,
        horizon=62,
        step_size=62,  # Non-overlapping folds
        retrain=True
    )

    # Run backtest
    results = backtest.run_backtest(forecaster, y_full)

    # Analyze results
    print(f"Overall MAE: {results['overall_metrics']['mae']:.3f}")
    print(f"Overall RMSE: {results['overall_metrics']['rmse']:.3f}")
    print(f"Avg training time: {results['avg_train_time']:.2f}s")
    print(f"Number of folds: {len(results['fold_metrics'])}")

    # Analyze degradation over time
    for i, fold_metric in enumerate(results['fold_metrics']):
        print(f"Fold {i}: MAE={fold_metric['mae']:.3f}, "
              f"RMSE={fold_metric['rmse']:.3f}, "
              f"train_size={fold_metric['train_size']}")
    ```

Performance Considerations:
- IOPS dataset: 295K samples, typical horizon: 62 steps (21 hours at 20x subsampling)
- Use step_size=horizon for non-overlapping folds (faster, independent folds)
- Use retrain=False for quick baseline assessment (faster, less accurate)
- Use expanding_window=False for non-stationary series (more stable)
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .baselines import BaseForecaster
from .evaluation import compute_all_metrics


@dataclass
class BacktestResults:
    """
    Results from walk-forward backtesting.

    This dataclass encapsulates all results from a backtesting run, including
    forecasts, actuals, metrics by fold, and overall aggregated metrics.

    Attributes:
        forecasts: All forecasts across folds, shape (n_total_forecasts,)
        actuals: Corresponding actual values, shape (n_total_forecasts,)
        fold_metrics: List of metrics dicts for each fold
        overall_metrics: Aggregated metrics across all folds
        train_sizes: Training set size at each fold, shape (n_folds,)
        train_times: Training time for each fold in seconds, shape (n_folds,)
        avg_train_time: Average training time across folds in seconds
        n_folds: Number of folds completed

    Example:
        ```python
        results = backtest.run_backtest(forecaster, y_full)

        # Access overall metrics
        print(f"Overall MAE: {results.overall_metrics['mae']:.4f}")

        # Access per-fold metrics
        for i, metrics in enumerate(results.fold_metrics):
            print(f"Fold {i}: MAE={metrics['mae']:.4f}, "
                  f"RMSE={metrics['rmse']:.4f}")

        # Access raw forecasts and actuals
        import matplotlib.pyplot as plt
        plt.plot(results.actuals, label='Actual', alpha=0.7)
        plt.plot(results.forecasts, label='Forecast', alpha=0.7)
        plt.legend()
        plt.show()
        ```
    """

    forecasts: NDArray[np.floating]
    actuals: NDArray[np.floating]
    fold_metrics: list[dict[str, float]]
    overall_metrics: dict[str, float]
    train_sizes: NDArray[np.integer]
    train_times: NDArray[np.floating]
    avg_train_time: float
    n_folds: int


class WalkForwardBacktest:
    """
    Walk-forward backtesting for time series forecasting evaluation.

    This class implements walk-forward validation, a realistic evaluation methodology
    that simulates production forecasting workflows. The process:

    1. Start with initial training window
    2. Train forecaster on historical data
    3. Generate forecast for next horizon steps
    4. Move window forward by step_size
    5. Repeat until data exhausted

    Window Strategies:
    - Expanding window (default): Training data grows over time
      - Use for: Stationary series, more data = better performance
      - Example: [0:1000] → [0:1062] → [0:1124] → ...
    - Sliding window: Fixed training size, window slides forward
      - Use for: Non-stationary series, concept drift
      - Example: [0:1000] → [62:1062] → [124:1124] → ...

    Retraining Strategies:
    - retrain=True (default): Refit forecaster at each fold
      - Use for: Production evaluation, best accuracy
      - Trade-off: Slower, more realistic
    - retrain=False: Use initial model for all forecasts
      - Use for: Quick baseline assessment, model stability testing
      - Trade-off: Faster, less accurate over time

    Args:
        initial_train_size: Size of initial training window (must be > 0)
        horizon: Forecast horizon in steps (must be > 0)
        step_size: Steps to move forward each iteration (default: 1)
        retrain: Whether to retrain model at each step (default: True)
        expanding_window: If True, use expanding window; if False, use sliding
            window of size initial_train_size (default: True)

    Raises:
        ValueError: If initial_train_size, horizon, or step_size <= 0

    Example:
        ```python
        # Non-overlapping folds (faster, independent)
        backtest = WalkForwardBacktest(
            initial_train_size=5000,
            horizon=62,
            step_size=62  # Move forward by entire horizon
        )

        # Overlapping folds (slower, more data)
        backtest = WalkForwardBacktest(
            initial_train_size=5000,
            horizon=62,
            step_size=1  # Move forward one step at a time
        )

        # Sliding window for non-stationary data
        backtest = WalkForwardBacktest(
            initial_train_size=5000,
            horizon=62,
            step_size=62,
            expanding_window=False  # Fixed training size
        )

        # Quick baseline without retraining
        backtest = WalkForwardBacktest(
            initial_train_size=5000,
            horizon=62,
            step_size=62,
            retrain=False  # Use initial model only
        )
        ```
    """

    def __init__(
        self,
        initial_train_size: int,
        horizon: int,
        step_size: int = 1,
        retrain: bool = True,
        expanding_window: bool = True,
    ):
        """
        Initialize the walk-forward backtest configuration.

        Args:
            initial_train_size: Size of initial training window (must be > 0)
            horizon: Forecast horizon in steps (must be > 0)
            step_size: Steps to move forward each iteration (default: 1)
            retrain: Whether to retrain model at each step (default: True)
            expanding_window: If True, use expanding window; if False, use
                sliding window (default: True)

        Raises:
            ValueError: If initial_train_size, horizon, or step_size <= 0
        """
        if initial_train_size <= 0:
            raise ValueError(f"initial_train_size must be > 0, got {initial_train_size}")
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

        self.initial_train_size = initial_train_size
        self.horizon = horizon
        self.step_size = step_size
        self.retrain = retrain
        self.expanding_window = expanding_window

        logger.info(
            f"Initialized WalkForwardBacktest: "
            f"initial_train_size={initial_train_size}, "
            f"horizon={horizon}, "
            f"step_size={step_size}, "
            f"retrain={retrain}, "
            f"expanding_window={expanding_window}"
        )

    def run_backtest(
        self, forecaster: BaseForecaster, y: NDArray[np.floating], **fit_params: Any
    ) -> BacktestResults:
        """
        Run walk-forward backtesting on the given time series.

        This method performs the core walk-forward validation logic:
        1. Validates sufficient data exists for at least one fold
        2. Iterates through folds, training and forecasting at each step
        3. Tracks forecasts, actuals, and metrics for each fold
        4. Computes overall aggregated metrics
        5. Returns comprehensive BacktestResults

        Args:
            forecaster: Any forecaster implementing BaseForecaster interface
            y: Full time series data, shape (n_samples,)
            **fit_params: Optional keyword arguments passed to forecaster.fit()
                (e.g., exogenous variables, seasonal period)

        Returns:
            BacktestResults containing all forecasts, actuals, and metrics

        Raises:
            ValueError: If insufficient data for initial training and at least
                one forecast horizon
            RuntimeError: If forecaster.fit() fails for all folds (logged as warnings
                for individual fold failures)

        Example:
            ```python
            from hellocloud.modeling.forecasting import NaiveForecaster
            import numpy as np

            # Generate data
            y_full = np.random.randn(10000)

            # Setup and run
            forecaster = NaiveForecaster()
            backtest = WalkForwardBacktest(
                initial_train_size=5000,
                horizon=62,
                step_size=62
            )
            results = backtest.run_backtest(forecaster, y_full)

            # Analyze
            print(f"Completed {results.n_folds} folds")
            print(f"Overall MAE: {results.overall_metrics['mae']:.4f}")
            ```

        Notes:
            - If forecaster.fit() fails for a fold, it's skipped with a warning
            - Training time is tracked per fold for performance analysis
            - Forecasts and actuals are concatenated across all folds
            - Metrics are computed both per-fold and overall
        """
        # Validate sufficient data
        min_required_size = self.initial_train_size + self.horizon
        if len(y) < min_required_size:
            raise ValueError(
                f"Time series too short for backtesting. "
                f"Need at least {min_required_size} samples "
                f"(initial_train_size={self.initial_train_size} + horizon={self.horizon}), "
                f"got {len(y)}"
            )

        logger.info(
            f"Starting backtest on time series with {len(y)} samples. "
            f"Expecting ~{self._estimate_n_folds(len(y))} folds."
        )

        # Initialize result containers
        all_forecasts = []
        all_actuals = []
        fold_metrics_list = []
        train_sizes = []
        train_times = []

        # Initial model fitting (for retrain=False case)
        initial_model_fitted = False
        fold_idx = 0

        # Walk-forward loop
        train_start = 0
        train_end = self.initial_train_size

        while train_end + self.horizon <= len(y):
            # Determine training window
            if self.expanding_window:
                # Expanding: Always start from beginning
                train_window = y[train_start:train_end]
            else:
                # Sliding: Fixed window size
                train_window = y[train_end - self.initial_train_size : train_end]

            # Determine test window
            test_start = train_end
            test_end = train_end + self.horizon
            test_window = y[test_start:test_end]

            # Fit forecaster (if retraining or first fold)
            should_fit = self.retrain or not initial_model_fitted
            if should_fit:
                try:
                    start_time = time.time()
                    forecaster.fit(train_window, **fit_params)
                    train_time = time.time() - start_time
                    initial_model_fitted = True

                    logger.debug(
                        f"Fold {fold_idx}: Fitted forecaster on "
                        f"{len(train_window)} samples in {train_time:.2f}s"
                    )
                except Exception as e:
                    logger.warning(
                        f"Fold {fold_idx}: Forecaster fit failed with error: {e}. "
                        f"Skipping this fold."
                    )
                    # Move to next fold
                    train_end += self.step_size
                    fold_idx += 1
                    continue
            else:
                # Not retraining, use time from initial fit
                train_time = train_times[0] if train_times else 0.0

            # Generate forecast
            try:
                forecast = forecaster.forecast(self.horizon)
            except Exception as e:
                logger.warning(
                    f"Fold {fold_idx}: Forecaster forecast failed with error: {e}. "
                    f"Skipping this fold."
                )
                # Move to next fold
                train_end += self.step_size
                fold_idx += 1
                continue

            # Store results
            all_forecasts.append(forecast)
            all_actuals.append(test_window)
            train_sizes.append(len(train_window))
            train_times.append(train_time)

            # Compute fold metrics (no y_train needed for individual fold)
            fold_metrics = compute_all_metrics(test_window, forecast, y_train=None)
            fold_metrics["train_size"] = len(train_window)
            fold_metrics["train_time"] = train_time
            fold_metrics_list.append(fold_metrics)

            logger.debug(
                f"Fold {fold_idx}: MAE={fold_metrics['mae']:.4f}, "
                f"RMSE={fold_metrics['rmse']:.4f}, "
                f"train_size={len(train_window)}"
            )

            # Move to next fold
            train_end += self.step_size
            fold_idx += 1

        # Validate we have at least one successful fold
        if len(all_forecasts) == 0:
            raise RuntimeError(
                "Backtesting failed: No successful folds completed. "
                "All forecaster.fit() calls failed."
            )

        # Concatenate all forecasts and actuals
        forecasts_concat = np.concatenate(all_forecasts)
        actuals_concat = np.concatenate(all_actuals)

        # Compute overall metrics (use full training data for MASE)
        y_train_for_mase = y[: self.initial_train_size]
        overall_metrics = compute_all_metrics(
            actuals_concat, forecasts_concat, y_train=y_train_for_mase
        )

        # Compute average training time
        avg_train_time = float(np.mean(train_times))

        logger.info(
            f"Backtest completed: {len(fold_metrics_list)} folds, "
            f"Overall MAE={overall_metrics['mae']:.4f}, "
            f"RMSE={overall_metrics['rmse']:.4f}, "
            f"Avg train time={avg_train_time:.2f}s"
        )

        return BacktestResults(
            forecasts=forecasts_concat,
            actuals=actuals_concat,
            fold_metrics=fold_metrics_list,
            overall_metrics=overall_metrics,
            train_sizes=np.array(train_sizes, dtype=np.int64),
            train_times=np.array(train_times, dtype=np.float64),
            avg_train_time=avg_train_time,
            n_folds=len(fold_metrics_list),
        )

    def _estimate_n_folds(self, n_samples: int) -> int:
        """
        Estimate the number of folds given time series length.

        Args:
            n_samples: Length of time series

        Returns:
            Estimated number of folds
        """
        available_for_testing = n_samples - self.initial_train_size
        n_folds = max(0, available_for_testing // self.step_size)
        return n_folds
