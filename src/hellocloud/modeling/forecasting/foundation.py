"""
Foundation model forecasters for time series prediction.

This module provides wrappers around pre-trained foundation models for zero-shot
time series forecasting. Foundation models are trained on large corpora of diverse
time series and can generalize to new domains without fine-tuning.

Models included:
- TimesFMForecaster: Google's TimesFM (Time Series Foundation Model)
- ChronosForecaster: Amazon's Chronos (coming soon)

All forecasters follow the BaseForecaster API:
1. fit(y_train): Store context window (no training required - zero-shot)
2. forecast(horizon, return_quantiles): Generate predictions with optional quantiles

Example:
    ```python
    import numpy as np
    from hellocloud.modeling.forecasting.foundation import TimesFMForecaster

    # Training data (context window)
    y_train = np.random.randn(1000)

    # Zero-shot forecasting with TimesFM
    timesfm = TimesFMForecaster(
        model_name="google/timesfm-2.5-200m-pytorch",
        context_len=512,
        horizon_len=128
    )
    timesfm.fit(y_train)

    # Point forecasts (median)
    preds = timesfm.forecast(horizon=50)

    # Forecasts with quantiles
    preds, lower, upper = timesfm.forecast(horizon=50, return_quantiles=True)
    ```

References:
    TimesFM: https://github.com/google-research/timesfm
    Paper: "A decoder-only foundation model for time-series forecasting"
"""

import warnings

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .baselines import BaseForecaster

# Optional dependency
try:
    import timesfm

    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False


class TimesFMForecaster(BaseForecaster):
    """
    TimesFM (Time Series Foundation Model) forecaster wrapper.

    TimesFM is Google's pre-trained foundation model for time series forecasting.
    It uses a decoder-only transformer architecture trained on a large corpus of
    diverse time series data. The model provides zero-shot forecasting without
    requiring fine-tuning on new datasets.

    Key features:
    - Zero-shot forecasting (no training required)
    - Supports context windows up to 16k timesteps (v2.5 model)
    - Probabilistic forecasts via quantiles
    - Batch inference for efficiency
    - Pre-trained on 100B+ time series datapoints

    Model variants:
    - google/timesfm-2.5-200m-pytorch: 200M parameters, context_len up to 16k
    - Smaller variants available for faster inference

    Use cases:
    - Quick forecasting without model training
    - Cold-start scenarios with limited historical data
    - Benchmarking against trained models (GP, ARIMA)
    - Domains similar to training data (cloud metrics, web traffic, etc.)

    Performance characteristics:
    - Excellent zero-shot performance on diverse domains
    - Fast inference (no training required)
    - Memory-intensive (requires 32GB+ RAM for 200M model)
    - Outperforms statistical baselines on most datasets
    - May underperform domain-specific trained models

    Example:
        ```python
        # IOPS time series (1000 samples)
        y_train = load_iops_data()  # Shape: (1000,)

        # Zero-shot forecast with TimesFM
        timesfm = TimesFMForecaster(context_len=512, horizon_len=128)
        timesfm.fit(y_train)  # No training - just stores context

        # Point forecast (median)
        forecast = timesfm.forecast(horizon=50)

        # Probabilistic forecast (10th, 50th, 90th percentiles)
        forecast, lower, upper = timesfm.forecast(horizon=50, return_quantiles=True)
        ```

    Attributes:
        model_name: HuggingFace model ID for TimesFM
        context_len: Maximum context window length (default: 512, max: 16k for v2.5)
        horizon_len: Maximum forecast horizon (default: 128, max: 1k for v2.5)
        backend: Inference backend ("torch" or "jax")
        quantiles: Quantile levels for probabilistic forecasts
        model_: Loaded TimesFM model (set in constructor)
        is_fitted_: Whether fit() has been called
        y_context_: Stored context window (last context_len points)
    """

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
        backend: str = "torch",
        quantiles: list[float] | None = None,
    ):
        """
        Initialize the TimesFM forecaster.

        Args:
            model_name: HuggingFace model ID. Default is the 200M parameter v2.5 model.
                Available models:
                - "google/timesfm-2.5-200m-pytorch" (recommended)
                - Smaller variants available on HuggingFace Hub
            context_len: Context window size (must be > 0, max 16k for v2.5).
                Longer contexts capture more history but require more memory.
                Recommended: 512 for fast inference, 2048+ for complex patterns.
            horizon_len: Maximum forecast horizon (must be > 0, max 1k for v2.5).
                This is the maximum number of steps TimesFM can forecast.
            backend: Inference backend ("torch" or "jax"). Default is "torch".
                Torch backend is more compatible, JAX may be faster.
            quantiles: Quantile levels for probabilistic forecasts.
                Default: [0.1, 0.5, 0.9] (10th, 50th, 90th percentiles).
                The 0.5 quantile is the median (point forecast).

        Raises:
            ImportError: If timesfm package is not installed
            ValueError: If context_len or horizon_len exceed model limits
            ValueError: If backend is not "torch" or "jax"
        """
        self._check_timesfm_installed()
        self._validate_params(context_len, horizon_len, backend)
        self._check_model_limits(model_name, context_len, horizon_len)

        self.model_name = model_name
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]

        self._validate_quantiles()

        # Load model (do this once in constructor)
        logger.info(f"Loading TimesFM model: {model_name} (backend={backend})")
        try:
            self.model_ = timesfm.TimesFM.from_pretrained(
                model_name=model_name,
                backend=backend,
            )
            logger.info("TimesFM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TimesFM model: {e}")
            raise RuntimeError(f"Failed to load TimesFM model '{model_name}': {e}") from e

        # Will be set after fitting
        self.is_fitted_ = False
        self.y_context_: NDArray[np.floating] | None = None

    @staticmethod
    def _check_timesfm_installed():
        """Check if TimesFM is installed."""
        if not HAS_TIMESFM:
            raise ImportError(
                "timesfm package is not installed. Install with:\n"
                "  uv sync --group foundation --group gpu\n"
                "Or directly:\n"
                "  pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'"
            )

    @staticmethod
    def _validate_params(context_len: int, horizon_len: int, backend: str):
        """Validate basic parameters."""
        if context_len <= 0:
            raise ValueError(f"context_len must be > 0, got {context_len}")
        if horizon_len <= 0:
            raise ValueError(f"horizon_len must be > 0, got {horizon_len}")
        if backend not in ("torch", "jax"):
            raise ValueError(f"backend must be 'torch' or 'jax', got {backend}")

    @staticmethod
    def _check_model_limits(model_name: str, context_len: int, horizon_len: int):
        """Check model-specific limits and warn if exceeded."""
        if "2.5" in model_name:
            if context_len > 16384:
                logger.warning(
                    f"context_len={context_len} exceeds v2.5 limit (16k). "
                    "Model may fail or truncate."
                )
            if horizon_len > 1024:
                logger.warning(
                    f"horizon_len={horizon_len} exceeds v2.5 limit (1k). "
                    "Model may fail or produce poor forecasts."
                )

    def _validate_quantiles(self):
        """Validate quantile values."""
        if not all(0.0 <= q <= 1.0 for q in self.quantiles):
            raise ValueError(f"quantiles must be in [0, 1], got {self.quantiles}")
        if 0.5 not in self.quantiles:
            logger.warning(
                "0.5 (median) not in quantiles. Point forecasts will use first quantile."
            )

    def fit(self, y_train: NDArray[np.floating]) -> "TimesFMForecaster":
        """
        Store context window for zero-shot forecasting.

        TimesFM is a pre-trained model and does not require training. This method
        simply stores the context window (last context_len points) that will be
        used for forecasting.

        If y_train exceeds context_len, only the last context_len points are used.
        This is logged as a warning to inform the user about data truncation.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance (for method chaining)

        Raises:
            ValueError: If y_train is empty or contains NaN/Inf
        """
        # Validate input
        if len(y_train) == 0:
            raise ValueError("y_train cannot be empty")

        if not np.all(np.isfinite(y_train)):
            raise ValueError("y_train contains NaN or Inf values")

        # Store context window (truncate if needed)
        if len(y_train) > self.context_len:
            logger.warning(
                f"y_train has {len(y_train)} samples but context_len={self.context_len}. "
                f"Using last {self.context_len} points."
            )
            self.y_context_ = y_train[-self.context_len :].copy()
        else:
            self.y_context_ = y_train.copy()

        self.is_fitted_ = True
        logger.info(
            f"TimesFM context stored: {len(self.y_context_)} samples " f"(original: {len(y_train)})"
        )

        # Warn if context is very short
        if len(self.y_context_) < 50:
            logger.warning(
                f"Context window has only {len(self.y_context_)} samples. "
                "TimesFM may not perform well with very short contexts. "
                "Consider using more historical data if available."
            )

        return self

    def forecast(
        self, horizon: int, return_quantiles: bool = False
    ) -> (
        NDArray[np.floating]
        | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    ):
        """
        Generate zero-shot forecasts using TimesFM.

        Args:
            horizon: Number of steps ahead to forecast (must be > 0 and <= horizon_len)
            return_quantiles: If True, return (forecast, lower, upper) tuple with
                10th and 90th percentile bounds. If False, return only median forecast.

        Returns:
            If return_quantiles=False:
                Forecasts of shape (horizon,) - median predictions
            If return_quantiles=True:
                Tuple of (forecast, lower, upper), each of shape (horizon,)
                - forecast: 50th percentile (median)
                - lower: 10th percentile (lower bound)
                - upper: 90th percentile (upper bound)

        Raises:
            ValueError: If horizon <= 0 or > horizon_len
            RuntimeError: If called before fit()
            RuntimeError: If TimesFM inference fails

        Note:
            TimesFM returns probabilistic forecasts by default. The point forecast
            is the median (50th percentile). Quantile forecasts provide uncertainty
            estimates for risk-aware decision making.
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast()")

        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        if horizon > self.horizon_len:
            raise ValueError(
                f"horizon={horizon} exceeds horizon_len={self.horizon_len}. "
                "Increase horizon_len or reduce forecast horizon."
            )

        # Run inference
        try:
            # TimesFM expects list of time series (batch inference)
            # We have a single series, so wrap in list
            inputs = [self.y_context_]

            # Generate forecasts (suppress any warnings from TimesFM)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # TimesFM.forecast returns (point_forecast, quantile_forecast)
                # point_forecast: shape (n_series, horizon)
                # quantile_forecast: shape (n_series, horizon, n_quantiles)
                point_forecast, quantile_forecast = self.model_.forecast(
                    inputs=inputs,
                    freq=[0],  # Frequency indicator (0 for unknown/irregular)
                    horizon=horizon,
                )

            # Extract forecasts for our single series
            # point_forecast is median (50th percentile)
            forecast = point_forecast[0]  # Shape: (horizon,)

            if return_quantiles:
                # Extract quantile forecasts
                quantile_array = quantile_forecast[0]  # Shape: (horizon, n_quantiles)

                # Find indices for 10th, 50th, 90th percentiles
                # TimesFM uses self.quantiles ordering
                try:
                    idx_10 = self.quantiles.index(0.1)
                    idx_50 = self.quantiles.index(0.5)
                    idx_90 = self.quantiles.index(0.9)
                except ValueError:
                    logger.warning(
                        f"Required quantiles (0.1, 0.5, 0.9) not found in {self.quantiles}. "
                        "Using first, middle, last quantiles as approximation."
                    )
                    idx_10 = 0
                    idx_50 = len(self.quantiles) // 2
                    idx_90 = len(self.quantiles) - 1

                lower = quantile_array[:, idx_10]
                median = quantile_array[:, idx_50]
                upper = quantile_array[:, idx_90]

                # Use median from quantiles (more accurate than point_forecast)
                return median, lower, upper
            else:
                return forecast

        except Exception as e:
            logger.error(f"TimesFM forecasting failed: {e}")
            raise RuntimeError(f"Failed to generate forecast: {e}") from e

    def forecast_batch(
        self,
        horizons: list[int],
        return_quantiles: bool = False,
    ) -> (
        list[NDArray[np.floating]]
        | list[tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]]
    ):
        """
        Generate forecasts for multiple horizons in one call (batch inference).

        This is more efficient than calling forecast() multiple times, especially
        for walk-forward backtesting where you need forecasts at many horizons.

        Args:
            horizons: List of forecast horizons (e.g., [10, 20, 50])
            return_quantiles: If True, return quantile forecasts for each horizon

        Returns:
            List of forecasts, one per horizon. If return_quantiles=False, each
            element is a numpy array. If return_quantiles=True, each element is
            a (forecast, lower, upper) tuple.

        Raises:
            ValueError: If any horizon <= 0 or > horizon_len
            RuntimeError: If called before fit()

        Example:
            ```python
            # Forecast at multiple horizons
            forecasts = timesfm.forecast_batch([10, 20, 50])
            # forecasts[0]: 10-step forecast
            # forecasts[1]: 20-step forecast
            # forecasts[2]: 50-step forecast
            ```

        Note:
            This implementation calls forecast() for each horizon sequentially.
            Future optimization: Use TimesFM's batch inference API directly.
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast_batch()")

        # Validate all horizons
        for h in horizons:
            if h <= 0:
                raise ValueError(f"All horizons must be > 0, got {h}")
            if h > self.horizon_len:
                raise ValueError(
                    f"horizon={h} exceeds horizon_len={self.horizon_len}. "
                    "Increase horizon_len or reduce forecast horizons."
                )

        # Generate forecasts for each horizon
        # TODO: Optimize by using TimesFM's batch API directly
        forecasts = []
        for h in horizons:
            forecast_result = self.forecast(horizon=h, return_quantiles=return_quantiles)
            forecasts.append(forecast_result)

        logger.info(f"Generated forecasts for {len(horizons)} horizons")
        return forecasts
