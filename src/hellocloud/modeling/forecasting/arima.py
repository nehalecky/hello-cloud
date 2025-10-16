"""
ARIMA forecasting models for time series prediction.

This module provides ARIMA (AutoRegressive Integrated Moving Average) forecasters
with automatic order selection capabilities. ARIMA models capture temporal dependencies,
trends, and seasonality in time series data.

Models included:
- ARIMAForecaster: ARIMA(p,d,q) with optional seasonal component SARIMA(p,d,q)(P,D,Q,m)
- Auto-selection via pmdarima (optional dependency) or simple grid search

All forecasters follow the BaseForecaster API:
1. fit(y_train): Learn from training data
2. forecast(horizon, return_conf_int): Generate predictions with optional confidence intervals

Example:
    ```python
    import numpy as np
    from hellocloud.modeling.forecasting.arima import ARIMAForecaster

    # Training data with trend and seasonality
    t = np.arange(500)
    y_train = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(500)

    # Manual order specification
    arima = ARIMAForecaster(order=(2, 1, 2))
    arima.fit(y_train)
    preds = arima.forecast(horizon=50)

    # Automatic order selection (requires pmdarima)
    auto_arima = ARIMAForecaster(auto_select=True)
    auto_arima.fit(y_train)
    preds, lower, upper = auto_arima.forecast(horizon=50, return_conf_int=True)

    # Seasonal ARIMA
    sarima = ARIMAForecaster(
        order=(1, 1, 1),
        seasonal=True,
        seasonal_order=(1, 1, 1, 50)
    )
    sarima.fit(y_train)
    preds = sarima.forecast(horizon=50)
    ```
"""

import warnings

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

from .baselines import BaseForecaster

# Optional dependency
try:
    from pmdarima import auto_arima as pmdarima_auto_arima

    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecaster with automatic order selection.

    ARIMA (AutoRegressive Integrated Moving Average) models are classical
    statistical models for time series forecasting. They combine three components:
    - AR(p): Autoregression - linear combination of past values
    - I(d): Integration - differencing to achieve stationarity
    - MA(q): Moving Average - linear combination of past forecast errors

    Optionally supports seasonal ARIMA (SARIMA) with seasonal components (P,D,Q,m).

    Formula:
        (1 - φ₁L - ... - φₚLᵖ)(1 - L)ᵈ yₜ = (1 + θ₁L + ... + θᵧLᵧ)εₜ

    Where:
        - φ: AR coefficients, θ: MA coefficients
        - L: Lag operator, d: differencing order
        - εₜ: White noise error term

    Use cases:
        - Series with trends (use d > 0)
        - Series with autocorrelation (use p > 0)
        - Series with MA patterns (use q > 0)
        - Seasonal series (use seasonal=True)
        - When unsure about orders, use auto_select=True

    Performance characteristics:
        - Excellent for linear trends and autocorrelated series
        - Handles seasonality with SARIMA extension
        - Requires stationarity (achieved via differencing)
        - Can struggle with highly nonlinear patterns
        - Computationally more expensive than naive baselines

    Example:
        ```python
        # Trend + seasonal pattern
        t = np.arange(1000)
        y = 10 + 0.1*t + 5*np.sin(2*np.pi*t/250) + np.random.randn(1000)

        # Auto-select orders
        arima = ARIMAForecaster(auto_select=True)
        arima.fit(y)
        forecast, lower, upper = arima.forecast(50, return_conf_int=True)

        # Manual specification
        arima = ARIMAForecaster(order=(2, 1, 2), seasonal=True,
                                seasonal_order=(1, 0, 1, 250))
        arima.fit(y)
        forecast = arima.forecast(50)
        ```

    Attributes:
        order: (p, d, q) order for non-seasonal component
        seasonal: Whether to use seasonal ARIMA
        seasonal_order: (P, D, Q, m) order for seasonal component
        auto_select: Use automatic order selection (requires pmdarima)
        model_: Fitted statsmodels ARIMA model (set after fit())
        is_fitted_: Whether the model has been fitted
        selected_order_: Automatically selected order (if auto_select=True)
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal: bool = False,
        seasonal_order: tuple[int, int, int, int] | None = None,
        auto_select: bool = False,
        alpha: float = 0.05,
    ):
        """
        Initialize the ARIMA forecaster.

        Args:
            order: (p, d, q) order for ARIMA model. If None and auto_select=False,
                defaults to (1, 1, 1). Ignored if auto_select=True.
            seasonal: Enable seasonal ARIMA (SARIMA)
            seasonal_order: (P, D, Q, m) seasonal order. Required if seasonal=True.
                m is the seasonal period (e.g., 250 for IOPS dataset).
            auto_select: Use automatic order selection via pmdarima or grid search.
                If True, overrides the order parameter.
            alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)

        Raises:
            ValueError: If seasonal=True but seasonal_order is None
            ValueError: If seasonal_order provided but m <= 0
            ImportError: If auto_select=True but pmdarima not installed (falls back to grid search)
        """
        # Validate seasonal parameters
        if seasonal and seasonal_order is None:
            raise ValueError("seasonal_order must be provided when seasonal=True")

        if seasonal_order is not None:
            if len(seasonal_order) != 4:
                raise ValueError(f"seasonal_order must be (P, D, Q, m), got {seasonal_order}")
            if seasonal_order[3] <= 0:
                raise ValueError(f"Seasonal period m must be > 0, got {seasonal_order[3]}")

        # Set default order if not auto-selecting
        if order is None and not auto_select:
            order = (1, 1, 1)

        self.order = order
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.auto_select = auto_select
        self.alpha = alpha

        # Will be set after fitting
        self.model_: StatsmodelsARIMA | None = None
        self.is_fitted_ = False
        self.selected_order_: tuple[int, int, int] | None = None
        self._y_train: NDArray[np.floating] | None = None

        # Check pmdarima availability if auto_select requested
        if auto_select and not HAS_PMDARIMA:
            logger.warning(
                "pmdarima not installed. Will use simple grid search for order selection. "
                "Install with: uv add pmdarima"
            )

    def fit(self, y_train: NDArray[np.floating]) -> "ARIMAForecaster":
        """
        Fit the ARIMA model to training data.

        Args:
            y_train: Training time series of shape (n_samples,)

        Returns:
            self: Fitted forecaster instance (for method chaining)

        Raises:
            ValueError: If y_train is empty or contains NaN/Inf
            ValueError: If y_train is too short for the requested order
            RuntimeError: If model fitting fails to converge
        """
        # Validate input
        if len(y_train) == 0:
            raise ValueError("y_train cannot be empty")

        if not np.all(np.isfinite(y_train)):
            raise ValueError("y_train contains NaN or Inf values")

        # Store training data for diagnostics
        self._y_train = y_train.copy()

        # Determine order (auto-select or use provided)
        if self.auto_select:
            self.selected_order_ = self._auto_select_order(y_train)
            order_to_use = self.selected_order_
            logger.info(f"Auto-selected ARIMA order: {order_to_use}")
        else:
            order_to_use = self.order
            self.selected_order_ = order_to_use

        # Validate series length vs order requirements
        self._validate_series_length(y_train, order_to_use)

        # Fit ARIMA model
        try:
            # Suppress convergence warnings (we'll handle them)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                # Build kwargs for ARIMA (only include seasonal_order if needed)
                arima_kwargs = {
                    "order": order_to_use,
                    "enforce_stationarity": False,
                    "enforce_invertibility": False,
                }
                if self.seasonal and self.seasonal_order is not None:
                    arima_kwargs["seasonal_order"] = self.seasonal_order

                self.model_ = StatsmodelsARIMA(y_train, **arima_kwargs)
                fitted_model = self.model_.fit()
                self.model_ = fitted_model  # Store fitted model

                self.is_fitted_ = True
                logger.info(f"ARIMA{order_to_use} model fitted successfully")

        except Exception as e:
            logger.error(f"ARIMA model fitting failed: {e}")
            raise RuntimeError(f"Failed to fit ARIMA model: {e}") from e

        return self

    def forecast(
        self, horizon: int, return_conf_int: bool = False
    ) -> (
        NDArray[np.floating]
        | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    ):
        """
        Generate forecasts for future timesteps.

        Args:
            horizon: Number of steps ahead to forecast (must be > 0)
            return_conf_int: If True, return (forecast, lower, upper) tuple
                with confidence intervals. If False, return only point forecasts.

        Returns:
            If return_conf_int=False:
                Forecasts of shape (horizon,)
            If return_conf_int=True:
                Tuple of (forecast, lower, upper), each of shape (horizon,)
                Lower and upper bounds are 95% confidence intervals by default.

        Raises:
            ValueError: If horizon <= 0
            RuntimeError: If called before fit()
        """
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before forecast()")

        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        # Generate forecast
        try:
            forecast_result = self.model_.get_forecast(steps=horizon)

            # Point forecasts (handle both Series and ndarray)
            point_forecast = forecast_result.predicted_mean
            if hasattr(point_forecast, "values"):
                point_forecast = point_forecast.values
            else:
                point_forecast = np.asarray(point_forecast)

            if return_conf_int:
                # Get confidence intervals
                conf_int = forecast_result.conf_int(alpha=self.alpha)
                if hasattr(conf_int, "iloc"):
                    # DataFrame
                    lower = conf_int.iloc[:, 0].values
                    upper = conf_int.iloc[:, 1].values
                else:
                    # ndarray
                    lower = conf_int[:, 0]
                    upper = conf_int[:, 1]

                return point_forecast, lower, upper
            else:
                return point_forecast

        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            raise RuntimeError(f"Failed to generate forecast: {e}") from e

    def _auto_select_order(self, y_train: NDArray[np.floating]) -> tuple[int, int, int]:
        """
        Automatically select ARIMA order using pmdarima or grid search.

        Args:
            y_train: Training time series

        Returns:
            Optimal (p, d, q) order
        """
        if HAS_PMDARIMA:
            return self._auto_select_pmdarima(y_train)
        else:
            return self._auto_select_grid_search(y_train)

    def _auto_select_pmdarima(self, y_train: NDArray[np.floating]) -> tuple[int, int, int]:
        """
        Use pmdarima's auto_arima for optimal order selection.

        Args:
            y_train: Training time series

        Returns:
            Optimal (p, d, q) order selected by AIC
        """
        logger.info("Using pmdarima auto_arima for order selection")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                auto_model = pmdarima_auto_arima(
                    y_train,
                    start_p=0,
                    start_q=0,
                    max_p=5,
                    max_q=5,
                    max_d=2,
                    seasonal=self.seasonal,
                    m=self.seasonal_order[3] if self.seasonal else 1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )

                order = auto_model.order
                logger.info(f"pmdarima selected order: {order} (AIC={auto_model.aic():.2f})")
                return order

        except Exception as e:
            logger.warning(f"pmdarima auto_arima failed: {e}. Falling back to (1,1,1)")
            return (1, 1, 1)

    def _auto_select_grid_search(self, y_train: NDArray[np.floating]) -> tuple[int, int, int]:
        """
        Simple grid search for order selection (fallback when pmdarima unavailable).

        Tries common order combinations and selects based on AIC.

        Args:
            y_train: Training time series

        Returns:
            Best (p, d, q) order found
        """
        logger.info("Using simple grid search for order selection")

        # Common orders to try (limited to keep it fast)
        candidate_orders = [
            (0, 1, 0),  # Random walk
            (1, 0, 0),  # AR(1)
            (0, 0, 1),  # MA(1)
            (1, 1, 1),  # ARIMA(1,1,1) - default
            (2, 1, 2),  # ARIMA(2,1,2)
            (1, 1, 0),  # ARIMA(1,1,0)
            (0, 1, 1),  # ARIMA(0,1,1)
        ]

        best_aic = np.inf
        best_order = (1, 1, 1)

        for order in candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)

                    model = StatsmodelsARIMA(y_train, order=order)
                    fitted = model.fit()
                    aic = fitted.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = order

            except Exception:
                # Skip orders that fail to fit
                continue

        logger.info(f"Grid search selected order: {best_order} (AIC={best_aic:.2f})")
        return best_order

    def _validate_series_length(
        self, y_train: NDArray[np.floating], order: tuple[int, int, int]
    ) -> None:
        """
        Validate that series is long enough for the requested order.

        Args:
            y_train: Training time series
            order: (p, d, q) order

        Raises:
            ValueError: If series too short
        """
        p, d, q = order
        min_length = max(p, q) + d + 1

        if self.seasonal and self.seasonal_order is not None:
            P, D, Q, m = self.seasonal_order
            min_length += m * (max(P, Q) + D)

        if len(y_train) < min_length:
            raise ValueError(
                f"y_train has {len(y_train)} samples but order {order} "
                f"requires at least {min_length} samples"
            )

        # Warn if series is very short
        if len(y_train) < 50:
            logger.warning(
                f"Series has only {len(y_train)} samples. "
                "ARIMA may not perform well on very short series."
            )
