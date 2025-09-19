"""
Advanced Time Series Forecasting with Foundation Models
Integrates TimesFM, Chronos, and TiReX for cloud cost prediction
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import torch
from loguru import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod

# HuggingFace imports
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from chronos import ChronosPipeline
from datasets import Dataset

@dataclass
class ForecastResult:
    """Container for forecast results"""
    point_forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    quantiles: Optional[Dict[float, np.ndarray]] = None
    model_name: str = ""
    metrics: Optional[Dict[str, float]] = None

class BaseForecaster(ABC):
    """Base class for time series forecasting models"""

    @abstractmethod
    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Fit the model on training data"""
        pass

    @abstractmethod
    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate predictions for specified horizon"""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return model name"""
        pass

class ChronosForecaster(BaseForecaster):
    """Amazon Chronos for zero-shot time series forecasting"""

    def __init__(self, model_size: str = "small"):
        """
        Initialize Chronos model

        Args:
            model_size: "small", "base", or "large"
        """
        self.model_size = model_size
        self.model_name = f"amazon/chronos-t5-{model_size}"
        logger.info(f"Loading Chronos model: {self.model_name}")

        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32
        )

        self.context_data = None
        self.timestamps = None

    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Store context for prediction"""
        self.context_data = data
        self.timestamps = timestamps
        logger.info(f"Chronos context set with {len(data)} points")

    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate zero-shot forecast"""
        if self.context_data is None:
            raise ValueError("Must call fit() before predict()")

        # Prepare context tensor
        context = torch.tensor(self.context_data, dtype=torch.float32).unsqueeze(0)

        # Generate forecast with quantiles
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        forecast = self.pipeline.predict(
            context,
            prediction_length=horizon,
            num_samples=100,
            quantiles=quantiles
        )

        # Extract predictions
        point_forecast = forecast.quantile(0.5).squeeze().numpy()

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower_bound = forecast.quantile(lower_quantile).squeeze().numpy()
        upper_bound = forecast.quantile(upper_quantile).squeeze().numpy()

        # Extract all quantiles
        quantile_dict = {
            q: forecast.quantile(q).squeeze().numpy()
            for q in quantiles
        }

        return ForecastResult(
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=quantile_dict,
            model_name=self.name()
        )

    def name(self) -> str:
        return f"Chronos-{self.model_size}"

class TimesFMForecaster(BaseForecaster):
    """Google's TimesFM for foundation model forecasting"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize TimesFM model"""
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.context_data = None

        try:
            import timesfm
            logger.info("Loading TimesFM model...")

            # Initialize TimesFM
            self.model = timesfm.TimesFM(
                context_len=512,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend="cpu"  # Change to "gpu" if available
            )

            if checkpoint_path:
                self.model.load_from_checkpoint(checkpoint_path)
            else:
                logger.warning("No checkpoint provided, using random weights")

        except ImportError:
            logger.warning("TimesFM not installed, skipping initialization")
            logger.info("Install with: pip install git+https://github.com/google-research/timesfm.git")

    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Store context for prediction"""
        self.context_data = data
        logger.info(f"TimesFM context set with {len(data)} points")

    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate TimesFM forecast"""
        if self.model is None:
            logger.error("TimesFM model not initialized")
            return ForecastResult(
                point_forecast=np.zeros(horizon),
                model_name=self.name()
            )

        if self.context_data is None:
            raise ValueError("Must call fit() before predict()")

        # TimesFM expects list of arrays
        forecast_input = [self.context_data]

        # Generate point and quantile forecasts
        point_forecast, quantile_forecast = self.model.forecast(
            forecast_input,
            freq=[0],  # Frequency hint
            horizon=[horizon],
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )

        # Extract results
        point_forecast = point_forecast[0]

        # Calculate confidence intervals
        if quantile_forecast is not None:
            lower_bound = quantile_forecast[0][0]  # 10th percentile
            upper_bound = quantile_forecast[0][4]  # 90th percentile
            quantile_dict = {
                0.1: quantile_forecast[0][0],
                0.25: quantile_forecast[0][1],
                0.5: quantile_forecast[0][2],
                0.75: quantile_forecast[0][3],
                0.9: quantile_forecast[0][4]
            }
        else:
            lower_bound = None
            upper_bound = None
            quantile_dict = None

        return ForecastResult(
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            quantiles=quantile_dict,
            model_name=self.name()
        )

    def name(self) -> str:
        return "TimesFM"

class TiReXForecaster(BaseForecaster):
    """TiReX xLSTM-based forecasting model"""

    def __init__(self):
        """Initialize TiReX model"""
        self.model = None
        self.context_data = None

        try:
            from tirex import load_model
            logger.info("Loading TiReX model from HuggingFace...")
            self.model = load_model("NX-AI/TiReX")
        except ImportError:
            logger.warning("TiReX not installed")
            logger.info("Install with: pip install git+https://github.com/NX-AI/tirex.git")

    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Store context for prediction"""
        self.context_data = data
        logger.info(f"TiReX context set with {len(data)} points")

    def predict(self, horizon: int, confidence_level: float = 0.95) -> ForecastResult:
        """Generate TiReX forecast"""
        if self.model is None:
            logger.error("TiReX model not initialized")
            return ForecastResult(
                point_forecast=np.zeros(horizon),
                model_name=self.name()
            )

        if self.context_data is None:
            raise ValueError("Must call fit() before predict()")

        # Prepare input tensor
        context = torch.tensor(self.context_data, dtype=torch.float32).unsqueeze(0)

        # Generate forecast
        with torch.no_grad():
            forecast = self.model.forecast(
                context=context,
                prediction_length=horizon
            )

        # Convert to numpy
        point_forecast = forecast.squeeze().cpu().numpy()

        return ForecastResult(
            point_forecast=point_forecast,
            model_name=self.name()
        )

    def name(self) -> str:
        return "TiReX"

class EnsembleForecaster:
    """Ensemble multiple foundation models for robust forecasting"""

    def __init__(self, models: Optional[List[BaseForecaster]] = None):
        """
        Initialize ensemble

        Args:
            models: List of forecasting models to ensemble
        """
        if models is None:
            # Default ensemble
            self.models = [
                ChronosForecaster("small"),
                # TimesFMForecaster(),  # Uncomment if installed
                # TiReXForecaster(),    # Uncomment if installed
            ]
        else:
            self.models = models

        logger.info(f"Ensemble initialized with {len(self.models)} models")

    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """Fit all models"""
        for model in self.models:
            try:
                model.fit(data, timestamps)
            except Exception as e:
                logger.error(f"Failed to fit {model.name()}: {e}")

    def predict(
        self,
        horizon: int,
        confidence_level: float = 0.95,
        aggregation: str = "mean"
    ) -> ForecastResult:
        """Generate ensemble forecast"""

        forecasts = []
        weights = []

        for model in self.models:
            try:
                result = model.predict(horizon, confidence_level)
                forecasts.append(result.point_forecast)

                # Simple equal weighting for now
                weights.append(1.0 / len(self.models))
            except Exception as e:
                logger.error(f"Failed to predict with {model.name()}: {e}")

        if not forecasts:
            raise ValueError("No models produced valid forecasts")

        # Aggregate forecasts
        forecasts_array = np.array(forecasts)
        weights_array = np.array(weights)

        if aggregation == "mean":
            ensemble_forecast = np.average(forecasts_array, axis=0, weights=weights_array)
        elif aggregation == "median":
            ensemble_forecast = np.median(forecasts_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Calculate ensemble uncertainty
        forecast_std = np.std(forecasts_array, axis=0)
        z_score = 1.96 if confidence_level == 0.95 else 2.58

        return ForecastResult(
            point_forecast=ensemble_forecast,
            lower_bound=ensemble_forecast - z_score * forecast_std,
            upper_bound=ensemble_forecast + z_score * forecast_std,
            model_name=f"Ensemble-{aggregation}",
            metrics={"n_models": len(forecasts)}
        )

class CloudCostForecaster:
    """Specialized forecaster for cloud cost optimization"""

    def __init__(self, ensemble: bool = True):
        """Initialize cloud cost forecaster"""
        if ensemble:
            self.forecaster = EnsembleForecaster()
        else:
            self.forecaster = ChronosForecaster("small")

        self.scaler_params = {}

    def prepare_data(self, df: pl.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for forecasting"""

        # Extract time series
        values = df[target_column].to_numpy()
        timestamps = df["timestamp"].to_numpy()

        # Normalize data (important for neural models)
        mean = np.mean(values)
        std = np.std(values)
        normalized_values = (values - mean) / (std + 1e-8)

        # Store scaling parameters
        self.scaler_params[target_column] = {"mean": mean, "std": std}

        return normalized_values, timestamps

    def forecast_cost(
        self,
        df: pl.DataFrame,
        target_column: str,
        horizon: int,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Generate cost forecast with business context"""

        # Prepare data
        values, timestamps = self.prepare_data(df, target_column)

        # Fit model
        self.forecaster.fit(values, timestamps)

        # Generate forecast
        result = self.forecaster.predict(horizon, confidence_level)

        # Denormalize predictions
        params = self.scaler_params[target_column]
        point_forecast = result.point_forecast * params["std"] + params["mean"]

        if result.lower_bound is not None:
            lower_bound = result.lower_bound * params["std"] + params["mean"]
            upper_bound = result.upper_bound * params["std"] + params["mean"]
        else:
            lower_bound = None
            upper_bound = None

        # Calculate business metrics
        total_predicted_cost = np.sum(point_forecast)
        avg_hourly_cost = np.mean(point_forecast)
        peak_cost = np.max(point_forecast)
        cost_trend = "increasing" if point_forecast[-1] > point_forecast[0] else "decreasing"

        return {
            "forecast": point_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "total_cost": total_predicted_cost,
            "avg_hourly_cost": avg_hourly_cost,
            "peak_cost": peak_cost,
            "trend": cost_trend,
            "model": result.model_name
        }

    def forecast_unit_economics(
        self,
        df: pl.DataFrame,
        horizon: int = 24,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Forecast multiple unit economics metrics"""

        if metrics is None:
            metrics = ["hourly_cost", "cpu_utilization", "memory_utilization", "efficiency_score"]

        results = {}

        for metric in metrics:
            if metric in df.columns:
                try:
                    forecast_result = self.forecast_cost(
                        df=df,
                        target_column=metric,
                        horizon=horizon,
                        confidence_level=0.95
                    )
                    results[metric] = forecast_result
                    logger.info(f"Forecasted {metric}: {forecast_result['trend']} trend")
                except Exception as e:
                    logger.error(f"Failed to forecast {metric}: {e}")

        return results

def demo_forecasting():
    """Demo advanced forecasting capabilities"""
    logger.info("Demonstrating advanced time series forecasting...")

    # Generate sample data
    dates = pl.date_range(
        datetime.now() - timedelta(days=30),
        datetime.now(),
        "1h",
        eager=True
    )

    # Simulate cloud cost data with patterns
    np.random.seed(42)
    trend = np.linspace(100, 150, len(dates))
    seasonal = 20 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    noise = np.random.normal(0, 5, len(dates))
    costs = trend + seasonal + noise

    # Create DataFrame
    df = pl.DataFrame({
        "timestamp": dates,
        "hourly_cost": costs,
        "cpu_utilization": 30 + 10 * np.random.randn(len(dates)),
        "memory_utilization": 40 + 5 * np.random.randn(len(dates)),
        "efficiency_score": 60 + 15 * np.random.randn(len(dates))
    })

    # Initialize forecaster
    forecaster = CloudCostForecaster(ensemble=True)

    # Generate forecasts
    results = forecaster.forecast_unit_economics(df, horizon=48)

    # Display results
    print("\n=== Cloud Cost Forecasting Results ===")
    for metric, result in results.items():
        print(f"\n{metric}:")
        print(f"  - 48-hour total: ${result['total_cost']:.2f}")
        print(f"  - Average: ${result['avg_hourly_cost']:.2f}/hour")
        print(f"  - Peak: ${result['peak_cost']:.2f}")
        print(f"  - Trend: {result['trend']}")
        print(f"  - Model: {result['model']}")

if __name__ == "__main__":
    demo_forecasting()