"""Tests for advanced time series forecasting models."""

import pytest
import numpy as np
import polars as pl
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from cloud_sim.ml_models.advanced_forecasting import (
    ForecastResult,
    BaseForecaster,
    ChronosForecaster,
    TimesFMForecaster,
    TiReXForecaster,
    EnsembleForecaster,
    CloudCostForecaster,
)


class TestForecastResult:
    """Test ForecastResult data class."""

    def test_basic_forecast_result(self):
        """Test creating a basic forecast result."""
        result = ForecastResult(
            point_forecast=np.array([10.0, 11.0, 12.0]),
            model_name="TestModel"
        )
        assert len(result.point_forecast) == 3
        assert result.model_name == "TestModel"
        assert result.lower_bound is None
        assert result.upper_bound is None

    def test_complete_forecast_result(self):
        """Test creating a complete forecast result with all fields."""
        result = ForecastResult(
            point_forecast=np.array([10.0, 11.0, 12.0]),
            lower_bound=np.array([9.0, 10.0, 11.0]),
            upper_bound=np.array([11.0, 12.0, 13.0]),
            quantiles={
                0.1: np.array([9.5, 10.5, 11.5]),
                0.9: np.array([10.5, 11.5, 12.5])
            },
            model_name="TestModel",
            metrics={"mape": 0.05, "rmse": 1.2}
        )

        assert len(result.point_forecast) == 3
        assert len(result.lower_bound) == 3
        assert len(result.upper_bound) == 3
        assert 0.1 in result.quantiles
        assert 0.9 in result.quantiles
        assert result.metrics["mape"] == 0.05


class TestChronosForecaster:
    """Test Amazon Chronos forecasting model."""

    @patch('cloud_sim.ml_models.advanced_forecasting.AutoModelForSeq2SeqLM')
    @patch('cloud_sim.ml_models.advanced_forecasting.AutoTokenizer')
    @patch('cloud_sim.ml_models.advanced_forecasting.ChronosPipeline')
    def test_initialization(self, mock_pipeline, mock_tokenizer, mock_model):
        """Test Chronos model initialization."""
        # Setup mocks
        mock_pipeline.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        forecaster = ChronosForecaster(model_size="small")

        assert forecaster.model_size == "small"
        assert forecaster.model_name == "amazon/chronos-t5-small"
        assert forecaster.context_data is None

        # Verify model loading was attempted
        mock_model.from_pretrained.assert_called_once_with("amazon/chronos-t5-small")
        mock_tokenizer.from_pretrained.assert_called_once_with("amazon/chronos-t5-small")

    def test_fit_stores_context(self):
        """Test that fit method stores context data."""
        with patch('cloud_sim.ml_models.advanced_forecasting.AutoModelForSeq2SeqLM'), \
             patch('cloud_sim.ml_models.advanced_forecasting.AutoTokenizer'), \
             patch('cloud_sim.ml_models.advanced_forecasting.ChronosPipeline'):

            forecaster = ChronosForecaster()
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            timestamps = np.array([datetime.now() + timedelta(hours=i) for i in range(5)])

            forecaster.fit(data, timestamps)

            assert forecaster.context_data is not None
            assert len(forecaster.context_data) == 5
            assert forecaster.timestamps is not None

    @patch('cloud_sim.ml_models.advanced_forecasting.ChronosPipeline')
    def test_predict_without_fit_raises_error(self, mock_pipeline):
        """Test that predict without fit raises error."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        forecaster = ChronosForecaster()

        with pytest.raises(ValueError, match="Must call fit"):
            forecaster.predict(horizon=10)

    @patch('cloud_sim.ml_models.advanced_forecasting.ChronosPipeline')
    @patch('cloud_sim.ml_models.advanced_forecasting.AutoModelForSeq2SeqLM')
    @patch('cloud_sim.ml_models.advanced_forecasting.AutoTokenizer')
    def test_predict_returns_forecast(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test that predict returns proper forecast result."""
        # Setup mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        # Mock forecast output
        mock_forecast = MagicMock()
        mock_forecast.quantile.return_value.squeeze.return_value.numpy.return_value = \
            np.array([10.0, 11.0, 12.0])
        mock_pipeline_instance.predict.return_value = mock_forecast

        forecaster = ChronosForecaster()
        forecaster.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        result = forecaster.predict(horizon=3, confidence_level=0.95)

        assert isinstance(result, ForecastResult)
        assert len(result.point_forecast) == 3
        assert result.model_name == "Chronos-small"
        assert result.quantiles is not None

    def test_model_sizes(self):
        """Test different model sizes."""
        for size in ["small", "base", "large"]:
            with patch('cloud_sim.ml_models.advanced_forecasting.AutoModelForSeq2SeqLM'), \
                 patch('cloud_sim.ml_models.advanced_forecasting.AutoTokenizer'), \
                 patch('cloud_sim.ml_models.advanced_forecasting.ChronosPipeline'):

                forecaster = ChronosForecaster(model_size=size)
                assert forecaster.model_size == size
                assert f"chronos-t5-{size}" in forecaster.model_name


class TestTimesFMForecaster:
    """Test Google's TimesFM forecasting model."""

    def test_initialization_without_timesfm(self):
        """Test initialization when TimesFM is not installed."""
        with patch('cloud_sim.ml_models.advanced_forecasting.timesfm', side_effect=ImportError):
            forecaster = TimesFMForecaster()
            assert forecaster.model is None
            assert forecaster.context_data is None

    @patch('cloud_sim.ml_models.advanced_forecasting.timesfm')
    def test_initialization_with_timesfm(self, mock_timesfm):
        """Test initialization when TimesFM is installed."""
        mock_model = MagicMock()
        mock_timesfm.TimesFM.return_value = mock_model

        forecaster = TimesFMForecaster()

        assert forecaster.model is not None
        mock_timesfm.TimesFM.assert_called_once()

    @patch('cloud_sim.ml_models.advanced_forecasting.timesfm')
    def test_load_checkpoint(self, mock_timesfm):
        """Test loading from checkpoint."""
        mock_model = MagicMock()
        mock_timesfm.TimesFM.return_value = mock_model

        forecaster = TimesFMForecaster(checkpoint_path="test.ckpt")

        mock_model.load_from_checkpoint.assert_called_once_with("test.ckpt")

    def test_fit_stores_context(self):
        """Test that fit stores context data."""
        forecaster = TimesFMForecaster()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        forecaster.fit(data)

        assert forecaster.context_data is not None
        assert len(forecaster.context_data) == 5

    def test_predict_without_model_returns_zeros(self):
        """Test that predict without model returns zero forecast."""
        forecaster = TimesFMForecaster()
        forecaster.fit(np.array([1.0, 2.0, 3.0]))

        result = forecaster.predict(horizon=5)

        assert len(result.point_forecast) == 5
        assert np.all(result.point_forecast == 0)
        assert result.model_name == "TimesFM"

    @patch('cloud_sim.ml_models.advanced_forecasting.timesfm')
    def test_predict_with_model(self, mock_timesfm):
        """Test prediction with initialized model."""
        # Setup mock
        mock_model = MagicMock()
        mock_timesfm.TimesFM.return_value = mock_model

        point_forecast = np.array([10.0, 11.0, 12.0])
        quantile_forecast = [
            [9.0, 9.5, 10.0, 10.5, 11.0],  # Quantiles for each time step
            [10.0, 10.5, 11.0, 11.5, 12.0],
            [11.0, 11.5, 12.0, 12.5, 13.0]
        ]
        mock_model.forecast.return_value = ([point_forecast], [quantile_forecast])

        forecaster = TimesFMForecaster()
        forecaster.fit(np.array([1.0, 2.0, 3.0]))

        result = forecaster.predict(horizon=3)

        assert len(result.point_forecast) == 3
        assert result.lower_bound is not None
        assert result.upper_bound is not None
        assert result.quantiles is not None


class TestTiReXForecaster:
    """Test TiReX xLSTM forecasting model."""

    def test_initialization_without_tirex(self):
        """Test initialization when TiReX is not installed."""
        with patch('cloud_sim.ml_models.advanced_forecasting.load_model', side_effect=ImportError):
            forecaster = TiReXForecaster()
            assert forecaster.model is None

    @patch('cloud_sim.ml_models.advanced_forecasting.load_model')
    def test_initialization_with_tirex(self, mock_load_model):
        """Test initialization when TiReX is installed."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        forecaster = TiReXForecaster()

        assert forecaster.model is not None
        mock_load_model.assert_called_once_with("NX-AI/TiReX")

    def test_fit_stores_context(self):
        """Test that fit stores context data."""
        forecaster = TiReXForecaster()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        forecaster.fit(data)

        assert forecaster.context_data is not None
        assert len(forecaster.context_data) == 5

    def test_predict_without_model_returns_zeros(self):
        """Test that predict without model returns zero forecast."""
        forecaster = TiReXForecaster()
        forecaster.fit(np.array([1.0, 2.0, 3.0]))

        result = forecaster.predict(horizon=5)

        assert len(result.point_forecast) == 5
        assert np.all(result.point_forecast == 0)
        assert result.model_name == "TiReX"

    @patch('cloud_sim.ml_models.advanced_forecasting.load_model')
    def test_predict_with_model(self, mock_load_model):
        """Test prediction with initialized model."""
        mock_model = MagicMock()
        mock_forecast = torch.tensor([[10.0, 11.0, 12.0]])
        mock_model.forecast.return_value = mock_forecast
        mock_load_model.return_value = mock_model

        forecaster = TiReXForecaster()
        forecaster.fit(np.array([1.0, 2.0, 3.0]))

        with patch('torch.no_grad'):
            result = forecaster.predict(horizon=3)

        assert len(result.point_forecast) == 3
        assert result.model_name == "TiReX"


class TestEnsembleForecaster:
    """Test ensemble forecasting."""

    @patch('cloud_sim.ml_models.advanced_forecasting.ChronosForecaster')
    def test_default_initialization(self, mock_chronos):
        """Test ensemble with default models."""
        mock_chronos_instance = MagicMock()
        mock_chronos.return_value = mock_chronos_instance

        ensemble = EnsembleForecaster()

        assert len(ensemble.models) == 1  # Only Chronos by default
        mock_chronos.assert_called_once_with("small")

    def test_custom_models(self):
        """Test ensemble with custom models."""
        mock_model1 = MagicMock(spec=BaseForecaster)
        mock_model2 = MagicMock(spec=BaseForecaster)

        ensemble = EnsembleForecaster(models=[mock_model1, mock_model2])

        assert len(ensemble.models) == 2
        assert mock_model1 in ensemble.models
        assert mock_model2 in ensemble.models

    def test_fit_all_models(self):
        """Test that fit is called on all models."""
        mock_model1 = MagicMock(spec=BaseForecaster)
        mock_model2 = MagicMock(spec=BaseForecaster)

        ensemble = EnsembleForecaster(models=[mock_model1, mock_model2])
        data = np.array([1.0, 2.0, 3.0])

        ensemble.fit(data)

        mock_model1.fit.assert_called_once_with(data, None)
        mock_model2.fit.assert_called_once_with(data, None)

    def test_predict_mean_aggregation(self):
        """Test prediction with mean aggregation."""
        # Create mock models
        mock_model1 = MagicMock(spec=BaseForecaster)
        mock_model1.name.return_value = "Model1"
        mock_model1.predict.return_value = ForecastResult(
            point_forecast=np.array([10.0, 11.0, 12.0]),
            model_name="Model1"
        )

        mock_model2 = MagicMock(spec=BaseForecaster)
        mock_model2.name.return_value = "Model2"
        mock_model2.predict.return_value = ForecastResult(
            point_forecast=np.array([12.0, 13.0, 14.0]),
            model_name="Model2"
        )

        ensemble = EnsembleForecaster(models=[mock_model1, mock_model2])
        result = ensemble.predict(horizon=3, aggregation="mean")

        # Should be mean of two models
        expected = np.array([11.0, 12.0, 13.0])
        np.testing.assert_array_almost_equal(result.point_forecast, expected)
        assert result.model_name == "Ensemble-mean"

    def test_predict_median_aggregation(self):
        """Test prediction with median aggregation."""
        # Create mock models
        models = []
        for i in range(3):
            mock_model = MagicMock(spec=BaseForecaster)
            mock_model.name.return_value = f"Model{i}"
            mock_model.predict.return_value = ForecastResult(
                point_forecast=np.array([10.0 + i, 11.0 + i, 12.0 + i]),
                model_name=f"Model{i}"
            )
            models.append(mock_model)

        ensemble = EnsembleForecaster(models=models)
        result = ensemble.predict(horizon=3, aggregation="median")

        # Should be median of three models
        expected = np.array([11.0, 12.0, 13.0])
        np.testing.assert_array_almost_equal(result.point_forecast, expected)
        assert result.model_name == "Ensemble-median"

    def test_no_valid_forecasts_raises_error(self):
        """Test that error is raised when no models produce valid forecasts."""
        mock_model = MagicMock(spec=BaseForecaster)
        mock_model.predict.side_effect = Exception("Model failed")
        mock_model.name.return_value = "FailingModel"

        ensemble = EnsembleForecaster(models=[mock_model])

        with pytest.raises(ValueError, match="No models produced valid forecasts"):
            ensemble.predict(horizon=3)


class TestCloudCostForecaster:
    """Test cloud cost forecasting functionality."""

    @patch('cloud_sim.ml_models.advanced_forecasting.EnsembleForecaster')
    def test_initialization_with_ensemble(self, mock_ensemble):
        """Test initialization with ensemble forecaster."""
        mock_ensemble_instance = MagicMock()
        mock_ensemble.return_value = mock_ensemble_instance

        forecaster = CloudCostForecaster(ensemble=True)

        assert forecaster.forecaster == mock_ensemble_instance
        mock_ensemble.assert_called_once()

    @patch('cloud_sim.ml_models.advanced_forecasting.ChronosForecaster')
    def test_initialization_without_ensemble(self, mock_chronos):
        """Test initialization with single forecaster."""
        mock_chronos_instance = MagicMock()
        mock_chronos.return_value = mock_chronos_instance

        forecaster = CloudCostForecaster(ensemble=False)

        assert forecaster.forecaster == mock_chronos_instance
        mock_chronos.assert_called_once_with("small")

    def test_prepare_data_normalization(self):
        """Test data preparation and normalization."""
        forecaster = CloudCostForecaster(ensemble=False)

        # Create sample data
        df = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(10)],
            "hourly_cost": [100.0 + i * 10 for i in range(10)]
        })

        normalized, timestamps = forecaster.prepare_data(df, "hourly_cost")

        # Check normalization
        assert len(normalized) == 10
        assert len(timestamps) == 10
        assert abs(np.mean(normalized)) < 0.1  # Should be close to 0
        assert abs(np.std(normalized) - 1.0) < 0.1  # Should be close to 1

        # Check scaling parameters were stored
        assert "hourly_cost" in forecaster.scaler_params
        assert "mean" in forecaster.scaler_params["hourly_cost"]
        assert "std" in forecaster.scaler_params["hourly_cost"]

    def test_forecast_cost(self):
        """Test cost forecasting with business metrics."""
        # Create mock forecaster
        mock_forecaster = MagicMock()
        mock_forecaster.fit = MagicMock()
        mock_forecaster.predict.return_value = ForecastResult(
            point_forecast=np.array([0.0, 0.5, 1.0]),  # Normalized values
            lower_bound=np.array([-0.5, 0.0, 0.5]),
            upper_bound=np.array([0.5, 1.0, 1.5]),
            model_name="TestModel"
        )

        forecaster = CloudCostForecaster(ensemble=False)
        forecaster.forecaster = mock_forecaster

        # Create sample data
        df = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(10)],
            "hourly_cost": [100.0] * 10  # Constant for simplicity
        })

        result = forecaster.forecast_cost(df, "hourly_cost", horizon=3)

        assert "forecast" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "total_cost" in result
        assert "avg_hourly_cost" in result
        assert "peak_cost" in result
        assert "trend" in result
        assert "model" in result

        # Check denormalization happened
        assert result["forecast"][0] != 0.0  # Should be denormalized
        assert result["total_cost"] > 0
        assert result["avg_hourly_cost"] > 0
        assert result["peak_cost"] > 0

    def test_forecast_unit_economics(self):
        """Test forecasting multiple unit economics metrics."""
        # Create mock forecaster
        mock_forecaster = MagicMock()
        mock_forecaster.fit = MagicMock()
        mock_forecaster.predict.return_value = ForecastResult(
            point_forecast=np.array([0.0, 0.5, 1.0]),
            model_name="TestModel"
        )

        forecaster = CloudCostForecaster(ensemble=False)
        forecaster.forecaster = mock_forecaster

        # Create sample data with multiple metrics
        df = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(10)],
            "hourly_cost": [100.0] * 10,
            "cpu_utilization": [15.0] * 10,
            "memory_utilization": [25.0] * 10,
            "efficiency_score": [70.0] * 10,
            "other_metric": [50.0] * 10  # This won't be forecasted by default
        })

        results = forecaster.forecast_unit_economics(df, horizon=3)

        # Should have forecasts for default metrics
        assert "hourly_cost" in results
        assert "cpu_utilization" in results
        assert "memory_utilization" in results
        assert "efficiency_score" in results
        assert "other_metric" not in results

        # Each result should have expected structure
        for metric, result in results.items():
            assert "forecast" in result
            assert "total_cost" in result  # Even for non-cost metrics (sum)
            assert "trend" in result

    def test_trend_detection(self):
        """Test that trend is correctly identified."""
        mock_forecaster = MagicMock()
        mock_forecaster.fit = MagicMock()

        # Test increasing trend
        mock_forecaster.predict.return_value = ForecastResult(
            point_forecast=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            model_name="TestModel"
        )

        forecaster = CloudCostForecaster(ensemble=False)
        forecaster.forecaster = mock_forecaster

        df = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(5)],
            "hourly_cost": [100.0] * 5
        })

        result = forecaster.forecast_cost(df, "hourly_cost", horizon=5)
        assert result["trend"] == "increasing"

        # Test decreasing trend
        mock_forecaster.predict.return_value = ForecastResult(
            point_forecast=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            model_name="TestModel"
        )

        result = forecaster.forecast_cost(df, "hourly_cost", horizon=5)
        assert result["trend"] == "decreasing"


@pytest.mark.parametrize("horizon", [1, 24, 168, 720])
def test_forecast_horizons(horizon):
    """Test different forecast horizons."""
    with patch('cloud_sim.ml_models.advanced_forecasting.ChronosForecaster'):
        forecaster = CloudCostForecaster(ensemble=False)

        # Mock the internal forecaster
        mock_result = ForecastResult(
            point_forecast=np.random.randn(horizon),
            model_name="TestModel"
        )
        forecaster.forecaster.predict = MagicMock(return_value=mock_result)
        forecaster.forecaster.fit = MagicMock()

        df = pl.DataFrame({
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(100)],
            "hourly_cost": np.random.gamma(2, 50, 100)
        })

        result = forecaster.forecast_cost(df, "hourly_cost", horizon=horizon)

        assert len(result["forecast"]) == horizon


def test_demo_forecasting():
    """Test the demo forecasting function."""
    with patch('cloud_sim.ml_models.advanced_forecasting.CloudCostForecaster') as mock_forecaster:
        from cloud_sim.ml_models.advanced_forecasting import demo_forecasting

        # Mock the forecaster
        mock_instance = MagicMock()
        mock_forecaster.return_value = mock_instance
        mock_instance.forecast_unit_economics.return_value = {
            "hourly_cost": {
                "total_cost": 1000.0,
                "avg_hourly_cost": 20.0,
                "peak_cost": 50.0,
                "trend": "increasing",
                "model": "TestModel"
            }
        }

        # Should run without errors
        demo_forecasting()

        # Verify forecaster was created and used
        mock_forecaster.assert_called_once_with(ensemble=True)
        mock_instance.forecast_unit_economics.assert_called_once()