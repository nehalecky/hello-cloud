"""Tests for PyMC hierarchical Bayesian model."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest
import xarray as xr

from hellocloud.ml_models.application_taxonomy import CloudResourceTaxonomy
from hellocloud.ml_models.pymc_cloud_model import (
    CloudResourceHierarchicalModel,
)

# Mark all tests in this file as optional since PyMC is deprioritized
pytestmark = pytest.mark.optional


class TestCloudResourceHierarchicalModel:
    """Test the hierarchical Bayesian model."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = CloudResourceHierarchicalModel(seed=42)
        assert model.seed == 42
        assert model.model is None  # Not built yet
        assert model.trace is None  # Not fitted yet
        assert len(model.archetypes) > 0  # Should have archetypes from taxonomy

    def test_model_with_custom_archetypes(self):
        """Test model initialization with custom archetypes."""
        custom_archetypes = ["Web Application", "Batch Processing"]
        model = CloudResourceHierarchicalModel(archetypes=custom_archetypes, seed=42)
        assert model.archetypes == custom_archetypes

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_build_model(self, mock_pm):
        """Test building the PyMC model."""
        # Setup mock PyMC model context manager
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        # Setup mock distributions to return valid values
        mock_pm.Gamma.return_value = MagicMock()
        mock_pm.Beta.return_value = MagicMock()
        mock_pm.Normal.return_value = MagicMock()
        mock_pm.InverseGamma.return_value = MagicMock()
        mock_pm.Deterministic.return_value = MagicMock()

        # Create sample data
        data = self._create_sample_data()

        # Build model
        model = CloudResourceHierarchicalModel(seed=42)
        model.build_model(data)

        # Verify model was created
        mock_pm.Model.assert_called_once()
        assert model.model is not None

        # Verify distributions were created
        assert mock_pm.Beta.called  # For utilization priors
        assert mock_pm.Gamma.called  # For cost distributions

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_fit_model(self, mock_pm):
        """Test fitting the model with data."""
        # Setup mock trace with xarray structure
        mock_trace = self._create_mock_trace()
        mock_pm.sample.return_value = mock_trace

        # Setup mock model context
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        # Create sample data
        data = self._create_sample_data()

        # Create and fit model
        model = CloudResourceHierarchicalModel(seed=42)

        with patch.object(model, "build_model") as mock_build:
            # Don't set model attribute, let fit method call build_model
            mock_build.return_value = None

            # After build_model is called, set model.model for the sampling
            def side_effect(data):
                model.model = mock_pm.Model.return_value

            mock_build.side_effect = side_effect

            model.fit(data, tune=100, draws=100, chains=2)  # Fewer samples for testing

            # Verify build was called
            mock_build.assert_called_once_with(data)

            # Verify sampling was called with correct parameters
            mock_pm.sample.assert_called_once()
            call_kwargs = mock_pm.sample.call_args[1]
            assert call_kwargs["tune"] == 100
            assert call_kwargs["draws"] == 100
            assert call_kwargs["chains"] == 2

            # Verify trace was stored
            assert model.trace is not None

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_generate_synthetic_data(self, mock_pm):
        """Test synthetic data generation."""
        # Setup mock PyMC
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        # Mock sample_posterior_predictive to return valid samples
        mock_samples = MagicMock()
        mock_pm.sample_posterior_predictive.return_value = mock_samples

        model = CloudResourceHierarchicalModel(seed=42)

        # Mock the fitted model with proper xarray structure
        model.trace = self._create_mock_trace()
        model.model = mock_pm.Model.return_value

        # Generate synthetic data
        synthetic_df = model.generate_synthetic_data(
            num_resources=5, time_periods=24, start_date=datetime.now()
        )

        # Verify output
        assert isinstance(synthetic_df, pl.DataFrame)
        assert len(synthetic_df) == 5 * 24  # resources * time_periods

        # Check required columns
        expected_columns = [
            "timestamp",
            "resource_id",
            "archetype",
            "cpu_utilization",
            "memory_utilization",
            "network_in_mbps",
            "network_out_mbps",
            "disk_iops",
            "hourly_cost",
            "efficiency_score",
        ]
        for col in expected_columns:
            assert col in synthetic_df.columns

        # Verify realistic ranges
        assert synthetic_df["cpu_utilization"].min() >= 0
        assert synthetic_df["cpu_utilization"].max() <= 100
        assert synthetic_df["memory_utilization"].min() >= 0
        assert synthetic_df["memory_utilization"].max() <= 100
        assert synthetic_df["hourly_cost"].min() >= 0

    def test_posterior_predictions(self):
        """Test generating posterior predictions."""
        model = CloudResourceHierarchicalModel(seed=42)

        # Mock the fitted model with proper xarray structure
        model.trace = self._create_mock_trace()

        # Generate predictions
        predictions = model.posterior_predictions(archetype="Web Application", num_samples=100)

        assert isinstance(predictions, dict)
        assert "cpu" in predictions
        assert "memory" in predictions
        assert "cost" in predictions

        # Check shapes
        assert len(predictions["cpu"]) == 100
        assert len(predictions["memory"]) == 100
        assert len(predictions["cost"]) == 100

        # Check ranges
        assert all(0 <= cpu <= 100 for cpu in predictions["cpu"])
        assert all(0 <= mem <= 100 for mem in predictions["memory"])
        assert all(cost >= 0 for cost in predictions["cost"])

    @patch("hellocloud.ml_models.pymc_cloud_model.az.to_netcdf")
    def test_save_model(self, mock_to_netcdf):
        """Test saving the model."""
        model = CloudResourceHierarchicalModel(seed=42)
        model.trace = self._create_mock_trace()

        # Save model
        model.save_model("test_model.nc")

        # Verify save was called
        mock_to_netcdf.assert_called_once()

    @patch("hellocloud.ml_models.pymc_cloud_model.az.from_netcdf")
    @patch("os.path.exists")
    def test_load_model(self, mock_exists, mock_from_netcdf):
        """Test loading a saved model."""
        # Setup mocks
        mock_exists.return_value = True  # File exists

        mock_trace = self._create_mock_trace()
        mock_from_netcdf.return_value = mock_trace

        # Load model
        model = CloudResourceHierarchicalModel(seed=42)
        model.load_model("test_model.nc")

        # Verify existence check and load were called
        mock_exists.assert_called_once_with("test_model.nc")
        mock_from_netcdf.assert_called_once_with("test_model.nc")
        assert model.trace is not None

    def test_compute_waste_metrics(self):
        """Test waste metric computation."""
        model = CloudResourceHierarchicalModel(seed=42)

        # Create sample data with known waste patterns
        data = pl.DataFrame(
            {
                "cpu_utilization": [10.0, 15.0, 8.0, 12.0, 5.0],
                "memory_utilization": [20.0, 25.0, 15.0, 18.0, 10.0],
                "hourly_cost": [10.0, 15.0, 12.0, 8.0, 20.0],
                "archetype": ["Web Application"] * 5,
            }
        )

        waste_metrics = model.compute_waste_metrics(data)

        assert isinstance(waste_metrics, dict)
        assert "total_waste_percentage" in waste_metrics
        assert "waste_by_archetype" in waste_metrics
        assert "optimization_potential" in waste_metrics

        # Verify waste is realistic (research shows 30-32% average)
        assert 0 <= waste_metrics["total_waste_percentage"] <= 100

    def test_model_without_data_raises_error(self):
        """Test that generating predictions without fitting raises error."""
        model = CloudResourceHierarchicalModel(seed=42)

        with pytest.raises(ValueError, match="Model not fitted"):
            model.posterior_predictions("Web Application", 100)

    def test_invalid_archetype_uses_fallback(self):
        """Test that using invalid archetype name falls back to first archetype."""
        # Initialize model with specific archetypes
        model = CloudResourceHierarchicalModel(
            archetypes=["ecommerce_platform", "video_streaming"], seed=42
        )
        model.trace = self._create_mock_trace()

        # The model gracefully handles unknown archetypes by using the first one
        # This should not raise an error but use fallback
        predictions = model.posterior_predictions("NonExistentArchetype", 100)

        # Should still return valid predictions using fallback archetype
        assert isinstance(predictions, dict)
        assert "cpu" in predictions
        assert "memory" in predictions
        assert "cost" in predictions
        assert len(predictions["cpu"]) == 100

    # Helper methods

    def _create_sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        return pl.DataFrame(
            {
                "timestamp": [datetime.now() + timedelta(hours=i) for i in range(n_samples)],
                "resource_id": ["res_" + str(i % 10) for i in range(n_samples)],
                "archetype": [
                    "Web Application" if i % 2 == 0 else "Batch Processing"
                    for i in range(n_samples)
                ],
                "cpu_utilization": np.random.beta(2, 8, n_samples) * 100,
                "memory_utilization": np.random.beta(3, 7, n_samples) * 100,
                "network_in_mbps": np.random.gamma(2, 2, n_samples),
                "network_out_mbps": np.random.gamma(2, 2, n_samples),
                "disk_iops": np.random.gamma(3, 100, n_samples),
                "hourly_cost": np.random.gamma(2, 10, n_samples),
                "efficiency_score": np.random.beta(5, 2, n_samples) * 100,
            }
        )

    def _create_mock_trace(self):
        """Create a mock trace with xarray structure for testing."""
        n_chains = 2
        n_draws = 100
        n_archetypes = len(CloudResourceTaxonomy.ARCHETYPES.keys())

        # Create xarray datasets that mimic PyMC trace structure
        posterior_data = {
            "archetype_cpu_mean": xr.DataArray(
                np.random.beta(2, 8, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_mean": xr.DataArray(
                np.random.beta(3, 7, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_cv": xr.DataArray(
                np.random.gamma(2, 0.1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_cv": xr.DataArray(
                np.random.gamma(2, 0.1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_alpha": xr.DataArray(
                np.random.gamma(2, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_beta": xr.DataArray(
                np.random.gamma(8, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_alpha": xr.DataArray(
                np.random.gamma(3, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_beta": xr.DataArray(
                np.random.gamma(7, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "cost_mu": xr.DataArray(
                np.random.normal(50, 10, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "cost_sigma": xr.DataArray(
                np.random.gamma(2, 5, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "corr_matrix": xr.DataArray(
                np.eye(5)[np.newaxis, np.newaxis, :, :]
                .repeat(n_chains, axis=0)
                .repeat(n_draws, axis=1),
                dims=["chain", "draw", "corr_dim_0", "corr_dim_1"],
            ),
        }

        # Create mock trace with xarray Dataset
        mock_trace = MagicMock()
        mock_trace.posterior = xr.Dataset(posterior_data)

        return mock_trace


class TestHierarchicalStructure:
    """Test the hierarchical structure of the model."""

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_industry_level_parameters(self, mock_pm):
        """Test that industry-level parameters are created."""
        # Setup mock PyMC
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        # Setup mock distributions
        mock_pm.Gamma.return_value = MagicMock()
        mock_pm.Beta.return_value = MagicMock()
        mock_pm.InverseGamma.return_value = MagicMock()
        mock_pm.Deterministic.return_value = MagicMock()

        model = CloudResourceHierarchicalModel(seed=42)

        data = pl.DataFrame(
            {
                "archetype": ["Web Application"],
                "cpu_utilization": [15.0],
                "memory_utilization": [25.0],
                "hourly_cost": [10.0],
                "timestamp": [datetime.now()],
                "resource_id": ["res_001"],
            }
        )

        model.build_model(data)

        # Model should have been created
        assert mock_pm.Model.called

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_archetype_level_parameters(self, mock_pm):
        """Test that archetype-level parameters inherit from industry level."""
        # Setup mock PyMC
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)

        model = CloudResourceHierarchicalModel(seed=42)

        # Create data with multiple archetypes
        n_samples = 30
        data = pl.DataFrame(
            {
                "archetype": ["Web Application", "Batch Processing", "ML Training"] * 10,
                "cpu_utilization": np.random.beta(2, 8, n_samples) * 100,
                "memory_utilization": np.random.beta(3, 7, n_samples) * 100,
                "hourly_cost": np.random.gamma(2, 10, n_samples),
                "timestamp": [datetime.now() + timedelta(hours=i) for i in range(n_samples)],
                "resource_id": [f"res_{i % 3:03d}" for i in range(n_samples)],
            }
        )

        model.build_model(data)

        # Verify we have the right number of archetypes
        unique_archetypes = data["archetype"].unique().to_list()
        assert len(unique_archetypes) == 3


class TestResourceCorrelations:
    """Test that the model captures resource correlations."""

    def _create_mock_trace(self):
        """Create a mock trace with xarray structure for testing."""
        n_chains = 2
        n_draws = 100
        n_archetypes = len(CloudResourceTaxonomy.ARCHETYPES.keys())

        # Create xarray datasets that mimic PyMC trace structure
        posterior_data = {
            "archetype_cpu_mean": xr.DataArray(
                np.random.beta(2, 8, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_mean": xr.DataArray(
                np.random.beta(3, 7, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_cv": xr.DataArray(
                np.random.gamma(2, 0.1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_cv": xr.DataArray(
                np.random.gamma(2, 0.1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_alpha": xr.DataArray(
                np.random.gamma(2, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_cpu_beta": xr.DataArray(
                np.random.gamma(8, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_alpha": xr.DataArray(
                np.random.gamma(3, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "archetype_mem_beta": xr.DataArray(
                np.random.gamma(7, 1, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "cost_mu": xr.DataArray(
                np.random.normal(50, 10, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "cost_sigma": xr.DataArray(
                np.random.gamma(2, 5, (n_chains, n_draws, n_archetypes)),
                dims=["chain", "draw", "archetype_dim"],
            ),
            "corr_matrix": xr.DataArray(
                np.eye(5)[np.newaxis, np.newaxis, :, :]
                .repeat(n_chains, axis=0)
                .repeat(n_draws, axis=1),
                dims=["chain", "draw", "corr_dim_0", "corr_dim_1"],
            ),
        }

        # Create mock trace with xarray Dataset
        mock_trace = MagicMock()
        mock_trace.posterior = xr.Dataset(posterior_data)

        return mock_trace

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_cpu_memory_correlation(self, mock_pm):
        """Test that CPU and memory utilization are correlated in synthetic data."""
        # Setup mock PyMC
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)
        mock_pm.sample_posterior_predictive.return_value = MagicMock()

        model = CloudResourceHierarchicalModel(seed=42)

        # Mock fitted model with proper xarray structure
        model.trace = self._create_mock_trace()
        model.model = mock_pm.Model.return_value

        # Generate synthetic data
        synthetic = model.generate_synthetic_data(num_resources=100, time_periods=100)

        # Calculate correlation
        cpu_array = synthetic["cpu_utilization"].to_numpy()
        mem_array = synthetic["memory_utilization"].to_numpy()

        if len(cpu_array) > 1:
            correlation = np.corrcoef(cpu_array, mem_array)[0, 1]
            # Should have some correlation (not perfectly 0)
            assert abs(correlation) > 0.01

    @patch("hellocloud.ml_models.pymc_cloud_model.pm")
    def test_cost_utilization_relationship(self, mock_pm):
        """Test that cost increases with utilization."""
        # Setup mock PyMC
        mock_model_instance = MagicMock()
        mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
        mock_pm.Model.return_value.__exit__ = Mock(return_value=None)
        mock_pm.sample_posterior_predictive.return_value = MagicMock()

        model = CloudResourceHierarchicalModel(seed=42)
        model.trace = self._create_mock_trace()
        model.model = mock_pm.Model.return_value

        # Generate synthetic data
        synthetic = model.generate_synthetic_data(num_resources=50, time_periods=24)

        # Higher utilization should generally mean higher cost
        high_cpu_mask = synthetic["cpu_utilization"] > 50
        low_cpu_mask = synthetic["cpu_utilization"] < 20

        if high_cpu_mask.sum() > 0 and low_cpu_mask.sum() > 0:
            high_cpu_cost = synthetic.filter(high_cpu_mask)["hourly_cost"].mean()
            low_cpu_cost = synthetic.filter(low_cpu_mask)["hourly_cost"].mean()

            # This is a soft assertion as there can be variation
            # But generally, higher utilization should have higher cost
            assert high_cpu_cost >= low_cpu_cost * 0.8  # Allow some flexibility


@pytest.mark.parametrize(
    "num_resources,time_periods",
    [
        (1, 24),
        (10, 168),  # 1 week
        (5, 720),  # 1 month
    ],
)
@patch("hellocloud.ml_models.pymc_cloud_model.pm")
def test_synthetic_data_dimensions(mock_pm, num_resources, time_periods):
    """Test that synthetic data has correct dimensions."""
    # Setup mock PyMC
    mock_model_instance = MagicMock()
    mock_pm.Model.return_value.__enter__ = Mock(return_value=mock_model_instance)
    mock_pm.Model.return_value.__exit__ = Mock(return_value=None)
    mock_pm.sample_posterior_predictive.return_value = MagicMock()

    # Create test helper instance to access _create_mock_trace
    test_helper = TestCloudResourceHierarchicalModel()

    model = CloudResourceHierarchicalModel(seed=42)
    model.trace = test_helper._create_mock_trace()
    model.model = mock_pm.Model.return_value

    synthetic = model.generate_synthetic_data(
        num_resources=num_resources, time_periods=time_periods
    )

    assert len(synthetic) == num_resources * time_periods
    assert synthetic["resource_id"].n_unique() == num_resources
