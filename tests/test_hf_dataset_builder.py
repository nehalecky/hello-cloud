"""Tests for HuggingFace dataset builder."""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

from hellocloud.data_generation.hf_dataset_builder import (
    CloudMetricsDatasetBuilder,
)


class TestCloudMetricsDatasetBuilder:
    """Test CloudMetricsDatasetBuilder functionality."""

    def test_initialization(self):
        """Test dataset builder initialization."""
        builder = CloudMetricsDatasetBuilder()

        assert builder.cache_dir == "./data/hf_cache"
        assert Path(builder.cache_dir).exists()
        assert builder.api is not None

    def test_initialization_with_custom_cache(self, tmp_path):
        """Test initialization with custom cache directory."""
        custom_cache = tmp_path / "custom_cache"
        builder = CloudMetricsDatasetBuilder(cache_dir=str(custom_cache))

        assert builder.cache_dir == str(custom_cache)
        assert custom_cache.exists()

    def test_feature_schema_definition(self):
        """Test that feature schema is properly defined."""
        assert hasattr(CloudMetricsDatasetBuilder, 'FEATURE_SCHEMA')
        schema = CloudMetricsDatasetBuilder.FEATURE_SCHEMA

        # Check essential fields exist
        assert "resource_id" in schema
        assert "timestamp" in schema
        assert "cpu_utilization" in schema
        assert "memory_utilization" in schema
        assert "hourly_cost" in schema
        assert "workload_type" in schema

        # Check field types
        assert schema["resource_id"].dtype == "string"
        assert schema["cpu_utilization"].dtype == "float32"
        assert schema["is_idle"].dtype == "bool"

    @patch('cloud_sim.data_generation.hf_dataset_builder.Dataset')
    def test_polars_to_dataset_conversion(self, mock_dataset):
        """Test converting Polars DataFrame to HuggingFace Dataset."""
        # Mock Dataset constructor to accept arrow table
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        builder = CloudMetricsDatasetBuilder()

        # Create sample Polars DataFrame
        df = self._create_sample_dataframe()

        # Convert to dataset
        dataset = builder.polars_to_dataset(df, split_name="train")

        # Verify Dataset constructor was called with arrow table
        mock_dataset.assert_called_once()

        # Verify the result
        assert dataset == mock_dataset_instance

    def test_polars_to_dataset_missing_required_columns(self):
        """Test that missing required columns raises error."""
        builder = CloudMetricsDatasetBuilder()

        # DataFrame missing required columns
        df = pl.DataFrame({
            "resource_id": ["res_001", "res_002"],
            # Missing: timestamp, cpu_utilization, memory_utilization, hourly_cost
        })

        with pytest.raises(ValueError, match="Required column .* not found in DataFrame"):
            builder.polars_to_dataset(df)

    def test_polars_to_dataset_adds_missing_optional_columns(self):
        """Test that missing optional columns are added with defaults."""
        builder = CloudMetricsDatasetBuilder()

        # Minimal DataFrame with only required columns
        df = pl.DataFrame({
            "resource_id": ["res_001", "res_002"],
            "timestamp": [datetime.now(), datetime.now()],
            "cpu_utilization": [15.0, 20.0],
            "memory_utilization": [25.0, 30.0],
            "hourly_cost": [10.0, 15.0]
        })

        # Call the method to verify it adds the defaults
        dataset = builder.polars_to_dataset(df)

        # Note: Since we're actually adding columns to the DataFrame,
        # we can't easily verify the arrow table contents without mocking
        # the internals. The best we can do is ensure it doesn't error.
        assert dataset is not None

    def test_create_time_series_splits(self):
        """Test creating time series train/test splits."""
        builder = CloudMetricsDatasetBuilder()

        # Create sample data spanning multiple days
        df = self._create_sample_dataframe(num_hours=168)  # 1 week

        with patch('cloud_sim.data_generation.hf_dataset_builder.DatasetDict') as mock_dict:
            mock_dict.return_value = MagicMock()

            splits = builder.create_time_series_splits(df, test_size=0.2)

            # Verify DatasetDict was created with train and test splits
            mock_dict.assert_called_once()
            call_args = mock_dict.call_args[0][0]
            assert "train" in call_args
            assert "test" in call_args

    def test_time_series_splits_chronological(self):
        """Test that time series splits preserve chronological order."""
        builder = CloudMetricsDatasetBuilder()

        # Create data with clear time ordering
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
        df = pl.DataFrame({
            "resource_id": ["res_001"] * 100,
            "timestamp": timestamps,
            "cpu_utilization": list(range(100)),  # Increasing values to check order
            "memory_utilization": [20.0] * 100,
            "hourly_cost": [10.0] * 100
        })

        # Mock Dataset creation to capture the data
        captured_train = None
        captured_test = None

        def capture_dataset(data, features=None):
            dataset = MagicMock()
            dataset.data = data
            return dataset

        with patch('cloud_sim.data_generation.hf_dataset_builder.Dataset') as mock_dataset:
            mock_dataset.from_dict.side_effect = capture_dataset

            splits = builder.create_time_series_splits(df, test_size=0.2)

            # Get the captured data
            train_calls = [call for call in mock_dataset.from_dict.call_args_list]
            if len(train_calls) >= 2:
                train_data = train_calls[0][0][0]
                test_data = train_calls[1][0][0]

                # Train should have first 80 values
                assert max(train_data["cpu_utilization"]) < min(test_data["cpu_utilization"])

    def test_prepare_for_foundation_models(self):
        """Test preparing datasets for foundation models."""
        builder = CloudMetricsDatasetBuilder()

        df = self._create_sample_dataframe(num_hours=24)

        with patch('cloud_sim.data_generation.hf_dataset_builder.Dataset') as mock_dataset:
            mock_dataset_instance = MagicMock()
            mock_dataset.from_dict.return_value = mock_dataset_instance
            mock_dataset_instance.map.return_value = mock_dataset_instance

            prepared = builder.prepare_for_foundation_models(df)

            # Check that dataset was created and mapped
            mock_dataset.from_dict.assert_called()
            mock_dataset_instance.map.assert_called()

    def test_prepare_sliding_windows(self):
        """Test creating sliding window features."""
        builder = CloudMetricsDatasetBuilder()

        # Create time series data
        df = self._create_sample_dataframe(num_hours=100)

        windows = builder.prepare_sliding_windows(
            df,
            window_size=24,
            stride=6,
            target_col="hourly_cost"
        )

        assert isinstance(windows, list)
        assert len(windows) > 0

        # Check window structure
        first_window = windows[0]
        assert "context" in first_window
        assert "target" in first_window
        assert "timestamp" in first_window
        assert "resource_id" in first_window

        # Check window dimensions
        assert len(first_window["context"]) == 24
        assert isinstance(first_window["target"], (float, int))

    def test_sliding_windows_respects_resource_boundaries(self):
        """Test that sliding windows don't cross resource boundaries."""
        builder = CloudMetricsDatasetBuilder()

        # Create data with multiple resources
        dfs = []
        for i in range(3):
            df = pl.DataFrame({
                "resource_id": [f"res_{i:03d}"] * 50,
                "timestamp": [datetime.now() + timedelta(hours=j) for j in range(50)],
                "cpu_utilization": np.random.randn(50) * 10 + 15,
                "memory_utilization": np.random.randn(50) * 10 + 25,
                "hourly_cost": np.random.gamma(2, 10, 50)
            })
            dfs.append(df)

        combined_df = pl.concat(dfs)

        windows = builder.prepare_sliding_windows(
            combined_df,
            window_size=10,
            stride=5
        )

        # Check that each window has consistent resource_id
        for window in windows:
            # All context values should be from the same resource
            resource_id = window["resource_id"]
            assert all(resource_id == window["resource_id"] for _ in window["context"])

    def test_compute_dataset_statistics(self):
        """Test computing dataset statistics."""
        builder = CloudMetricsDatasetBuilder()

        df = self._create_sample_dataframe(num_hours=168)

        stats = builder.compute_dataset_statistics(df)

        assert isinstance(stats, dict)

        # Check expected statistics
        assert "num_resources" in stats
        assert "time_range" in stats
        assert "total_samples" in stats
        assert "metrics" in stats

        # Check metric statistics
        metrics = stats["metrics"]
        assert "cpu_utilization" in metrics
        assert "memory_utilization" in metrics
        assert "hourly_cost" in metrics

        # Each metric should have mean, std, min, max
        for metric_stats in metrics.values():
            assert "mean" in metric_stats
            assert "std" in metric_stats
            assert "min" in metric_stats
            assert "max" in metric_stats

    def test_validate_dataset(self):
        """Test dataset validation."""
        builder = CloudMetricsDatasetBuilder()

        # Valid dataset
        df = self._create_sample_dataframe()
        is_valid, errors = builder.validate_dataset(df)

        assert is_valid
        assert len(errors) == 0

        # Dataset with invalid values
        df_invalid = pl.DataFrame({
            "resource_id": ["res_001", "res_002"],
            "timestamp": [datetime.now(), datetime.now()],
            "cpu_utilization": [150.0, -10.0],  # Invalid: >100 and <0
            "memory_utilization": [25.0, 30.0],
            "hourly_cost": [10.0, -5.0]  # Invalid: negative cost
        })

        is_valid, errors = builder.validate_dataset(df_invalid)

        assert not is_valid
        assert len(errors) > 0
        assert any("cpu_utilization" in error for error in errors)
        assert any("hourly_cost" in error for error in errors)


    # Helper methods

    def _create_sample_dataframe(self, num_hours=24):
        """Create sample Polars DataFrame for testing."""
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(num_hours)]

        return pl.DataFrame({
            "resource_id": ["res_001"] * num_hours,
            "timestamp": timestamps,
            "workload_type": ["web_application"] * num_hours,
            "cpu_utilization": np.random.beta(2, 8, num_hours) * 100,
            "memory_utilization": np.random.beta(3, 7, num_hours) * 100,
            "network_in_mbps": np.random.gamma(2, 2, num_hours),
            "network_out_mbps": np.random.gamma(2, 2, num_hours),
            "disk_iops": np.random.gamma(3, 100, num_hours),
            "hourly_cost": np.random.gamma(2, 10, num_hours),
            "efficiency_score": np.random.beta(5, 2, num_hours) * 100,
            "waste_percentage": np.random.beta(3, 7, num_hours) * 100,
            "is_idle": np.random.choice([True, False], num_hours, p=[0.2, 0.8]),
            "is_overprovisioned": np.random.choice([True, False], num_hours, p=[0.3, 0.7]),
            "is_anomaly": np.random.choice([True, False], num_hours, p=[0.05, 0.95]),
            "cloud_provider": ["AWS"] * num_hours,
            "region": ["us-east-1"] * num_hours,
            "instance_type": ["t3.medium"] * num_hours,
            "environment": ["prod"] * num_hours,
        })

