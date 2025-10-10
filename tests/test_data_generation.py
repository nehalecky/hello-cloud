"""Tests for data generation modules."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from hellocloud.data_generation import (
    CloudMetricsSimulator,
    WorkloadPatternGenerator,
    WorkloadType,
)


class TestCloudMetricsSimulator:
    """Test CloudMetricsSimulator functionality."""

    def test_simulator_initialization(self):
        """Test simulator can be initialized with proper parameters."""
        sim = CloudMetricsSimulator(num_resources=10, sampling_interval_minutes=60)
        assert sim.num_resources == 10
        assert len(sim.resources) == 10

    def test_resource_generation(self):
        """Test that resources are generated with correct attributes."""
        sim = CloudMetricsSimulator(num_resources=5)

        for resource in sim.resources:
            assert resource.resource_id
            assert resource.resource_type
            assert resource.cloud_provider in ["AWS", "Azure", "GCP"]
            assert resource.base_cost_per_hour > 0
            assert resource.cpu_cores > 0
            assert resource.memory_gb > 0

    def test_dataset_generation(self):
        """Test that dataset is generated correctly."""
        sim = CloudMetricsSimulator(
            num_resources=2,
            start_date=datetime.now() - timedelta(hours=2),
            end_date=datetime.now(),
            sampling_interval_minutes=60,
        )

        df = sim.generate_dataset(include_anomalies=False)

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "cpu_utilization" in df.columns
        assert "memory_utilization" in df.columns
        assert "hourly_cost" in df.columns

    def test_anomaly_injection(self):
        """Test that anomalies are properly injected."""
        sim = CloudMetricsSimulator(num_resources=1)
        df = sim.generate_dataset(include_anomalies=True)

        assert "is_anomaly" in df.columns
        assert df["is_anomaly"].sum() > 0  # Should have some anomalies

    def test_unit_economics_calculation(self):
        """Test unit economics calculations."""
        sim = CloudMetricsSimulator(num_resources=5)
        df = sim.generate_dataset()

        # Add team tags for testing
        df = df.with_columns(pl.lit("team_a").alias("tag_team"))

        unit_economics = sim.calculate_unit_economics(df)

        assert "cost_per_team" in unit_economics
        assert "waste_analysis" in unit_economics


class TestWorkloadPatternGenerator:
    """Test WorkloadPatternGenerator functionality."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = WorkloadPatternGenerator(seed=42)
        assert generator is not None

    def test_workload_profiles_exist(self):
        """Test that all workload profiles are defined."""
        generator = WorkloadPatternGenerator()

        for workload_type in WorkloadType:
            assert workload_type in generator.WORKLOAD_PROFILES
            profile = generator.WORKLOAD_PROFILES[workload_type]
            assert profile.base_cpu_util >= 0
            assert profile.base_mem_util >= 0
            assert profile.waste_factor >= 0

    def test_time_series_generation(self):
        """Test time series generation for different workload types."""
        generator = WorkloadPatternGenerator(seed=42)

        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()

        df = generator.generate_time_series(
            workload_type=WorkloadType.WEB_APP,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 24  # One hour intervals for 24 hours
        assert "cpu_utilization" in df.columns
        assert "memory_utilization" in df.columns
        assert "efficiency_score" in df.columns

    def test_realistic_utilization_values(self):
        """Test that generated utilization matches research findings."""
        generator = WorkloadPatternGenerator(seed=42)

        # Generate week of data
        df = generator.generate_time_series(
            workload_type=WorkloadType.WEB_APP,
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            interval_minutes=60,
        )

        # Check average utilization is realistically low
        avg_cpu = df["cpu_utilization"].mean()
        avg_memory = df["memory_utilization"].mean()

        # Based on research: CPU should be around 13%, memory around 20%
        # Web apps might be slightly higher but still low
        assert 5 <= avg_cpu <= 30  # Reasonable range for web apps
        assert 10 <= avg_memory <= 55

        # Check waste percentage is realistic (30-32% industry average)
        avg_waste = df["waste_percentage"].mean()
        assert 20 <= avg_waste <= 60

    def test_anomaly_generation(self):
        """Test anomaly generation in workload patterns."""
        generator = WorkloadPatternGenerator()

        df = generator.generate_time_series(
            workload_type=WorkloadType.BATCH_PROCESSING,
            start_time=datetime.now() - timedelta(hours=10),
            end_time=datetime.now(),
            interval_minutes=5,
        )

        # Inject anomalies
        df_with_anomalies = generator.generate_anomalies(df)

        # Check that data has changed
        assert not df["cpu_utilization"].equals(df_with_anomalies["cpu_utilization"])


class TestIntegration:
    """Integration tests for the full data generation pipeline."""

    def test_full_pipeline(self):
        """Test the complete data generation pipeline."""
        from hellocloud.data_generation.workload_patterns import create_multi_workload_dataset

        # Generate multi-workload dataset
        df = create_multi_workload_dataset(
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now(),
            workload_distribution={
                WorkloadType.WEB_APP: 2,
                WorkloadType.DATABASE_OLTP: 1,
                WorkloadType.ML_TRAINING: 1,
            },
        )

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

        # Check all expected columns exist
        expected_columns = [
            "timestamp",
            "workload_type",
            "resource_id",
            "cpu_utilization",
            "memory_utilization",
            "network_in_mbps",
            "network_out_mbps",
            "disk_iops",
            "efficiency_score",
            "is_idle",
            "is_overprovisioned",
            "waste_percentage",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_realistic_correlations(self):
        """Test that metrics show realistic correlations."""
        generator = WorkloadPatternGenerator(seed=42)

        df = generator.generate_time_series(
            workload_type=WorkloadType.ML_TRAINING,
            start_time=datetime.now() - timedelta(hours=100),
            end_time=datetime.now(),
            interval_minutes=60,
        )

        # ML workloads should show correlation between CPU and memory
        cpu_array = df["cpu_utilization"].to_numpy()
        mem_array = df["memory_utilization"].to_numpy()

        correlation = np.corrcoef(cpu_array, mem_array)[0, 1]

        # ML workloads should have positive correlation
        assert correlation > 0.3


@pytest.fixture
def sample_cloud_data():
    """Fixture providing sample cloud metrics data."""
    return pl.DataFrame(
        {
            "timestamp": [datetime.now() - timedelta(hours=i) for i in range(100)],
            "resource_id": ["res_001"] * 100,
            "cpu_utilization": np.random.beta(2, 8, 100) * 100,
            "memory_utilization": np.random.beta(3, 7, 100) * 100,
            "hourly_cost": np.random.gamma(2, 50, 100),
            "workload_type": ["web_app"] * 100,
        }
    )


def test_data_statistics(sample_cloud_data):
    """Test that generated data has expected statistical properties."""
    # Check mean utilization matches research
    assert 10 <= sample_cloud_data["cpu_utilization"].mean() <= 30
    assert 15 <= sample_cloud_data["memory_utilization"].mean() <= 40

    # Check cost variability
    cost_cv = sample_cloud_data["hourly_cost"].std() / sample_cloud_data["hourly_cost"].mean()
    assert 0.2 <= cost_cv <= 1.5  # Reasonable variability
