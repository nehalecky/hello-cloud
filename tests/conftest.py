"""Shared fixtures and configuration for tests."""

import os

# MUST set MPLBACKEND before any matplotlib imports
os.environ["MPLBACKEND"] = "Agg"

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import polars as pl
import pytest

# Check if torch is available (for GP tests)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip GP test collection if torch not available
if not TORCH_AVAILABLE:
    collect_ignore_glob = ["**/test_gaussian_process_*.py"]

from hellocloud.modeling.application_taxonomy import (
    ApplicationArchetype,
    ApplicationDomain,
    CostProfile,
    OptimizationPotential,
    ResourcePattern,
    ScalingBehavior,
)

# Test configuration
pytest.TEST_SEED = 42
pytest.TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Automatically set random seed for reproducible tests."""
    np.random.seed(pytest.TEST_SEED)
    if TORCH_AVAILABLE:
        torch.manual_seed(pytest.TEST_SEED)
    yield
    # Reset after test
    np.random.seed(None)
    if TORCH_AVAILABLE:
        torch.manual_seed(torch.seed())


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    from hellocloud.spark import get_spark_session

    spark = get_spark_session(app_name="pytest")
    yield spark
    spark.stop()


@pytest.fixture(autouse=True)
def configure_altair_renderer():
    """Configure Altair to use non-interactive renderer for testing."""
    try:
        import altair as alt

        alt.renderers.enable("json")  # Non-interactive renderer for testing
        alt.data_transformers.disable_max_rows()  # Allow large datasets in tests
    except ImportError:
        pass  # Altair not installed, skip configuration
    yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


# Data Generation Fixtures


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data."""
    num_hours = 168  # 1 week
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(num_hours)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "resource_id": ["res_001"] * num_hours,
            "cpu_utilization": np.random.beta(2, 8, num_hours) * 100,
            "memory_utilization": np.random.beta(3, 7, num_hours) * 100,
            "network_in_mbps": np.random.gamma(2, 2, num_hours),
            "network_out_mbps": np.random.gamma(2, 2, num_hours),
            "disk_iops": np.random.gamma(3, 100, num_hours),
            "hourly_cost": np.random.gamma(2, 10, num_hours),
            "efficiency_score": np.random.beta(5, 2, num_hours) * 100,
            "waste_percentage": np.random.beta(3, 7, num_hours) * 100,
        }
    )


@pytest.fixture
def multi_resource_data():
    """Generate data with multiple resources."""
    num_resources = 5
    hours_per_resource = 24
    dfs = []

    for i in range(num_resources):
        timestamps = [datetime.now() + timedelta(hours=j) for j in range(hours_per_resource)]
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "resource_id": [f"res_{i:03d}"] * hours_per_resource,
                "cpu_utilization": np.random.beta(2, 8, hours_per_resource) * 100,
                "memory_utilization": np.random.beta(3, 7, hours_per_resource) * 100,
                "hourly_cost": np.random.gamma(2, 10 + i * 5, hours_per_resource),
            }
        )
        dfs.append(df)

    return pl.concat(dfs)


@pytest.fixture
def anomalous_data():
    """Generate data with anomalies."""
    num_hours = 100
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(num_hours)]

    # Normal data with some anomalies
    cpu_util = np.random.beta(2, 8, num_hours) * 100
    memory_util = np.random.beta(3, 7, num_hours) * 100
    hourly_cost = np.random.gamma(2, 10, num_hours)

    # Inject anomalies
    anomaly_indices = np.random.choice(num_hours, size=10, replace=False)
    cpu_util[anomaly_indices] *= 3  # Spike CPU
    memory_util[anomaly_indices] *= 2.5  # Spike memory
    hourly_cost[anomaly_indices] *= 5  # Cost spike

    is_anomaly = np.zeros(num_hours, dtype=bool)
    is_anomaly[anomaly_indices] = True

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "resource_id": ["res_001"] * num_hours,
            "cpu_utilization": np.clip(cpu_util, 0, 100),
            "memory_utilization": np.clip(memory_util, 0, 100),
            "hourly_cost": hourly_cost,
            "is_anomaly": is_anomaly,
        }
    )


# Model Fixtures


@pytest.fixture
def mock_pymc_model():
    """Create a mock PyMC model."""
    mock_model = MagicMock()
    mock_model.__enter__ = Mock(return_value=mock_model)
    mock_model.__exit__ = Mock(return_value=None)
    return mock_model


@pytest.fixture
def mock_pymc_trace():
    """Create a mock PyMC trace."""
    trace = MagicMock()
    trace.posterior = {
        "cpu_alpha": np.random.gamma(2, 1, (2, 100, 3)),  # chains, draws, dimensions
        "cpu_beta": np.random.gamma(8, 1, (2, 100, 3)),
        "memory_alpha": np.random.gamma(3, 1, (2, 100, 3)),
        "memory_beta": np.random.gamma(7, 1, (2, 100, 3)),
        "cost_mu": np.random.normal(50, 10, (2, 100, 3)),
        "cost_sigma": np.random.gamma(2, 5, (2, 100, 3)),
    }
    return trace


@pytest.fixture
def sample_resource_pattern():
    """Create a sample ResourcePattern for testing."""
    return ResourcePattern(
        cpu_p50=15.0,
        cpu_p95=45.0,
        memory_p50=25.0,
        memory_p95=60.0,
        cpu_cv=0.8,
        memory_cv=0.5,
        correlation_matrix=np.eye(5),
        daily_pattern_type="business_hours",
        weekly_pattern_type="weekday_heavy",
        seasonality_strength=0.3,
        burst_frequency=5.0,
        burst_amplitude=2.5,
        burst_duration_minutes=30,
    )


@pytest.fixture
def sample_cost_profile():
    """Create a sample CostProfile for testing."""
    return CostProfile(
        avg_hourly_cost=25.50,
        cost_variability=0.3,
        waste_percentage=32.0,
        optimization_difficulty=0.6,
        business_criticality=0.8,
    )


@pytest.fixture
def sample_application_archetype(sample_resource_pattern, sample_cost_profile):
    """Create a sample ApplicationArchetype for testing."""
    return ApplicationArchetype(
        name="Test Application",
        domain=ApplicationDomain.CUSTOMER_FACING,
        description="Test application for unit tests",
        resource_pattern=sample_resource_pattern,
        scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
        optimization_potential=OptimizationPotential.HIGH,
        cost_profile=sample_cost_profile,
        typical_stack=["Python", "Django", "PostgreSQL"],
        cloud_services=["EC2", "RDS", "ELB"],
        typical_instance_types=["t3.medium", "t3.large"],
        deployment_frequency="continuous",
        maintenance_windows=[(6, 2), (0, 3)],
        availability_target=99.9,
        latency_p99_ms=500.0,
        data_volume_gb_per_day=100.0,
        data_retention_days=90,
        example_companies=["TestCorp"],
        market_size_percentage=10.0,
    )


# Forecasting Fixtures (removed - foundation models deleted)


# HuggingFace Fixtures


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace Dataset."""
    dataset = MagicMock()
    dataset.num_rows = 1000
    dataset.column_names = ["resource_id", "timestamp", "cpu_utilization"]
    dataset.features = {
        "resource_id": "string",
        "timestamp": "timestamp[ns]",
        "cpu_utilization": "float32",
    }
    return dataset


@pytest.fixture
def mock_hf_api():
    """Create a mock HuggingFace API client."""
    api = MagicMock()
    api.create_repo = MagicMock(return_value="test-user/test-repo")
    api.upload_file = MagicMock()
    api.list_datasets = MagicMock(return_value=[])
    return api


# Utility Functions


def create_synthetic_metrics(num_samples: int, seed: int = None) -> pl.DataFrame:
    """Create synthetic cloud metrics for testing."""
    if seed is not None:
        np.random.seed(seed)

    return pl.DataFrame(
        {
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(num_samples)],
            "cpu_utilization": np.random.beta(2, 8, num_samples) * 100,
            "memory_utilization": np.random.beta(3, 7, num_samples) * 100,
            "hourly_cost": np.random.gamma(2, 10, num_samples),
            "efficiency_score": np.random.beta(5, 2, num_samples) * 100,
        }
    )


def assert_valid_utilization(
    df: pl.DataFrame, cpu_col: str = "cpu_utilization", memory_col: str = "memory_utilization"
):
    """Assert that utilization values are valid (0-100%)."""
    assert df[cpu_col].min() >= 0, "CPU utilization cannot be negative"
    assert df[cpu_col].max() <= 100, "CPU utilization cannot exceed 100%"
    assert df[memory_col].min() >= 0, "Memory utilization cannot be negative"
    assert df[memory_col].max() <= 100, "Memory utilization cannot exceed 100%"


def assert_realistic_waste(df: pl.DataFrame, waste_col: str = "waste_percentage"):
    """Assert that waste percentage is realistic (based on research: 30-32% average)."""
    if waste_col in df.columns:
        avg_waste = df[waste_col].mean()
        assert 15 <= avg_waste <= 60, f"Average waste {avg_waste}% outside realistic range"


def assert_time_series_continuity(
    df: pl.DataFrame, timestamp_col: str = "timestamp", resource_col: str = "resource_id"
):
    """Assert that time series data is continuous for each resource."""
    for resource in df[resource_col].unique():
        resource_data = df.filter(pl.col(resource_col) == resource)
        timestamps = resource_data[timestamp_col].sort()

        # Check for gaps
        time_diffs = timestamps.diff().drop_nulls()
        if len(time_diffs) > 0:
            # All differences should be the same (regular interval)
            unique_diffs = time_diffs.unique()
            assert len(unique_diffs) <= 2, f"Irregular time intervals for resource {resource}"


# Performance Testing Utilities


@pytest.fixture
def benchmark_timer():
    """Simple timer for performance benchmarking."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start_time

    return Timer()


# Parametrized Test Data


def pytest_generate_tests(metafunc):
    """Generate parametrized test cases."""
    if "application_domain" in metafunc.fixturenames:
        metafunc.parametrize("application_domain", list(ApplicationDomain))

    if "optimization_potential" in metafunc.fixturenames:
        metafunc.parametrize("optimization_potential", list(OptimizationPotential))


# Markers for Test Categories


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "requires_hf: marks tests that require HuggingFace Hub access"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GP tests if torch is not available."""
    if TORCH_AVAILABLE:
        return  # torch available, run all tests

    skip_gpu = pytest.mark.skip(reason="Requires torch (install with: uv sync --group gpu)")
    for item in items:
        # Skip all tests in GP test modules
        if "test_gaussian_process" in str(item.fspath):
            item.add_marker(skip_gpu)
