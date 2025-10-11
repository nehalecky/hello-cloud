"""
Tests for time series transformation functions using Ibis pipe patterns.

All tests validate that transformations:
1. Return valid Ibis tables
2. Preserve original columns (add new ones)
3. Calculate correct values
4. Handle edge cases (nulls, partitions, empty windows)
"""

from datetime import datetime

import ibis
import pandas as pd
import pytest
from ibis import _

from hellocloud.transforms import (
    add_lag_features,
    add_rolling_stats,
    add_z_score,
    cumulative_sum,
    pct_change,
    rolling_average,
    rolling_std,
    time_features,
)


@pytest.fixture
def con():
    """Create in-memory DuckDB connection."""
    return ibis.duckdb.connect()


@pytest.fixture
def sample_timeseries(con):
    """
    Create sample time series data for testing.

    Returns table with:
    - date: Daily timestamps
    - entity_id: Partition key (A, B)
    - value: Numeric values
    """
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D").tolist() * 2,
            "entity_id": ["A"] * 10 + ["B"] * 10,
            "value": [10, 12, 15, 14, 16, 18, 17, 19, 20, 22] + [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        }
    )
    return con.create_table("sample_ts", data)


class TestPctChange:
    """Test percentage change transformation."""

    def test_basic_pct_change(self, sample_timeseries):
        """Test basic percentage change calculation."""
        result = sample_timeseries.pipe(
            pct_change("value", "date", partition_by="entity_id")
        ).execute()

        # Check column exists
        assert "value_pct_change" in result.columns

        # First row should be null (no previous value)
        assert pd.isna(result.loc[result["entity_id"] == "A", "value_pct_change"].iloc[0])

        # Second row: (12 - 10) / 10 = 0.20 (20% as fraction)
        assert result.loc[result["entity_id"] == "A", "value_pct_change"].iloc[1] == pytest.approx(
            0.20
        )

    def test_pct_change_custom_suffix(self, sample_timeseries):
        """Test custom suffix for output column."""
        result = sample_timeseries.pipe(pct_change("value", "date", suffix="_change_pct")).execute()

        assert "value_change_pct" in result.columns

    def test_pct_change_multiple_periods(self, sample_timeseries):
        """Test percentage change with periods > 1."""
        result = sample_timeseries.pipe(
            pct_change("value", "date", partition_by="entity_id", periods=2)
        ).execute()

        # First two rows should be null
        assert pd.isna(result.loc[result["entity_id"] == "A", "value_pct_change"].iloc[0])
        assert pd.isna(result.loc[result["entity_id"] == "A", "value_pct_change"].iloc[1])

        # Third row: (15 - 10) / 10 = 0.50 (50% as fraction)
        assert result.loc[result["entity_id"] == "A", "value_pct_change"].iloc[2] == pytest.approx(
            0.50
        )

    def test_pct_change_respects_partitions(self, sample_timeseries):
        """Test that partitions are respected (no cross-entity calculation)."""
        result = sample_timeseries.pipe(
            pct_change("value", "date", partition_by="entity_id")
        ).execute()

        # Entity A and B should have independent calculations
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Both first rows should be null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])
        assert pd.isna(entity_b["value_pct_change"].iloc[0])


class TestRollingAverage:
    """Test rolling average transformation."""

    def test_basic_rolling_average(self, sample_timeseries):
        """Test basic rolling average calculation."""
        result = sample_timeseries.pipe(
            rolling_average("value", "date", window_size=3, partition_by="entity_id")
        ).execute()

        assert "value_rolling_3" in result.columns

        # Entity A, row 2: mean([10, 12, 15]) = 12.33...
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        assert entity_a["value_rolling_3"].iloc[2] == pytest.approx(12.333, abs=0.01)

    def test_rolling_average_custom_suffix(self, sample_timeseries):
        """Test custom suffix for output column."""
        result = sample_timeseries.pipe(
            rolling_average("value", "date", window_size=3, suffix="_ma3")
        ).execute()

        assert "value_ma3" in result.columns

    def test_rolling_average_respects_partitions(self, sample_timeseries):
        """Test that rolling average respects partitions."""
        result = sample_timeseries.pipe(
            rolling_average("value", "date", window_size=3, partition_by="entity_id")
        ).execute()

        # Entity A and B should have different rolling averages
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Entity A, row 2: mean([10, 12, 15])
        assert entity_a["value_rolling_3"].iloc[2] == pytest.approx(12.333, abs=0.01)

        # Entity B, row 2: mean([5, 6, 7]) = 6.0
        assert entity_b["value_rolling_3"].iloc[2] == pytest.approx(6.0)


class TestRollingStd:
    """Test rolling standard deviation transformation."""

    def test_basic_rolling_std(self, sample_timeseries):
        """Test basic rolling std calculation."""
        result = sample_timeseries.pipe(
            rolling_std("value", "date", window_size=3, partition_by="entity_id")
        ).execute()

        assert "value_rolling_std_3" in result.columns

        # Std should be > 0 for varying values
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        assert entity_a["value_rolling_std_3"].iloc[2] > 0


class TestCumulativeSum:
    """Test cumulative sum transformation."""

    def test_basic_cumsum(self, sample_timeseries):
        """Test basic cumulative sum calculation."""
        result = sample_timeseries.pipe(
            cumulative_sum("value", "date", partition_by="entity_id")
        ).execute()

        assert "value_cumsum" in result.columns

        # Entity A cumsum at row 2: 10 + 12 + 15 = 37
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        assert entity_a["value_cumsum"].iloc[2] == 37

    def test_cumsum_respects_partitions(self, sample_timeseries):
        """Test that cumsum resets per partition."""
        result = sample_timeseries.pipe(
            cumulative_sum("value", "date", partition_by="entity_id")
        ).execute()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Entity A starts at 10
        assert entity_a["value_cumsum"].iloc[0] == 10

        # Entity B starts at 5 (not continuing from A)
        assert entity_b["value_cumsum"].iloc[0] == 5


class TestAddLagFeatures:
    """Test lag features transformation."""

    def test_basic_lag_features(self, sample_timeseries):
        """Test adding multiple lag features."""
        result = sample_timeseries.pipe(
            add_lag_features("value", "date", lags=[1, 2], partition_by="entity_id")
        ).execute()

        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns

        # Entity A, row 2: lag_1 should be 12, lag_2 should be 10
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        assert entity_a["value_lag_1"].iloc[2] == 12
        assert entity_a["value_lag_2"].iloc[2] == 10

    def test_lag_features_respects_partitions(self, sample_timeseries):
        """Test that lag features respect partitions."""
        result = sample_timeseries.pipe(
            add_lag_features("value", "date", lags=[1], partition_by="entity_id")
        ).execute()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # First lag should be null for both
        assert pd.isna(entity_a["value_lag_1"].iloc[0])
        assert pd.isna(entity_b["value_lag_1"].iloc[0])

        # Second values should be different
        assert entity_a["value_lag_1"].iloc[1] == 10
        assert entity_b["value_lag_1"].iloc[1] == 5


class TestTimeFeatures:
    """Test time feature extraction."""

    def test_basic_time_features(self, con):
        """Test extracting time features."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
                "value": [1, 2, 3, 4, 5],
            }
        )
        table = con.create_table("time_test", data)

        result = table.pipe(
            time_features("timestamp", components=["hour", "day_of_week", "month"])
        ).execute()

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns

        # First timestamp is 2024-01-01 00:00:00 (Monday)
        assert result["hour"].iloc[0] == 0
        assert result["day_of_week"].iloc[0] == 0  # Monday
        assert result["month"].iloc[0] == 1  # January

    def test_time_features_quarter(self, con):
        """Test quarter extraction."""
        data = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),
                    datetime(2024, 4, 1),
                    datetime(2024, 7, 1),
                    datetime(2024, 10, 1),
                ],
                "value": [1, 2, 3, 4],
            }
        )
        table = con.create_table("quarter_test", data)

        result = table.pipe(time_features("timestamp", components=["quarter"])).execute()

        assert result["quarter"].tolist() == [1, 2, 3, 4]


class TestPipeComposition:
    """Test chaining multiple transformations."""

    def test_chain_multiple_transforms(self, sample_timeseries):
        """Test composing multiple transformations in a pipeline."""
        result = (
            sample_timeseries.pipe(pct_change("value", "date", partition_by="entity_id"))
            .pipe(rolling_average("value", "date", window_size=3, partition_by="entity_id"))
            .pipe(cumulative_sum("value", "date", partition_by="entity_id"))
            .execute()
        )

        # All columns should exist
        assert "value_pct_change" in result.columns
        assert "value_rolling_3" in result.columns
        assert "value_cumsum" in result.columns

        # Original column should still exist
        assert "value" in result.columns

    def test_readable_pipeline(self, sample_timeseries):
        """Test that pipe creates readable, self-documenting pipelines."""
        # This test demonstrates the readability benefit
        result = (
            sample_timeseries.pipe(
                add_lag_features("value", "date", lags=[1, 7], partition_by="entity_id")
            )
            .pipe(pct_change("value", "date", partition_by="entity_id"))
            .pipe(rolling_average("value", "date", window_size=7, partition_by="entity_id"))
            .filter(_.date >= "2024-01-08")  # Keep only rows with full window
            .execute()
        )

        # Verify all transformations applied
        assert "value_lag_1" in result.columns
        assert "value_lag_7" in result.columns
        assert "value_pct_change" in result.columns
        assert "value_rolling_7" in result.columns


class TestRollingStats:
    """Test rolling statistics transformation."""

    def test_default_rolling_stats(self, sample_timeseries):
        """Test default rolling stats (mean and std)."""
        result = sample_timeseries.pipe(
            add_rolling_stats("value", "date", window_size=3, partition_by="entity_id")
        ).execute()

        # Default stats should be mean and std
        assert "value_rolling_mean_3" in result.columns
        assert "value_rolling_std_3" in result.columns

    def test_all_rolling_stats(self, sample_timeseries):
        """Test all available rolling statistics."""
        result = sample_timeseries.pipe(
            add_rolling_stats(
                "value",
                "date",
                window_size=3,
                partition_by="entity_id",
                stats=["mean", "std", "min", "max", "median"],
            )
        ).execute()

        # All stats should be present
        assert "value_rolling_mean_3" in result.columns
        assert "value_rolling_std_3" in result.columns
        assert "value_rolling_min_3" in result.columns
        assert "value_rolling_max_3" in result.columns
        assert "value_rolling_median_3" in result.columns

        # Verify min/max make sense for Entity A
        entity_a = result[result["entity_id"] == "A"].sort_values("date")

        # Row 2: values [10, 12, 15]
        assert entity_a["value_rolling_min_3"].iloc[2] == 10
        assert entity_a["value_rolling_max_3"].iloc[2] == 15


class TestZScore:
    """Test z-score transformation."""

    def test_basic_z_score(self, sample_timeseries):
        """Test basic z-score calculation."""
        result = sample_timeseries.pipe(
            add_z_score("value", "date", window_size=5, partition_by="entity_id")
        ).execute()

        assert "value_zscore" in result.columns

        # Z-scores should be numeric (can be null for early rows)
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        assert entity_a["value_zscore"].dtype in ["float64", "float32", "object"]

    def test_z_score_with_anomaly_flag(self, sample_timeseries):
        """Test creating anomaly flags from z-scores."""
        result = (
            sample_timeseries.pipe(
                add_z_score("value", "date", window_size=5, partition_by="entity_id")
            )
            .mutate(is_anomaly=(_.value_zscore.abs() > 2))
            .execute()
        )

        # Should have both z-score and anomaly flag columns
        assert "value_zscore" in result.columns
        assert "is_anomaly" in result.columns

        # Anomaly flag should be boolean
        assert result["is_anomaly"].dtype in ["bool", "object"]
