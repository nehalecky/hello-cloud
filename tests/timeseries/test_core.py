"""Tests for TimeSeries core functionality."""

import pytest

from hellocloud.timeseries import TimeSeries
from hellocloud.timeseries.core import TimeSeriesError


class TestTimeSeriesInitialization:
    """Test TimeSeries initialization and basic properties."""

    def test_create_from_dataframe(self, spark):
        """Should create TimeSeries from PySpark DataFrame."""
        # Arrange
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 110.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        # Act
        ts = TimeSeries(
            df=df, hierarchy=["provider", "account", "region"], metric_col="cost", time_col="date"
        )

        # Assert
        assert ts.df is not None
        assert ts.hierarchy == ["provider", "account", "region"]
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"

    def test_stores_dataframe_reference(self, spark):
        """Should store reference to DataFrame, not copy."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries(
            df=df, hierarchy=["provider", "account"], metric_col="cost", time_col="date"
        )

        assert ts.df.count() == 1
        assert "date" in ts.df.columns
        assert "cost" in ts.df.columns


class TestTimeSeriesValidation:
    """Test TimeSeries validation and error handling."""

    def test_missing_time_column_raises_error(self, spark):
        """Should raise error if time column not in DataFrame."""
        df = spark.createDataFrame(
            [
                ("AWS", "acc1", 100.0),
            ],
            ["provider", "account", "cost"],
        )

        with pytest.raises(TimeSeriesError, match="time_col 'date' not found"):
            TimeSeries(df=df, hierarchy=["provider", "account"], metric_col="cost", time_col="date")

    def test_missing_metric_column_raises_error(self, spark):
        """Should raise error if metric column not in DataFrame."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1"),
            ],
            ["date", "provider", "account"],
        )

        with pytest.raises(TimeSeriesError, match="metric_col 'cost' not found"):
            TimeSeries(df=df, hierarchy=["provider", "account"], metric_col="cost", time_col="date")

    def test_missing_hierarchy_column_raises_error(self, spark):
        """Should raise error if hierarchy column not in DataFrame."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", 100.0),
            ],
            ["date", "provider", "cost"],
        )

        with pytest.raises(TimeSeriesError, match="hierarchy column 'account' not found"):
            TimeSeries(df=df, hierarchy=["provider", "account"], metric_col="cost", time_col="date")

    def test_empty_dataframe_does_not_raise_error(self, spark):
        """Should not raise error for empty DataFrame (but logs warning)."""
        df = spark.createDataFrame([], "date STRING, provider STRING, cost DOUBLE")

        # Should not raise error
        ts = TimeSeries(df=df, hierarchy=["provider"], metric_col="cost", time_col="date")

        # Should create empty TimeSeries
        assert ts.df.count() == 0
        assert ts.hierarchy == ["provider"]
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"
