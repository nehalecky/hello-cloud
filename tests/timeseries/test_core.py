"""Tests for TimeSeries core functionality."""

from hellocloud.timeseries import TimeSeries


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
