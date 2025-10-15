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


class TestTimeSeriesFactoryMethods:
    """Test TimeSeries factory methods."""

    def test_from_dataframe_creates_instance(self, spark):
        """Should create TimeSeries from DataFrame via factory method."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(
            df, hierarchy=["provider", "account", "region"], metric_col="cost", time_col="date"
        )

        assert isinstance(ts, TimeSeries)
        assert ts.hierarchy == ["provider", "account", "region"]
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"

    def test_from_dataframe_with_defaults(self, spark):
        """Should use default column names if not specified."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", 100.0),
            ],
            ["date", "provider", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider"])

        # Should default to metric_col="cost", time_col="date"
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"


class TestGrainResolution:
    """Test grain resolution and validation."""

    def test_resolve_grain_returns_ordered_columns(self, spark):
        """Should return grain columns in hierarchy order."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", "Compute", 100.0),
            ],
            ["date", "provider", "account", "region", "product", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region", "product"])

        # Request out-of-order grain
        grain = ts._resolve_grain(["region", "provider"])

        # Should return in hierarchy order
        assert grain == ["provider", "region"]

    def test_resolve_grain_validates_subset(self, spark):
        """Should raise error if grain contains columns not in hierarchy."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        with pytest.raises(TimeSeriesError, match="Invalid grain columns"):
            ts._resolve_grain(["provider", "invalid_column"])

    def test_resolve_grain_handles_partial_hierarchy(self, spark):
        """Should handle grain that is partial subset of hierarchy."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        grain = ts._resolve_grain(["account"])
        assert grain == ["account"]


class TestTimeSeriesFilter:
    """Test TimeSeries filtering operations."""

    def test_filter_single_entity(self, spark):
        """Should filter to specific entity and return new TimeSeries."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
                ("2025-01-01", "GCP", "acc3", 300.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        filtered = ts.filter(provider="AWS", account="acc1")

        assert isinstance(filtered, TimeSeries)
        assert filtered.df.count() == 1
        result = filtered.df.collect()[0]
        assert result["provider"] == "AWS"
        assert result["account"] == "acc1"

    def test_filter_returns_new_instance(self, spark):
        """Should return new TimeSeries instance, not modify original."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        filtered = ts.filter(account="acc1")

        assert ts.df.count() == 2  # Original unchanged
        assert filtered.df.count() == 1
        assert ts is not filtered

    def test_filter_multiple_criteria(self, spark):
        """Should filter on multiple hierarchy columns."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
                ("2025-01-01", "AWS", "acc2", "us-east-1", 300.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        filtered = ts.filter(provider="AWS", account="acc1", region="us-east-1")

        assert filtered.df.count() == 1
        result = filtered.df.collect()[0]
        assert result["region"] == "us-east-1"

    def test_filter_invalid_column_raises_error(self, spark):
        """Should raise error if filter column not in hierarchy."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        with pytest.raises(TimeSeriesError, match="Invalid filter column"):
            ts.filter(invalid_column="value")


class TestTimeSeriesSample:
    """Test TimeSeries sampling operations."""

    def test_sample_single_entity(self, spark):
        """Should sample single entity at specified grain."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 110.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
                ("2025-01-02", "AWS", "acc2", 220.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"], n=1)

        assert isinstance(sampled, TimeSeries)
        # Should have 2 rows (both dates) for the sampled account
        assert sampled.df.count() == 2
        # Should only have 1 unique account
        assert sampled.df.select("account").distinct().count() == 1

    def test_sample_multiple_entities(self, spark):
        """Should sample N entities at specified grain."""
        df = spark.createDataFrame(
            [("2025-01-01", "AWS", f"acc{i}", 100.0 * i) for i in range(10)],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"], n=3)

        # Should have 3 unique accounts
        assert sampled.df.select("account").distinct().count() == 3

    def test_sample_more_than_available(self, spark):
        """Should return all entities if n > available."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        # Request more than available - should return all entities
        sampled = ts.sample(grain=["account"], n=10)

        # Should return all 2 available accounts (not raise error)
        assert sampled.df.select("account").distinct().count() == 2
        # Should return all rows
        assert sampled.df.count() == 2

    def test_sample_default_n_equals_1(self, spark):
        """Should default to n=1 if not specified."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"])

        assert sampled.df.select("account").distinct().count() == 1

    def test_sample_hierarchical_grain(self, spark):
        """Should sample at multi-level grain correctly."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
                ("2025-01-01", "AWS", "acc2", "us-east-1", 300.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        sampled = ts.sample(grain=["account", "region"], n=2)

        # Should have 2 unique account+region combinations
        assert sampled.df.select("account", "region").distinct().count() == 2


class TestTimeSeriesAggregate:
    """Test TimeSeries aggregation operations."""

    def test_aggregate_to_coarser_grain(self, spark):
        """Should aggregate metric to coarser grain level."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Aggregate from account+region to just account
        agg = ts.aggregate(grain=["account"])

        assert isinstance(agg, TimeSeries)
        # Should have 2 rows (2 dates for acc1)
        assert agg.df.count() == 2
        # Region and provider columns should be removed from DataFrame
        assert "region" not in agg.df.columns
        assert "provider" not in agg.df.columns
        # Hierarchy should only include grain columns
        assert agg.hierarchy == ["account"]
        # Should sum costs: 100+200=300 for date 1, 150 for date 2
        results = agg.df.orderBy("date").collect()
        assert results[0]["cost"] == 300.0
        assert results[1]["cost"] == 150.0

    def test_aggregate_preserves_time_dimension(self, spark):
        """Should preserve time column in aggregation."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        agg = ts.aggregate(grain=["provider"])

        assert "date" in agg.df.columns
        assert agg.df.count() == 2  # One per date
        assert agg.hierarchy == ["provider"]

    def test_aggregate_to_top_level(self, spark):
        """Should aggregate all the way to top of hierarchy."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-01", "AWS", "acc2", "us-west-1", 200.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Aggregate to just provider level
        agg = ts.aggregate(grain=["provider"])

        assert agg.df.count() == 2  # 2 dates
        assert "account" not in agg.df.columns
        assert "region" not in agg.df.columns
        assert agg.hierarchy == ["provider"]
        results = agg.df.orderBy("date").collect()
        assert results[0]["cost"] == 300.0  # 100 + 200
        assert results[1]["cost"] == 150.0

    def test_aggregate_same_grain_returns_copy(self, spark):
        """Should return copy and log info if already at requested grain."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        # Test that aggregating to the same grain returns a copy
        agg = ts.aggregate(grain=["provider", "account"])

        # Should return same row count (already at grain)
        assert agg.df.count() == ts.df.count()
        # Should be new instance
        assert agg is not ts


class TestTimeSeriesSummaryStats:
    """Test TimeSeries summary statistics."""

    def test_summary_stats_returns_dataframe(self, spark):
        """Should return DataFrame with summary statistics."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 110.0),
                ("2025-01-03", "AWS", "acc1", 105.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()

        assert stats is not None
        # Should have stats columns
        stats_cols = stats.columns
        assert "mean" in stats_cols
        assert "std" in stats_cols
        assert "min" in stats_cols
        assert "max" in stats_cols
        assert "count" in stats_cols

    def test_summary_stats_includes_entity_keys(self, spark):
        """Should include entity identifier columns in stats output."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 110.0),
                ("2025-01-01", "AWS", "acc2", 200.0),
                ("2025-01-02", "AWS", "acc2", 220.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()

        # Should have entity keys
        assert "provider" in stats.columns
        assert "account" in stats.columns
        # Should have one row per entity
        assert stats.count() == 2

    def test_summary_stats_at_different_grain(self, spark):
        """Should compute stats at specified grain level."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 110.0),
                ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
                ("2025-01-02", "AWS", "acc1", "us-west-1", 220.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Stats at account level (should aggregate regions first)
        stats = ts.summary_stats(grain=["account"])

        assert stats.count() == 1  # One account
        # Should aggregate: date1: 100+200=300, date2: 110+220=330
        result = stats.collect()[0]
        assert result["mean"] == 315.0  # (300 + 330) / 2

    def test_summary_stats_correct_calculations(self, spark):
        """Should calculate statistics correctly."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 200.0),
                ("2025-01-03", "AWS", "acc1", 150.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()
        result = stats.collect()[0]

        assert result["count"] == 3
        assert result["mean"] == 150.0
        assert result["min"] == 100.0
        assert result["max"] == 200.0
        assert result["std"] > 0  # Should have standard deviation


class TestTimeSeriesPlotTemporalDensity:
    """Test TimeSeries plot_temporal_density visualization method."""

    def test_returns_matplotlib_figure(self, spark):
        """Should return matplotlib Figure object."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 110.0),
                ("2025-01-03", "AWS", "acc1", 105.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        fig = ts.plot_temporal_density()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grain_context_in_subtitle(self, spark):
        """Should include grain context in subtitle."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-02", "AWS", "acc1", "us-east-1", 110.0),
                ("2025-01-01", "AWS", "acc2", "us-west-1", 200.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        fig = ts.plot_temporal_density()

        # Verify subtitle contains grain info
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Should have subtitle with grain format like "provider-account-region"
        subtitle_found = any("provider-account-region" in text for text in texts)
        assert subtitle_found, "Grain context not found in subtitle"

        # Should mention entity count
        entity_count_found = any("entities" in text.lower() for text in texts)
        assert entity_count_found, "Entity count not found in subtitle"

        plt.close(fig)

    def test_passes_kwargs_to_eda_function(self, spark):
        """Should pass kwargs to underlying eda.plot_temporal_density function."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 1000.0),  # Large value for log scale testing
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        # Test that log_scale parameter is passed through
        fig = ts.plot_temporal_density(log_scale=True)

        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

        plt.close(fig)

    def test_custom_title_overrides_default(self, spark):
        """Should allow custom title to override default."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        custom_title = "My Custom Plot Title"
        fig = ts.plot_temporal_density(title=custom_title)

        ax = fig.axes[0]
        assert ax.get_title() == custom_title

        plt.close(fig)


class TestTimeSeriesWithDf:
    """Test TimeSeries with_df helper method."""

    def test_preserves_metadata(self, spark):
        """Should preserve hierarchy, metric_col, time_col metadata."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-02", "AWS", "acc1", 110.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        # Create filtered DataFrame
        filtered_df = df.filter("cost > 100")
        ts_filtered = ts.with_df(filtered_df)

        # Should preserve metadata
        assert ts_filtered.hierarchy == ["provider", "account"]
        assert ts_filtered.metric_col == "cost"
        assert ts_filtered.time_col == "date"
        assert ts_filtered.df.count() == 1

    def test_returns_new_instance(self, spark):
        """Should return new TimeSeries instance, not modify original."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        filtered_df = df.filter("cost > 50")
        ts_new = ts.with_df(filtered_df)

        assert ts is not ts_new
        assert ts.df.count() == 1
        assert ts_new.df.count() == 1


class TestTimeSeriesFilterTime:
    """Test TimeSeries filter_time method."""

    def test_filter_before(self, spark):
        """Should filter to times before specified date (exclusive)."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-05", "AWS", "acc1", 110.0),
                ("2025-01-10", "AWS", "acc1", 120.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        ts_filtered = ts.filter_time(before="2025-01-10")

        assert ts_filtered.df.count() == 2
        dates = [row["date"] for row in ts_filtered.df.collect()]
        assert "2025-01-01" in dates
        assert "2025-01-05" in dates
        assert "2025-01-10" not in dates

    def test_filter_after(self, spark):
        """Should filter to times after specified date (inclusive)."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-05", "AWS", "acc1", 110.0),
                ("2025-01-10", "AWS", "acc1", 120.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        ts_filtered = ts.filter_time(after="2025-01-05")

        assert ts_filtered.df.count() == 2
        dates = [row["date"] for row in ts_filtered.df.collect()]
        assert "2025-01-01" not in dates
        assert "2025-01-05" in dates
        assert "2025-01-10" in dates

    def test_filter_range_start_end(self, spark):
        """Should filter to time range with start (inclusive) and end (exclusive)."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-05", "AWS", "acc1", 110.0),
                ("2025-01-10", "AWS", "acc1", 120.0),
                ("2025-01-15", "AWS", "acc1", 130.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        ts_filtered = ts.filter_time(start="2025-01-05", end="2025-01-15")

        assert ts_filtered.df.count() == 2
        dates = [row["date"] for row in ts_filtered.df.collect()]
        assert "2025-01-01" not in dates
        assert "2025-01-05" in dates
        assert "2025-01-10" in dates
        assert "2025-01-15" not in dates

    def test_preserves_metadata(self, spark):
        """Should preserve hierarchy, metric_col, time_col."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
                ("2025-01-10", "AWS", "acc1", "us-east-1", 110.0),
            ],
            ["date", "provider", "account", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])
        ts_filtered = ts.filter_time(before="2025-01-05")

        assert ts_filtered.hierarchy == ["provider", "account", "region"]
        assert ts_filtered.metric_col == "cost"
        assert ts_filtered.time_col == "date"

    def test_returns_new_instance(self, spark):
        """Should return new instance, not modify original."""
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "acc1", 100.0),
                ("2025-01-10", "AWS", "acc1", 110.0),
            ],
            ["date", "provider", "account", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        ts_filtered = ts.filter_time(before="2025-01-05")

        assert ts is not ts_filtered
        assert ts.df.count() == 2  # Original unchanged
        assert ts_filtered.df.count() == 1  # Filtered has less


class TestTimeSeriesPlotEntityCounts:
    """Test TimeSeries plot_entity_counts method."""

    def test_returns_matplotlib_figure(self, spark):
        """Should return matplotlib Figure object."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "us-east-1", "EC2", 100.0),
                ("2025-01-01", "AWS", "us-west-1", "S3", 50.0),
                ("2025-01-02", "AWS", "us-east-1", "EC2", 110.0),
            ],
            ["date", "provider", "region", "product", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "region", "product"])

        fig = ts.plot_entity_counts(["region", "product"])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_box_plot_for_each_dimension(self, spark):
        """Should create box plot with one box per dimension."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "us-east-1", "EC2", 100.0),
                ("2025-01-02", "AWS", "us-east-1", "EC2", 110.0),
            ],
            ["date", "provider", "region", "product", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "region", "product"])

        fig = ts.plot_entity_counts(["region", "product"])

        # Should have created axes with box plots
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Dimension"
        assert ax.get_ylabel() == "Entity Count"

        plt.close(fig)

    def test_log_scale_parameter(self, spark):
        """Should apply log scale when requested."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "us-east-1", "EC2", 100.0),
                ("2025-01-02", "AWS", "us-east-1", "EC2", 110.0),
            ],
            ["date", "provider", "region", "product", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "region", "product"])

        fig = ts.plot_entity_counts(["region", "product"], log_scale=True)

        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

        plt.close(fig)

    def test_custom_title(self, spark):
        """Should allow custom title."""
        import matplotlib.pyplot as plt

        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "us-east-1", "EC2", 100.0),
            ],
            ["date", "provider", "region", "product", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "region", "product"])

        custom_title = "My Custom Entity Count Plot"
        fig = ts.plot_entity_counts(["region"], title=custom_title)

        ax = fig.axes[0]
        assert ax.get_title() == custom_title

        plt.close(fig)


class TestPlotCostDistribution:
    """Test cost distribution plotting."""

    def test_min_cost_filter(self, spark):
        """Should filter out daily costs below min_cost threshold."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Create data with mix of high and low costs
        df = spark.createDataFrame(
            [
                ("2025-01-01", "AWS", "us-east-1", 100.0),
                ("2025-01-02", "AWS", "us-east-1", 5.0),  # Below threshold
                ("2025-01-03", "AWS", "us-east-1", 120.0),
                ("2025-01-04", "AWS", "us-east-1", 3.0),  # Below threshold
                ("2025-01-05", "AWS", "us-east-1", 110.0),
                ("2025-01-01", "Azure", "us-west-2", 50.0),
                ("2025-01-02", "Azure", "us-west-2", 2.0),  # Below threshold
                ("2025-01-03", "Azure", "us-west-2", 55.0),
            ],
            ["date", "provider", "region", "cost"],
        )

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "region"])

        # Plot with min_cost filter
        fig = ts.plot_cost_distribution(["provider"], top_n=2, min_cost=10.0)

        # Verify plot was created
        assert isinstance(fig, plt.Figure)

        # Get the box plot data
        ax = fig.axes[0]

        # Each box plot has a "boxes" collection with quartile data
        # We should verify that the min values displayed are >= 10.0
        # The y-data from the box plot should not contain values below 10.0

        # Get all y-data from the plot (lines, boxes, whiskers, etc.)
        all_y_data = []
        for line in ax.get_lines():
            y_data = line.get_ydata()
            all_y_data.extend(y_data)

        # Filter out NaN values
        all_y_data = [y for y in all_y_data if not pd.isna(y)]

        # Verify no values below threshold (with small tolerance for floating point)
        min_value = min(all_y_data) if all_y_data else float("inf")
        assert min_value >= 10.0, f"Found value {min_value} below min_cost threshold of 10.0"

        plt.close(fig)
