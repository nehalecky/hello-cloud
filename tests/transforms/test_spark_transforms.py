"""Tests for PySpark transform functions."""

import pytest


@pytest.fixture(scope="module")
def spark():
    """Spark session for testing."""
    from hellocloud.spark.session import get_spark_session

    return get_spark_session(app_name="test-transforms")


def test_pct_change_basic(spark):
    """Test pct_change on simple time series."""
    from hellocloud.transforms.spark import pct_change

    data = [
        ("A", "2024-01-01", 100.0),
        ("A", "2024-01-02", 110.0),
        ("A", "2024-01-03", 99.0),
    ]
    df = spark.createDataFrame(data, ["group", "date", "value"])

    result = df.transform(pct_change(value_col="value", order_col="date"))

    # First row should be null, then 10% increase, then 10% decrease
    values = [row.pct_change for row in result.collect()]
    assert values[0] is None
    assert abs(values[1] - 0.10) < 0.001
    assert abs(values[2] - (-0.10)) < 0.001


def test_pct_change_with_groups(spark):
    """Test pct_change respects group_cols."""
    from hellocloud.transforms.spark import pct_change

    data = [
        ("A", "2024-01-01", 100.0),
        ("A", "2024-01-02", 110.0),
        ("B", "2024-01-01", 200.0),
        ("B", "2024-01-02", 220.0),
    ]
    df = spark.createDataFrame(data, ["group", "date", "value"])

    result = df.transform(pct_change(value_col="value", order_col="date", group_cols=["group"]))

    # Each group's first row should be null
    rows = result.orderBy("group", "date").collect()
    assert rows[0].pct_change is None  # A, first row
    assert abs(rows[1].pct_change - 0.10) < 0.001  # A, 10% increase
    assert rows[2].pct_change is None  # B, first row
    assert abs(rows[3].pct_change - 0.10) < 0.001  # B, 10% increase


def test_pct_change_handles_nulls(spark):
    """Test pct_change propagates nulls correctly."""
    from hellocloud.transforms.spark import pct_change

    data = [
        ("A", "2024-01-01", 100.0),
        ("A", "2024-01-02", None),
        ("A", "2024-01-03", 120.0),
    ]
    df = spark.createDataFrame(data, ["group", "date", "value"])

    result = df.transform(pct_change(value_col="value", order_col="date"))

    values = [row.pct_change for row in result.collect()]
    assert values[0] is None  # First row
    assert values[1] is None  # Null value
    assert values[2] is None  # Previous was null


def test_pct_change_with_unsorted_groups(spark):
    """Test pct_change works with unsorted input."""
    from hellocloud.transforms.spark import pct_change

    # Deliberately unsorted data
    data = [
        ("A", "2024-01-03", 99.0),
        ("A", "2024-01-01", 100.0),
        ("A", "2024-01-02", 110.0),
    ]
    df = spark.createDataFrame(data, ["group", "date", "value"])

    result = df.transform(pct_change(value_col="value", order_col="date"))

    # Should sort by date and calculate correctly
    sorted_result = result.orderBy("date").collect()
    assert sorted_result[0].pct_change is None  # 2024-01-01
    assert abs(sorted_result[1].pct_change - 0.10) < 0.001  # 2024-01-02
    assert abs(sorted_result[2].pct_change - (-0.10)) < 0.001  # 2024-01-03


def test_summary_stats_basic(spark):
    """Test summary_stats on simple dataset."""
    from hellocloud.transforms.spark import summary_stats

    data = [
        ("A", 100.0),
        ("A", 150.0),
        ("A", 200.0),
        ("B", 50.0),
        ("B", 100.0),
    ]
    df = spark.createDataFrame(data, ["group", "value"])

    result = df.transform(summary_stats(value_col="value", group_col="group"))

    # Should have statistics for both groups
    rows = {row.group: row for row in result.collect()}

    # Group A stats
    assert rows["A"]["count"] == 3
    assert abs(rows["A"]["mean"] - 150.0) < 0.001
    assert rows["A"]["min"] == 100.0
    assert rows["A"]["max"] == 200.0

    # Group B stats
    assert rows["B"]["count"] == 2
    assert abs(rows["B"]["mean"] - 75.0) < 0.001
    assert rows["B"]["min"] == 50.0
    assert rows["B"]["max"] == 100.0


def test_summary_stats_handles_nulls(spark):
    """Test summary_stats excludes nulls from calculations."""
    from hellocloud.transforms.spark import summary_stats

    data = [
        ("A", 100.0),
        ("A", None),
        ("A", 200.0),
    ]
    df = spark.createDataFrame(data, ["group", "value"])

    result = df.transform(summary_stats(value_col="value", group_col="group"))

    row = result.collect()[0]
    assert row["count"] == 2  # Nulls excluded
    assert abs(row["mean"] - 150.0) < 0.001


def test_summary_stats_custom_columns(spark):
    """Test summary_stats with custom column names."""
    from hellocloud.transforms.spark import summary_stats

    data = [("A", 100.0), ("A", 200.0)]
    df = spark.createDataFrame(data, ["category", "amount"])

    result = df.transform(summary_stats(value_col="amount", group_col="category"))

    row = result.collect()[0]
    assert row["count"] == 2
    assert abs(row["mean"] - 150.0) < 0.001
