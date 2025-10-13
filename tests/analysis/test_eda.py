"""Tests for exploratory data analysis functions."""

import pytest
from pyspark.sql import functions as F

from hellocloud.analysis.eda import correlation_pairs


class TestCorrelationPairs:
    """Test correlation_pairs function."""

    def test_returns_pyspark_dataframe_with_correct_schema(self, spark):
        """Returns PySpark DataFrame with (col1, col2, correlation) schema."""
        # Arrange: Create test data with known correlations
        data = [
            (1.0, 2.0, 3.0),
            (2.0, 4.0, 6.0),
            (3.0, 6.0, 9.0),
            (4.0, 8.0, 12.0),
        ]
        df = spark.createDataFrame(data, ["a", "b", "c"])

        # Act: Compute correlations
        result = correlation_pairs(df, ["a", "b", "c"])

        # Assert: Check schema
        assert result.columns == ["col1", "col2", "correlation"]

        # Assert: Check we have the right number of pairs (3 choose 2 = 3)
        assert result.count() == 3

        # Assert: All combinations present (order doesn't matter)
        pairs = {(row["col1"], row["col2"]) for row in result.collect()}
        expected_pairs = {("a", "b"), ("a", "c"), ("b", "c")}
        assert pairs == expected_pairs

    def test_computes_perfect_positive_correlation(self, spark):
        """Computes correlation = 1.0 for perfectly correlated columns."""
        # Arrange: b = 2*a (perfect positive correlation)
        data = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)]
        df = spark.createDataFrame(data, ["a", "b"])

        # Act
        result = correlation_pairs(df, ["a", "b"])

        # Assert: Should be exactly 1.0
        corr_value = result.select("correlation").first()[0]
        assert abs(corr_value - 1.0) < 0.0001

    def test_computes_perfect_negative_correlation(self, spark):
        """Computes correlation = -1.0 for perfectly anti-correlated columns."""
        # Arrange: b = -a (perfect negative correlation)
        data = [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)]
        df = spark.createDataFrame(data, ["a", "b"])

        # Act
        result = correlation_pairs(df, ["a", "b"])

        # Assert: Should be exactly -1.0
        corr_value = result.select("correlation").first()[0]
        assert abs(corr_value - (-1.0)) < 0.0001

    def test_accepts_method_parameter(self, spark):
        """Accepts method parameter for pearson correlation."""
        # Arrange
        data = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
        df = spark.createDataFrame(data, ["a", "b"])

        # Act: Explicitly specify pearson method
        result = correlation_pairs(df, ["a", "b"], method="pearson")

        # Assert: Should succeed and return correlation
        assert result.count() == 1
        assert "correlation" in result.columns
