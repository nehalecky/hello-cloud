"""Tests for dataset loaders."""

from hellocloud.io import PiedPiperLoader
from hellocloud.timeseries import TimeSeries


class TestPiedPiperLoader:
    """Test PiedPiperLoader functionality."""

    def test_load_creates_timeseries(self, spark):
        """Should load DataFrame and create TimeSeries."""
        df = spark.createDataFrame(
            [
                ("2025-09-01", "AWS", "acc1", "us-east-1", "Compute", "Standard", 100.0),
            ],
            [
                "usage_date",
                "cloud_provider",
                "cloud_account_id",
                "region",
                "product_family",
                "usage_type",
                "materialized_cost",
            ],
        )

        ts = PiedPiperLoader.load(df)

        assert isinstance(ts, TimeSeries)

    def test_load_renames_columns(self, spark):
        """Should rename usage_date->date and materialized_cost->cost."""
        df = spark.createDataFrame(
            [
                ("2025-09-01", "AWS", 100.0),
            ],
            ["usage_date", "cloud_provider", "materialized_cost"],
        )

        ts = PiedPiperLoader.load(
            df, hierarchy=["cloud_provider"], drop_cols=[]  # Don't drop anything for this test
        )

        assert "date" in ts.df.columns
        assert "cost" in ts.df.columns
        assert "usage_date" not in ts.df.columns
        assert "materialized_cost" not in ts.df.columns

    def test_load_drops_low_info_columns(self, spark):
        """Should drop UUID and redundant cost columns."""
        df = spark.createDataFrame(
            [
                ("2025-09-01", "AWS", "uuid-123", 100.0, 95.0, 98.0, 100.0, 105.0),
            ],
            [
                "usage_date",
                "cloud_provider",
                "billing_event_id",
                "materialized_cost",
                "materialized_discounted_cost",
                "materialized_amortized_cost",
                "materialized_invoiced_cost",
                "materialized_public_cost",
            ],
        )

        ts = PiedPiperLoader.load(df, hierarchy=["cloud_provider"])

        # Should drop redundant cost columns
        assert "materialized_discounted_cost" not in ts.df.columns
        assert "materialized_amortized_cost" not in ts.df.columns
        assert "materialized_invoiced_cost" not in ts.df.columns
        assert "materialized_public_cost" not in ts.df.columns
        # Should drop UUID column
        assert "billing_event_id" not in ts.df.columns
        # Should keep the base cost (renamed)
        assert "cost" in ts.df.columns

    def test_load_uses_default_hierarchy(self, spark):
        """Should use default PiedPiper hierarchy if not specified."""
        df = spark.createDataFrame(
            [
                ("2025-09-01", "AWS", "acc1", "us-east-1", "Compute", "Standard", 100.0),
            ],
            [
                "usage_date",
                "cloud_provider",
                "cloud_account_id",
                "region",
                "product_family",
                "usage_type",
                "materialized_cost",
            ],
        )

        ts = PiedPiperLoader.load(df)

        assert ts.hierarchy == [
            "cloud_provider",
            "cloud_account_id",
            "region",
            "product_family",
            "usage_type",
        ]

    def test_load_accepts_custom_hierarchy(self, spark):
        """Should allow overriding default hierarchy."""
        df = spark.createDataFrame(
            [
                ("2025-09-01", "AWS", "acc1", 100.0),
            ],
            ["usage_date", "cloud_provider", "cloud_account_id", "materialized_cost"],
        )

        ts = PiedPiperLoader.load(df, hierarchy=["cloud_provider", "cloud_account_id"])

        assert ts.hierarchy == ["cloud_provider", "cloud_account_id"]
