"""Dataset loaders for creating TimeSeries instances."""

from loguru import logger
from pyspark.sql import DataFrame

from hellocloud.timeseries import TimeSeries


class PiedPiperLoader:
    """
    Load PiedPiper billing data with EDA-informed defaults.

    Applies column renames, drops low-information columns,
    and creates TimeSeries with standard hierarchy.
    """

    # Default hierarchy from EDA analysis (using renamed column names)
    DEFAULT_HIERARCHY = [
        "provider",
        "account",
        "region",
        "product",
        "usage",
    ]

    # Column renames for standardization
    COLUMN_RENAMES = {
        "usage_date": "date",
        "materialized_cost": "cost",
        "cloud_provider": "provider",
        "cloud_account_id": "account",
        "product_family": "product",
        "usage_type": "usage",
    }

    # Low-information columns to drop (from EDA)
    DROP_COLUMNS = [
        # UUID/primary keys (>90% cardinality)
        "billing_event_id",
        # Redundant cost variants (>0.95 correlation)
        "materialized_discounted_cost",
        "materialized_amortized_cost",
        "materialized_invoiced_cost",
        "materialized_public_cost",
    ]

    @staticmethod
    def load(
        df: DataFrame,
        hierarchy: list[str] | None = None,
        metric_col: str = "cost",
        time_col: str = "date",
        drop_cols: list[str] | None = None,
    ) -> TimeSeries:
        """
        Load PiedPiper data into TimeSeries.

        Args:
            df: PySpark DataFrame with PiedPiper billing data
            hierarchy: Custom hierarchy (default: DEFAULT_HIERARCHY)
            metric_col: Metric column name after rename (default: "cost")
            time_col: Time column name after rename (default: "date")
            drop_cols: Columns to drop (default: DROP_COLUMNS)

        Returns:
            TimeSeries instance with cleaned data
        """
        initial_count = df.count()
        initial_cols = len(df.columns)

        logger.info(f"Loading PiedPiper data: {initial_cols} columns, {initial_count:,} records")

        # Log column mapping usage
        logger.info("Using default column mapping")

        # Apply column renames
        rename_map = {}
        for old_name, new_name in PiedPiperLoader.COLUMN_RENAMES.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)
                rename_map[old_name] = new_name

        # Drop low-info columns
        cols_to_drop = drop_cols if drop_cols is not None else PiedPiperLoader.DROP_COLUMNS
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(*existing_cols_to_drop)
            logger.info(
                f"Filtered to {len(df.columns)} columns (removed {len(existing_cols_to_drop)})"
            )

        # Log renames
        if rename_map:
            rename_strs = [f"{old} → {new}" for old, new in rename_map.items()]
            logger.info(f"Renamed {len(rename_map)} columns: {', '.join(rename_strs)}")

        # Create TimeSeries
        final_hierarchy = hierarchy if hierarchy is not None else PiedPiperLoader.DEFAULT_HIERARCHY
        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=final_hierarchy,
            metric_col=metric_col,
            time_col=time_col,
        )

        logger.info(
            f"TimeSeries created: hierarchy={final_hierarchy}, metric={metric_col}, "
            f"time={time_col}, records={initial_count:,}"
        )

        return ts
