"""Dataset loaders for creating TimeSeries instances."""

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pyspark.sql import DataFrame

from hellocloud.timeseries import TimeSeries


class PiedPiperLoader:
    """
    Load PiedPiper billing data with EDA-informed defaults.

    Applies column renames, drops low-information columns,
    and creates TimeSeries with standard hierarchy.
    """

    # Default hierarchy from EDA analysis
    DEFAULT_HIERARCHY = [
        "cloud_provider",
        "cloud_account_id",
        "region",
        "product_family",
        "usage_type",
    ]

    # Column renames for standardization
    COLUMN_RENAMES = {"usage_date": "date", "materialized_cost": "cost"}

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
        # Apply column renames
        for old_name, new_name in PiedPiperLoader.COLUMN_RENAMES.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # Drop low-info columns
        cols_to_drop = drop_cols if drop_cols is not None else PiedPiperLoader.DROP_COLUMNS
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(*existing_cols_to_drop)

        # Create TimeSeries
        return TimeSeries.from_dataframe(
            df,
            hierarchy=hierarchy if hierarchy is not None else PiedPiperLoader.DEFAULT_HIERARCHY,
            metric_col=metric_col,
            time_col=time_col,
        )


class IOPSLoader:
    """
    Load and preprocess IOPS time series data.

    Provides standard train/val/test splitting for consistent experimentation.
    """

    @staticmethod
    def get_standard_splits(
        y_train_full: NDArray,
        subsample_factor: int = 20,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        random_seed: int = 42,
    ) -> dict:
        """Standard train/val/test splits for IOPS experiments.

        Used across all forecasting notebooks for consistency.

        Args:
            y_train_full: Full training data (146K samples)
            subsample_factor: Take every Nth sample (default: 20)
            train_frac: Fraction for training (default: 0.8)
            val_frac: Fraction for validation (default: 0.1)
            random_seed: Random seed for reproducibility

        Returns:
            dict with keys:
                - y_train: Subsampled data
                - y_train_split: Training portion
                - y_val_split: Validation portion
                - y_test_split: Test portion
                - subsample_indices: Indices used
        """
        # Set seed
        np.random.seed(random_seed)

        # Subsample
        subsample_indices = np.arange(0, len(y_train_full), subsample_factor)
        y_train = y_train_full[subsample_indices]

        logger.info(f"Subsampling: {len(y_train_full):,} → {len(y_train):,} ({subsample_factor}x)")

        # Split
        n_train = int(train_frac * len(y_train))
        n_val = int(val_frac * len(y_train))

        y_train_split = y_train[:n_train]
        y_val_split = y_train[n_train : n_train + n_val]
        y_test_split = y_train[n_train + n_val :]

        logger.info(
            f"Splits - Train: {len(y_train_split):,} | Val: {len(y_val_split):,} | Test: {len(y_test_split):,}"
        )

        return {
            "y_train": y_train,
            "y_train_split": y_train_split,
            "y_val_split": y_val_split,
            "y_test_split": y_test_split,
            "subsample_indices": subsample_indices,
        }
