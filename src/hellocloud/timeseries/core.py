"""Core TimeSeries class for hierarchical time series analysis."""

from loguru import logger
from pyspark.sql import DataFrame


class TimeSeriesError(Exception):
    """Base exception for TimeSeries operations."""

    pass


class TimeSeries:
    """
    Wrapper around PySpark DataFrame for hierarchical time series analysis.

    Attributes:
        df: PySpark DataFrame containing time series data
        hierarchy: Ordered list of key columns (coarsest to finest grain)
        metric_col: Name of the metric/value column
        time_col: Name of the timestamp column
    """

    def __init__(self, df: DataFrame, hierarchy: list[str], metric_col: str, time_col: str):
        """
        Initialize TimeSeries wrapper.

        Args:
            df: PySpark DataFrame with time series data
            hierarchy: Ordered key columns (e.g., ["provider", "account", "region"])
            metric_col: Name of metric column (e.g., "cost")
            time_col: Name of timestamp column (e.g., "date")

        Raises:
            TimeSeriesError: If required columns missing from DataFrame
        """
        self.df = df
        self.hierarchy = hierarchy
        self.metric_col = metric_col
        self.time_col = time_col
        self._cached_stats = {}

        # Validate columns exist
        self._validate_columns()

        # Warn if empty
        if df.count() == 0:
            logger.warning(
                "Creating TimeSeries from empty DataFrame. Operations will return empty results."
            )

    @classmethod
    def from_dataframe(
        cls, df: DataFrame, hierarchy: list[str], metric_col: str = "cost", time_col: str = "date"
    ) -> "TimeSeries":
        """
        Factory method to create TimeSeries from DataFrame.

        Args:
            df: PySpark DataFrame with time series data
            hierarchy: Ordered key columns (e.g., ["provider", "account"])
            metric_col: Name of metric column (default: "cost")
            time_col: Name of timestamp column (default: "date")

        Returns:
            TimeSeries instance
        """
        return cls(df=df, hierarchy=hierarchy, metric_col=metric_col, time_col=time_col)

    def _validate_columns(self) -> None:
        """Validate that required columns exist in DataFrame."""
        df_cols = set(self.df.columns)

        # Check time column
        if self.time_col not in df_cols:
            raise TimeSeriesError(
                f"time_col '{self.time_col}' not found in DataFrame columns: {list(df_cols)}"
            )

        # Check metric column
        if self.metric_col not in df_cols:
            raise TimeSeriesError(
                f"metric_col '{self.metric_col}' not found in DataFrame columns: {list(df_cols)}"
            )

        # Check hierarchy columns
        for col in self.hierarchy:
            if col not in df_cols:
                raise TimeSeriesError(
                    f"hierarchy column '{col}' not found in DataFrame columns: {list(df_cols)}"
                )
