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

    def _resolve_grain(self, grain: list[str]) -> list[str]:
        """
        Validate grain is subset of hierarchy and return in hierarchy order.

        Args:
            grain: List of column names defining the grain

        Returns:
            Grain columns in hierarchy order

        Raises:
            TimeSeriesError: If grain contains columns not in hierarchy
        """
        grain_set = set(grain)
        hierarchy_set = set(self.hierarchy)

        # Check for invalid columns
        invalid = grain_set - hierarchy_set
        if invalid:
            raise TimeSeriesError(
                f"Invalid grain columns: {invalid}. "
                f"Must be subset of hierarchy: {self.hierarchy}"
            )

        # Return in hierarchy order
        return [col for col in self.hierarchy if col in grain_set]

    def filter(self, **entity_keys) -> "TimeSeries":
        """
        Filter to specific entity by hierarchy column values.

        Args:
            **entity_keys: Column name/value pairs to filter on
                          (must be columns in hierarchy)

        Returns:
            New TimeSeries with filtered DataFrame

        Raises:
            TimeSeriesError: If filter column not in hierarchy

        Example:
            ts.filter(provider="AWS", account="acc1")
        """
        from pyspark.sql import functions as F

        # Validate all filter columns are in hierarchy
        invalid = set(entity_keys.keys()) - set(self.hierarchy)
        if invalid:
            raise TimeSeriesError(
                f"Invalid filter column(s): {invalid}. "
                f"Must be columns in hierarchy: {self.hierarchy}"
            )

        # Apply filters
        filtered_df = self.df
        for col, value in entity_keys.items():
            filtered_df = filtered_df.filter(F.col(col) == value)

        # Return new TimeSeries with filtered data
        return TimeSeries(
            df=filtered_df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def sample(self, grain: list[str], n: int = 1) -> "TimeSeries":
        """
        Sample n random entities at specified grain level.

        Args:
            grain: Column names defining the grain (must be subset of hierarchy)
            n: Number of entities to sample (default: 1)

        Returns:
            New TimeSeries with sampled entities

        Example:
            ts.sample(grain=["account", "region"], n=10)
        """
        from pyspark.sql import functions as F

        # Validate and resolve grain
        grain_cols = self._resolve_grain(grain)

        # Get unique entities at grain
        entities_df = self.df.select(*grain_cols).distinct()
        total_entities = entities_df.count()

        # Warn if requesting more than available
        if n > total_entities:
            logger.warning(
                f"Requested {n} entities but only {total_entities} exist at grain {grain}. "
                f"Returning all {total_entities}."
            )
            n = total_entities

        # Sample entities randomly
        sampled_entities = entities_df.orderBy(F.rand()).limit(n)

        # Join back to get full time series for sampled entities
        sampled_df = self.df.join(sampled_entities, on=grain_cols, how="inner")

        return TimeSeries(
            df=sampled_df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def aggregate(self, grain: list[str]) -> "TimeSeries":
        """
        Aggregate metric to coarser grain level.

        Sums metric values across entities, grouping by grain + time.

        Args:
            grain: Column names defining target grain (must be subset of hierarchy)

        Returns:
            New TimeSeries aggregated to specified grain

        Example:
            # Aggregate from account+region to just account
            ts.aggregate(grain=["account"])
        """
        from pyspark.sql import functions as F

        # Validate and resolve grain
        grain_cols = self._resolve_grain(grain)

        # Check if already at requested grain
        current_grain = [col for col in self.hierarchy if col in self.df.columns]
        if set(grain_cols) == set(current_grain):
            logger.info(f"Data already at grain {grain}. Returning copy.")
            return TimeSeries(
                df=self.df,
                hierarchy=self.hierarchy,
                metric_col=self.metric_col,
                time_col=self.time_col,
            )

        # Group by grain + time, sum metric
        group_cols = grain_cols + [self.time_col]
        agg_df = self.df.groupBy(*group_cols).agg(F.sum(self.metric_col).alias(self.metric_col))

        # New hierarchy only includes grain columns (in original hierarchy order)
        new_hierarchy = grain_cols

        return TimeSeries(
            df=agg_df,
            hierarchy=new_hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def summary_stats(self, grain: list[str] | None = None) -> DataFrame:
        """
        Compute summary statistics for the time series.

        Args:
            grain: Optional grain to aggregate to before computing stats.
                  If None, uses current grain of the data.

        Returns:
            PySpark DataFrame with entity keys and summary statistics
            (count, mean, std, min, max)

        Example:
            stats = ts.summary_stats()  # Stats at current grain
            stats = ts.summary_stats(grain=["account"])  # Aggregate first
        """
        from pyspark.sql import functions as F

        # Aggregate to target grain if specified
        if grain is not None:
            ts_for_stats = self.aggregate(grain)
        else:
            ts_for_stats = self

        # Identify entity columns (hierarchy columns present in data)
        entity_cols = [col for col in ts_for_stats.hierarchy if col in ts_for_stats.df.columns]

        # Group by entity and compute stats on metric over time
        stats_df = ts_for_stats.df.groupBy(*entity_cols).agg(
            F.count(self.metric_col).alias("count"),
            F.mean(self.metric_col).alias("mean"),
            F.stddev(self.metric_col).alias("std"),
            F.min(self.metric_col).alias("min"),
            F.max(self.metric_col).alias("max"),
        )

        return stats_df
