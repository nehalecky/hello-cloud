"""PySpark DataFrame transforms for time series and analytics."""

from collections.abc import Callable

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def pct_change(
    value_col: str,
    order_col: str,
    group_cols: list[str] | None = None,
    output_col: str = "pct_change",
) -> Callable[[DataFrame], DataFrame]:
    """
    Calculate percentage change from previous row.

    Returns fractional (decimal) change: 0.10 for 10% increase, -0.30 for 30% decrease.
    This follows pandas convention, not percentage points.

    Formula: (current - previous) / previous

    Behavior:
    - First row in each group returns NULL (no previous value)
    - Uses window functions with lag() for previous value
    - Null values propagate (null previous → null result)
    - Groups are partitioned independently when group_cols specified

    Args:
        value_col: Column with numeric values to calculate change on
        order_col: Column to order by (typically time/date)
        group_cols: Optional list of grouping columns (e.g., ["resource_id"])
                   If None, treats entire DataFrame as single group
        output_col: Name for output column (default: "pct_change")

    Returns:
        Transform function for DataFrame.transform()

    Example:
        >>> # Basic usage (whole DataFrame)
        >>> df.transform(pct_change("cost", "date"))

        >>> # With grouping (per-resource change)
        >>> df.transform(pct_change("cost", "date", ["resource_id"]))

        >>> # Custom output column
        >>> df.transform(pct_change("cost", "date", output_col="cost_change"))
    """

    def transform(df: DataFrame) -> DataFrame:
        # Define window: partition by groups, order by time
        window_spec = Window.orderBy(order_col)
        if group_cols:
            window_spec = Window.partitionBy(*group_cols).orderBy(order_col)

        # Get previous value using lag
        prev_col = F.lag(value_col, 1).over(window_spec)

        # Calculate percentage change: (current - prev) / prev
        pct_change_expr = (F.col(value_col) - prev_col) / prev_col

        return df.withColumn(output_col, pct_change_expr)

    return transform


def summary_stats(
    value_col: str,
    group_col: str,
) -> Callable[[DataFrame], DataFrame]:
    """
    Calculate summary statistics by group.

    Computes standard statistical aggregations for a numeric column grouped by
    a categorical column. Returns one row per unique group value.

    Statistics computed:
    - count: Number of non-null values in group
    - mean: Arithmetic mean of values
    - stddev: Sample standard deviation (N-1 denominator)
    - min: Minimum value in group
    - max: Maximum value in group

    Behavior:
    - Null values excluded from all calculations
    - Groups with all nulls return: count=0, mean=null, stddev=null, min=null, max=null
    - Original DataFrame columns not preserved (returns aggregated result)

    Args:
        value_col: Column with numeric values to aggregate
        group_col: Column to group by (typically categorical)

    Returns:
        Transform function for DataFrame.transform()
        Result DataFrame has columns: [group_col, count, mean, stddev, min, max]

    Example:
        >>> # Daily cost statistics
        >>> df.transform(summary_stats("cost", "date"))

        >>> # Per-region resource statistics
        >>> df.transform(summary_stats("cpu_utilization", "region"))
    """

    def transform(df: DataFrame) -> DataFrame:
        return df.groupBy(group_col).agg(
            F.count(value_col).alias("count"),
            F.mean(value_col).alias("mean"),
            F.stddev(value_col).alias("stddev"),
            F.min(value_col).alias("min"),
            F.max(value_col).alias("max"),
        )

    return transform
