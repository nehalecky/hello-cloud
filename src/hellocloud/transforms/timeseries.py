"""
Time series transformation functions using Ibis pipe patterns.

All functions return closures that accept an Ibis table and return a transformed table.
This enables clean composition via the .pipe() method.

Examples:
    >>> # Calculate percentage change
    >>> df.pipe(pct_change('cost', 'date', partition_by='resource_id'))

    >>> # Chain multiple transformations
    >>> result = (
    ...     df
    ...     .pipe(pct_change('cost', 'date', partition_by='resource_id'))
    ...     .pipe(rolling_average('cost', 'date', window_size=7, partition_by='resource_id'))
    ... )
"""

import ibis
from ibis import _


def pct_change(
    value_col: str,
    time_col: str,
    partition_by: str | list[str] | None = None,
    periods: int = 1,
    suffix: str = "_pct_change",
):
    """
    Calculate fractional change over time using window functions.

    Formula: (value - lag(value, periods)) / lag(value, periods)

    Returns fraction (e.g., 0.25 for 25% increase, -0.30 for 30% decrease).

    Args:
        value_col: Column to calculate percentage change for
        time_col: Time column for ordering
        partition_by: Column(s) to partition by (e.g., 'resource_id')
        periods: Number of periods to shift (default: 1)
        suffix: Suffix for new column name (default: '_pct_change')

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(pct_change('cpu_usage', 'timestamp', partition_by='resource_id'))
        >>> # Returns fractional change: 0.20 = 20% increase, -0.30 = 30% decrease
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        # Create lag expression
        lag_expr = _[value_col].lag(periods).over(group_by=partition_cols, order_by=_[time_col])

        # Calculate fractional change (not multiplied by 100)
        pct_change_expr = ((_[value_col] - lag_expr) / lag_expr).name(f"{value_col}{suffix}")

        return t.mutate(pct_change_expr)

    return inner


def rolling_average(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: str | list[str] | None = None,
    suffix: str | None = None,
):
    """
    Calculate rolling average over time window.

    Args:
        value_col: Column to calculate rolling average for
        time_col: Time column for ordering
        window_size: Number of periods to include in window
        partition_by: Column(s) to partition by
        suffix: Suffix for new column name (default: '_rolling_{window_size}')

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(rolling_average('cpu_usage', 'timestamp',
        ...                         window_size=7, partition_by='resource_id'))
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        col_suffix = suffix or f"_rolling_{window_size}"

        rolling_expr = (
            _[value_col]
            .mean()
            .over(
                group_by=partition_cols,
                order_by=_[time_col],
                rows=(-(window_size - 1), 0),  # preceding rows to current
            )
            .name(f"{value_col}{col_suffix}")
        )

        return t.mutate(rolling_expr)

    return inner


def rolling_std(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: str | list[str] | None = None,
    suffix: str | None = None,
):
    """
    Calculate rolling standard deviation over time window.

    Args:
        value_col: Column to calculate rolling std for
        time_col: Time column for ordering
        window_size: Number of periods to include in window
        partition_by: Column(s) to partition by
        suffix: Suffix for new column name (default: '_rolling_std_{window_size}')

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(rolling_std('cost', 'date', window_size=30, partition_by='account_id'))
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        col_suffix = suffix or f"_rolling_std_{window_size}"

        rolling_expr = (
            _[value_col]
            .std()
            .over(group_by=partition_cols, order_by=_[time_col], rows=(-(window_size - 1), 0))
            .name(f"{value_col}{col_suffix}")
        )

        return t.mutate(rolling_expr)

    return inner


def cumulative_sum(
    value_col: str,
    time_col: str,
    partition_by: str | list[str] | None = None,
    suffix: str = "_cumsum",
):
    """
    Calculate cumulative sum over time.

    Args:
        value_col: Column to calculate cumulative sum for
        time_col: Time column for ordering
        partition_by: Column(s) to partition by
        suffix: Suffix for new column name (default: '_cumsum')

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(cumulative_sum('cost', 'date', partition_by='resource_id'))
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        cumsum_expr = (
            _[value_col]
            .sum()
            .over(
                group_by=partition_cols,
                order_by=_[time_col],
                rows=(None, 0),  # All preceding rows to current
            )
            .name(f"{value_col}{suffix}")
        )

        return t.mutate(cumsum_expr)

    return inner


def add_lag_features(
    value_col: str,
    time_col: str,
    lags: list[int],
    partition_by: str | list[str] | None = None,
):
    """
    Add multiple lag features for a column.

    Args:
        value_col: Column to create lag features for
        time_col: Time column for ordering
        lags: List of lag periods (e.g., [1, 7, 30])
        partition_by: Column(s) to partition by

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(add_lag_features('cost', 'date',
        ...                          lags=[1, 7, 30], partition_by='resource_id'))
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        mutations = {}
        for lag in lags:
            lag_expr = _[value_col].lag(lag).over(group_by=partition_cols, order_by=_[time_col])
            mutations[f"{value_col}_lag_{lag}"] = lag_expr

        return t.mutate(**mutations)

    return inner


def time_features(
    time_col: str,
    components: list[str] | None = None,
):
    """
    Extract common time-based features from a timestamp column.

    Args:
        time_col: Timestamp column to extract features from
        components: List of components to extract. Options:
            - 'hour': Hour of day (0-23)
            - 'day_of_week': Day of week (0=Monday, 6=Sunday)
            - 'day_of_month': Day of month (1-31)
            - 'month': Month (1-12)
            - 'quarter': Quarter (1-4)
            - 'year': Year
            Default: ['hour', 'day_of_week', 'month']

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(time_features('timestamp', components=['hour', 'day_of_week', 'quarter']))
    """
    if components is None:
        components = ["hour", "day_of_week", "month"]

    def inner(t: ibis.Table) -> ibis.Table:
        mutations = {}

        if "hour" in components:
            mutations["hour"] = _[time_col].hour()
        if "day_of_week" in components:
            mutations["day_of_week"] = _[time_col].day_of_week.index()
        if "day_of_month" in components:
            mutations["day_of_month"] = _[time_col].day()
        if "month" in components:
            mutations["month"] = _[time_col].month()
        if "quarter" in components:
            mutations["quarter"] = _[time_col].quarter()
        if "year" in components:
            mutations["year"] = _[time_col].year()

        return t.mutate(**mutations)

    return inner


def summary_stats(  # noqa: C901
    value_col: str | None = None,
    group_by: str | list[str] | None = None,
    stats: list[str] | None = None,
):
    """
    Aggregate to point statistics (mean, median, std, min, max, quantiles).

    If group_by is provided, computes row count per group then summarizes those counts.
    Otherwise, summarizes the value_col directly.

    Args:
        value_col: Column to compute statistics for (optional if group_by specified)
        group_by: Optional column(s) to group by. If provided, counts rows per group
                  then summarizes the counts. Enables: df.pipe(summary_stats(group_by='date'))
        stats: List of statistics to compute. Options:
            - 'mean': Average value
            - 'median': Median value
            - 'std': Standard deviation
            - 'min': Minimum value
            - 'max': Maximum value
            - 'q25': 25th percentile
            - 'q75': 75th percentile
            Default: ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']

    Returns:
        Closure that accepts Ibis table and returns single-row summary table

    Example:
        >>> # Idiomatic: group by date and summarize counts
        >>> summary = df.pipe(summary_stats(group_by='date'))
        >>> # Returns: count_mean, count_median, count_std, count_min, count_max, count_q25, count_q75

        >>> # Or summarize existing column
        >>> summary = (
        ...     df.group_by('date')
        ...     .agg(cost=_.cost.sum())
        ...     .pipe(summary_stats('cost'))
        ... )
    """
    if stats is None:
        stats = ["mean", "median", "std", "min", "max", "q25", "q75"]

    def inner(t: ibis.Table) -> ibis.Table:
        # If group_by specified, first aggregate to counts
        if group_by is not None:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
            t = t.group_by(group_cols).agg(count=_.count())
            target_col = "count"
        else:
            target_col = value_col

        agg_exprs = {}

        if "mean" in stats:
            agg_exprs[f"{target_col}_mean"] = _[target_col].mean()
        if "median" in stats:
            agg_exprs[f"{target_col}_median"] = _[target_col].median()
        if "std" in stats:
            agg_exprs[f"{target_col}_std"] = _[target_col].std()
        if "min" in stats:
            agg_exprs[f"{target_col}_min"] = _[target_col].min()
        if "max" in stats:
            agg_exprs[f"{target_col}_max"] = _[target_col].max()
        if "q25" in stats:
            agg_exprs[f"{target_col}_q25"] = _[target_col].quantile(0.25)
        if "q75" in stats:
            agg_exprs[f"{target_col}_q75"] = _[target_col].quantile(0.75)

        return t.agg(**agg_exprs)

    return inner


def add_rolling_stats(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: str | list[str] | None = None,
    stats: list[str] | None = None,
):
    """
    Add multiple rolling statistics at once (mean, std, min, max, median).

    Args:
        value_col: Column to calculate statistics for
        time_col: Time column for ordering
        window_size: Number of periods to include in window
        partition_by: Column(s) to partition by
        stats: List of statistics to compute. Options:
            - 'mean': Rolling average
            - 'std': Rolling standard deviation
            - 'min': Rolling minimum
            - 'max': Rolling maximum
            - 'median': Rolling median
            Default: ['mean', 'std']

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> df.pipe(add_rolling_stats('cost', 'date', window_size=30,
        ...                           partition_by='resource_id',
        ...                           stats=['mean', 'std', 'min', 'max']))
    """
    if stats is None:
        stats = ["mean", "std"]

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        mutations = {}
        window_spec = {
            "group_by": partition_cols,
            "order_by": _[time_col],
            "rows": (-(window_size - 1), 0),
        }

        if "mean" in stats:
            mutations[f"{value_col}_rolling_mean_{window_size}"] = (
                _[value_col].mean().over(**window_spec)
            )
        if "std" in stats:
            mutations[f"{value_col}_rolling_std_{window_size}"] = (
                _[value_col].std().over(**window_spec)
            )
        if "min" in stats:
            mutations[f"{value_col}_rolling_min_{window_size}"] = (
                _[value_col].min().over(**window_spec)
            )
        if "max" in stats:
            mutations[f"{value_col}_rolling_max_{window_size}"] = (
                _[value_col].max().over(**window_spec)
            )
        if "median" in stats:
            mutations[f"{value_col}_rolling_median_{window_size}"] = (
                _[value_col].median().over(**window_spec)
            )

        return t.mutate(**mutations)

    return inner


def add_z_score(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: str | list[str] | None = None,
    suffix: str = "_zscore",
):
    """
    Calculate rolling z-score for anomaly detection.

    Formula: (value - rolling_mean) / rolling_std

    Args:
        value_col: Column to calculate z-score for
        time_col: Time column for ordering
        window_size: Number of periods for rolling statistics
        partition_by: Column(s) to partition by
        suffix: Suffix for new column name (default: '_zscore')

    Returns:
        Closure that accepts Ibis table and returns transformed table

    Example:
        >>> # Flag anomalies as |z-score| > 3
        >>> result = (
        ...     df.pipe(add_z_score('cost', 'date', window_size=30,
        ...                        partition_by='resource_id'))
        ...     .mutate(is_anomaly=(_.cost_zscore.abs() > 3))
        ... )
    """

    def inner(t: ibis.Table) -> ibis.Table:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by or []

        window_spec = {
            "group_by": partition_cols,
            "order_by": _[time_col],
            "rows": (-(window_size - 1), 0),
        }

        rolling_mean = _[value_col].mean().over(**window_spec)
        rolling_std = _[value_col].std().over(**window_spec)

        z_score_expr = ((_[value_col] - rolling_mean) / rolling_std).name(f"{value_col}{suffix}")

        return t.mutate(z_score_expr)

    return inner
