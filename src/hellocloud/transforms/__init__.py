"""
Composable transformation functions for Ibis tables using pipe patterns.

This module provides parameterized transformation functions that integrate
seamlessly with Ibis's pipe() method for clean, readable data pipelines.

All functions follow the closure pattern:
    def transform(params):
        def inner(t: ibis.Table) -> ibis.Table:
            return t.mutate(...)
        return inner

Usage:
    result = (
        table
        .pipe(pct_change('value', 'time', partition_by='id'))
        .pipe(rolling_average('value', 'time', window_size=7))
    )
"""

from .timeseries import (
    add_lag_features,
    add_rolling_stats,
    add_z_score,
    cumulative_sum,
    pct_change,
    rolling_average,
    rolling_std,
    summary_stats,
    time_features,
)

__all__ = [
    # Time series transformations
    "pct_change",
    "rolling_average",
    "rolling_std",
    "cumulative_sum",
    "add_lag_features",
    "time_features",
    # Descriptive statistics
    "summary_stats",
    "add_rolling_stats",
    "add_z_score",
]
