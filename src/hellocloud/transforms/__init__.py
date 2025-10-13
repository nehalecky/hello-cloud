"""Data transformations for time series and analytics.

Backends:
- spark: PySpark transforms (preferred, install with: uv sync)
- ibis: Ibis transforms (DEPRECATED, backward compatibility only)
"""

import warnings

# Try importing Spark transforms
try:
    from hellocloud.transforms.spark import pct_change, summary_stats

    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

# Fallback to Ibis with deprecation warning
if not _SPARK_AVAILABLE:
    warnings.warn(
        "Using deprecated Ibis transforms. Migrate to PySpark: uv sync",
        DeprecationWarning,
        stacklevel=2,
    )
    from hellocloud.transforms.ibis import (
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
    "pct_change",
    "summary_stats",
]
