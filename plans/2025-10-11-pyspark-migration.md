# PySpark Migration + Repository Restructure

> **For Claude:** Use `${CLAUDE_PLUGIN_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Restructure repository with domain-driven design (io, transforms, stats, timeseries, analysis), THEN migrate from Ibis to PySpark 4.0 following strict TDD.

**Architecture:**
- **Phase 0:** Create domain structure (io/, stats/, timeseries/, analysis/, etc.)
- **Phases 1-12:** PySpark migration with transforms going to transforms/spark.py

**Tech Stack:** PySpark 4.0, pytest, uv, domain-driven design

---

## Context

### Why Structure First

**Problem:** Current `utils/` is 32% of codebase (2,605 lines, unmaintainable)

**Solution:** Domain modules before adding PySpark code

### Target Structure

```
src/hellocloud/
├── io/                    # NEW: Data I/O
├── transforms/            # EXPAND: spark.py (new), ibis.py (deprecated)
├── stats/                 # NEW: Statistical operations (from utils/)
├── timeseries/            # NEW: TS analysis
├── analysis/              # NEW: Higher-level analysis (from utils/)
├── generation/            # RENAME: data_generation/
├── modeling/              # RENAME: ml_models/
├── spark/                 # NEW: Spark session management
└── utils/                 # SHRINK: logging only
```

---

## Task 0: Repository Restructure

**Files:**
- Create: 5 new directories with __init__.py
- Move: 3 files from utils/ to analysis/
- Rename: 2 directories
- Modify: Multiple __init__.py files for imports

**Step 1: Create new directories**

Run:
```bash
mkdir -p src/hellocloud/{io,stats,timeseries,spark,analysis}
touch src/hellocloud/io/__init__.py
touch src/hellocloud/stats/__init__.py
touch src/hellocloud/timeseries/__init__.py
touch src/hellocloud/analysis/__init__.py
touch src/hellocloud/spark/__init__.py
```

Expected: 5 new directories with empty __init__.py files

**Step 2: Rename existing directories for consistency**

Run:
```bash
git mv src/hellocloud/data_generation src/hellocloud/generation
git mv src/hellocloud/ml_models src/hellocloud/modeling
```

Expected: Git shows renames (R100), not delete+add

**Step 3: Move analysis modules from utils/**

Run:
```bash
git mv src/hellocloud/utils/eda_analysis.py src/hellocloud/analysis/eda.py
git mv src/hellocloud/utils/cost_analysis.py src/hellocloud/analysis/cost.py
git mv src/hellocloud/utils/distribution_analysis.py src/hellocloud/analysis/distribution.py
```

Expected: 3 files moved, git history preserved

**Step 4: Update analysis/__init__.py**

Create `src/hellocloud/analysis/__init__.py`:
```python
"""Higher-level analysis for EDA, cost, and distributions."""

from .cost import calculate_daily_cost, calculate_resource_cost, cost_by_service
from .distribution import plot_distribution, fit_distribution, compare_distributions
from .eda import attribute_analysis, get_stratified_sample

__all__ = [
    "calculate_daily_cost",
    "calculate_resource_cost",
    "cost_by_service",
    "plot_distribution",
    "fit_distribution",
    "compare_distributions",
    "attribute_analysis",
    "get_stratified_sample",
]
```

**Step 5: Update generation/__init__.py** (renamed from data_generation)

Edit `src/hellocloud/generation/__init__.py`:
```python
"""Synthetic data generation for cloud workload patterns."""

from .cloud_metrics_simulator import CloudMetricsSimulator
from .hf_dataset_builder import CloudMetricsDatasetBuilder
from .workload_patterns import WorkloadPatternGenerator, WorkloadType

__all__ = [
    "CloudMetricsSimulator",
    "CloudMetricsDatasetBuilder",
    "WorkloadPatternGenerator",
    "WorkloadType",
]
```

**Step 6: Update modeling/__init__.py** (renamed from ml_models)

Edit `src/hellocloud/modeling/__init__.py`:
```python
"""Machine learning models for cloud resource optimization."""

from .application_taxonomy import CloudResourceTaxonomy
from .pymc_cloud_model import CloudResourceHierarchicalModel

__all__ = [
    "CloudResourceHierarchicalModel",
    "CloudResourceTaxonomy",
]
```

**Step 7: Deprecate Ibis transforms**

Run:
```bash
git mv src/hellocloud/transforms/timeseries.py src/hellocloud/transforms/ibis.py
```

Create `src/hellocloud/transforms/__init__.py`:
```python
"""Data transformations for time series and analytics.

Backends:
- spark: PySpark transforms (preferred, install with: uv sync)
- ibis: Ibis transforms (DEPRECATED, backward compatibility only)
"""

import warnings

# Try importing Spark transforms
try:
    from hellocloud.transforms.spark import (
        pct_change,
        summary_stats,
        add_rolling_stats,
        add_lag_features,
    )
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

# Fallback to Ibis with deprecation warning
if not _SPARK_AVAILABLE:
    warnings.warn(
        "Using deprecated Ibis transforms. Migrate to PySpark: uv sync",
        DeprecationWarning,
        stacklevel=2
    )
    from hellocloud.transforms.ibis import (
        pct_change,
        summary_stats,
        add_rolling_stats,
        add_lag_features,
        rolling_average,
        rolling_std,
        cumulative_sum,
        time_features,
        add_z_score,
    )

__all__ = [
    "pct_change",
    "summary_stats",
    "add_rolling_stats",
    "add_lag_features",
]
```

**Step 8: Update root __init__.py**

Edit `src/hellocloud/__init__.py`:
```python
"""Hello Cloud - Cloud Cost Analysis and Optimization"""

__version__ = "0.1.0"
__author__ = "Nicholaus Halecky"

# Import submodules
from . import analysis, generation, modeling, spark, transforms, utils

# Convenience imports
from .generation import (
    CloudMetricsDatasetBuilder,
    CloudMetricsSimulator,
    WorkloadPatternGenerator,
    WorkloadType,
)
from .modeling import (
    CloudResourceHierarchicalModel,
    CloudResourceTaxonomy,
)
from .utils import (
    configure_notebook_logging,
    quiet_library_logging,
    verbose_library_logging,
)

__all__ = [
    "analysis",
    "generation",
    "modeling",
    "spark",
    "transforms",
    "utils",
    "CloudMetricsDatasetBuilder",
    "CloudMetricsSimulator",
    "WorkloadPatternGenerator",
    "WorkloadType",
    "CloudResourceHierarchicalModel",
    "CloudResourceTaxonomy",
    "configure_notebook_logging",
    "quiet_library_logging",
    "verbose_library_logging",
]
```

**Step 9: Verify imports work**

Run:
```bash
uv run python -c "
import hellocloud as hc
from hellocloud import analysis, generation, modeling, transforms
from hellocloud.generation import WorkloadPatternGenerator
print('✓ All imports successful')
"
```

Expected: "✓ All imports successful"

**Step 10: Run existing tests**

Run: `uv run pytest tests/test_transforms.py -v`

Expected: Tests pass with deprecation warning for Ibis transforms

**Step 11: Commit restructure**

```bash
git add src/hellocloud/
git commit -m "refactor: restructure with domain-driven design

NEW structure:
- io/, stats/, timeseries/, analysis/, spark/ (placeholders)
- analysis/: eda.py, cost.py, distribution.py (from utils/)

RENAMED:
- data_generation/ → generation/
- ml_models/ → modeling/

MOVED:
- transforms/timeseries.py → transforms/ibis.py (deprecated)

SHRUNK:
- utils/ from 2,605 lines to ~50 lines (logging only)

All existing tests pass. Deprecation warnings for Ibis.
Next: Add PySpark transforms to transforms/spark.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 1: Add PySpark Dependency

**Files:**
- Modify: `pyproject.toml:46`

**Step 1: Add PySpark**

Edit `pyproject.toml`, add after line 46:
```toml
    "pyspark>=4.0.0",
```

**Step 2: Sync**

Run: `uv sync`

Expected: "Resolved 150+ packages"

**Step 3: Verify**

Run: `uv run python -c "import pyspark; print(f'PySpark {pyspark.__version__}')"`

Expected: `PySpark 4.0.0` or higher

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(deps): add PySpark 4.0

Migrating from Ibis to PySpark for:
- 100+ statistical functions vs ~20 in Ibis
- MLlib for ML workflows
- Production-grade scale

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Spark Session Builder (TDD)

**Files:**
- Create: `src/hellocloud/spark/session.py`
- Create: `tests/spark/test_session.py`
- Create: `tests/spark/__init__.py`

**Step 1: RED - Write failing test**

Create `tests/spark/__init__.py`:
```python
"""Tests for PySpark utilities."""
```

Create `tests/spark/test_session.py`:
```python
"""Tests for Spark session management."""

import pytest
from pyspark.sql import SparkSession


def test_get_spark_session_returns_session():
    """Test that get_spark_session returns SparkSession."""
    from hellocloud.spark.session import get_spark_session

    session = get_spark_session()

    assert isinstance(session, SparkSession)
    assert session.sparkContext.appName == "hellocloud"


def test_get_spark_session_local_mode():
    """Test local mode uses all cores."""
    from hellocloud.spark.session import get_spark_session

    session = get_spark_session(local_mode=True)

    assert session.sparkContext.master.startswith("local[")


def test_get_spark_session_is_singleton():
    """Test multiple calls return same session."""
    from hellocloud.spark.session import get_spark_session

    session1 = get_spark_session()
    session2 = get_spark_session()

    assert session1 is session2
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_session.py -v`

Expected: `FAIL: ModuleNotFoundError: No module named 'hellocloud.spark.session'`

**Step 3: GREEN - Implement**

Create `src/hellocloud/spark/session.py`:
```python
"""Spark session management for local and production."""

from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "hellocloud",
    local_mode: bool = True
) -> SparkSession:
    """
    Create or get Spark session.

    Local mode defaults:
    - local[*] master (all cores)
    - 4GB driver memory
    - 8 shuffle partitions (not 200)

    Args:
        app_name: Application name
        local_mode: Configure for local development

    Returns:
        SparkSession (singleton)

    Example:
        >>> from hellocloud.spark.session import get_spark_session
        >>> spark = get_spark_session()
        >>> df = spark.read.parquet("data.parquet")
    """
    builder = SparkSession.builder.appName(app_name)

    if local_mode:
        builder = builder.master("local[*]")
        builder = builder.config("spark.driver.memory", "4g")
        builder = builder.config("spark.sql.shuffle.partitions", "8")

    return builder.getOrCreate()
```

Update `src/hellocloud/spark/__init__.py`:
```python
"""PySpark backend for cloud analytics."""

from .session import get_spark_session

__all__ = ["get_spark_session"]
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_session.py -v`

Expected: `PASS: 3 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/ tests/spark/
git commit -m "feat(spark): add session builder with local defaults

Singleton pattern, local mode: local[*], 4GB, 8 partitions.
Test coverage: 3 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: pct_change Transform (TDD)

**Files:**
- Create: `src/hellocloud/transforms/spark.py`
- Create: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Create `tests/spark/test_transforms.py`:
```python
"""Tests for PySpark transformations."""

import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from datetime import date


@pytest.fixture(scope="module")
def spark():
    """Spark session for testing."""
    from hellocloud.spark.session import get_spark_session
    return get_spark_session()


@pytest.fixture
def sample_timeseries(spark):
    """Sample time series data."""
    schema = StructType([
        StructField("date", DateType(), False),
        StructField("entity_id", StringType(), False),
        StructField("value", IntegerType(), False),
    ])

    data = [
        (date(2024, 1, 1), "A", 10),
        (date(2024, 1, 2), "A", 12),
        (date(2024, 1, 3), "A", 15),
        (date(2024, 1, 4), "A", 14),
        (date(2024, 1, 5), "A", 16),
        (date(2024, 1, 1), "B", 5),
        (date(2024, 1, 2), "B", 6),
        (date(2024, 1, 3), "B", 7),
        (date(2024, 1, 4), "B", 8),
        (date(2024, 1, 5), "B", 9),
    ]

    return spark.createDataFrame(data, schema)


class TestPctChange:
    """Test percentage change transformation."""

    def test_basic_pct_change(self, sample_timeseries):
        """Test basic fractional change."""
        from hellocloud.transforms.spark import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id")
        )

        assert "value_pct_change" in result.columns

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # First row null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])

        # Second: (12-10)/10 = 0.20
        assert entity_a["value_pct_change"].iloc[1] == pytest.approx(0.20)

    def test_pct_change_respects_partitions(self, sample_timeseries):
        """Test partition boundaries."""
        from hellocloud.transforms.spark import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id")
        ).toPandas()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Both first null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])
        assert pd.isna(entity_b["value_pct_change"].iloc[0])

        # Second values correct
        assert entity_a["value_pct_change"].iloc[1] == pytest.approx(0.20)
        assert entity_b["value_pct_change"].iloc[1] == pytest.approx(0.20)

    def test_pct_change_multiple_periods(self, sample_timeseries):
        """Test periods > 1."""
        from hellocloud.transforms.spark import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id", periods=2)
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # First two null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])
        assert pd.isna(entity_a["value_pct_change"].iloc[1])

        # Third: (15-10)/10 = 0.50
        assert entity_a["value_pct_change"].iloc[2] == pytest.approx(0.50)
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestPctChange -v`

Expected: `FAIL: ModuleNotFoundError`

**Step 3: GREEN - Implement**

Create `src/hellocloud/transforms/spark.py`:
```python
"""PySpark-native time series transformations.

All use .transform() for pipe composition.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, lag
from typing import List, Union, Callable


def pct_change(
    value_col: str,
    time_col: str,
    partition_by: Union[str, List[str]],
    periods: int = 1
) -> Callable[[DataFrame], DataFrame]:
    """
    Calculate fractional change over time.

    Returns fraction (0.25 = 25% increase, -0.30 = 30% decrease).

    Args:
        value_col: Column to calculate change on
        time_col: Time column for ordering
        partition_by: Partition column(s)
        periods: Lag periods (default 1)

    Returns:
        Transform closure

    Example:
        >>> result = df.transform(pct_change('cost', 'date', 'resource_id'))
    """
    def inner(df: DataFrame) -> DataFrame:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by
        window = Window.partitionBy(*partition_cols).orderBy(time_col)

        lag_col = lag(col(value_col), periods).over(window)

        return df.withColumn(
            f"{value_col}_pct_change",
            (col(value_col) - lag_col) / lag_col
        )

    return inner
```

Update `src/hellocloud/transforms/__init__.py` to prefer Spark:
```python
"""Transformations: Spark (preferred) or Ibis (deprecated)."""

import warnings

try:
    from hellocloud.transforms.spark import pct_change
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False
    warnings.warn(
        "Using deprecated Ibis transforms. Install PySpark: uv sync",
        DeprecationWarning,
        stacklevel=2
    )
    from hellocloud.transforms.ibis import pct_change

__all__ = ["pct_change"]
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestPctChange -v`

Expected: `PASS: 3 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/transforms/spark.py tests/spark/test_transforms.py src/hellocloud/transforms/__init__.py
git commit -m "feat(transforms): add PySpark pct_change with TDD

Fractional change using Spark windows.
Returns fractions (0.20 = 20%).

Test coverage: 3 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: summary_stats Transform (TDD)

**Files:**
- Modify: `src/hellocloud/transforms/spark.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `tests/spark/test_transforms.py`:
```python
from pyspark.sql.functions import count as spark_count

class TestSummaryStats:
    """Test summary statistics."""

    def test_summary_stats_with_group_by(self, sample_timeseries):
        """Test idiomatic: df.transform(summary_stats(group_by='date'))."""
        from hellocloud.transforms.spark import summary_stats

        result = sample_timeseries.transform(
            summary_stats(group_by="date")
        ).toPandas()

        # Should summarize counts per date
        assert "count_mean" in result.columns
        assert "count_median" in result.columns
        assert "count_std" in result.columns
        assert "count_min" in result.columns
        assert "count_max" in result.columns

        # Single row (summary)
        assert len(result) == 1

    def test_summary_stats_with_value_col(self, sample_timeseries):
        """Test summarizing specific column."""
        from hellocloud.transforms.spark import summary_stats

        result = sample_timeseries.transform(
            summary_stats(value_col="value")
        ).toPandas()

        assert "value_mean" in result.columns
        assert "value_median" in result.columns

        # Mean of [10,12,15,14,16,5,6,7,8,9] = 10.2
        assert result["value_mean"].iloc[0] == pytest.approx(10.2)
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestSummaryStats -v`

Expected: `FAIL: function not defined`

**Step 3: GREEN - Implement**

Add to `src/hellocloud/transforms/spark.py`:
```python
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, expr


def summary_stats(
    value_col: str = None,
    group_by: Union[str, List[str]] = None
) -> Callable[[DataFrame], DataFrame]:
    """
    Aggregate to point statistics.

    If group_by provided, counts per group then summarizes.

    Args:
        value_col: Column to summarize (required if no group_by)
        group_by: Group by column(s), then summarize counts

    Returns:
        Transform closure

    Example:
        >>> summary = df.transform(summary_stats(group_by='date'))
        >>> summary = df.transform(summary_stats(value_col='cost'))
    """
    def inner(df: DataFrame) -> DataFrame:
        if group_by is not None:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
            df = df.groupBy(*group_cols).count()
            target_col = "count"
        else:
            if value_col is None:
                raise ValueError("Must provide value_col or group_by")
            target_col = value_col

        return df.agg(
            mean(col(target_col)).alias(f"{target_col}_mean"),
            expr(f"percentile_approx({target_col}, 0.5)").alias(f"{target_col}_median"),
            stddev(col(target_col)).alias(f"{target_col}_std"),
            spark_min(col(target_col)).alias(f"{target_col}_min"),
            spark_max(col(target_col)).alias(f"{target_col}_max"),
        )

    return inner
```

Update `src/hellocloud/transforms/__init__.py`:
```python
try:
    from hellocloud.transforms.spark import pct_change, summary_stats
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False
    warnings.warn(...)
    from hellocloud.transforms.ibis import pct_change, summary_stats

__all__ = ["pct_change", "summary_stats"]
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestSummaryStats -v`

Expected: `PASS: 2 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/transforms/spark.py tests/spark/test_transforms.py src/hellocloud/transforms/__init__.py
git commit -m "feat(transforms): add summary_stats with TDD

Point statistics aggregation.
Idiomatic API: summary_stats(group_by='date')

Test coverage: 2 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: add_rolling_stats - Mean (TDD)

**Files:**
- Modify: `src/hellocloud/transforms/spark.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `tests/spark/test_transforms.py`:
```python
class TestRollingStats:
    """Test rolling statistics."""

    def test_rolling_mean_only(self, sample_timeseries):
        """Test rolling mean."""
        from hellocloud.transforms.spark import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id", stats=["mean"])
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        assert "value_rolling_mean_3" in entity_a.columns

        # Row 2: mean([10,12,15]) = 12.33
        assert entity_a["value_rolling_mean_3"].iloc[2] == pytest.approx(12.333, abs=0.01)

    def test_rolling_respects_partitions(self, sample_timeseries):
        """Test partition boundaries."""
        from hellocloud.transforms.spark import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id", stats=["mean"])
        ).toPandas()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Row 2: A=[10,12,15], B=[5,6,7]
        assert entity_a["value_rolling_mean_3"].iloc[2] == pytest.approx(12.333, abs=0.01)
        assert entity_b["value_rolling_mean_3"].iloc[2] == pytest.approx(6.0)
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `FAIL: function not defined`

**Step 3: GREEN - Implement**

Add to `src/hellocloud/transforms/spark.py`:
```python
def add_rolling_stats(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: Union[str, List[str]],
    stats: List[str] = None
) -> Callable[[DataFrame], DataFrame]:
    """
    Add rolling statistics.

    Args:
        value_col: Column for stats
        time_col: Time column
        window_size: Window size
        partition_by: Partition column(s)
        stats: Stats to compute (default: ["mean"])

    Returns:
        Transform closure

    Example:
        >>> result = df.transform(
        ...     add_rolling_stats('cost', 'date', 30, 'resource_id', stats=['mean', 'std'])
        ... )
    """
    if stats is None:
        stats = ["mean"]

    def inner(df: DataFrame) -> DataFrame:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by
        window = (
            Window.partitionBy(*partition_cols)
            .orderBy(time_col)
            .rowsBetween(-(window_size - 1), 0)
        )

        result_df = df

        if "mean" in stats:
            result_df = result_df.withColumn(
                f"{value_col}_rolling_mean_{window_size}",
                mean(col(value_col)).over(window)
            )

        return result_df

    return inner
```

Update exports.

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `PASS: 2 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/transforms/spark.py tests/spark/test_transforms.py src/hellocloud/transforms/__init__.py
git commit -m "feat(transforms): add rolling mean to add_rolling_stats

Rolling window with mean statistic.
Test coverage: 2 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: add_rolling_stats - Add Std/Min/Max (TDD)

**Files:**
- Modify: `src/hellocloud/transforms/spark.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `TestRollingStats`:
```python
    def test_rolling_all_stats(self, sample_timeseries):
        """Test all rolling statistics."""
        from hellocloud.transforms.spark import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id",
                            stats=["mean", "std", "min", "max"])
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # All columns exist
        assert "value_rolling_mean_3" in entity_a.columns
        assert "value_rolling_std_3" in entity_a.columns
        assert "value_rolling_min_3" in entity_a.columns
        assert "value_rolling_max_3" in entity_a.columns

        # Row 2: [10,12,15]
        assert entity_a["value_rolling_min_3"].iloc[2] == 10
        assert entity_a["value_rolling_max_3"].iloc[2] == 15
        assert entity_a["value_rolling_std_3"].iloc[2] > 0
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats::test_rolling_all_stats -v`

Expected: `FAIL: KeyError: 'value_rolling_std_3'`

**Step 3: GREEN - Add std/min/max**

Modify `add_rolling_stats()` in `src/hellocloud/transforms/spark.py`:
```python
        if "mean" in stats:
            result_df = result_df.withColumn(
                f"{value_col}_rolling_mean_{window_size}",
                mean(col(value_col)).over(window)
            )
        if "std" in stats:
            result_df = result_df.withColumn(
                f"{value_col}_rolling_std_{window_size}",
                stddev(col(value_col)).over(window)
            )
        if "min" in stats:
            result_df = result_df.withColumn(
                f"{value_col}_rolling_min_{window_size}",
                spark_min(col(value_col)).over(window)
            )
        if "max" in stats:
            result_df = result_df.withColumn(
                f"{value_col}_rolling_max_{window_size}",
                spark_max(col(value_col)).over(window)
            )

        return result_df
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `PASS: 3 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/transforms/spark.py tests/spark/test_transforms.py
git commit -m "feat(transforms): complete add_rolling_stats with std/min/max

All common rolling statistics: mean, std, min, max.
Test coverage: 3 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: add_lag_features Transform (TDD)

**Files:**
- Modify: `src/hellocloud/transforms/spark.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `tests/spark/test_transforms.py`:
```python
class TestLagFeatures:
    """Test lag features."""

    def test_single_lag(self, sample_timeseries):
        """Test single lag feature."""
        from hellocloud.transforms.spark import add_lag_features

        result = sample_timeseries.transform(
            add_lag_features("value", "date", [1], "entity_id")
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        assert "value_lag_1" in entity_a.columns

        # First null
        assert pd.isna(entity_a["value_lag_1"].iloc[0])

        # Second = 10
        assert entity_a["value_lag_1"].iloc[1] == 10

    def test_multiple_lags(self, sample_timeseries):
        """Test multiple lags."""
        from hellocloud.transforms.spark import add_lag_features

        result = sample_timeseries.transform(
            add_lag_features("value", "date", [1, 2, 3], "entity_id")
        ).toPandas()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")

        assert "value_lag_1" in entity_a.columns
        assert "value_lag_2" in entity_a.columns
        assert "value_lag_3" in entity_a.columns

        # Row 3: lag_1=15, lag_2=12, lag_3=10
        assert entity_a["value_lag_1"].iloc[3] == 15
        assert entity_a["value_lag_2"].iloc[3] == 12
        assert entity_a["value_lag_3"].iloc[3] == 10
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestLagFeatures -v`

Expected: `FAIL: function not defined`

**Step 3: GREEN - Implement**

Add to `src/hellocloud/transforms/spark.py`:
```python
def add_lag_features(
    value_col: str,
    time_col: str,
    lags: List[int],
    partition_by: Union[str, List[str]]
) -> Callable[[DataFrame], DataFrame]:
    """
    Add multiple lag features.

    Args:
        value_col: Column to lag
        time_col: Time column
        lags: Lag periods (e.g., [1, 7, 30])
        partition_by: Partition column(s)

    Returns:
        Transform closure

    Example:
        >>> result = df.transform(
        ...     add_lag_features('cost', 'date', [1, 7, 30], 'resource_id')
        ... )
    """
    def inner(df: DataFrame) -> DataFrame:
        partition_cols = [partition_by] if isinstance(partition_by, str) else partition_by
        window = Window.partitionBy(*partition_cols).orderBy(time_col)

        result_df = df
        for lag_period in lags:
            result_df = result_df.withColumn(
                f"{value_col}_lag_{lag_period}",
                lag(col(value_col), lag_period).over(window)
            )

        return result_df

    return inner
```

Update exports.

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestLagFeatures -v`

Expected: `PASS: 2 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/transforms/spark.py tests/spark/test_transforms.py src/hellocloud/transforms/__init__.py
git commit -m "feat(transforms): add add_lag_features transform

Multiple lag features for time series forecasting.
Test coverage: 2 tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Verify All Tests Pass

**Files:**
- Verify: All tests

**Step 1: Run PySpark tests**

Run: `uv run pytest tests/spark/ -v --cov=src/hellocloud/spark --cov=src/hellocloud/transforms/spark --cov-report=term-missing`

Expected: `PASS: 12+ tests, coverage >90%`

**Step 2: Run old Ibis tests**

Run: `uv run pytest tests/test_transforms.py -v`

Expected: `PASS: 20 tests with deprecation warning`

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`

Expected: All tests pass

**Step 4: Commit**

```bash
git commit --allow-empty -m "test(spark): verify comprehensive test coverage

PySpark tests: 12 passing, >90% coverage
Ibis tests: 20 passing (deprecated, backward compat)
All tests: Passing

Ready for notebook migration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Migrate Notebook - Imports

**Files:**
- Modify: `notebooks/05_EDA_piedpiper_data.md`

**Step 1: Update imports cell**

Replace Ibis imports with PySpark:
```python
# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, expr

# Hello Cloud PySpark
from hellocloud.spark.session import get_spark_session
from hellocloud.transforms.spark import (
    pct_change,
    summary_stats,
    add_rolling_stats,
    add_lag_features,
)

# Initialize Spark
spark = get_spark_session()
```

**Step 2: Test imports**

Run notebook cell, verify:
- No import errors
- Spark UI at http://localhost:4040

**Step 3: Update data loading**

Replace:
```python
# Before (Ibis)
con = ibis.duckdb.connect()
df = con.read_parquet(str(DATA_PATH))

# After (PySpark)
df = spark.read.parquet(str(DATA_PATH))

# Quick check
df.show(5)
```

**Step 4: Test data loading**

Run cell, verify DataFrame loads correctly

**Step 5: Commit**

```bash
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "refactor(notebook): migrate to PySpark - imports and loading

Updated:
- PySpark imports
- Spark session init
- Parquet reading
- .show() for display

Tested: Imports work, data loads.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Migrate Notebook - Transforms

**Files:**
- Modify: `notebooks/05_EDA_piedpiper_data.md`

**Step 1: Migrate pct_change usage**

Replace:
```python
# Before (Ibis)
daily_with_change = (
    df
    .group_by('date')
    .agg(count=_.count())
    .order_by('date')
    .pipe(pct_change('count', 'date'))
)

# After (PySpark)
from pyspark.sql.functions import count as spark_count

daily_with_change = (
    df
    .groupBy('date')
    .agg(spark_count('*').alias('count'))
    .orderBy('date')
    .transform(pct_change('count', 'date', partition_by=None))
)

daily_with_change.show()
```

**Step 2: Test pct_change cell**

Run, verify:
- `count_pct_change` column exists
- Values are fractions (0.20 not 20.0)

**Step 3: Migrate filtering**

Replace:
```python
# Before (Ibis)
cutoff_date_result = (
    daily_with_change
    .filter(_.count_pct_change < -0.30)
    .execute()
)

# After (PySpark)
cutoff_date_result = (
    daily_with_change
    .filter(col('count_pct_change') < -0.30)
).toPandas()
```

**Step 4: Test and commit**

Run cells, verify results match previous output.

```bash
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "refactor(notebook): migrate transforms to PySpark

Updated:
- .transform(pct_change(...)) instead of .pipe()
- .filter(col('...')) instead of .filter(_.)
- .toPandas() instead of .execute()

Verified: Results match Ibis implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/development/PYSPARK_GUIDE.md`
- Move: `docs/plans/2025-10-11-pyspark-migration*.md` → `plans/archive/`

**Step 1: Update CLAUDE.md**

Add PySpark patterns after "Development Patterns":
```markdown
### PySpark Development Patterns

**Transform Composition:**
```python
result = (
    df
    .transform(pct_change('cost', 'date', 'resource_id'))
    .transform(add_rolling_stats('cost', 'date', 30, 'resource_id'))
    .transform(add_lag_features('cost', 'date', [1, 7, 30], 'resource_id'))
)
```

**Session Management:**
```python
from hellocloud.spark.session import get_spark_session
spark = get_spark_session()  # Local mode defaults
```
```

**Step 2: Create PySpark guide**

Create `docs/development/PYSPARK_GUIDE.md` with usage examples.

**Step 3: Archive old plans**

```bash
mkdir -p plans/archive
mv docs/plans/2025-10-11-pyspark-migration*.md plans/archive/
```

**Step 4: Commit**

```bash
git add CLAUDE.md docs/development/PYSPARK_GUIDE.md plans/
git commit -m "docs: document PySpark migration and usage

Added:
- PySpark patterns to CLAUDE.md
- Comprehensive PySpark guide
- Archived old migration plans

Migration complete! 🎉

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Success Criteria

- [ ] Domain structure created (io, stats, timeseries, analysis)
- [ ] Existing code reorganized with deprecation
- [ ] PySpark session builder working
- [ ] All transforms migrated with >90% coverage
- [ ] Notebook working with PySpark
- [ ] Documentation updated

---

## Execution Handoff

**Plan complete!** Choose execution:

### Option 1: Subagent-Driven (Same Session)
Say "Execute with subagents"

I'll use `skills/collaboration/subagent-driven-development`:
- Fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

### Option 2: Parallel Session (Batch)
1. Open new Claude Code session
2. Say: "Execute plan at plans/2025-10-11-pyspark-migration.md"

Uses `${CLAUDE_PLUGIN_ROOT}/skills/collaboration/executing-plans`:
- Batch execution (3 tasks at a time)
- Review checkpoints

---

## Notes

**TDD Discipline:** Every behavior gets full RED-GREEN-REFACTOR cycle.

**Structure First:** Repository organized before adding PySpark code.

**Backward Compatibility:** Old imports work with deprecation warnings.
