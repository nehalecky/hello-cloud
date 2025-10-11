# PySpark Migration Implementation Plan

> **For Claude:** Use `${CLAUDE_PLUGIN_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Migrate from Ibis to PySpark 4.0 for ETL and EDA, following strict TDD practices with one RED-GREEN-REFACTOR cycle per transform.

**Architecture:** Create `src/hellocloud/spark/` module with session management, display helpers, and composable transforms using PySpark's `.transform()` method. Preserve pipe-based composition patterns from Ibis while gaining PySpark's full statistical/ML ecosystem.

**Tech Stack:** PySpark 4.0, pytest, uv package manager, existing DuckDB for comparison testing

---

## Context

### Why This Migration

This is **graduating to the right tool**, not a sunk cost pivot:

1. **Polars → Ibis**: Learned we need SQL-like semantics and lazy evaluation ✓
2. **Ibis → PySpark**: Learned we need full statistical/ML toolkit and production scale ✓

### What We're Keeping

**Conceptual patterns from Ibis work:**
- Pipe-based composition (PySpark has `.transform()`)
- Lazy evaluation (PySpark does this natively)
- Clean aggregation patterns (translate directly)
- Thinking about composability and reusability

**Transform library concepts:**
- `pct_change()` - Fractional change over time
- `summary_stats()` - Point statistics aggregation
- `add_rolling_stats()` - Multiple rolling statistics
- `add_lag_features()` - Lag feature engineering
- `time_features()` - Temporal feature extraction

---

## Task 1: Add PySpark Dependency

**Files:**
- Modify: `pyproject.toml:20-47` (dependencies section)

**Step 1: Add PySpark to dependencies**

Edit `pyproject.toml`, add to dependencies array after line 46:

```toml
dependencies = [
    # ... existing dependencies ...
    "scipy>=1.16.2",
    "scikit-learn>=1.7.2",
    "pyspark>=4.0.0",  # Add this line
]
```

**Step 2: Sync dependencies**

Run: `uv sync`

Expected: "Resolved 150+ packages" (will add PySpark + transitive dependencies)

**Step 3: Verify import works**

Run: `uv run python -c "import pyspark; print(f'PySpark {pyspark.__version__}')"`

Expected: `PySpark 4.0.0` (or higher)

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(deps): add PySpark 4.0 for ETL and EDA work

Migrating from Ibis to PySpark to gain:
- 100+ statistical functions vs Ibis ~20
- MLlib for feature engineering and modeling
- Battle-tested production patterns
- 10+ years of community support

Preserving pipe composition concepts via .transform() method.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Spark Session Builder (TDD)

**Files:**
- Create: `src/hellocloud/spark/__init__.py`
- Create: `tests/spark/test_session.py`
- Create: `tests/spark/__init__.py`

**Step 1: RED - Write failing test**

Create `tests/spark/__init__.py`:
```python
"""Tests for PySpark utilities and transforms."""
```

Create `tests/spark/test_session.py`:
```python
"""Tests for Spark session management."""

import pytest
from pyspark.sql import SparkSession


def test_get_spark_session_returns_session():
    """Test that get_spark_session returns a SparkSession."""
    from hellocloud.spark import get_spark_session

    session = get_spark_session()

    assert isinstance(session, SparkSession)
    assert session.sparkContext.appName == "hellocloud"


def test_get_spark_session_local_mode():
    """Test that local mode uses all cores."""
    from hellocloud.spark import get_spark_session

    session = get_spark_session(local_mode=True)

    # Check master is set to local[*]
    assert session.sparkContext.master.startswith("local[")


def test_get_spark_session_is_singleton():
    """Test that multiple calls return same session."""
    from hellocloud.spark import get_spark_session

    session1 = get_spark_session()
    session2 = get_spark_session()

    assert session1 is session2  # Same object instance
```

**Step 2: Verify RED - Watch it fail**

Run: `uv run pytest tests/spark/test_session.py -v`

Expected: `FAIL: ModuleNotFoundError: No module named 'hellocloud.spark'`

**Step 3: GREEN - Implement minimal code**

Create `src/hellocloud/spark/__init__.py`:
```python
"""PySpark utilities for local development and production scale.

This module provides:
- Session management with sensible local defaults
- Display helpers for notebooks
- Composable transforms following pipe pattern
"""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "hellocloud", local_mode: bool = True) -> SparkSession:
    """
    Create or get existing Spark session.

    For local development, configures sensible defaults:
    - Uses all available cores: local[*]
    - 4GB driver memory
    - 8 shuffle partitions (not 200 default)

    Args:
        app_name: Application name for Spark UI
        local_mode: If True, configure for local development

    Returns:
        SparkSession (singleton)

    Example:
        >>> spark = get_spark_session()
        >>> df = spark.read.parquet("data.parquet")
    """
    builder = SparkSession.builder.appName(app_name)

    if local_mode:
        builder = builder.master("local[*]")  # Use all cores
        builder = builder.config("spark.driver.memory", "4g")
        builder = builder.config("spark.sql.shuffle.partitions", "8")

    return builder.getOrCreate()


__all__ = ["get_spark_session"]
```

**Step 4: Verify GREEN - Watch it pass**

Run: `uv run pytest tests/spark/test_session.py -v`

Expected: `PASS: 3 tests passed in <1s`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/__init__.py tests/spark/
git commit -m "feat(spark): add session builder with local dev defaults

Implements get_spark_session() with:
- Singleton pattern via getOrCreate()
- Local mode: local[*] master, 4GB memory, 8 partitions
- Production mode: no defaults, let cluster configure

Test coverage: 3 tests (creation, local mode, singleton)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Display Helpers (TDD)

**Files:**
- Create: `src/hellocloud/spark/display.py`
- Modify: `tests/spark/test_session.py` (rename to `test_spark_utils.py` and expand)

**Step 1: RED - Write failing tests**

Rename and expand test file to `tests/spark/test_spark_utils.py`:
```python
"""Tests for Spark session management and display utilities."""

import pytest
from pyspark.sql import SparkSession
import pandas as pd


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    from hellocloud.spark import get_spark_session
    return get_spark_session()


class TestSparkSession:
    """Tests for get_spark_session()."""

    def test_returns_session(self, spark):
        """Test that get_spark_session returns a SparkSession."""
        assert isinstance(spark, SparkSession)
        assert spark.sparkContext.appName == "hellocloud"

    def test_local_mode(self, spark):
        """Test that local mode uses all cores."""
        assert spark.sparkContext.master.startswith("local[")

    def test_is_singleton(self, spark):
        """Test that multiple calls return same session."""
        from hellocloud.spark import get_spark_session
        session2 = get_spark_session()
        assert spark is session2


class TestDisplayHelpers:
    """Tests for notebook display helpers."""

    def test_show_pretty_returns_pandas(self, spark):
        """Test that show_pretty returns pandas DataFrame."""
        from hellocloud.spark.display import show_pretty

        # Create test data
        data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        df = spark.createDataFrame(data, ["name", "age"])

        result = show_pretty(df, n=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Limited to n=2
        assert list(result.columns) == ["name", "age"]

    def test_show_pretty_default_limit(self, spark):
        """Test default limit of 10 rows."""
        from hellocloud.spark.display import show_pretty

        # Create 20 rows
        data = [(i, i*10) for i in range(20)]
        df = spark.createDataFrame(data, ["id", "value"])

        result = show_pretty(df)

        assert len(result) == 10  # Default limit

    def test_describe_pretty(self, spark):
        """Test describe_pretty returns summary statistics."""
        from hellocloud.spark.display import describe_pretty

        data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        df = spark.createDataFrame(data, ["id", "value"])

        result = describe_pretty(df)

        assert isinstance(result, pd.DataFrame)
        assert "count" in result["summary"].values
        assert "mean" in result["summary"].values
        assert "stddev" in result["summary"].values
```

**Step 2: Verify RED - Watch it fail**

Run: `uv run pytest tests/spark/test_spark_utils.py::TestDisplayHelpers -v`

Expected: `FAIL: ModuleNotFoundError: No module named 'hellocloud.spark.display'`

**Step 3: GREEN - Implement display helpers**

Create `src/hellocloud/spark/display.py`:
```python
"""Display utilities for PySpark DataFrames in notebooks.

Makes PySpark results render beautifully in Jupyter notebooks
by converting to pandas for display while keeping Spark for computation.
"""

from pyspark.sql import DataFrame
import pandas as pd


def show_pretty(df: DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Convert PySpark DataFrame to pandas for beautiful notebook display.

    Args:
        df: PySpark DataFrame to display
        n: Number of rows to show (default 10)

    Returns:
        pandas DataFrame with n rows

    Example:
        >>> df = spark.read.parquet("data.parquet")
        >>> show_pretty(df, 5)  # Shows first 5 rows as pandas
    """
    return df.limit(n).toPandas()


def describe_pretty(df: DataFrame) -> pd.DataFrame:
    """
    Pretty summary statistics for PySpark DataFrame.

    Args:
        df: PySpark DataFrame to describe

    Returns:
        pandas DataFrame with summary statistics

    Example:
        >>> df = spark.read.parquet("data.parquet")
        >>> describe_pretty(df)  # Shows count, mean, std, min, max
    """
    return df.describe().toPandas()
```

Update `src/hellocloud/spark/__init__.py` to export display functions:
```python
"""PySpark utilities for local development and production scale."""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "hellocloud", local_mode: bool = True) -> SparkSession:
    """Create or get existing Spark session."""
    # ... (existing implementation)


# Import display helpers for convenient access
from hellocloud.spark.display import show_pretty, describe_pretty

__all__ = ["get_spark_session", "show_pretty", "describe_pretty"]
```

**Step 4: Verify GREEN - Watch it pass**

Run: `uv run pytest tests/spark/test_spark_utils.py -v`

Expected: `PASS: 6 tests passed in <2s`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/display.py src/hellocloud/spark/__init__.py tests/spark/test_spark_utils.py
git commit -m "feat(spark): add notebook display helpers

Implements show_pretty() and describe_pretty() for converting
PySpark DataFrames to pandas for beautiful Jupyter display.

- show_pretty(df, n): First n rows as pandas
- describe_pretty(df): Summary stats as pandas

Test coverage: 3 tests (pandas conversion, limits, describe)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: pct_change Transform (TDD)

**Files:**
- Create: `src/hellocloud/spark/transforms.py`
- Create: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Create `tests/spark/test_transforms.py`:
```python
"""Tests for PySpark time series transformations."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from datetime import date


@pytest.fixture(scope="module")
def spark():
    """Create Spark session for testing."""
    from hellocloud.spark import get_spark_session
    return get_spark_session()


@pytest.fixture
def sample_timeseries(spark):
    """
    Create sample time series data for testing.

    Returns DataFrame with:
    - date: Daily timestamps
    - entity_id: Partition key (A, B)
    - value: Numeric values
    """
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
        """Test basic percentage change calculation."""
        from hellocloud.spark.transforms import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id")
        )

        # Check column exists
        assert "value_pct_change" in result.columns

        # Get entity A results
        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # First row should be null (no previous value)
        assert pd.isna(entity_a["value_pct_change"].iloc[0])

        # Second row: (12 - 10) / 10 = 0.20 (20% as fraction)
        assert entity_a["value_pct_change"].iloc[1] == pytest.approx(0.20)

        # Third row: (15 - 12) / 12 = 0.25 (25% as fraction)
        assert entity_a["value_pct_change"].iloc[2] == pytest.approx(0.25)

    def test_pct_change_respects_partitions(self, sample_timeseries):
        """Test that partitions are respected (no cross-entity calculation)."""
        from hellocloud.spark.transforms import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id")
        ).toPandas()

        # Entity A and B should have independent calculations
        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Both first rows should be null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])
        assert pd.isna(entity_b["value_pct_change"].iloc[0])

        # Second values should be different
        assert entity_a["value_pct_change"].iloc[1] == pytest.approx(0.20)  # (12-10)/10
        assert entity_b["value_pct_change"].iloc[1] == pytest.approx(0.20)  # (6-5)/5

    def test_pct_change_multiple_periods(self, sample_timeseries):
        """Test percentage change with periods > 1."""
        from hellocloud.spark.transforms import pct_change

        result = sample_timeseries.transform(
            pct_change("value", "date", "entity_id", periods=2)
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # First two rows should be null
        assert pd.isna(entity_a["value_pct_change"].iloc[0])
        assert pd.isna(entity_a["value_pct_change"].iloc[1])

        # Third row: (15 - 10) / 10 = 0.50 (50% as fraction)
        assert entity_a["value_pct_change"].iloc[2] == pytest.approx(0.50)


import pandas as pd  # Need pandas for isna checks
```

**Step 2: Verify RED - Watch it fail**

Run: `uv run pytest tests/spark/test_transforms.py::TestPctChange -v`

Expected: `FAIL: ModuleNotFoundError: No module named 'hellocloud.spark.transforms'`

**Step 3: GREEN - Implement pct_change**

Create `src/hellocloud/spark/transforms.py`:
```python
"""
Time series transformations for PySpark DataFrames.

All transforms follow pipe-based composition pattern using .transform() method.
Each function returns a closure that takes and returns a DataFrame.

Example:
    >>> from hellocloud.spark.transforms import pct_change
    >>> result = df.transform(pct_change('cost', 'date', 'resource_id'))
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

    Returns fraction (e.g., 0.25 for 25% increase, -0.30 for 30% decrease).

    Args:
        value_col: Column to calculate change on
        time_col: Time column for ordering
        partition_by: Column(s) to partition by (e.g., 'resource_id')
        periods: Number of periods to shift (default 1)

    Returns:
        Closure that transforms DataFrame

    Example:
        >>> result = df.transform(pct_change('cost', 'date', 'resource_id'))
        >>> # Adds 'cost_pct_change' column with fractional changes
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

Update `src/hellocloud/spark/__init__.py` to export transforms:
```python
# ... (existing code)

# Import transforms for convenient access
from hellocloud.spark.transforms import pct_change

__all__ = [
    "get_spark_session",
    "show_pretty",
    "describe_pretty",
    "pct_change",
]
```

**Step 4: Verify GREEN - Watch it pass**

Run: `uv run pytest tests/spark/test_transforms.py::TestPctChange -v`

Expected: `PASS: 3 tests passed in <2s`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/transforms.py src/hellocloud/spark/__init__.py tests/spark/test_transforms.py
git commit -m "feat(spark): add pct_change transform with TDD

Implements fractional change over time using PySpark windows.
Returns fractions (0.20 for 20% increase, not 20.0).

- Respects partition boundaries
- Handles multiple periods (lag > 1)
- Follows .transform() composition pattern

Test coverage: 3 tests (basic, partitions, periods)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: summary_stats Transform (TDD)

**Files:**
- Modify: `src/hellocloud/spark/transforms.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `tests/spark/test_transforms.py`:
```python
class TestSummaryStats:
    """Test summary statistics transformation."""

    def test_summary_stats_with_group_by(self, sample_timeseries):
        """Test idiomatic usage: df.transform(summary_stats(group_by='date'))."""
        from hellocloud.spark.transforms import summary_stats

        result = sample_timeseries.transform(
            summary_stats(group_by="date")
        ).toPandas()

        # Should aggregate to counts per date, then summarize
        assert "count_mean" in result.columns
        assert "count_median" in result.columns
        assert "count_std" in result.columns
        assert "count_min" in result.columns
        assert "count_max" in result.columns

        # Should be single row (summary of counts)
        assert len(result) == 1

    def test_summary_stats_with_value_col(self, sample_timeseries):
        """Test summarizing specific column."""
        from hellocloud.spark.transforms import summary_stats

        result = sample_timeseries.transform(
            summary_stats(value_col="value")
        ).toPandas()

        # Should summarize 'value' column directly
        assert "value_mean" in result.columns
        assert "value_median" in result.columns

        # Mean of [10,12,15,14,16,5,6,7,8,9] = 10.2
        assert result["value_mean"].iloc[0] == pytest.approx(10.2)
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestSummaryStats -v`

Expected: `FAIL: AttributeError: 'NoneType' object has no attribute 'transform'` (function doesn't exist)

**Step 3: GREEN - Implement summary_stats**

Add to `src/hellocloud/spark/transforms.py`:
```python
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max, expr


def summary_stats(
    value_col: str = None,
    group_by: Union[str, List[str]] = None
) -> Callable[[DataFrame], DataFrame]:
    """
    Aggregate to point statistics (mean, median, std, min, max, quantiles).

    If group_by provided, counts rows per group then summarizes.

    Args:
        value_col: Column to summarize (required if no group_by)
        group_by: Column(s) to group by, then summarize counts

    Returns:
        Closure that transforms DataFrame to single-row summary

    Example:
        >>> # Idiomatic: group by date and summarize counts
        >>> summary = df.transform(summary_stats(group_by='date'))
        >>>
        >>> # Or summarize specific column
        >>> summary = df.transform(summary_stats(value_col='cost'))
    """
    def inner(df: DataFrame) -> DataFrame:
        # If group_by specified, first aggregate to counts
        if group_by is not None:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
            df = df.groupBy(*group_cols).count()
            target_col = "count"
        else:
            if value_col is None:
                raise ValueError("Must provide either value_col or group_by")
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

Update `src/hellocloud/spark/__init__.py`:
```python
from hellocloud.spark.transforms import pct_change, summary_stats

__all__ = [
    "get_spark_session",
    "show_pretty",
    "describe_pretty",
    "pct_change",
    "summary_stats",
]
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestSummaryStats -v`

Expected: `PASS: 2 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/transforms.py src/hellocloud/spark/__init__.py tests/spark/test_transforms.py
git commit -m "feat(spark): add summary_stats transform with TDD

Implements point statistics aggregation with idiomatic API:
- df.transform(summary_stats(group_by='date'))
- df.transform(summary_stats(value_col='cost'))

Calculates mean, median, std, min, max.

Test coverage: 2 tests (group_by mode, value_col mode)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: add_rolling_stats Transform - Mean (TDD)

**Files:**
- Modify: `src/hellocloud/spark/transforms.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing test**

Add to `tests/spark/test_transforms.py`:
```python
class TestRollingStats:
    """Test rolling statistics transformation."""

    def test_rolling_mean_only(self, sample_timeseries):
        """Test rolling mean calculation."""
        from hellocloud.spark.transforms import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id", stats=["mean"])
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # Column should exist
        assert "value_rolling_mean_3" in entity_a.columns

        # Row 2: mean([10, 12, 15]) = 12.33...
        assert entity_a["value_rolling_mean_3"].iloc[2] == pytest.approx(12.333, abs=0.01)

    def test_rolling_respects_partitions(self, sample_timeseries):
        """Test that rolling stats respect partitions."""
        from hellocloud.spark.transforms import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id", stats=["mean"])
        ).toPandas()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")
        entity_b = result[result["entity_id"] == "B"].sort_values("date")

        # Row 2: Entity A mean([10,12,15]), Entity B mean([5,6,7])
        assert entity_a["value_rolling_mean_3"].iloc[2] == pytest.approx(12.333, abs=0.01)
        assert entity_b["value_rolling_mean_3"].iloc[2] == pytest.approx(6.0)
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `FAIL: function not defined`

**Step 3: GREEN - Implement add_rolling_stats (mean only)**

Add to `src/hellocloud/spark/transforms.py`:
```python
def add_rolling_stats(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: Union[str, List[str]],
    stats: List[str] = None
) -> Callable[[DataFrame], DataFrame]:
    """
    Add multiple rolling statistics at once.

    Args:
        value_col: Column to calculate stats on
        time_col: Time column for ordering
        window_size: Number of rows in window
        partition_by: Column(s) to partition by
        stats: List of stats to compute (default: ["mean"])

    Returns:
        Closure that transforms DataFrame

    Example:
        >>> result = df.transform(
        ...     add_rolling_stats('cost', 'date', 30, 'resource_id',
        ...                      stats=['mean', 'std'])
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

Update exports in `src/hellocloud/spark/__init__.py`.

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `PASS: 2 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/transforms.py src/hellocloud/spark/__init__.py tests/spark/test_transforms.py
git commit -m "feat(spark): add rolling mean to add_rolling_stats

Implements rolling mean with configurable window size.
Uses PySpark window functions with rowsBetween for correct frame.

Test coverage: 2 tests (basic calculation, partition respect)

Next: Add std, min, max stats.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: add_rolling_stats - Add Std, Min, Max (TDD)

**Files:**
- Modify: `src/hellocloud/spark/transforms.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing tests**

Add to `tests/spark/test_transforms.py` under `TestRollingStats`:
```python
    def test_rolling_all_stats(self, sample_timeseries):
        """Test all available rolling statistics."""
        from hellocloud.spark.transforms import add_rolling_stats

        result = sample_timeseries.transform(
            add_rolling_stats("value", "date", 3, "entity_id",
                            stats=["mean", "std", "min", "max"])
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        # All columns should exist
        assert "value_rolling_mean_3" in entity_a.columns
        assert "value_rolling_std_3" in entity_a.columns
        assert "value_rolling_min_3" in entity_a.columns
        assert "value_rolling_max_3" in entity_a.columns

        # Row 2: values [10, 12, 15]
        assert entity_a["value_rolling_min_3"].iloc[2] == 10
        assert entity_a["value_rolling_max_3"].iloc[2] == 15
        assert entity_a["value_rolling_std_3"].iloc[2] > 0  # Should be ~2.5
```

**Step 2: Verify RED**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats::test_rolling_all_stats -v`

Expected: `FAIL: KeyError: 'value_rolling_std_3'` (columns don't exist)

**Step 3: GREEN - Add std, min, max**

Modify `add_rolling_stats()` in `src/hellocloud/spark/transforms.py`:
```python
def add_rolling_stats(
    value_col: str,
    time_col: str,
    window_size: int,
    partition_by: Union[str, List[str]],
    stats: List[str] = None
) -> Callable[[DataFrame], DataFrame]:
    """Add multiple rolling statistics at once."""
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

    return inner
```

**Step 4: Verify GREEN**

Run: `uv run pytest tests/spark/test_transforms.py::TestRollingStats -v`

Expected: `PASS: 3 tests passed`

**Step 5: Commit**

```bash
git add src/hellocloud/spark/transforms.py tests/spark/test_transforms.py
git commit -m "feat(spark): add std/min/max to rolling stats

Completes add_rolling_stats with all common statistics:
- mean, std, min, max over configurable windows

Test coverage: 3 tests (mean, partitions, all stats)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: add_lag_features Transform (TDD)

**Files:**
- Modify: `src/hellocloud/spark/transforms.py`
- Modify: `tests/spark/test_transforms.py`

**Step 1: RED - Write failing tests**

Add to `tests/spark/test_transforms.py`:
```python
class TestLagFeatures:
    """Test lag features transformation."""

    def test_single_lag(self, sample_timeseries):
        """Test adding single lag feature."""
        from hellocloud.spark.transforms import add_lag_features

        result = sample_timeseries.transform(
            add_lag_features("value", "date", [1], "entity_id")
        )

        entity_a = result.filter("entity_id = 'A'").orderBy("date").toPandas()

        assert "value_lag_1" in entity_a.columns

        # First row should be null
        assert pd.isna(entity_a["value_lag_1"].iloc[0])

        # Second row should be 10 (previous value)
        assert entity_a["value_lag_1"].iloc[1] == 10

    def test_multiple_lags(self, sample_timeseries):
        """Test adding multiple lag features."""
        from hellocloud.spark.transforms import add_lag_features

        result = sample_timeseries.transform(
            add_lag_features("value", "date", [1, 2, 3], "entity_id")
        ).toPandas()

        entity_a = result[result["entity_id"] == "A"].sort_values("date")

        # All lag columns should exist
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

**Step 3: GREEN - Implement add_lag_features**

Add to `src/hellocloud/spark/transforms.py`:
```python
def add_lag_features(
    value_col: str,
    time_col: str,
    lags: List[int],
    partition_by: Union[str, List[str]]
) -> Callable[[DataFrame], DataFrame]:
    """
    Add multiple lag features at once.

    Args:
        value_col: Column to lag
        time_col: Time column for ordering
        lags: List of lag periods (e.g., [1, 7, 30])
        partition_by: Column(s) to partition by

    Returns:
        Closure that transforms DataFrame

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
git add src/hellocloud/spark/transforms.py src/hellocloud/spark/__init__.py tests/spark/test_transforms.py
git commit -m "feat(spark): add add_lag_features transform

Implements multiple lag feature creation in one transform.
Useful for time series forecasting feature engineering.

Test coverage: 2 tests (single lag, multiple lags)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Verify All Tests Pass

**Files:**
- Verify: All test files

**Step 1: Run full test suite**

Run: `uv run pytest tests/spark/ -v --cov=src/hellocloud/spark --cov-report=term-missing`

Expected: `PASS: 12+ tests passed, coverage >90%`

**Step 2: Verify old Ibis tests still pass**

Run: `uv run pytest tests/test_transforms.py -v`

Expected: `PASS: 20 tests passed` (Ibis transforms still work)

**Step 3: Check for any warnings or deprecations**

Run: `uv run pytest tests/spark/ -v --strict-markers`

Expected: Clean output, no warnings

**Step 4: Commit verification results**

```bash
git add tests/spark/
git commit -m "test(spark): verify all transforms with comprehensive coverage

Test suite results:
- 12 PySpark transform tests passing
- Coverage >90% on spark module
- 20 Ibis tests still passing (backward compatible)
- No warnings or deprecations

Ready for notebook migration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Migrate Notebook - Setup Cells

**Files:**
- Modify: `notebooks/05_EDA_piedpiper_data.md`

**Note:** This task and subsequent notebook tasks will be detailed but require manual verification of notebook execution. Each cell should be tested individually.

**Step 1: Update imports section**

Find the imports cell (typically first code cell) and replace Ibis imports:

```python
# Before (Ibis)
import ibis
from ibis import _
import pandas as pd

# After (PySpark)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, expr
import pandas as pd

from hellocloud.spark import (
    get_spark_session,
    show_pretty,
    describe_pretty,
)
from hellocloud.spark.transforms import (
    pct_change,
    summary_stats,
    add_rolling_stats,
    add_lag_features,
)

# Initialize Spark
spark = get_spark_session()
```

**Step 2: Test imports cell**

Run cell in notebook, verify:
- No import errors
- Spark session starts successfully
- Can see Spark UI at http://localhost:4040

**Step 3: Update data loading cell**

```python
# Before (Ibis)
con = ibis.duckdb.connect()
df = con.read_parquet(str(DATA_PATH))

# After (PySpark)
df = spark.read.parquet(str(DATA_PATH))

# Quick verification
show_pretty(df, 5)
```

**Step 4: Test data loading cell**

Run cell, verify:
- DataFrame loads successfully
- show_pretty displays first 5 rows as pandas
- Column names and types look correct

**Step 5: Commit**

```bash
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "refactor(notebook): migrate to PySpark - imports and data loading

Updated notebook setup:
- PySpark imports instead of Ibis
- Spark session initialization
- Parquet reading with Spark
- Display helpers for pandas output

Tested: Imports run, data loads, displays correctly.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Migrate Notebook - Transform Cells

**Files:**
- Modify: `notebooks/05_EDA_piedpiper_data.md`

**Step 1: Migrate pct_change usage**

Find this pattern:
```python
# Before (Ibis)
daily_with_change = (
    df
    .group_by('date')
    .agg(count=_.count())
    .order_by('date')
    .pipe(pct_change('count', 'date'))
)
```

Replace with:
```python
# After (PySpark)
daily_with_change = (
    df
    .groupBy('date')
    .agg(count('*').alias('count'))
    .orderBy('date')
    .transform(pct_change('count', 'date', partition_by=None))
)

# Display
show_pretty(daily_with_change)
```

**Note:** PySpark's `pct_change` requires explicit `partition_by` parameter. Use `partition_by=None` for global (no partitions).

**Step 2: Test pct_change cell**

Run cell, verify:
- `count_pct_change` column exists
- Values are fractions (0.20 not 20.0)
- First row is null (no previous value)

**Step 3: Migrate filter with pct_change**

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

Run cells, verify results match previous Ibis output.

```bash
git add notebooks/05_EDA_piedpiper_data.md
git commit -m "refactor(notebook): migrate pct_change and filtering to PySpark

Updated transform usage:
- .transform(pct_change(...)) instead of .pipe()
- Explicit partition_by parameter
- .filter(col('...')) instead of .filter(_.)
- .toPandas() instead of .execute()

Verified: Results match previous Ibis implementation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `docs/PYSPARK_MIGRATION_PLAN.md` (move to archive)
- Modify: `CLAUDE.md`
- Create: `docs/development/PYSPARK_GUIDE.md`

**Step 1: Archive old migration plan**

```bash
mkdir -p docs/archive
mv docs/PYSPARK_MIGRATION_PLAN.md docs/archive/2025-10-11-pyspark-migration-plan-v1.md
```

**Step 2: Update CLAUDE.md with PySpark patterns**

Add section after "Development Patterns":

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

**Notebook Display:**
```python
from hellocloud.spark import show_pretty, describe_pretty

show_pretty(df, 10)  # First 10 rows as pandas
describe_pretty(df)  # Summary stats as pandas
```

**Session Management:**
```python
from hellocloud.spark import get_spark_session

spark = get_spark_session()  # Local mode with sensible defaults
```
```

**Step 3: Create PySpark guide**

Create `docs/development/PYSPARK_GUIDE.md` with:
- Local development setup
- Transform library overview
- Common patterns (window functions, aggregations)
- Testing with pytest
- MLlib integration basics
- Performance tips

**Step 4: Commit**

```bash
git add docs/ CLAUDE.md
git commit -m "docs: document PySpark migration and usage patterns

Added:
- PySpark development patterns to CLAUDE.md
- Comprehensive PySpark guide
- Archived old migration plan

Migration complete! 🎉

Next steps:
- Migrate remaining notebooks as needed
- Explore MLlib for feature engineering
- Consider Spark Structured Streaming for real-time

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Execution Handoff

**Plan complete!** Choose your execution approach:

### Option 1: Subagent-Driven (This Session)

I dispatch a fresh subagent for each task with code review between tasks. Advantages:
- Stay in this session (no context switch)
- Fast iteration with quality gates
- Code review after each task catches issues early
- Continuous progress

**To proceed:** Say "Execute with subagents"

I'll use `skills/collaboration/subagent-driven-development/SKILL.md`:
1. Load plan, create TodoWrite
2. For each task: dispatch implementation subagent → code review → fix issues → next task
3. Final code review after all tasks
4. Finish with `skills/collaboration/finishing-a-development-branch`

### Option 2: Parallel Session (Batch Execution)

Open a new Claude Code session in this directory and use executing-plans skill. Advantages:
- Fresh context dedicated to execution
- Batch execution with review checkpoints
- Can review plan first before starting

**To proceed:**
1. Open new Claude Code session in this directory
2. Say: "Execute plan at docs/plans/2025-10-11-pyspark-migration.md"
3. I'll load plan, review critically, execute in batches of 3 tasks
4. Report for feedback between batches

Uses `${CLAUDE_PLUGIN_ROOT}/skills/collaboration/executing-plans/SKILL.md`.

---

## Notes on TDD Discipline

This plan follows strict TDD (RED-GREEN-REFACTOR) for every behavior:

1. **RED**: Write failing test first
2. **Verify RED**: Watch it fail for the right reason
3. **GREEN**: Write minimal code to pass
4. **Verify GREEN**: Watch it pass
5. **Commit**: One behavior per commit

**No shortcuts.** Even for "simple" transforms like `pct_change`, we write tests first. This ensures:
- Tests actually test the right thing (we saw them fail)
- Implementation is driven by requirements (from tests)
- Refactoring is safe (tests catch breaks)

**If you're tempted to skip TDD:** That's the voice of rationalization. Delete code, start with test.

---

## Success Criteria

- [ ] All 12 tasks completed with passing tests
- [ ] PySpark transform library with >90% coverage
- [ ] Notebook migrated and executing correctly
- [ ] Documentation updated
- [ ] All Ibis tests still passing (backward compatible)
- [ ] No warnings or deprecations
- [ ] Clean commit history (one behavior per commit)
