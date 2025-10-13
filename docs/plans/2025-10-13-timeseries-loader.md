# TimeSeries Loader Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Create a TimeSeries wrapper class for hierarchical time series data with PySpark DataFrame backend, supporting sampling, aggregation, transformations, and visualization. Include PiedPiper dataset loader with EDA-informed defaults.

**Architecture:** Single `TimeSeries` class wraps PySpark DataFrame containing hierarchical time series data (timestamp + multiple key columns + metric). All operations return new immutable instances. PiedPiperLoader applies column renames, drops low-information columns, and creates TimeSeries with dataset-specific defaults.

**Tech Stack:** PySpark 4.0, pytest, existing hellocloud.transforms utilities, loguru for logging

---

## Task 1: TimeSeries Core Class - Basic Structure

**Files:**
- Create: `src/hellocloud/timeseries/core.py`
- Create: `tests/timeseries/test_core.py`
- Modify: `src/hellocloud/timeseries/__init__.py`

**Step 1: Write the failing test**

Create `tests/timeseries/test_core.py`:

```python
"""Tests for TimeSeries core functionality."""
import pytest
from pyspark.sql import functions as F
from hellocloud.timeseries import TimeSeries


class TestTimeSeriesInitialization:
    """Test TimeSeries initialization and basic properties."""

    def test_create_from_dataframe(self, spark):
        """Should create TimeSeries from PySpark DataFrame."""
        # Arrange
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-02", "AWS", "acc1", "us-east-1", 110.0),
        ], ["date", "provider", "account", "region", "cost"])

        # Act
        ts = TimeSeries(
            df=df,
            hierarchy=["provider", "account", "region"],
            metric_col="cost",
            time_col="date"
        )

        # Assert
        assert ts.df is not None
        assert ts.hierarchy == ["provider", "account", "region"]
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"

    def test_stores_dataframe_reference(self, spark):
        """Should store reference to DataFrame, not copy."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries(
            df=df,
            hierarchy=["provider", "account"],
            metric_col="cost",
            time_col="date"
        )

        assert ts.df.count() == 1
        assert "date" in ts.df.columns
        assert "cost" in ts.df.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesInitialization -v
```

Expected: FAIL with "cannot import name 'TimeSeries'"

**Step 3: Write minimal implementation**

Create `src/hellocloud/timeseries/core.py`:

```python
"""Core TimeSeries class for hierarchical time series analysis."""
from typing import List
from pyspark.sql import DataFrame


class TimeSeries:
    """
    Wrapper around PySpark DataFrame for hierarchical time series analysis.

    Attributes:
        df: PySpark DataFrame containing time series data
        hierarchy: Ordered list of key columns (coarsest to finest grain)
        metric_col: Name of the metric/value column
        time_col: Name of the timestamp column
    """

    def __init__(
        self,
        df: DataFrame,
        hierarchy: List[str],
        metric_col: str,
        time_col: str
    ):
        """
        Initialize TimeSeries wrapper.

        Args:
            df: PySpark DataFrame with time series data
            hierarchy: Ordered key columns (e.g., ["provider", "account", "region"])
            metric_col: Name of metric column (e.g., "cost")
            time_col: Name of timestamp column (e.g., "date")
        """
        self.df = df
        self.hierarchy = hierarchy
        self.metric_col = metric_col
        self.time_col = time_col
        self._cached_stats = {}
```

Update `src/hellocloud/timeseries/__init__.py`:

```python
"""TimeSeries analysis utilities."""
from hellocloud.timeseries.core import TimeSeries

__all__ = ["TimeSeries"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesInitialization -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py src/hellocloud/timeseries/__init__.py
git commit -m "feat: add TimeSeries core class with basic initialization

- Create TimeSeries wrapper around PySpark DataFrame
- Store hierarchy, metric_col, time_col metadata
- Add test fixtures for initialization

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Validation and Error Handling

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
from hellocloud.timeseries.core import TimeSeriesError


class TestTimeSeriesValidation:
    """Test TimeSeries validation and error handling."""

    def test_missing_time_column_raises_error(self, spark):
        """Should raise error if time column not in DataFrame."""
        df = spark.createDataFrame([
            ("AWS", "acc1", 100.0),
        ], ["provider", "account", "cost"])

        with pytest.raises(TimeSeriesError, match="time_col 'date' not found"):
            TimeSeries(
                df=df,
                hierarchy=["provider", "account"],
                metric_col="cost",
                time_col="date"
            )

    def test_missing_metric_column_raises_error(self, spark):
        """Should raise error if metric column not in DataFrame."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1"),
        ], ["date", "provider", "account"])

        with pytest.raises(TimeSeriesError, match="metric_col 'cost' not found"):
            TimeSeries(
                df=df,
                hierarchy=["provider", "account"],
                metric_col="cost",
                time_col="date"
            )

    def test_missing_hierarchy_column_raises_error(self, spark):
        """Should raise error if hierarchy column not in DataFrame."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", 100.0),
        ], ["date", "provider", "cost"])

        with pytest.raises(TimeSeriesError, match="hierarchy column 'account' not found"):
            TimeSeries(
                df=df,
                hierarchy=["provider", "account"],
                metric_col="cost",
                time_col="date"
            )

    def test_empty_dataframe_logs_warning(self, spark, caplog):
        """Should log warning for empty DataFrame but not raise error."""
        import logging
        df = spark.createDataFrame([], "date STRING, provider STRING, cost DOUBLE")

        with caplog.at_level(logging.WARNING):
            ts = TimeSeries(
                df=df,
                hierarchy=["provider"],
                metric_col="cost",
                time_col="date"
            )

        assert "empty DataFrame" in caplog.text.lower()
        assert ts.df.count() == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesValidation -v
```

Expected: FAIL with "TimeSeriesError not defined" and validation not implemented

**Step 3: Write minimal implementation**

Update `src/hellocloud/timeseries/core.py`:

```python
"""Core TimeSeries class for hierarchical time series analysis."""
from typing import List
from pyspark.sql import DataFrame
from loguru import logger


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

    def __init__(
        self,
        df: DataFrame,
        hierarchy: List[str],
        metric_col: str,
        time_col: str
    ):
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
            logger.warning("Creating TimeSeries from empty DataFrame. Operations will return empty results.")

    def _validate_columns(self) -> None:
        """Validate that required columns exist in DataFrame."""
        df_cols = set(self.df.columns)

        # Check time column
        if self.time_col not in df_cols:
            raise TimeSeriesError(f"time_col '{self.time_col}' not found in DataFrame columns: {list(df_cols)}")

        # Check metric column
        if self.metric_col not in df_cols:
            raise TimeSeriesError(f"metric_col '{self.metric_col}' not found in DataFrame columns: {list(df_cols)}")

        # Check hierarchy columns
        for col in self.hierarchy:
            if col not in df_cols:
                raise TimeSeriesError(f"hierarchy column '{col}' not found in DataFrame columns: {list(df_cols)}")
```

Update `src/hellocloud/timeseries/__init__.py`:

```python
"""TimeSeries analysis utilities."""
from hellocloud.timeseries.core import TimeSeries, TimeSeriesError

__all__ = ["TimeSeries", "TimeSeriesError"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesValidation -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py src/hellocloud/timeseries/__init__.py
git commit -m "feat: add TimeSeries column validation and error handling

- Validate time_col, metric_col, hierarchy columns exist
- Raise TimeSeriesError with clear messages for missing columns
- Log warning for empty DataFrames
- Add comprehensive validation tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Factory Method - from_dataframe

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestTimeSeriesFactoryMethods:
    """Test TimeSeries factory methods."""

    def test_from_dataframe_creates_instance(self, spark):
        """Should create TimeSeries from DataFrame via factory method."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=["provider", "account", "region"],
            metric_col="cost",
            time_col="date"
        )

        assert isinstance(ts, TimeSeries)
        assert ts.hierarchy == ["provider", "account", "region"]
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"

    def test_from_dataframe_with_defaults(self, spark):
        """Should use default column names if not specified."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", 100.0),
        ], ["date", "provider", "cost"])

        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=["provider"]
        )

        # Should default to metric_col="cost", time_col="date"
        assert ts.metric_col == "cost"
        assert ts.time_col == "date"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesFactoryMethods -v
```

Expected: FAIL with "TimeSeries has no attribute 'from_dataframe'"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        hierarchy: List[str],
        metric_col: str = "cost",
        time_col: str = "date"
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
        return cls(
            df=df,
            hierarchy=hierarchy,
            metric_col=metric_col,
            time_col=time_col
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesFactoryMethods -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add TimeSeries.from_dataframe factory method

- Add classmethod factory with sensible defaults
- Default metric_col='cost', time_col='date'
- Simplifies common case of creating TimeSeries

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Grain Resolution Helper

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestGrainResolution:
    """Test grain resolution and validation."""

    def test_resolve_grain_returns_ordered_columns(self, spark):
        """Should return grain columns in hierarchy order."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", "Compute", 100.0),
        ], ["date", "provider", "account", "region", "product", "cost"])

        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=["provider", "account", "region", "product"]
        )

        # Request out-of-order grain
        grain = ts._resolve_grain(["region", "provider"])

        # Should return in hierarchy order
        assert grain == ["provider", "region"]

    def test_resolve_grain_validates_subset(self, spark):
        """Should raise error if grain contains columns not in hierarchy."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=["provider", "account"]
        )

        with pytest.raises(TimeSeriesError, match="Invalid grain columns"):
            ts._resolve_grain(["provider", "invalid_column"])

    def test_resolve_grain_handles_partial_hierarchy(self, spark):
        """Should handle grain that is partial subset of hierarchy."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(
            df,
            hierarchy=["provider", "account", "region"]
        )

        grain = ts._resolve_grain(["account"])
        assert grain == ["account"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestGrainResolution -v
```

Expected: FAIL with "_resolve_grain not defined"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
    def _resolve_grain(self, grain: List[str]) -> List[str]:
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
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestGrainResolution -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add grain resolution helper method

- Validate grain is subset of hierarchy
- Return grain columns in hierarchy order
- Raise clear errors for invalid grain columns

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Filter Operation

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestTimeSeriesFilter:
    """Test TimeSeries filtering operations."""

    def test_filter_single_entity(self, spark):
        """Should filter to specific entity and return new TimeSeries."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
            ("2025-01-01", "GCP", "acc3", 300.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        filtered = ts.filter(provider="AWS", account="acc1")

        assert isinstance(filtered, TimeSeries)
        assert filtered.df.count() == 1
        result = filtered.df.collect()[0]
        assert result["provider"] == "AWS"
        assert result["account"] == "acc1"

    def test_filter_returns_new_instance(self, spark):
        """Should return new TimeSeries instance, not modify original."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])
        filtered = ts.filter(account="acc1")

        assert ts.df.count() == 2  # Original unchanged
        assert filtered.df.count() == 1
        assert ts is not filtered

    def test_filter_multiple_criteria(self, spark):
        """Should filter on multiple hierarchy columns."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
            ("2025-01-01", "AWS", "acc2", "us-east-1", 300.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        filtered = ts.filter(provider="AWS", account="acc1", region="us-east-1")

        assert filtered.df.count() == 1
        result = filtered.df.collect()[0]
        assert result["region"] == "us-east-1"

    def test_filter_invalid_column_raises_error(self, spark):
        """Should raise error if filter column not in hierarchy."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        with pytest.raises(TimeSeriesError, match="Invalid filter column"):
            ts.filter(invalid_column="value")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesFilter -v
```

Expected: FAIL with "TimeSeries has no attribute 'filter'"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
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
            time_col=self.time_col
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesFilter -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add TimeSeries filter operation

- Filter to specific entity by hierarchy column values
- Return new immutable TimeSeries instance
- Validate filter columns are in hierarchy
- Support multiple filter criteria

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Sampling Operations

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestTimeSeriesSample:
    """Test TimeSeries sampling operations."""

    def test_sample_single_entity(self, spark):
        """Should sample single entity at specified grain."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-02", "AWS", "acc1", 110.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
            ("2025-01-02", "AWS", "acc2", 220.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"], n=1)

        assert isinstance(sampled, TimeSeries)
        # Should have 2 rows (both dates) for the sampled account
        assert sampled.df.count() == 2
        # Should only have 1 unique account
        assert sampled.df.select("account").distinct().count() == 1

    def test_sample_multiple_entities(self, spark):
        """Should sample N entities at specified grain."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", f"acc{i}", 100.0 * i)
            for i in range(10)
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"], n=3)

        # Should have 3 unique accounts
        assert sampled.df.select("account").distinct().count() == 3

    def test_sample_more_than_available(self, spark, caplog):
        """Should return all entities and log warning if n > available."""
        import logging
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        with caplog.at_level(logging.WARNING):
            sampled = ts.sample(grain=["account"], n=10)

        assert "only 2 exist" in caplog.text.lower()
        assert sampled.df.select("account").distinct().count() == 2

    def test_sample_default_n_equals_1(self, spark):
        """Should default to n=1 if not specified."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        sampled = ts.sample(grain=["account"])

        assert sampled.df.select("account").distinct().count() == 1

    def test_sample_hierarchical_grain(self, spark):
        """Should sample at multi-level grain correctly."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
            ("2025-01-01", "AWS", "acc2", "us-east-1", 300.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        sampled = ts.sample(grain=["account", "region"], n=2)

        # Should have 2 unique account+region combinations
        assert sampled.df.select("account", "region").distinct().count() == 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesSample -v
```

Expected: FAIL with "TimeSeries has no attribute 'sample'"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
    def sample(self, grain: List[str], n: int = 1) -> "TimeSeries":
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
            time_col=self.time_col
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesSample -v
```

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add TimeSeries sampling operation

- Sample n random entities at specified grain level
- Support single and multi-level grains
- Warn and return all if n > available entities
- Default to n=1 for single entity sampling

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Aggregation Operations

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestTimeSeriesAggregate:
    """Test TimeSeries aggregation operations."""

    def test_aggregate_to_coarser_grain(self, spark):
        """Should aggregate metric to coarser grain level."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
            ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Aggregate from account+region to just account
        agg = ts.aggregate(grain=["account"])

        assert isinstance(agg, TimeSeries)
        # Should have 2 rows (2 dates for acc1)
        assert agg.df.count() == 2
        # Region column should be removed
        assert "region" not in agg.df.columns
        # Should sum costs: 100+200=300 for date 1, 150 for date 2
        results = agg.df.orderBy("date").collect()
        assert results[0]["cost"] == 300.0
        assert results[1]["cost"] == 150.0

    def test_aggregate_preserves_time_dimension(self, spark):
        """Should preserve time column in aggregation."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        agg = ts.aggregate(grain=["provider"])

        assert "date" in agg.df.columns
        assert agg.df.count() == 2  # One per date

    def test_aggregate_to_top_level(self, spark):
        """Should aggregate all the way to top of hierarchy."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-01", "AWS", "acc2", "us-west-1", 200.0),
            ("2025-01-02", "AWS", "acc1", "us-east-1", 150.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Aggregate to just provider level
        agg = ts.aggregate(grain=["provider"])

        assert agg.df.count() == 2  # 2 dates
        assert "account" not in agg.df.columns
        assert "region" not in agg.df.columns
        results = agg.df.orderBy("date").collect()
        assert results[0]["cost"] == 300.0  # 100 + 200
        assert results[1]["cost"] == 150.0

    def test_aggregate_same_grain_returns_copy(self, spark, caplog):
        """Should return copy and log info if already at requested grain."""
        import logging
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        with caplog.at_level(logging.INFO):
            agg = ts.aggregate(grain=["provider", "account"])

        assert "already at grain" in caplog.text.lower()
        assert agg.df.count() == ts.df.count()
        assert agg is not ts  # Should be new instance
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesAggregate -v
```

Expected: FAIL with "TimeSeries has no attribute 'aggregate'"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
    def aggregate(self, grain: List[str]) -> "TimeSeries":
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
                time_col=self.time_col
            )

        # Group by grain + time, sum metric
        group_cols = grain_cols + [self.time_col]
        agg_df = self.df.groupBy(*group_cols).agg(
            F.sum(self.metric_col).alias(self.metric_col)
        )

        return TimeSeries(
            df=agg_df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col
        )
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesAggregate -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add TimeSeries aggregation operation

- Aggregate metric to coarser grain via groupBy + sum
- Preserve time dimension in aggregation
- Support full hierarchy roll-up
- Log info if already at requested grain

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: PiedPiper Loader

**Files:**
- Create: `src/hellocloud/io/loaders.py`
- Create: `tests/io/test_loaders.py`
- Modify: `src/hellocloud/io/__init__.py`

**Step 1: Write the failing test**

Create `tests/io/test_loaders.py`:

```python
"""Tests for dataset loaders."""
import pytest
from pyspark.sql import functions as F
from hellocloud.io import PiedPiperLoader
from hellocloud.timeseries import TimeSeries


class TestPiedPiperLoader:
    """Test PiedPiperLoader functionality."""

    def test_load_creates_timeseries(self, spark):
        """Should load DataFrame and create TimeSeries."""
        df = spark.createDataFrame([
            ("2025-09-01", "AWS", "acc1", "us-east-1", "Compute", "Standard", 100.0),
        ], ["usage_date", "cloud_provider", "cloud_account_id", "region",
            "product_family", "usage_type", "materialized_cost"])

        ts = PiedPiperLoader.load(df)

        assert isinstance(ts, TimeSeries)

    def test_load_renames_columns(self, spark):
        """Should rename usage_date->date and materialized_cost->cost."""
        df = spark.createDataFrame([
            ("2025-09-01", "AWS", 100.0),
        ], ["usage_date", "cloud_provider", "materialized_cost"])

        ts = PiedPiperLoader.load(
            df,
            hierarchy=["cloud_provider"],
            drop_cols=[]  # Don't drop anything for this test
        )

        assert "date" in ts.df.columns
        assert "cost" in ts.df.columns
        assert "usage_date" not in ts.df.columns
        assert "materialized_cost" not in ts.df.columns

    def test_load_drops_low_info_columns(self, spark):
        """Should drop UUID and redundant cost columns."""
        df = spark.createDataFrame([
            ("2025-09-01", "AWS", "uuid-123", 100.0, 95.0, 98.0, 100.0, 105.0),
        ], ["usage_date", "cloud_provider", "billing_event_id",
            "materialized_cost", "materialized_discounted_cost",
            "materialized_amortized_cost", "materialized_invoiced_cost",
            "materialized_public_cost"])

        ts = PiedPiperLoader.load(df, hierarchy=["cloud_provider"])

        # Should drop redundant cost columns
        assert "materialized_discounted_cost" not in ts.df.columns
        assert "materialized_amortized_cost" not in ts.df.columns
        assert "materialized_invoiced_cost" not in ts.df.columns
        assert "materialized_public_cost" not in ts.df.columns
        # Should drop UUID column
        assert "billing_event_id" not in ts.df.columns
        # Should keep the base cost (renamed)
        assert "cost" in ts.df.columns

    def test_load_uses_default_hierarchy(self, spark):
        """Should use default PiedPiper hierarchy if not specified."""
        df = spark.createDataFrame([
            ("2025-09-01", "AWS", "acc1", "us-east-1", "Compute", "Standard", 100.0),
        ], ["usage_date", "cloud_provider", "cloud_account_id", "region",
            "product_family", "usage_type", "materialized_cost"])

        ts = PiedPiperLoader.load(df)

        assert ts.hierarchy == [
            "cloud_provider",
            "cloud_account_id",
            "region",
            "product_family",
            "usage_type"
        ]

    def test_load_accepts_custom_hierarchy(self, spark):
        """Should allow overriding default hierarchy."""
        df = spark.createDataFrame([
            ("2025-09-01", "AWS", "acc1", 100.0),
        ], ["usage_date", "cloud_provider", "cloud_account_id", "materialized_cost"])

        ts = PiedPiperLoader.load(
            df,
            hierarchy=["cloud_provider", "cloud_account_id"]
        )

        assert ts.hierarchy == ["cloud_provider", "cloud_account_id"]
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/io/test_loaders.py -v
```

Expected: FAIL with "cannot import name 'PiedPiperLoader'"

**Step 3: Write minimal implementation**

Create `src/hellocloud/io/loaders.py`:

```python
"""Dataset loaders for creating TimeSeries instances."""
from typing import List, Optional
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
        "usage_type"
    ]

    # Column renames for standardization
    COLUMN_RENAMES = {
        "usage_date": "date",
        "materialized_cost": "cost"
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
        hierarchy: Optional[List[str]] = None,
        metric_col: str = "cost",
        time_col: str = "date",
        drop_cols: Optional[List[str]] = None
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
            time_col=time_col
        )
```

Update `src/hellocloud/io/__init__.py`:

```python
"""Data loading utilities."""
from hellocloud.io.loaders import PiedPiperLoader

__all__ = ["PiedPiperLoader"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/io/test_loaders.py -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/io/loaders.py tests/io/test_loaders.py src/hellocloud/io/__init__.py
git commit -m "feat: add PiedPiperLoader for billing data

- Apply EDA-informed column renames (usage_date->date, materialized_cost->cost)
- Drop redundant cost variants and UUID columns
- Use default hierarchy from EDA analysis
- Support custom hierarchy and drop columns
- Create TimeSeries with cleaned data

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Summary Statistics

**Files:**
- Modify: `src/hellocloud/timeseries/core.py`
- Modify: `tests/timeseries/test_core.py`

**Step 1: Write the failing test**

Add to `tests/timeseries/test_core.py`:

```python
class TestTimeSeriesSummaryStats:
    """Test TimeSeries summary statistics."""

    def test_summary_stats_returns_dataframe(self, spark):
        """Should return DataFrame with summary statistics."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-02", "AWS", "acc1", 110.0),
            ("2025-01-03", "AWS", "acc1", 105.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()

        assert stats is not None
        # Should have stats columns
        stats_cols = stats.columns
        assert "mean" in stats_cols
        assert "std" in stats_cols
        assert "min" in stats_cols
        assert "max" in stats_cols
        assert "count" in stats_cols

    def test_summary_stats_includes_entity_keys(self, spark):
        """Should include entity identifier columns in stats output."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-02", "AWS", "acc1", 110.0),
            ("2025-01-01", "AWS", "acc2", 200.0),
            ("2025-01-02", "AWS", "acc2", 220.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()

        # Should have entity keys
        assert "provider" in stats.columns
        assert "account" in stats.columns
        # Should have one row per entity
        assert stats.count() == 2

    def test_summary_stats_at_different_grain(self, spark):
        """Should compute stats at specified grain level."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", "us-east-1", 100.0),
            ("2025-01-02", "AWS", "acc1", "us-east-1", 110.0),
            ("2025-01-01", "AWS", "acc1", "us-west-1", 200.0),
            ("2025-01-02", "AWS", "acc1", "us-west-1", 220.0),
        ], ["date", "provider", "account", "region", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account", "region"])

        # Stats at account level (should aggregate regions first)
        stats = ts.summary_stats(grain=["account"])

        assert stats.count() == 1  # One account
        # Should aggregate: date1: 100+200=300, date2: 110+220=330
        result = stats.collect()[0]
        assert result["mean"] == 315.0  # (300 + 330) / 2

    def test_summary_stats_correct_calculations(self, spark):
        """Should calculate statistics correctly."""
        df = spark.createDataFrame([
            ("2025-01-01", "AWS", "acc1", 100.0),
            ("2025-01-02", "AWS", "acc1", 200.0),
            ("2025-01-03", "AWS", "acc1", 150.0),
        ], ["date", "provider", "account", "cost"])

        ts = TimeSeries.from_dataframe(df, hierarchy=["provider", "account"])

        stats = ts.summary_stats()
        result = stats.collect()[0]

        assert result["count"] == 3
        assert result["mean"] == 150.0
        assert result["min"] == 100.0
        assert result["max"] == 200.0
        assert result["std"] > 0  # Should have standard deviation
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesSummaryStats -v
```

Expected: FAIL with "TimeSeries has no attribute 'summary_stats'"

**Step 3: Write minimal implementation**

Add to `src/hellocloud/timeseries/core.py` (inside TimeSeries class):

```python
    def summary_stats(self, grain: Optional[List[str]] = None) -> DataFrame:
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
            F.max(self.metric_col).alias("max")
        )

        return stats_df
```

Also add to imports at top of file:

```python
from typing import List, Optional
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/timeseries/test_core.py::TestTimeSeriesSummaryStats -v
```

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/hellocloud/timeseries/core.py tests/timeseries/test_core.py
git commit -m "feat: add TimeSeries summary statistics

- Compute count, mean, std, min, max by entity
- Support stats at different grain levels
- Return DataFrame with entity keys + stats
- Aggregate to target grain before computing stats

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite

**Files:**
- None (verification step)

**Step 1: Run all TimeSeries and loader tests**

```bash
uv run pytest tests/timeseries/ tests/io/ -v
```

Expected: All tests PASS

**Step 2: Run with coverage**

```bash
uv run pytest tests/timeseries/ tests/io/ -v --cov=src/hellocloud/timeseries --cov=src/hellocloud/io --cov-report=term-missing
```

Expected: Coverage > 70% for new modules

**Step 3: Verify no regressions in existing tests**

```bash
uv run pytest tests/ -v --maxfail=5
```

Expected: Same number of passing/failing tests as baseline (180 pass, 4 fail on notebooks)

**Step 4: Update documentation**

Update project README or create docs/how-to/use-timeseries-loader.md with usage examples.

**Step 5: Final commit**

```bash
git add -A
git commit -m "docs: add TimeSeries loader documentation and examples

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Implementation Notes

### Testing Strategy
- Use pytest fixtures from `tests/conftest.py` for spark session
- Test one behavior per test method
- Follow Arrange-Act-Assert pattern
- Use descriptive test names

### Code Style
- Follow existing hellocloud conventions (2-space indent in docs, PEP 8 in code)
- Use type hints for all public methods
- Add docstrings with examples
- Use loguru for logging

### Integration with Existing Code
- Reuse `hellocloud.transforms.pct_change` for transformations (Task 11, if needed)
- Consider reusing `hellocloud.analysis.eda.align_entity_timeseries` (Task 12, if needed)
- PySpark 4.0 conventions throughout

### Future Enhancements (Not in this plan)
- Plotting utilities (`ts.plot()`)
- Percent change transformation wrapper
- Entity alignment utilities
- More sophisticated sampling strategies
- Caching optimization for expensive operations

---

**Plan Complete!** Ready for execution via `skills/collaboration/executing-plans/SKILL.md`.
