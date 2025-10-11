"""
Exploratory Data Analysis utilities for billing and time series data.

Provides reusable functions for schema analysis, information theory calculations,
temporal normalization patterns, and outlier detection.
Designed for Ibis Tables with DuckDB backend, focusing on cloud billing data.
"""

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibis import _
from sklearn.ensemble import IsolationForest

# ============================================================================
# Schema & Information Theory
# ============================================================================


def attribute_analysis(df: ibis.Table, sample_size: int = 50_000) -> ibis.Table:
    """
    Comprehensive attribute analysis with information density metrics.

    Fully Ibis-native implementation - takes Ibis Table, returns Ibis Table.

    Computes metrics for grain discovery and feature selection:
    - Value density: proportion of non-null values (completeness)
    - Nonzero density: proportion of non-zero values (for numeric columns)
    - Cardinality ratio: unique_count / total_rows (uniqueness)
    - Entropy: Shannon entropy measuring value distribution confusion
    - Information score: harmonic mean of (value_density, nonzero_density,
                         cardinality_ratio, entropy) - all "higher is better"

    Args:
        df: Input Ibis Table to analyze
        sample_size: Sample size for entropy calculation (default 50k)

    Returns:
        Ibis Table with columns: column, dtype, value_density, nonzero_density,
        cardinality_ratio, entropy, information_score, sample_value
        Sorted by information_score descending.

    Example:
        >>> attrs = hc.utils.attribute_analysis(df)
        >>> attrs  # Beautiful Ibis table rendering
    """
    schema = df.schema()  # Returns dict of {name: dtype}
    total_rows = df.count().execute()

    # Sample for entropy calculation - keep as Ibis Table
    df_sample_table = df.limit(sample_size)

    # Identify numeric columns for nonzero_density calculation
    numeric_types = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "decimal",
    )

    # Compute per-column metrics
    results = []
    for col in schema.keys():
        # Null count
        null_count = (
            df.select((df[col].isnull()).sum().name("null_count")).execute()["null_count"].iloc[0]
        )

        # Unique count
        unique_count = df[col].nunique().execute()

        # Value density (complement of null ratio, "higher is better")
        value_density = (total_rows - null_count) / total_rows

        # Nonzero density (numeric columns only) - higher is better
        dtype_str = str(schema[col])
        is_numeric = any(nt in dtype_str.lower() for nt in numeric_types)

        if is_numeric:
            # Count zeros - use pandas to count after executing
            # Some columns might have all nulls, handle gracefully
            try:
                zero_result = df.select((df[col] == 0).sum()).execute()
                # Get the value, handling various return formats
                if hasattr(zero_result, "iloc"):
                    val = zero_result.iloc[0, 0] if len(zero_result) > 0 else 0
                else:
                    val = zero_result
                zero_count = int(val) if val is not None else 0
            except Exception:
                # If comparison fails (e.g., all nulls), treat as all nonzero
                zero_count = 0
            nonzero_density = (total_rows - zero_count) / total_rows
        else:
            nonzero_density = 1.0  # Non-numeric: treat as "all nonzero" for scoring

        # Cardinality ratio
        cardinality_ratio = unique_count / total_rows

        # Shannon entropy (on sample) - using Ibis Table
        entropy = shannon_entropy_ibis(df_sample_table, col)

        # Information score: harmonic mean of (value_density, nonzero_density,
        # cardinality_ratio, entropy) - all metrics are "higher is better"
        epsilon = 1e-10
        information_score = 4.0 / (
            1.0 / (value_density + epsilon)
            + 1.0 / (nonzero_density + epsilon)
            + 1.0 / (cardinality_ratio + epsilon)
            + 1.0 / (entropy + epsilon)
        )

        # Sample value - get first non-null value from sample
        sample_result = (
            df_sample_table.select(col).filter(df_sample_table[col].notnull()).limit(1).execute()
        )
        sample_val = str(sample_result[col].iloc[0])[:50] if len(sample_result) > 0 else "<null>"

        results.append(
            {
                "column": col,
                "dtype": str(schema[col]),
                "value_density": round(value_density, 6),
                "nonzero_density": round(nonzero_density, 6),
                "cardinality_ratio": round(cardinality_ratio, 6),
                "entropy": round(entropy, 4),
                "information_score": round(information_score, 6),
                "sample_value": sample_val,
            }
        )

    # Build pandas DataFrame, sort, then convert to Ibis Table for return
    results_df = (
        pd.DataFrame(results)
        .sort_values("information_score", ascending=False)
        .reset_index(drop=True)
    )

    # Return as Ibis Table for beautiful rendering and composability
    return ibis.memtable(results_df)


def stratified_column_filter(
    attrs: ibis.Table,
    primary_key_threshold: float = 0.9,
    sparse_threshold: float = 0.6,
    grouping_cardinality: float = 0.1,
    grouping_completeness: float = 0.95,
    resource_id_min: float = 0.5,
    resource_id_max: float = 0.9,
    resource_id_completeness: float = 0.95,
    composite_min: float = 0.1,
    composite_max: float = 0.5,
    composite_info_score: float = 0.3,
) -> tuple[list[str], list[str]]:
    """
    Apply stratified filtering to attribute analysis results using Ibis operations.

    Different filtering criteria based on cardinality class:
    - Primary keys (>90%): Always drop (no analytical value)
    - High cardinality (50-90%): Keep if complete (potential resource IDs)
    - Medium cardinality (10-50%): Keep if info score > threshold
    - Grouping dimensions (<10%): Keep if highly complete
    - Sparse columns: Drop if value_density < threshold

    Args:
        attrs: Output from attribute_analysis() (Ibis Table)
        primary_key_threshold: Cardinality ratio above which to drop (primary keys)
        sparse_threshold: Value density below which to drop (too many nulls)
        grouping_cardinality: Max cardinality for grouping dimensions
        grouping_completeness: Min value density for grouping dimensions
        resource_id_min: Min cardinality for resource IDs
        resource_id_max: Max cardinality for resource IDs
        resource_id_completeness: Min value density for resource IDs
        composite_min: Min cardinality for composite key candidates
        composite_max: Max cardinality for composite key candidates
        composite_info_score: Min information score for composite candidates

    Returns:
        (drop_cols, keep_cols): Tuple of column name lists

    Example:
        >>> attrs = hc.utils.attribute_analysis(df)
        >>> drop_cols, keep_cols = hc.utils.stratified_column_filter(attrs)
        >>> df_filtered = df.select([c for c in df.schema().names if c not in drop_cols])
    """
    # DROP: Primary keys (high cardinality, no grouping utility)
    drop_primary_keys = (
        attrs.filter(_.cardinality_ratio > primary_key_threshold)
        .select("column")
        .execute()["column"]
        .tolist()
    )

    # DROP: Sparse columns (too many nulls)
    drop_sparse = (
        attrs.filter(_.value_density < sparse_threshold)
        .select("column")
        .execute()["column"]
        .tolist()
    )

    # KEEP: Grouping dimensions (low cardinality, highly complete)
    keep_grouping = (
        attrs.filter(
            (_.cardinality_ratio <= grouping_cardinality)
            & (_.value_density > grouping_completeness)
        )
        .select("column")
        .execute()["column"]
        .tolist()
    )

    # KEEP: High cardinality resource IDs (if complete)
    keep_resource_ids = (
        attrs.filter(
            (_.cardinality_ratio > resource_id_min)
            & (_.cardinality_ratio <= resource_id_max)
            & (_.value_density > resource_id_completeness)
        )
        .select("column")
        .execute()["column"]
        .tolist()
    )

    # KEEP: Medium cardinality composite candidates (if good info score)
    keep_composite_candidates = (
        attrs.filter(
            (_.cardinality_ratio > composite_min)
            & (_.cardinality_ratio <= composite_max)
            & (_.information_score > composite_info_score)
        )
        .select("column")
        .execute()["column"]
        .tolist()
    )

    # Combine filters
    drop_cols = list(set(drop_primary_keys + drop_sparse))
    keep_cols = list(set(keep_grouping + keep_resource_ids + keep_composite_candidates))

    return drop_cols, keep_cols


# Backward compatibility alias
def comprehensive_schema_analysis(df: ibis.Table) -> ibis.Table:
    """
    Deprecated: Use attribute_analysis() instead.

    This function is maintained for backward compatibility but will be removed
    in a future version. Please update your code to use attribute_analysis().
    """
    return attribute_analysis(df)


def daily_observation_counts(df: ibis.Table, date_col: str | None = None) -> pd.DataFrame:
    """
    Count records per day - simple groupby for distribution analysis.

    Auto-detects date column if not specified (looks for Date/Datetime types).

    Args:
        df: Input Ibis Table
        date_col: Name of date column to group by (None = auto-detect)

    Returns:
        pandas DataFrame with columns: date, count
        Sorted by date ascending.

    Example:
        >>> daily = daily_observation_counts(df)  # Auto-detects date column
        >>> daily.describe()  # See distribution stats
    """
    # Auto-detect date column if not provided
    if date_col is None:
        schema = df.schema()
        date_cols = [name for name, dtype in schema.items() if dtype.is_temporal()]
        if not date_cols:
            raise ValueError("No Date or Datetime columns found in schema")
        date_col = date_cols[0]  # Use first date column found

    return df.group_by(date_col).agg(count=_.count()).order_by(date_col).execute()


def numeric_column_summary(df: ibis.Table, null_threshold: float = 95.0) -> pd.DataFrame:
    """
    Generate summary statistics for numeric columns, filtering high-null columns.

    Args:
        df: Input Ibis Table
        null_threshold: Exclude columns with null percentage > this value

    Returns:
        pandas DataFrame with numeric columns and their distribution statistics

    Example:
        >>> numeric_summary = numeric_column_summary(df, null_threshold=95.0)
        >>> display(numeric_summary)
    """
    schema = df.schema()
    total_rows = df.count().execute()

    # Identify numeric columns
    numeric_types = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "decimal",
    )
    numeric_cols = [
        col
        for col, dtype in schema.items()
        if any(nt in str(dtype).lower() for nt in numeric_types)
    ]

    if not numeric_cols:
        return pd.DataFrame()

    # Build summary DataFrame
    results = []
    for col in numeric_cols:
        # Compute statistics for this column
        col_expr = df[col]

        # Null count
        null_count = col_expr.isnull().sum().execute()
        null_pct = (null_count / total_rows) * 100

        # Filter by null threshold
        if null_pct > null_threshold:
            continue

        # Compute all statistics
        unique_count = col_expr.nunique().execute()
        min_val = col_expr.min().execute()
        max_val = col_expr.max().execute()
        mean_val = col_expr.mean().execute()
        std_val = col_expr.std().execute()
        median_val = col_expr.median().execute()
        q25_val = col_expr.quantile(0.25).execute()
        q75_val = col_expr.quantile(0.75).execute()

        results.append(
            {
                "column": col,
                "dtype": str(schema[col]),
                "null_pct": round(null_pct, 2),
                "unique": unique_count,
                "min": min_val,
                "q25": q25_val,
                "median": median_val,
                "q75": q75_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def categorical_column_summary(
    df: ibis.Table, null_threshold: float = 95.0, sample_size: int = 100_000
) -> pd.DataFrame:
    """
    Generate summary statistics for categorical (string) columns, filtering high-null columns.

    Args:
        df: Input Ibis Table
        null_threshold: Exclude columns with null percentage > this value
        sample_size: Sample size for top value calculation

    Returns:
        pandas DataFrame with categorical columns and their characteristics

    Example:
        >>> cat_summary = categorical_column_summary(df, null_threshold=95.0)
        >>> display(cat_summary)
    """
    schema = df.schema()
    total_rows = df.count().execute()

    # Identify categorical (string) columns
    categorical_cols = [
        col for col, dtype in schema.items() if str(dtype).lower() in ("string", "category")
    ]

    if not categorical_cols:
        return pd.DataFrame()

    # Sample for top value calculation
    df_sample = df.select(categorical_cols).limit(sample_size).execute()

    # Build summary DataFrame
    results = []
    for col in categorical_cols:
        # Null count from full table
        null_count = df[col].isnull().sum().execute()
        null_pct = (null_count / total_rows) * 100

        # Filter by null threshold
        if null_pct > null_threshold:
            continue

        # Unique count from full table
        unique_count = df[col].nunique().execute()
        cardinality_ratio = unique_count / total_rows

        # Get top value from sample
        value_counts = df_sample[col].value_counts()
        if len(value_counts) > 0:
            top_value = value_counts.index[0]
            top_count = value_counts.iloc[0]
            top_pct = round((top_count / len(df_sample)) * 100, 2)
        else:
            top_value = "<null>"
            top_pct = 0.0

        # Compute entropy on sample
        entropy = shannon_entropy_pandas(df_sample[col])

        results.append(
            {
                "column": col,
                "dtype": str(schema[col]),
                "null_pct": round(null_pct, 2),
                "unique": unique_count,
                "cardinality_ratio": round(cardinality_ratio, 6),
                "card_class": cardinality_classification(cardinality_ratio),
                "entropy": round(entropy, 4),
                "top_value": str(top_value)[:30],
                "top_value_pct": top_pct,
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def shannon_entropy_ibis(table: ibis.Table, column: str) -> float:
    """
    Compute Shannon entropy for an Ibis Table column using Ibis operations.

    Shannon entropy measures the information content or "confusion" in a distribution.
    Higher entropy indicates more uniform distribution (more information).
    Lower entropy indicates concentration in few values (less information).

    Formula: H(X) = -∑ p_i * log(p_i)

    Args:
        table: Input Ibis Table
        column: Column name to analyze

    Returns:
        Shannon entropy (non-negative float, 0 = no entropy)

    Example:
        >>> entropy = shannon_entropy_ibis(df, 'cloud_provider')
        >>> print(f"Entropy: {entropy:.3f}")
    """
    # Get value counts using Ibis aggregation
    value_counts = (
        table.group_by(column)
        .agg(count=ibis._.count())
        .execute()  # Small result (unique values), OK to materialize
    )

    if len(value_counts) == 0:
        return 0.0

    # Calculate probabilities and entropy
    total = value_counts["count"].sum()
    if total == 0:
        return 0.0

    probs = value_counts["count"] / total

    # Compute -∑ p * log(p), filtering out zeros
    log_probs = np.log(probs)
    entropy_val = -(probs * log_probs).sum()

    # Handle NaN (can occur if all values are null)
    return float(entropy_val) if not np.isnan(entropy_val) else 0.0


def shannon_entropy_pandas(series: pd.Series) -> float:
    """
    Compute Shannon entropy for a pandas Series.

    Shannon entropy measures the information content or "confusion" in a distribution.
    Higher entropy indicates more uniform distribution (more information).
    Lower entropy indicates concentration in few values (less information).

    Formula: H(X) = -∑ p_i * log(p_i)

    Args:
        series: Input pandas Series

    Returns:
        Shannon entropy (non-negative float, 0 = no entropy)

    Example:
        >>> entropy = shannon_entropy_pandas(df['cloud_provider'])
        >>> print(f"Entropy: {entropy:.3f}")
    """
    # Get value counts as probabilities
    vc = series.value_counts()
    total = vc.sum()

    if total == 0:
        return 0.0

    probs = vc / total

    # Compute -∑ p * log(p), filtering out zeros
    log_probs = np.log(probs)
    entropy_val = -(probs * log_probs).sum()

    # Handle NaN (can occur if all values are null)
    return float(entropy_val) if not np.isnan(entropy_val) else 0.0


# Backward compatibility alias for Polars (deprecated)
def shannon_entropy(series: pd.Series) -> float:
    """
    Deprecated: Use shannon_entropy_pandas() instead.
    Maintained for backward compatibility.
    """
    return shannon_entropy_pandas(series)


def calculate_attribute_scores(df: ibis.Table, sample_size: int = 100_000) -> pd.DataFrame:
    """
    Calculate information scores for all attributes using harmonic mean.

    Computes three metrics for each attribute:
    1. Value density: ρ_v = (non-null count) / N
    2. Cardinality ratio: ρ_c = (unique count) / N
    3. Shannon entropy: H = -∑ p_i log(p_i)

    Then combines via harmonic mean:
    I = 3 / (1/ρ_v + 1/ρ_c + 1/H)

    Harmonic mean ensures attributes must score well on ALL dimensions.

    Args:
        df: Input Ibis Table
        sample_size: Sample size for entropy calculation (expensive on full dataset)

    Returns:
        pandas DataFrame with columns: attribute, value_density, cardinality_ratio,
        entropy, information_score, card_class (sorted by score descending)

    Example:
        >>> scores = calculate_attribute_scores(df, sample_size=100_000)
        >>> scores.head(10)  # Top 10 most informative attributes
    """
    schema = df.schema()
    total_rows = df.count().execute()

    # Sample for entropy calculation
    df_sample = df.limit(sample_size).execute()

    # Compute metrics for each attribute
    results = []
    for col in schema.keys():
        # Value density
        non_null_count = total_rows - df[col].isnull().sum().execute()
        value_density = non_null_count / total_rows

        # Cardinality ratio
        unique_count = df[col].nunique().execute()
        cardinality_ratio = unique_count / total_rows

        # Shannon entropy (on sample)
        entropy = shannon_entropy_pandas(df_sample[col])

        # Harmonic mean (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        harmonic_mean = 3.0 / (
            1.0 / (value_density + epsilon)
            + 1.0 / (cardinality_ratio + epsilon)
            + 1.0 / (entropy + epsilon)
        )

        results.append(
            {
                "attribute": col,
                "value_density": round(value_density, 6),
                "cardinality_ratio": round(cardinality_ratio, 6),
                "entropy": round(entropy, 4),
                "information_score": round(harmonic_mean, 6),
                "card_class": cardinality_classification(cardinality_ratio),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("information_score", ascending=False)
        .reset_index(drop=True)
    )


def cardinality_classification(cardinality_ratio: float) -> str:
    """
    Classify cardinality ratio into High/Medium/Low categories.

    Args:
        cardinality_ratio: Ratio of unique values to total rows

    Returns:
        'High' (>0.01), 'Medium' (0.0001-0.01), or 'Low' (<0.0001)
    """
    if cardinality_ratio > 0.01:
        return "High"
    elif cardinality_ratio > 0.0001:
        return "Medium"
    else:
        return "Low"


def infer_column_semantics(column_name: str) -> dict:  # noqa: C901
    """
    Infer semantic meaning from column name patterns.

    Uses naming conventions to classify columns by their likely semantic role:
    - Cost/financial metrics
    - Usage/consumption metrics
    - Identifiers (keys, IDs, UUIDs)
    - Temporal attributes
    - Kubernetes/container metadata
    - Cloud provider hierarchy (account, region, product, etc.)

    Args:
        column_name: Column name to analyze

    Returns:
        Dictionary with inferred semantics: category, subcategory, expected_characteristics

    Example:
        >>> infer_column_semantics('materialized_discounted_cost')
        {'category': 'financial', 'subcategory': 'cost_discounted',
         'expected': 'non-negative numeric, right-skewed'}
    """
    col_lower = column_name.lower()

    # Financial metrics
    if "cost" in col_lower or "price" in col_lower or "charge" in col_lower:
        subcategory = "cost_unknown"
        if "discount" in col_lower:
            subcategory = "cost_discounted"
        elif "amortiz" in col_lower:
            subcategory = "cost_amortized"
        elif "invoice" in col_lower:
            subcategory = "cost_invoiced"
        elif "public" in col_lower or "demand" in col_lower:
            subcategory = "cost_list_price"

        return {
            "category": "financial",
            "subcategory": subcategory,
            "expected": "non-negative numeric (or small negative for refunds), right-skewed",
            "unit": "currency",
            "quality_checks": ["check for negative values", "check for extreme outliers"],
        }

    # Usage/consumption metrics
    if "usage" in col_lower or "consumption" in col_lower:
        return {
            "category": "consumption",
            "subcategory": "usage_metric",
            "expected": "non-negative numeric, possibly zero",
            "unit": "varies (GB, hours, requests)",
            "quality_checks": ["check for negative values", "check zero prevalence"],
        }

    # Identifiers
    if any(x in col_lower for x in ["id", "uuid", "arn", "key"]):
        return {
            "category": "identifier",
            "subcategory": "unique_key" if "uuid" in col_lower else "reference_key",
            "expected": "high cardinality string, minimal nulls",
            "unit": "none",
            "quality_checks": ["check uniqueness", "check null rate"],
        }

    # Temporal
    if any(x in col_lower for x in ["date", "time", "timestamp", "period"]):
        return {
            "category": "temporal",
            "subcategory": "date" if "date" in col_lower else "timestamp",
            "expected": "datetime type, sequential, no gaps",
            "unit": "datetime",
            "quality_checks": ["check for gaps", "check range validity"],
        }

    # Kubernetes/container
    if col_lower.startswith("_k8s") or "kubernetes" in col_lower:
        return {
            "category": "kubernetes",
            "subcategory": "container_metadata",
            "expected": "sparse (high nulls), categorical",
            "unit": "none",
            "quality_checks": ["expect high null percentage", "check cardinality"],
        }

    # Cloud hierarchy - accounts
    if "account" in col_lower:
        return {
            "category": "cloud_hierarchy",
            "subcategory": "account_identifier",
            "expected": "medium cardinality, categorical",
            "unit": "none",
            "quality_checks": ["check cardinality matches expected accounts"],
        }

    # Cloud hierarchy - products/services
    if any(x in col_lower for x in ["product", "service", "family"]):
        return {
            "category": "cloud_hierarchy",
            "subcategory": "product_taxonomy",
            "expected": "low-medium cardinality, categorical",
            "unit": "none",
            "quality_checks": ["check value set consistency"],
        }

    # Cloud hierarchy - provider
    if "provider" in col_lower or "cloud" in col_lower:
        return {
            "category": "cloud_hierarchy",
            "subcategory": "provider",
            "expected": "low cardinality (AWS/Azure/GCP), categorical",
            "unit": "none",
            "quality_checks": ["check for expected provider names"],
        }

    # Cloud hierarchy - region/zone
    if "region" in col_lower or "zone" in col_lower or "location" in col_lower:
        return {
            "category": "cloud_hierarchy",
            "subcategory": "geographic",
            "expected": "low-medium cardinality, categorical",
            "unit": "none",
            "quality_checks": ["check for valid region codes"],
        }

    # Descriptive metadata
    if any(x in col_lower for x in ["name", "description", "label", "tag"]):
        return {
            "category": "metadata",
            "subcategory": "descriptive",
            "expected": "high cardinality string, possibly sparse",
            "unit": "none",
            "quality_checks": ["check informativeness"],
        }

    # Aggregation indicators
    if "aggregat" in col_lower or "count" in col_lower:
        return {
            "category": "aggregation",
            "subcategory": "record_count",
            "expected": "positive integer",
            "unit": "count",
            "quality_checks": ["check for zeros", "check distribution"],
        }

    # Default: unknown
    return {
        "category": "unknown",
        "subcategory": "unclassified",
        "expected": "requires investigation",
        "unit": "unknown",
        "quality_checks": ["manual inspection needed"],
    }


def semantic_column_analysis(df: ibis.Table) -> pd.DataFrame:
    """
    Analyze all columns to infer semantic meaning from names.

    Args:
        df: Input Ibis Table

    Returns:
        pandas DataFrame with columns: column, semantic_category, semantic_subcategory,
        expected_characteristics, quality_checks

    Example:
        >>> semantic_analysis = semantic_column_analysis(df)
        >>> display(semantic_analysis)
    """
    schema = df.schema()

    results = []
    for col in schema.keys():
        semantics = infer_column_semantics(col)
        results.append(
            {
                "column": col,
                "dtype": str(schema[col]),
                "semantic_category": semantics["category"],
                "semantic_subcategory": semantics["subcategory"],
                "expected_characteristics": semantics["expected"],
                "unit": semantics["unit"],
                "quality_checks": ", ".join(semantics["quality_checks"]),
            }
        )

    return pd.DataFrame(results)


# ============================================================================
# Temporal Normalization
# ============================================================================


def time_normalized_size(df: ibis.Table, time_col: str, freq: str) -> pd.DataFrame:
    """
    Create normalized time series of record counts by frequency.

    Aggregate record counts by time period to understand data collection patterns.

    Args:
        df: Input Ibis Table with temporal data
        time_col: Name of datetime column
        freq: Frequency string for pandas (e.g., '1D', '1W', '4h')
            Note: Uses pandas frequency strings after execution

    Returns:
        pandas DataFrame with time index and record counts

    Example:
        >>> daily_counts = time_normalized_size(df, 'usage_date', '1D')
        >>> weekly_counts = time_normalized_size(df, 'usage_date', '1W')
    """
    # Execute to pandas and use pandas time operations
    # Ibis doesn't have a direct equivalent to Polars' dt.round
    pdf = df.select([time_col]).execute()

    # Round timestamps using pandas
    pdf["time"] = pdf[time_col].dt.floor(freq)

    # Count records per time bucket
    result = pdf.groupby("time").size().reset_index(name="record_count")
    result = result.sort_values("time").reset_index(drop=True)

    return result


def align_entity_timeseries(
    df: ibis.Table,
    entity_filter: dict[str, any],
    date_col: str,
    metric_col: str,
    cost_col_name: str = "cost",
) -> pd.DataFrame:
    """
    Align entity time series to complete date range, filling zeros only within entity's active period.

    Args:
        df: Input Ibis Table
        entity_filter: Dict mapping column names to values for filtering (e.g., {'account': '123', 'region': 'us-east-1'})
        date_col: Name of date column
        metric_col: Name of metric column to aggregate
        cost_col_name: Name for output cost column

    Returns:
        pandas DataFrame with columns: date, <cost_col_name>
        Only includes dates within entity's first→last observation range

    Example:
        >>> aligned = align_entity_timeseries(
        ...     df,
        ...     entity_filter={'cloud_provider': 'AWS', 'account': '123'},
        ...     date_col='date',
        ...     metric_col='cost'
        ... )
    """
    # Build filter condition
    filter_cond = ibis.literal(True)
    for col, val in entity_filter.items():
        filter_cond = filter_cond & (df[col] == val)

    # Get entity time series
    entity_ts = (
        df.filter(filter_cond).group_by(date_col).agg(entity_cost=df[metric_col].sum()).execute()
    )

    # Get entity's actual date range
    if len(entity_ts) == 0:
        return pd.DataFrame({date_col: [], cost_col_name: []})

    min_date = entity_ts[date_col].min()
    max_date = entity_ts[date_col].max()

    # Get all dates in dataset and filter to entity's active period
    all_dates = (
        df.select(date_col)
        .distinct()
        .filter((df[date_col] >= min_date) & (df[date_col] <= max_date))
        .order_by(date_col)
        .execute()
    )

    # Merge and fill zeros only within active range
    aligned = all_dates.merge(entity_ts, on=date_col, how="left")
    aligned[cost_col_name] = aligned["entity_cost"].fillna(0)
    aligned = aligned[[date_col, cost_col_name]].sort_values(date_col)

    return aligned


def entity_normalized_by_day(
    df: ibis.Table, entity_col: str, metric_col: str, date_col: str = "usage_date"
) -> pd.DataFrame:
    """
    Normalize entity metrics by total daily activity.

    For each entity e and day t, compute:
    x̂_e,t = x_e,t / ∑_e' x_e',t

    This normalization:
    1. Removes volume effects from data collection changes
    2. Creates comparable distributions (all entities sum to 1.0 per day)
    3. Stabilizes time series for forecasting

    Args:
        df: Input Ibis Table
        entity_col: Column to group by (e.g., 'account_id', 'product_family')
        metric_col: Column to normalize (e.g., 'materialized_discounted_cost')
        date_col: Date column for daily aggregation

    Returns:
        pandas DataFrame with: date, entity, metric_raw, metric_normalized, daily_total

    Example:
        >>> normalized = entity_normalized_by_day(
        ...     df,
        ...     entity_col='account_id',
        ...     metric_col='materialized_discounted_cost'
        ... )
    """
    # Aggregate by entity-day
    entity_day = df.group_by([date_col, entity_col]).agg(metric_raw=df[metric_col].sum())

    # Compute daily totals
    daily_totals = entity_day.group_by(date_col).agg(daily_total=entity_day["metric_raw"].sum())

    # Join and normalize
    result = (
        entity_day.join(daily_totals, date_col)
        .mutate(metric_normalized=entity_day["metric_raw"] / daily_totals["daily_total"])
        .order_by([date_col, entity_col])
        .execute()
    )

    return result


# ============================================================================
# Sampling
# ============================================================================


def smart_sample(
    df: ibis.Table, n: int, stratify_col: str | None = None, seed: int = 42
) -> pd.DataFrame:
    """
    Sample DataFrame with optional stratification.

    Simple random sampling when stratify_col=None.
    Stratified sampling (proportional within groups) when stratify_col is provided.

    Args:
        df: Input Ibis Table
        n: Target sample size
        stratify_col: Optional column for stratified sampling
        seed: Random seed for reproducibility

    Returns:
        Sampled pandas DataFrame

    Example:
        >>> # Simple random sample
        >>> sample = smart_sample(df, n=10_000)
        >>>
        >>> # Stratified by cloud provider
        >>> sample = smart_sample(df, n=100_000, stratify_col='cloud_provider')
    """
    # Execute to pandas for sampling operations
    total_count = df.count().execute()
    pdf = df.execute()

    if stratify_col:
        # Stratified sampling - proportional within groups
        fraction = min(n / total_count, 1.0)
        result = pdf.sample(frac=fraction, random_state=seed)
    else:
        # Simple random sampling
        sample_size = min(n, total_count)
        result = pdf.sample(n=sample_size, random_state=seed)

    return result.reset_index(drop=True)


# ============================================================================
# Outlier Detection
# ============================================================================


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are values outside:
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR]

    where IQR = Q3 - Q1

    Args:
        series: Input pandas Series (numeric)
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers)

    Returns:
        Boolean Series (True = outlier)

    Example:
        >>> outliers = detect_outliers_iqr(df['materialized_discounted_cost'])
        >>> df['is_outlier'] = outliers
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Outliers are values where |z| > threshold, where:
    z = (x - μ) / σ

    Args:
        series: Input pandas Series (numeric)
        threshold: Z-score threshold (3.0 = standard, 2.0 = more sensitive)

    Returns:
        Boolean Series (True = outlier)

    Example:
        >>> outliers = detect_outliers_zscore(df['materialized_discounted_cost'])
        >>> print(f"Outliers detected: {outliers.sum()}")
    """
    mean = series.mean()
    std = series.std()

    if std == 0:
        # All values are identical, no outliers
        return pd.Series([False] * len(series), index=series.index)

    z_scores = ((series - mean) / std).abs()
    outliers = z_scores > threshold

    return outliers


def detect_outliers_isolation_forest(
    df: pd.DataFrame, columns: list[str], contamination: float = 0.05, random_state: int = 42
) -> np.ndarray:
    """
    Detect multivariate outliers using Isolation Forest.

    Isolation Forest is effective for high-dimensional outlier detection.
    It isolates observations by randomly selecting features and split values.

    Args:
        df: Input pandas DataFrame
        columns: List of columns to use for outlier detection
        contamination: Expected proportion of outliers (0.05 = 5%)
        random_state: Random seed for reproducibility

    Returns:
        Boolean numpy array (True = outlier)

    Example:
        >>> cost_cols = ['materialized_discounted_cost', 'materialized_usage_amount']
        >>> outliers = detect_outliers_isolation_forest(df, cost_cols, contamination=0.05)
        >>> df['is_outlier'] = outliers
    """
    # Extract columns as numpy array
    X = df[columns].to_numpy()

    # Fit Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = clf.fit_predict(X)

    # Convert to boolean (Isolation Forest returns -1 for outliers, 1 for inliers)
    outliers = predictions == -1

    return outliers


# ============================================================================
# Visualization Helpers
# ============================================================================


def setup_seaborn_style(
    style: str = "whitegrid",
    palette: str = "husl",
    context: str = "notebook",
    font_scale: float = 1.0,
) -> None:
    """
    Configure seaborn styling for consistent, professional visualizations.

    Sets up theme, color palette, and context for all subsequent plots.
    Call this once at the beginning of a notebook or script.

    Args:
        style: Seaborn style preset ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        palette: Color palette name (e.g., 'husl', 'Set2', 'deep', 'muted', 'bright', 'pastel')
        context: Plot context sizing ('paper', 'notebook', 'talk', 'poster')
        font_scale: Scaling factor for font sizes

    Example:
        >>> setup_seaborn_style()  # Use defaults
        >>> setup_seaborn_style(style='darkgrid', palette='Set2', context='talk')
    """
    sns.set_style(style)
    sns.set_palette(palette)
    sns.set_context(context, font_scale=font_scale)


def get_categorical_palette(n_colors: int, palette: str = "husl") -> list:
    """
    Get a categorical color palette with specified number of colors.

    Ensures consistent, visually distinct colors across all categorical plots.
    Uses perceptually uniform color spaces for better accessibility.

    Args:
        n_colors: Number of colors needed
        palette: Palette name ('husl', 'Set2', 'Set3', 'tab10', 'tab20')

    Returns:
        List of RGB color tuples

    Example:
        >>> colors = get_categorical_palette(5)
        >>> colors = get_categorical_palette(12, palette='tab20')
    """
    return sns.color_palette(palette, n_colors=n_colors)


def plot_numeric_distributions(
    df: ibis.Table,
    columns: list[str] | None = None,
    group_by: str | None = None,
    sample_size: int = 50_000,
    figsize: tuple[int, int] = (14, 10),
    cols_per_row: int = 2,
) -> plt.Figure:
    """
    Create grouped boxplots for numeric columns to visualize distributions.

    Shows proper box-and-whisker plots with quartiles, medians, and outliers.
    Can group by a categorical variable (e.g., cloud_provider) to compare distributions.

    Args:
        df: Input Ibis Table
        columns: List of numeric columns to plot (None = auto-detect, max 6)
        group_by: Optional categorical column to group by (e.g., 'cloud_provider')
        sample_size: Sample size for plotting (reduces rendering time)
        figsize: Figure size (width, height)
        cols_per_row: Number of subplots per row

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Simple boxplots
        >>> fig = plot_numeric_distributions(df, sample_size=50_000)
        >>> # Grouped by provider
        >>> fig = plot_numeric_distributions(df, group_by='cloud_provider')
        >>> plt.show()
    """
    # Auto-detect numeric columns if not provided
    if columns is None:
        numeric_summary = numeric_column_summary(df, null_threshold=95.0)
        if len(numeric_summary) == 0:
            raise ValueError("No numeric columns found after filtering")
        # Limit to first 6 for readability
        columns = numeric_summary["column"].to_list()[:6]

    # Select columns for sampling
    select_cols = columns.copy()
    if group_by and group_by not in select_cols:
        select_cols.append(group_by)

    # Sample data - execute to pandas
    sample_df = df.select(select_cols).limit(sample_size).execute()

    # Calculate grid dimensions
    n_cols = len(columns)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row

    # Create figure
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]

    # Plot each column
    for idx, col in enumerate(columns):
        ax = axes[idx]

        # Remove nulls
        if group_by:
            plot_data = sample_df[[col, group_by]].dropna()
        else:
            plot_data = sample_df[[col]].dropna()

        if len(plot_data) > 0:
            if group_by:
                # Grouped boxplot
                sns.boxplot(data=plot_data, y=col, x=group_by, ax=ax, palette="Set2", fliersize=2)
                ax.set_xlabel("")
                ax.set_ylabel("Value", fontsize=10)
                ax.set_title(f"{col} by {group_by}", fontsize=11, fontweight="bold")
                # Rotate x labels if needed
                ax.tick_params(axis="x", rotation=45)
            else:
                # Single boxplot
                sns.boxplot(y=plot_data[col], ax=ax, color="steelblue", width=0.5, fliersize=3)
                ax.set_ylabel("Value", fontsize=10)
                ax.set_title(col, fontsize=11, fontweight="bold")

                # Add statistics annotation
                q25, median, q75 = plot_data[col].quantile([0.25, 0.5, 0.75])
                stats_text = (
                    f"Q1: {q25:.2e}\nMedian: {median:.2e}\nQ3: {q75:.2e}\nN: {len(plot_data):,}"
                )
                ax.text(
                    0.98,
                    0.97,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(col, fontsize=11, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_categorical_frequencies(  # noqa: C901
    df: ibis.Table,
    columns: list[str] | None = None,
    top_n: int = 10,
    figsize: tuple[int, int] = (14, 10),
    cols_per_row: int = 2,
    log_scale: bool = True,
    shared_xaxis: bool = True,
) -> plt.Figure:
    """
    Create horizontal bar charts for categorical columns showing top N values.

    This is the categorical analog to boxplots - visualizing the distribution
    of values by showing frequency of top categories.

    Args:
        df: Input Ibis Table
        columns: List of categorical columns to plot (None = auto-detect)
        top_n: Number of top values to display per column
        figsize: Figure size (width, height)
        cols_per_row: Number of subplots per row
        log_scale: Use logarithmic scale for x-axis (default True)
        shared_xaxis: Use same x-axis limits across all subplots (default True)

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_categorical_frequencies(df, top_n=10, log_scale=True)
        >>> plt.show()
    """
    # Auto-detect categorical columns if not provided
    if columns is None:
        categorical_summary = categorical_column_summary(df, null_threshold=95.0)
        if len(categorical_summary) == 0:
            raise ValueError("No categorical columns found after filtering")
        columns = categorical_summary["column"].to_list()

    # Calculate grid dimensions
    n_cols = len(columns)
    n_rows = (n_cols + cols_per_row - 1) // cols_per_row

    # Create figure
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]

    # Compute value counts using Ibis groupby on full dataset
    all_value_counts = {}
    global_max = 0
    for col in columns:
        # Group by column and count, then get top N
        value_counts_df = (
            df.group_by(col)
            .agg(count=df.count())
            .order_by(ibis.desc("count"))
            .limit(top_n)
            .execute()
        )
        all_value_counts[col] = value_counts_df
        if len(value_counts_df) > 0:
            col_max = value_counts_df["count"].max()
            global_max = max(global_max, col_max)

    # Plot each column
    for idx, col in enumerate(columns):
        ax = axes[idx]
        value_counts_df = all_value_counts[col]

        if len(value_counts_df) > 0:
            # Prepare data for seaborn
            values = value_counts_df[col].tolist()
            counts = value_counts_df["count"].tolist()

            # Truncate long labels
            labels = [str(v)[:30] + "..." if len(str(v)) > 30 else str(v) for v in values]

            # Create DataFrame for seaborn
            plot_df = pd.DataFrame({"category": labels, "count": counts})

            # Create horizontal bar chart with seaborn
            sns.barplot(
                data=plot_df,
                y="category",
                x="count",
                ax=ax,
                color="steelblue",
                alpha=0.8,
                orient="h",
            )

            # Apply log scale if requested
            if log_scale:
                ax.set_xscale("log")
                ax.set_xlabel("Frequency (log scale)", fontsize=10)
            else:
                ax.set_xlabel("Frequency", fontsize=10)

            # Apply shared x-axis limits
            if shared_xaxis:
                if log_scale:
                    ax.set_xlim(1, global_max * 1.5)
                else:
                    ax.set_xlim(0, global_max * 1.1)

            ax.set_ylabel("")  # Remove y-label (category name is in labels)
            ax.set_title(f"{col} (Top {min(top_n, len(values))})", fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add percentage annotations
            total = sum(counts)
            for i, count in enumerate(counts):
                pct = (count / total) * 100
                ax.text(count, i, f" {pct:.1f}%", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(col, fontsize=11, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def create_info_score_chart(
    attribute_scores: pd.DataFrame, figsize: tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create horizontal bar chart of attribute information scores.

    Args:
        attribute_scores: Output from calculate_attribute_scores() (pandas DataFrame)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> scores = calculate_attribute_scores(df)
        >>> fig = create_info_score_chart(scores)
        >>> plt.show()
    """
    # Filter out zero scores (log scale can't handle zeros)
    # Zero information scores indicate completely null or invariant attributes
    filtered_scores = attribute_scores[attribute_scores["information_score"] > 0].copy()

    if len(filtered_scores) == 0:
        raise ValueError("All attributes have zero information score - cannot visualize")

    # Sort by score
    plot_data = filtered_scores.sort_values("information_score", ascending=True)

    # Calculate median for reference line
    median_score = plot_data["information_score"].median()

    # Create figure
    height = max(10, len(plot_data) * 0.4)
    fig, ax = plt.subplots(figsize=(figsize[0], height))

    # Create color palette based on cardinality class
    card_class_colors = {"High": "#1f77b4", "Medium": "#ff7f0e", "Low": "#2ca02c"}

    # Horizontal bar chart with seaborn
    sns.barplot(
        data=plot_data,
        y="attribute",
        x="information_score",
        hue="card_class",
        palette=card_class_colors,
        ax=ax,
        alpha=0.8,
        orient="h",
        legend=False,  # We'll create custom legend below
    )

    # Add median reference line
    ax.axvline(
        median_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_score:.4f}",
        alpha=0.7,
    )

    # Log scale on x-axis
    ax.set_xscale("log")
    ax.set_xlabel("Information Score (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")  # Remove y-label (attribute names are on axis)
    ax.set_title("Attribute Information Scores", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Create legend for cardinality classes and median line
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=card_class_colors[cc], label=cc, alpha=0.8)
        for cc in ["High", "Medium", "Low"]
    ]
    legend_elements.append(
        plt.Line2D(
            [0], [0], color="red", linestyle="--", linewidth=2, label=f"Median: {median_score:.4f}"
        )
    )

    ax.legend(
        handles=legend_elements,
        title="Cardinality Class",
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig


def create_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Correlation Matrix",
    annotate: bool = True,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """
    Create Seaborn annotated correlation heatmap.

    Args:
        corr_df: Correlation matrix as pandas DataFrame
        title: Plot title
        annotate: If True, show correlation values in cells
        figsize: Figure size (width, height)
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Using pandas DataFrame
        >>> cost_cols = [col for col in df.columns if 'cost' in col.lower()]
        >>> corr = df[cost_cols].corr()
        >>> fig = create_correlation_heatmap(corr, title='Cost Metrics Correlation')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use pandas DataFrame directly
    corr_pd = corr_df

    # Create heatmap
    sns.heatmap(
        corr_pd,
        annot=annotate,
        fmt=".3f" if annotate else None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    return fig


def plot_temporal_density(  # noqa: C901
    df: ibis.Table,
    date_col: str,
    metric_col: str | None = None,
    log_scale: bool = False,
    title: str | None = None,
    figsize: tuple[int, int] = (14, 5),
    show_distribution: bool = False,
    dist_type: str = "violin",
    show_marginal: bool = False,
    marginal_type: str = "box",
) -> plt.Figure:
    """
    Plot temporal observation density or metric trends over time using seaborn.

    Creates either:
    - Line plot showing aggregated values (default)
    - Distribution plot showing value spread at each timestamp (show_distribution=True)
    - Can include marginal distribution (box/hist/kde) alongside time series

    Args:
        df: Input Ibis Table
        date_col: Name of date column
        metric_col: Optional metric column to aggregate (None = count records)
        log_scale: Use logarithmic scale for y-axis
        title: Plot title (None = auto-generate)
        figsize: Figure size (width, height)
        show_distribution: If True, show distribution per timestamp instead of aggregation
        dist_type: Distribution plot type ('violin' or 'box'), only used if show_distribution=True
        show_marginal: If True, add marginal distribution plot on the right side
        marginal_type: Marginal plot type ('box', 'hist', or 'kde'), only used if show_marginal=True

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Plot record counts over time (line)
        >>> fig = plot_temporal_density(df, 'usage_date', log_scale=True)
        >>> # Plot with marginal box plot showing distribution of counts
        >>> fig = plot_temporal_density(df, 'usage_date', show_marginal=True, marginal_type='box')
        >>> # Plot cost distribution per day (violin)
        >>> fig = plot_temporal_density(df, 'usage_date', metric_col='cost', show_distribution=True)
        >>> plt.show()
    """
    # Determine if we need to create subplot layout for marginal
    if show_marginal and not show_distribution:
        # Create figure with gridspec for main plot + marginal
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_marg = fig.add_subplot(gs[1], sharey=ax)
    else:
        # Standard single plot
        fig, ax = plt.subplots(figsize=figsize)
        ax_marg = None

    if show_distribution and metric_col:
        # Distribution mode: show value spread at each timestamp
        daily = df.select(date_col, metric_col).order_by(date_col).execute()

        auto_title = f"{metric_col.replace('_', ' ').title()} Distribution Over Time"
        y_label = metric_col.replace("_", " ").title()

        # Plot distribution
        if dist_type == "violin":
            sns.violinplot(
                data=daily,
                x=date_col,
                y=metric_col,
                ax=ax,
                color="steelblue",
                inner="quartile",
                cut=0,
            )
        elif dist_type == "box":
            sns.boxplot(
                data=daily,
                x=date_col,
                y=metric_col,
                ax=ax,
                color="steelblue",
            )
        else:
            raise ValueError(f"dist_type must be 'violin' or 'box', got '{dist_type}'")

    else:
        # Aggregation mode: show single value per timestamp (original behavior)
        if metric_col:
            daily = (
                df.group_by(date_col).agg(value=df[metric_col].sum()).order_by(date_col).execute()
            )
            y_col = "value"
            auto_title = f"{metric_col.replace('_', ' ').title()} Over Time"
            y_label = metric_col.replace("_", " ").title()
        else:
            daily = df.group_by(date_col).agg(value=_.count()).order_by(date_col).execute()
            y_col = "value"
            auto_title = "Temporal Observation Density"
            y_label = "Record Count"

        # Plot with seaborn
        sns.lineplot(
            data=daily,
            x=date_col,
            y=y_col,
            ax=ax,
            linewidth=2.5,
            color="steelblue",
            marker="o",
            markersize=4,
        )

        # Add marginal distribution if requested
        if show_marginal and ax_marg is not None:
            if marginal_type == "box":
                sns.boxplot(
                    data=daily,
                    y=y_col,
                    ax=ax_marg,
                    color="steelblue",
                    width=0.5,
                )
            elif marginal_type == "hist":
                ax_marg.hist(
                    daily[y_col],
                    bins=20,
                    orientation="horizontal",
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="black",
                )
            elif marginal_type == "kde":
                from scipy import stats

                density = stats.gaussian_kde(daily[y_col])
                y_vals = np.linspace(daily[y_col].min(), daily[y_col].max(), 100)
                ax_marg.plot(density(y_vals), y_vals, color="steelblue", linewidth=2)
                ax_marg.fill_betweenx(y_vals, 0, density(y_vals), alpha=0.3, color="steelblue")
            else:
                raise ValueError(
                    f"marginal_type must be 'box', 'hist', or 'kde', got '{marginal_type}'"
                )

            # Style marginal plot
            ax_marg.set_xlabel("")
            ax_marg.set_ylabel("")
            ax_marg.tick_params(labelleft=False)
            ax_marg.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale("log")
        if ax_marg:
            ax_marg.set_yscale("log")
        y_label += " (log scale)"

    # Styling
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title or auto_title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_daily_change_analysis(
    df: ibis.Table,
    date_col: str,
    metric_col: str | None = None,
    highlight_threshold: float | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot day-over-day percent changes with anomaly highlighting using seaborn.

    Useful for identifying pipeline issues, data quality problems, or significant
    business events through sharp changes in daily patterns.

    Args:
        df: Input Ibis Table
        date_col: Name of date column
        metric_col: Optional metric column to analyze (None = count records)
        highlight_threshold: Percent change threshold to highlight (e.g., 0.30 = 30% drop/gain)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Detect anomalies with >30% day-over-day change
        >>> fig = plot_daily_change_analysis(df, 'usage_date', highlight_threshold=0.30)
        >>> plt.show()
    """
    # Aggregate by date
    if metric_col:
        daily = df.group_by(date_col).agg(value=df[metric_col].sum()).order_by(date_col).execute()
        metric_name = metric_col.replace("_", " ").title()
    else:
        daily = df.group_by(date_col).agg(value=_.count()).order_by(date_col).execute()
        metric_name = "Record Count"

    # Compute day-over-day percent change
    daily["pct_change"] = daily["value"].pct_change()

    # Remove first row (no prior day)
    daily = daily.iloc[1:]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine threshold legend label
    if highlight_threshold:
        legend_label = f"|Change| ≥ {highlight_threshold*100:.0f}%"
    else:
        legend_label = None

    # Plot with seaborn scatter plot (better for highlighting individual points)
    sns.scatterplot(
        data=daily,
        x=date_col,
        y="pct_change",
        ax=ax,
        color="steelblue",
        s=80,
        alpha=0.7,
    )

    # Connect with lines
    ax.plot(
        daily[date_col],
        daily["pct_change"],
        linewidth=1.5,
        color="steelblue",
        alpha=0.5,
    )

    # Highlight anomalies if threshold set
    if highlight_threshold:
        anomalies = daily[daily["pct_change"].abs() >= highlight_threshold]
        if len(anomalies) > 0:
            ax.scatter(
                anomalies[date_col],
                anomalies["pct_change"],
                color="red",
                s=120,
                alpha=0.8,
                zorder=5,
                label=legend_label,
                marker="D",
            )

    # Add reference line at 0
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Day-over-Day % Change", fontsize=12, fontweight="bold")
    ax.set_title(f"{metric_name} - Day-over-Day Changes", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if legend_label:
        ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    return fig


def plot_dimension_cost_summary(
    df: ibis.Table,
    dimensions: list[str],
    cost_col: str,
    top_n: int = 10,
    figsize: tuple[int, int] | None = None,
    cols_per_row: int = 2,
) -> plt.Figure:
    """
    Create multi-panel horizontal bar charts showing cost distribution across dimensions.

    Shows top N values for each dimension with their total cost contribution.
    Uses seaborn for consistent styling and automatic color palettes.

    Args:
        df: Input Ibis Table
        dimensions: List of dimensional columns to analyze (e.g., ['provider', 'region', 'product'])
        cost_col: Cost column name
        top_n: Number of top values to show per dimension
        figsize: Figure size (None = auto-calculate based on dimensions)
        cols_per_row: Number of subplots per row

    Returns:
        Matplotlib Figure object

    Example:
        >>> fig = plot_dimension_cost_summary(
        ...     df,
        ...     dimensions=['cloud_provider', 'region', 'product_family'],
        ...     cost_col='cost',
        ...     top_n=10
        ... )
        >>> plt.show()
    """
    # Auto-calculate figsize if not provided
    if figsize is None:
        n_dims = len(dimensions)
        n_rows = (n_dims + cols_per_row - 1) // cols_per_row
        figsize = (14, max(4, n_rows * 4))

    # Calculate grid dimensions
    n_dims = len(dimensions)
    n_rows = (n_dims + cols_per_row - 1) // cols_per_row

    # Create figure
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    axes = axes.flatten() if n_dims > 1 else [axes]

    # Get color palette
    colors = get_categorical_palette(top_n, palette="husl")

    # Plot each dimension
    for idx, dim in enumerate(dimensions):
        ax = axes[idx]

        # Compute top N by cost
        dim_summary = (
            df.group_by(dim)
            .agg(total_cost=df[cost_col].sum())
            .order_by(ibis.desc("total_cost"))
            .limit(top_n)
            .execute()
        )

        if len(dim_summary) > 0:
            # Create horizontal bar chart with seaborn
            sns.barplot(
                data=dim_summary,
                y=dim,
                x="total_cost",
                ax=ax,
                palette=colors[: len(dim_summary)],
                orient="h",
            )

            # Styling
            ax.set_xlabel("Total Cost ($)", fontsize=11, fontweight="bold")
            ax.set_ylabel("")
            ax.set_title(
                f"{dim.replace('_', ' ').title()} (Top {len(dim_summary)})",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.3, linestyle="--")

            # Add cost labels
            for i, (_, row) in enumerate(dim_summary.iterrows()):
                cost = row["total_cost"]
                ax.text(cost, i, f" ${cost:,.0f}", va="center", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(dim.replace("_", " ").title(), fontsize=12, fontweight="bold")

    # Hide unused subplots
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_entity_timeseries(
    df: ibis.Table,
    entity_filters: list[dict[str, any]],
    date_col: str,
    metric_col: str,
    entity_labels: list[str] | None = None,
    mode: str = "line",
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot time series for multiple entities with seaborn styling.

    Supports both line plots (individual entities) and stacked area plots (cumulative contribution).
    Uses entity filters to select specific entities and plots their metric over time.

    Args:
        df: Input Ibis Table
        entity_filters: List of dicts mapping column names to values for filtering
                       Example: [{'provider': 'AWS', 'account': '123'}, {'provider': 'Azure', 'account': '456'}]
        date_col: Name of date column
        metric_col: Name of metric column to plot
        entity_labels: Optional custom labels for entities (None = use "Entity 1", "Entity 2", ...)
        mode: Plot mode - 'line' for individual trajectories, 'area' for stacked contribution
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Plot top 5 entities as individual lines
        >>> filters = [
        ...     {'cloud_provider': 'AWS', 'account': '12345'},
        ...     {'cloud_provider': 'AWS', 'account': '67890'},
        ... ]
        >>> fig = plot_entity_timeseries(df, filters, 'date', 'cost', mode='line')
        >>> plt.show()
    """
    # Get labels
    if entity_labels is None:
        entity_labels = [f"Entity {i+1}" for i in range(len(entity_filters))]

    # Collect aligned timeseries for all entities
    all_timeseries = []
    for i, entity_filter in enumerate(entity_filters):
        aligned = align_entity_timeseries(
            df,
            entity_filter=entity_filter,
            date_col=date_col,
            metric_col=metric_col,
            cost_col_name=metric_col,
        )
        aligned["entity"] = entity_labels[i]
        all_timeseries.append(aligned)

    # Combine into single DataFrame
    combined = pd.concat(all_timeseries, ignore_index=True)

    # Create figure with one or two subplots depending on mode
    if mode == "line":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
    else:  # mode == 'area'
        fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Get color palette
    colors = get_categorical_palette(len(entity_filters), palette="husl")

    # Plot 1: Individual trajectories (line mode) or top panel (area mode)
    ax1 = axes[0]

    if mode == "line":
        # Line plot with seaborn
        sns.lineplot(
            data=combined,
            x=date_col,
            y=metric_col,
            hue="entity",
            ax=ax1,
            linewidth=2.5,
            marker="o",
            markersize=5,
            palette=colors,
        )

        # Styling
        ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax1.set_ylabel(metric_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Entity Time Series - {metric_col.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(title="", fontsize=10, loc="best")

        # Rotate x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    else:  # mode == 'area'
        # Same line plot for top panel
        sns.lineplot(
            data=combined,
            x=date_col,
            y=metric_col,
            hue="entity",
            ax=ax1,
            linewidth=2.5,
            marker="o",
            markersize=5,
            palette=colors,
        )

        # Styling for top panel
        ax1.set_xlabel("")  # No x-label on top panel
        ax1.set_ylabel(metric_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax1.set_title(
            "Entity Time Series - Individual Trajectories", fontsize=14, fontweight="bold", pad=15
        )
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.legend(title="", fontsize=10, loc="best")

        # Bottom panel: Stacked area plot
        ax2 = axes[1]

        # Pivot data for stacking
        pivot = combined.pivot(index=date_col, columns="entity", values=metric_col).fillna(0)

        # Create stacked area
        ax2.stackplot(
            pivot.index,
            *[pivot[entity] for entity in entity_labels],
            labels=entity_labels,
            colors=colors,
            alpha=0.8,
        )

        # Styling for bottom panel
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.set_ylabel(metric_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax2.set_title(
            "Entity Time Series - Cumulative Contribution", fontsize=14, fontweight="bold", pad=15
        )
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.legend(title="", fontsize=10, loc="upper left")

        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_grain_persistence_comparison(
    grain_results_df: pd.DataFrame,
    stability_threshold: float = 70.0,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Visualize grain persistence comparison results with seaborn.

    Creates side-by-side bar charts showing:
    1. Entity count per grain (granularity)
    2. Stability percentage per grain (quality)

    Highlights grains meeting the stability threshold for optimal grain selection.

    Args:
        grain_results_df: DataFrame with columns: Grain, entities, stable_pct, median_days
        stability_threshold: Minimum stability percentage to highlight (default 70%)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> grain_results = pd.DataFrame({
        ...     'Grain': ['Provider', 'Provider + Account', 'Account + Region'],
        ...     'entities': [5, 127, 453],
        ...     'stable_pct': [100.0, 95.3, 72.4],
        ...     'median_days': [37, 37, 35]
        ... })
        >>> fig = plot_grain_persistence_comparison(grain_results, stability_threshold=70.0)
        >>> plt.show()
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Determine colors based on stability threshold
    colors = [
        "steelblue" if stable >= stability_threshold else "lightcoral"
        for stable in grain_results_df["stable_pct"]
    ]

    # Plot 1: Entity counts (granularity)
    sns.barplot(
        data=grain_results_df,
        x="entities",
        y="Grain",
        ax=ax1,
        palette=colors,
        orient="h",
    )

    ax1.set_xlabel("Number of Entities", fontsize=12, fontweight="bold")
    ax1.set_ylabel("")
    ax1.set_title("Grain Granularity\n(Entity Count)", fontsize=13, fontweight="bold", pad=15)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # Add entity count labels
    for i, row in grain_results_df.iterrows():
        count = row["entities"]
        ax1.text(count, i, f" {count:,}", va="center", fontsize=10, fontweight="bold")

    # Plot 2: Stability percentages (quality)
    sns.barplot(
        data=grain_results_df,
        x="stable_pct",
        y="Grain",
        ax=ax2,
        palette=colors,
        orient="h",
    )

    ax2.set_xlabel("Stability (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")
    ax2.set_title("Grain Stability\n(% Entities ≥30 days)", fontsize=13, fontweight="bold", pad=15)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.set_xlim(0, 100)

    # Add stability labels and highlight threshold
    for i, row in grain_results_df.iterrows():
        stable = row["stable_pct"]
        ax2.text(stable, i, f" {stable:.1f}%", va="center", fontsize=10, fontweight="bold")

    # Add threshold reference line
    ax2.axvline(
        stability_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Threshold: {stability_threshold:.0f}%",
    )
    ax2.legend(fontsize=10, loc="lower right")

    # Add overall title
    fig.suptitle(
        "Grain Persistence Analysis - Granularity vs. Stability Tradeoff",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()
    return fig
