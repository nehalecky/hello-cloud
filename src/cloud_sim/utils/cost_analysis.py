"""
Cost and time series analysis utilities for cloud billing datasets.

Provides atomic, composable functions for analyzing cost data with temporal
and entity dimensions. Designed to be parametric - users specify column names
and control all thresholds.

Philosophy: Small, testable, composable building blocks. Notebooks demonstrate
composition patterns.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import date
from scipy.stats import pearsonr


# ============================================================================
# Correlation & Redundancy Detection
# ============================================================================

def find_correlated_pairs(
    corr_matrix: pl.DataFrame,
    columns: List[str],
    threshold: float = 0.90
) -> List[Tuple[str, str, float]]:
    """
    Find correlated column pairs above threshold.

    Pure function - works on ANY correlation matrix from any source.

    Args:
        corr_matrix: Correlation matrix as Polars DataFrame
        columns: List of column names (must match corr_matrix cols)
        threshold: Absolute correlation threshold (0.90 = 90%)

    Returns:
        List of (col_i, col_j, correlation_value) tuples

    Example:
        >>> corr = df.select(['cost1', 'cost2', 'cost3']).corr()
        >>> pairs = find_correlated_pairs(corr, ['cost1', 'cost2', 'cost3'], 0.90)
        >>> # [(col_i, col_j, 0.95), ...]
    """
    corr_np = corr_matrix.to_numpy()
    pairs = [
        (columns[i], columns[j], abs(corr_np[i, j]))
        for i in range(len(columns))
        for j in range(i + 1, len(columns))
        if abs(corr_np[i, j]) > threshold
    ]
    return pairs


def select_from_pairs(
    pairs: List[Tuple[str, str, float]],
    score_map: Dict[str, float]
) -> set[str]:
    """
    Given correlated pairs and scores, select which columns to drop.

    Pure function - works with ANY scoring system (information score,
    variance, importance, etc.).

    For each pair, drops the column with LOWER score, keeps HIGHER score.

    Args:
        pairs: List of (col_i, col_j, correlation) from find_correlated_pairs()
        score_map: Dict mapping column names to scores {col: score, ...}

    Returns:
        Set of column names to drop

    Example:
        >>> pairs = [('cost_a', 'cost_b', 0.95)]
        >>> scores = {'cost_a': 0.8, 'cost_b': 0.3}
        >>> drops = select_from_pairs(pairs, scores)
        >>> # {'cost_b'}  # Lower score dropped
    """
    drops = set()
    for col_i, col_j, corr_val in pairs:
        score_i = score_map.get(col_i, 0)
        score_j = score_map.get(col_j, 0)
        to_drop = col_j if score_i > score_j else col_i
        drops.add(to_drop)
    return drops


# ============================================================================
# Temporal Quality Metrics
# ============================================================================

def temporal_quality_metrics(
    df: pl.LazyFrame,
    date_col: str,
    metric_col: str
) -> Dict:
    """
    Compute temporal quality indicators for time series data.

    Analyzes date coverage, volume stability, and autocorrelation to assess
    suitability for forecasting and time series modeling.

    Args:
        df: Input LazyFrame
        date_col: Name of date column
        metric_col: Name of metric column to analyze (e.g., cost, usage)

    Returns:
        Dictionary with:
            - date_range: (min_date, max_date)
            - coverage_days: Actual unique days observed
            - expected_days: Expected days based on date range
            - completeness_pct: coverage / expected * 100
            - record_volume_cv: Coefficient of variation for daily record counts
            - metric_lag1_autocorr: Lag-1 autocorrelation of daily metric
            - stability_class: 'stable' | 'variable' | 'volatile'

    Example:
        >>> quality = temporal_quality_metrics(df, 'usage_date', 'cost')
        >>> quality['stability_class']  # 'stable'
        >>> quality['metric_lag1_autocorr']  # 0.85 (sticky infrastructure)
    """
    # Date coverage
    date_stats = df.select([
        pl.col(date_col).min().alias('min_date'),
        pl.col(date_col).max().alias('max_date'),
        pl.col(date_col).n_unique().alias('unique_dates')
    ]).collect()

    min_date = date_stats['min_date'][0]
    max_date = date_stats['max_date'][0]
    actual_days = date_stats['unique_dates'][0]
    expected_days = (max_date - min_date).days + 1

    # Daily aggregations for stability analysis
    daily_agg = (
        df.group_by(date_col)
        .agg([
            pl.len().alias('records'),
            pl.col(metric_col).sum().alias('metric_sum')
        ])
        .sort(date_col)
        .collect()
    )

    # Record volume stability
    record_cv = daily_agg['records'].std() / daily_agg['records'].mean()

    # Metric autocorrelation (lag-1)
    metric_series = daily_agg['metric_sum'].to_numpy()
    if len(metric_series) > 1:
        lag1_corr, _ = pearsonr(metric_series[:-1], metric_series[1:])
    else:
        lag1_corr = 0.0

    # Stability classification
    if record_cv < 0.15 and lag1_corr > 0.7:
        stability = 'stable'
    elif record_cv < 0.30 and lag1_corr > 0.5:
        stability = 'variable'
    else:
        stability = 'volatile'

    return {
        'date_range': (min_date, max_date),
        'coverage_days': actual_days,
        'expected_days': expected_days,
        'completeness_pct': round((actual_days / expected_days) * 100, 2),
        'record_volume_cv': round(record_cv, 4),
        'metric_lag1_autocorr': round(lag1_corr, 4),
        'stability_class': stability
    }


# ============================================================================
# Cost Distribution Analysis
# ============================================================================

def cost_distribution_metrics(
    df: pl.LazyFrame,
    cost_col: str
) -> Dict:
    """
    Characterize cost distribution for modeling decisions.

    Computes percentiles, skewness, and outlier rates to inform
    transformation choices (log vs linear scale, robust methods).

    Args:
        df: Input LazyFrame
        cost_col: Name of cost column to analyze

    Returns:
        Dictionary with:
            - percentiles: Dict of {0, 1, 10, 25, 50, 75, 90, 99, 100: value}
            - skewness: Third moment / std^3 (>1 indicates right-skew)
            - outlier_count_iqr: Number of IQR outliers (k=1.5)
            - outlier_pct: Percentage of outliers
            - modeling_rec: 'log_transform' | 'linear' | 'robust'

    Example:
        >>> dist = cost_distribution_metrics(df, 'materialized_discounted_cost')
        >>> dist['skewness']  # 3.2 (highly right-skewed)
        >>> dist['modeling_rec']  # 'log_transform'
    """
    cost_series = df.select(pl.col(cost_col)).collect().to_series()

    # Percentiles
    percentile_values = [0, 1, 10, 25, 50, 75, 90, 99, 100]
    percentiles = {
        p: cost_series.quantile(p / 100)
        for p in percentile_values
    }

    # Skewness (third moment)
    mean = cost_series.mean()
    std = cost_series.std()
    if std > 0:
        skewness = ((cost_series - mean) ** 3).mean() / (std ** 3)
    else:
        skewness = 0.0

    # IQR outliers
    q1, q3 = percentiles[25], percentiles[75]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((cost_series < lower_bound) | (cost_series > upper_bound)).sum()
    outlier_pct = (outliers / len(cost_series)) * 100

    # Modeling recommendation
    if skewness > 1:
        modeling_rec = 'log_transform'
    elif outlier_pct > 10:
        modeling_rec = 'robust'
    else:
        modeling_rec = 'linear'

    return {
        'percentiles': percentiles,
        'skewness': round(skewness, 3),
        'outlier_count_iqr': int(outliers),
        'outlier_pct': round(outlier_pct, 2),
        'modeling_rec': modeling_rec
    }


# ============================================================================
# Entity Anomaly Detection
# ============================================================================

def detect_entity_anomalies(
    df: pl.LazyFrame,
    entity_col: str,
    date_col: str = 'usage_date',
    min_days: int = 10,
    top_n: int = 3
) -> pl.DataFrame:
    """
    Find entities with high temporal variability (anomaly candidates).

    For each entity, computes daily record count variation (CV) to identify
    entities driving dataset instability.

    Args:
        df: Input LazyFrame
        entity_col: Entity column to analyze (e.g., 'cloud_provider', 'account')
        date_col: Date column for temporal grouping
        min_days: Minimum days entity must appear (filter rare entities)
        top_n: Return top N most variable entities

    Returns:
        DataFrame with columns:
            - entity (from entity_col)
            - mean_daily_records
            - std_daily_records
            - cv (coefficient of variation)
            - days_present

    Example:
        >>> anomalies = detect_entity_anomalies(df, 'cloud_provider')
        >>> # Shows which provider has highest temporal variance
    """
    # Daily entity contributions
    entity_daily = (
        df.group_by([date_col, entity_col])
        .agg(pl.len().alias('daily_records'))
        .collect()
    )

    # Compute CV per entity
    entity_stats = (
        entity_daily
        .group_by(entity_col)
        .agg([
            pl.col('daily_records').mean().alias('mean_daily_records'),
            pl.col('daily_records').std().alias('std_daily_records'),
            pl.len().alias('days_present')
        ])
        .with_columns(
            (pl.col('std_daily_records') / pl.col('mean_daily_records')).alias('cv')
        )
        .filter(pl.col('days_present') >= min_days)
        .sort('cv', descending=True)
        .head(top_n)
        .rename({entity_col: 'entity'})
    )

    return entity_stats


# ============================================================================
# Entity Normalization
# ============================================================================

def normalize_by_period(
    df: pl.LazyFrame,
    entity_col: str,
    metric_col: str,
    time_col: str,
    freq: str = '1d'
) -> pl.DataFrame:
    """
    Entity-normalized time series: x_e,t / Σ_e' x_e',t

    For each time period, normalize entity metric by total metric across
    all entities. Removes volume effects from data collection changes.

    Pattern from reference notebook (cell 61).

    Args:
        df: Input LazyFrame
        entity_col: Entity column (e.g., 'account', 'product')
        metric_col: Metric to normalize (e.g., 'cost', 'usage')
        time_col: Time column for period grouping
        freq: Polars frequency string ('1d', '1w', '4h')

    Returns:
        DataFrame with columns:
            - time (rounded to freq)
            - entity
            - metric_raw (original sum)
            - metric_normalized (entity / total)
            - period_total

    Example:
        >>> normed = normalize_by_period(df, 'account', 'cost', 'usage_date', '1d')
        >>> # Each day, all accounts sum to 1.0
    """
    # Round time to frequency
    time_expr = pl.col(time_col).dt.round(freq).alias('time')

    # Entity-period aggregation
    entity_period = (
        df.group_by([time_expr, entity_col])
        .agg(pl.col(metric_col).sum().alias('metric_raw'))
    )

    # Period totals
    period_totals = (
        entity_period
        .group_by('time')
        .agg(pl.col('metric_raw').sum().alias('period_total'))
    )

    # Join and normalize
    result = (
        entity_period
        .join(period_totals, on='time')
        .with_columns(
            (pl.col('metric_raw') / pl.col('period_total')).alias('metric_normalized')
        )
        .sort(['time', entity_col])
        .collect()
    )

    return result


# ============================================================================
# Data Splitting (for analysis)
# ============================================================================

def split_at_date(
    df: pl.LazyFrame,
    date_col: str,
    split_date: date
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split dataset into pre/post at specified date.

    Simple utility for before/after analysis (e.g., detecting data quality
    changes, regime shifts).

    Args:
        df: Input LazyFrame
        date_col: Date column name
        split_date: Split point (datetime.date object)

    Returns:
        (pre_df, post_df) - both as collected DataFrames

    Example:
        >>> from datetime import date
        >>> pre, post = split_at_date(df, 'usage_date', date(2025, 10, 7))
        >>> # Analyze variance separately
    """
    pre_df = df.filter(pl.col(date_col) < split_date).collect()
    post_df = df.filter(pl.col(date_col) >= split_date).collect()
    return pre_df, post_df
