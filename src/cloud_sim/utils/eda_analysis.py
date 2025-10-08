"""
Exploratory Data Analysis utilities for billing and time series data.

Provides reusable functions for schema analysis, information theory calculations,
temporal normalization patterns (following 7Park methodology), and outlier detection.
Designed for Polars DataFrames with focus on cloud billing data.
"""

import numpy as np
import polars as pl
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


# ============================================================================
# Schema & Information Theory
# ============================================================================

def comprehensive_schema_analysis(df: pl.LazyFrame) -> pl.DataFrame:
    """
    Comprehensive schema analysis showing all columns with statistics.

    Computes null counts, cardinality, sample values, and quality indicators
    for every attribute in the dataset. Use pl.Config(tbl_rows=-1) to display
    all rows without truncation.

    Args:
        df: Input LazyFrame to analyze

    Returns:
        DataFrame with columns: column, dtype, null_count, null_pct,
        unique_count, cardinality_ratio, sample_value, quality, card_class

    Example:
        >>> with pl.Config(tbl_rows=-1):
        >>>     display(comprehensive_schema_analysis(df))
    """
    schema = df.collect_schema()
    total_rows = df.select(pl.len()).collect()[0, 0]

    # Collect statistics in parallel
    stats = df.select([
        pl.all().null_count().name.suffix('_nulls'),
        pl.all().n_unique().name.suffix('_unique'),
        pl.all().drop_nulls().first().cast(pl.Utf8).name.suffix('_sample')
    ]).collect()

    # Build comprehensive schema DataFrame
    schema_df = pl.DataFrame({
        'column': schema.names(),
        'dtype': [str(dt) for dt in schema.dtypes()],
        'null_count': [stats[f'{col}_nulls'][0] for col in schema.names()],
        'null_pct': [round(stats[f'{col}_nulls'][0] / total_rows * 100, 2) for col in schema.names()],
        'unique_count': [f"{stats[f'{col}_unique'][0]:,}" for col in schema.names()],
        'cardinality_ratio': [round(stats[f'{col}_unique'][0] / total_rows, 6) for col in schema.names()],
        'sample_value': [str(stats[f'{col}_sample'][0])[:50] if stats[f'{col}_sample'][0] is not None else '<null>'
                        for col in schema.names()],
    }).with_columns([
        # Quality indicator
        pl.when(pl.col('null_pct') < 1).then(pl.lit('✓'))
          .when(pl.col('null_pct') < 50).then(pl.lit('⚠'))
          .otherwise(pl.lit('✗')).alias('quality'),
        # Cardinality class
        pl.when(pl.col('cardinality_ratio') > 0.01).then(pl.lit('High'))
          .when(pl.col('cardinality_ratio') > 0.0001).then(pl.lit('Medium'))
          .otherwise(pl.lit('Low')).alias('card_class')
    ])

    return schema_df


def shannon_entropy(series: pl.Series) -> float:
    """
    Compute Shannon entropy for a Polars Series.

    Shannon entropy measures the information content or "confusion" in a distribution.
    Higher entropy indicates more uniform distribution (more information).
    Lower entropy indicates concentration in few values (less information).

    Formula: H(X) = -∑ p_i * log(p_i)

    Args:
        series: Input Polars Series

    Returns:
        Shannon entropy (non-negative float, 0 = no entropy)

    Example:
        >>> entropy = shannon_entropy(df['cloud_provider'])
        >>> print(f"Entropy: {entropy:.3f}")
    """
    # Get value counts as probabilities
    vc = series.value_counts()
    total = vc.select(pl.col('count').sum())[0, 0]
    probs = vc.select(pl.col('count') / total).to_series()

    # Compute -∑ p * log(p), filtering out zeros
    log_probs = probs.log()
    entropy_val = -(probs * log_probs).sum()

    # Handle NaN (can occur if all values are null)
    return float(entropy_val) if not np.isnan(entropy_val) else 0.0


def calculate_attribute_scores(df: pl.LazyFrame, sample_size: int = 100_000) -> pl.DataFrame:
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
        df: Input LazyFrame
        sample_size: Sample size for entropy calculation (expensive on full dataset)

    Returns:
        DataFrame with columns: attribute, value_density, cardinality_ratio,
        entropy, information_score, card_class (sorted by score descending)

    Example:
        >>> scores = calculate_attribute_scores(df, sample_size=100_000)
        >>> scores.head(10)  # Top 10 most informative attributes
    """
    schema = df.collect_schema()
    total_rows = df.select(pl.len()).collect()[0, 0]

    # Sample for entropy calculation
    df_sample = df.head(sample_size).collect()

    # Compute metrics for each attribute
    results = []
    for col in schema.names():
        # Value density
        non_null_count = df.select(pl.col(col).drop_nulls().len()).collect()[0, 0]
        value_density = non_null_count / total_rows

        # Cardinality ratio
        unique_count = df.select(pl.col(col).n_unique()).collect()[0, 0]
        cardinality_ratio = unique_count / total_rows

        # Shannon entropy (on sample)
        entropy = shannon_entropy(df_sample[col])

        # Harmonic mean (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        harmonic_mean = 3.0 / (1.0/(value_density + epsilon) +
                                1.0/(cardinality_ratio + epsilon) +
                                1.0/(entropy + epsilon))

        results.append({
            'attribute': col,
            'value_density': round(value_density, 6),
            'cardinality_ratio': round(cardinality_ratio, 6),
            'entropy': round(entropy, 4),
            'information_score': round(harmonic_mean, 6),
            'card_class': cardinality_classification(cardinality_ratio)
        })

    return pl.DataFrame(results).sort('information_score', descending=True)


def cardinality_classification(cardinality_ratio: float) -> str:
    """
    Classify cardinality ratio into High/Medium/Low categories.

    Args:
        cardinality_ratio: Ratio of unique values to total rows

    Returns:
        'High' (>0.01), 'Medium' (0.0001-0.01), or 'Low' (<0.0001)
    """
    if cardinality_ratio > 0.01:
        return 'High'
    elif cardinality_ratio > 0.0001:
        return 'Medium'
    else:
        return 'Low'


# ============================================================================
# Temporal Normalization (7Park Pattern)
# ============================================================================

def time_normalized_size(
    df: pl.LazyFrame,
    time_col: str,
    freq: str
) -> pl.DataFrame:
    """
    Create normalized time series of record counts by frequency.

    This follows the 7Park pattern for temporal analysis: aggregate record
    counts by time period to understand data collection patterns.

    Args:
        df: Input LazyFrame with temporal data
        time_col: Name of datetime column
        freq: Frequency string (e.g., '1d', '1w', '4h')

    Returns:
        DataFrame with time index and record counts

    Example:
        >>> daily_counts = time_normalized_size(df, 'usage_date', '1d')
        >>> weekly_counts = time_normalized_size(df, 'usage_date', '1w')
    """
    result = (df
        .group_by(pl.col(time_col).dt.round(freq).alias('time'))
        .agg(pl.len().alias('record_count'))
        .sort('time')
        .collect()
    )

    return result


def entity_normalized_by_day(
    df: pl.LazyFrame,
    entity_col: str,
    metric_col: str,
    date_col: str = 'usage_date'
) -> pl.DataFrame:
    """
    Normalize entity metrics by total daily activity (7Park pattern).

    For each entity e and day t, compute:
    x̂_e,t = x_e,t / ∑_e' x_e',t

    This normalization:
    1. Removes volume effects from data collection changes
    2. Creates comparable distributions (all entities sum to 1.0 per day)
    3. Stabilizes time series for forecasting

    Args:
        df: Input LazyFrame
        entity_col: Column to group by (e.g., 'account_id', 'product_family')
        metric_col: Column to normalize (e.g., 'materialized_discounted_cost')
        date_col: Date column for daily aggregation

    Returns:
        DataFrame with: date, entity, metric_raw, metric_normalized, daily_total

    Example:
        >>> normalized = entity_normalized_by_day(
        ...     df,
        ...     entity_col='account_id',
        ...     metric_col='materialized_discounted_cost'
        ... )
    """
    # Aggregate by entity-day
    entity_day = (df
        .group_by([date_col, entity_col])
        .agg(pl.col(metric_col).sum().alias('metric_raw'))
    )

    # Compute daily totals
    daily_totals = (entity_day
        .group_by(date_col)
        .agg(pl.col('metric_raw').sum().alias('daily_total'))
    )

    # Join and normalize
    result = (entity_day
        .join(daily_totals, on=date_col)
        .with_columns([
            (pl.col('metric_raw') / pl.col('daily_total')).alias('metric_normalized')
        ])
        .sort([date_col, entity_col])
        .collect()
    )

    return result


# ============================================================================
# Sampling
# ============================================================================

def smart_sample(
    df: pl.LazyFrame,
    n: int,
    stratify_col: Optional[str] = None,
    seed: int = 42
) -> pl.DataFrame:
    """
    Sample DataFrame with optional stratification.

    Simple random sampling when stratify_col=None.
    Stratified sampling (proportional within groups) when stratify_col is provided.

    Args:
        df: Input LazyFrame
        n: Target sample size
        stratify_col: Optional column for stratified sampling
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame

    Example:
        >>> # Simple random sample
        >>> sample = smart_sample(df, n=10_000)
        >>>
        >>> # Stratified by cloud provider
        >>> sample = smart_sample(df, n=100_000, stratify_col='cloud_provider')
    """
    if stratify_col:
        # Stratified sampling - proportional within groups
        total_count = df.select(pl.len()).collect()[0, 0]
        fraction = min(n / total_count, 1.0)

        result = (df
            .with_columns(pl.lit(seed).alias('_seed'))
            .collect()
            .sample(fraction=fraction, seed=seed)
        )

        return result
    else:
        # Simple random sampling via head (deterministic for reproducibility)
        # Note: Polars doesn't have built-in random sampling in LazyFrame
        # For true random sampling, collect first
        return df.collect().sample(n=min(n, df.select(pl.len()).collect()[0, 0]), seed=seed)


# ============================================================================
# Outlier Detection
# ============================================================================

def detect_outliers_iqr(
    series: pl.Series,
    multiplier: float = 1.5
) -> pl.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are values outside:
    [Q1 - multiplier*IQR, Q3 + multiplier*IQR]

    where IQR = Q3 - Q1

    Args:
        series: Input Polars Series (numeric)
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme outliers)

    Returns:
        Boolean Series (True = outlier)

    Example:
        >>> outliers = detect_outliers_iqr(df['materialized_discounted_cost'])
        >>> df_with_outliers = df.with_columns(outliers.alias('is_outlier'))
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers


def detect_outliers_zscore(
    series: pl.Series,
    threshold: float = 3.0
) -> pl.Series:
    """
    Detect outliers using Z-score method.

    Outliers are values where |z| > threshold, where:
    z = (x - μ) / σ

    Args:
        series: Input Polars Series (numeric)
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
        return pl.Series([False] * len(series))

    z_scores = ((series - mean) / std).abs()
    outliers = z_scores > threshold

    return outliers


def detect_outliers_isolation_forest(
    df: pl.DataFrame,
    columns: List[str],
    contamination: float = 0.05,
    random_state: int = 42
) -> np.ndarray:
    """
    Detect multivariate outliers using Isolation Forest.

    Isolation Forest is effective for high-dimensional outlier detection.
    It isolates observations by randomly selecting features and split values.

    Args:
        df: Input Polars DataFrame
        columns: List of columns to use for outlier detection
        contamination: Expected proportion of outliers (0.05 = 5%)
        random_state: Random seed for reproducibility

    Returns:
        Boolean numpy array (True = outlier)

    Example:
        >>> cost_cols = ['materialized_discounted_cost', 'materialized_usage_amount']
        >>> outliers = detect_outliers_isolation_forest(df, cost_cols, contamination=0.05)
        >>> df = df.with_columns(pl.Series('is_outlier', outliers))
    """
    # Extract columns as numpy array
    X = df.select(columns).to_numpy()

    # Fit Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = clf.fit_predict(X)

    # Convert to boolean (Isolation Forest returns -1 for outliers, 1 for inliers)
    outliers = predictions == -1

    return outliers


# ============================================================================
# Visualization Helpers
# ============================================================================

def create_info_score_chart(
    attribute_scores: pl.DataFrame,
    interactive: bool = True
) -> alt.Chart:
    """
    Create Altair bar chart of attribute information scores.

    Args:
        attribute_scores: Output from calculate_attribute_scores()
        interactive: If True, enable tooltips and interactivity

    Returns:
        Altair Chart object

    Example:
        >>> scores = calculate_attribute_scores(df)
        >>> chart = create_info_score_chart(scores)
        >>> chart.display()
    """
    # Calculate median for reference line
    median_score = attribute_scores['information_score'].median()

    base = alt.Chart(attribute_scores.to_pandas())

    bar = base.mark_bar().encode(
        x=alt.X('information_score:Q',
                title='Information Score (Harmonic Mean)',
                scale=alt.Scale(type='log')),
        y=alt.Y('attribute:N',
                sort='-x',
                title='Attribute'),
        color=alt.Color('card_class:N',
                       title='Cardinality Class',
                       scale=alt.Scale(scheme='category10')),
        tooltip=[
            'attribute',
            alt.Tooltip('information_score:Q', format='.6f', title='Score'),
            alt.Tooltip('value_density:Q', format='.6f', title='Value Density'),
            alt.Tooltip('cardinality_ratio:Q', format='.6f', title='Cardinality Ratio'),
            alt.Tooltip('entropy:Q', format='.4f', title='Entropy')
        ] if interactive else []
    )

    rule = base.mark_rule(color='red', strokeDash=[5, 5]).encode(
        x=alt.X('median:Q'),
        size=alt.value(2)
    ).transform_calculate(
        median=str(median_score)
    )

    chart = (bar + rule).properties(
        width=700,
        height=max(400, len(attribute_scores) * 15),
        title='Attribute Information Scores (Log Scale)'
    )

    if interactive:
        chart = chart.interactive()

    return chart


def create_correlation_heatmap(
    corr_df: pl.DataFrame,
    title: str = 'Correlation Matrix',
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu_r',
    vmin: float = -1.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Create Seaborn annotated correlation heatmap.

    Args:
        corr_df: Correlation matrix as Polars DataFrame
        title: Plot title
        annotate: If True, show correlation values in cells
        figsize: Figure size (width, height)
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib Figure object

    Example:
        >>> cost_cols = [col for col in df.columns if 'cost' in col.lower()]
        >>> corr = df.select(cost_cols).corr()
        >>> fig = create_correlation_heatmap(corr, title='Cost Metrics Correlation')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to pandas for seaborn
    corr_pd = corr_df.to_pandas()

    # Create heatmap
    sns.heatmap(
        corr_pd,
        annot=annotate,
        fmt='.3f' if annotate else None,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    return fig
