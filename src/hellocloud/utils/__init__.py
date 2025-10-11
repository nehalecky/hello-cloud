"""
Utility functions for cloud resource simulation.

Provides helpers for logging configuration, data processing, and notebook workflows.
"""

from .cost_analysis import (
    # Cost Distribution
    cost_distribution_metrics,
    # Entity Analysis
    detect_entity_anomalies,
    # Correlation & Redundancy
    find_correlated_pairs,
    normalize_by_period,
    select_from_pairs,
    # Data Splitting
    split_at_date,
    # Temporal Quality
    temporal_quality_metrics,
)
from .eda_analysis import (
    # Time Series
    align_entity_timeseries,
    # Schema & Information Theory
    attribute_analysis,
    calculate_attribute_scores,
    cardinality_classification,
    categorical_column_summary,
    comprehensive_schema_analysis,  # Deprecated alias
    create_correlation_heatmap,
    create_info_score_chart,
    daily_observation_counts,
    # Outlier Detection
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    entity_normalized_by_day,
    get_categorical_palette,
    infer_column_semantics,
    numeric_column_summary,
    plot_categorical_frequencies,
    plot_daily_change_analysis,
    plot_dimension_cost_summary,
    plot_entity_timeseries,
    plot_grain_persistence_comparison,
    # Visualization
    plot_numeric_distributions,
    plot_temporal_density,
    semantic_column_analysis,
    setup_seaborn_style,
    shannon_entropy,
    # Sampling
    smart_sample,
    stratified_column_filter,
    # Temporal Normalization
    time_normalized_size,
)
from .notebook_logging import (
    configure_notebook_logging,
    quiet_library_logging,
    verbose_library_logging,
)

__all__ = [
    # Logging
    "configure_notebook_logging",
    "quiet_library_logging",
    "verbose_library_logging",
    # EDA - Time Series
    "align_entity_timeseries",
    # EDA - Schema & Information Theory
    "attribute_analysis",
    "stratified_column_filter",
    "comprehensive_schema_analysis",  # Deprecated
    "daily_observation_counts",
    "numeric_column_summary",
    "categorical_column_summary",
    "semantic_column_analysis",
    "infer_column_semantics",
    "shannon_entropy",
    "calculate_attribute_scores",
    "cardinality_classification",
    # EDA - Temporal Normalization
    "time_normalized_size",
    "entity_normalized_by_day",
    # EDA - Sampling
    "smart_sample",
    # EDA - Outlier Detection
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_isolation_forest",
    # EDA - Visualization Config
    "setup_seaborn_style",
    "get_categorical_palette",
    # EDA - Visualization
    "plot_numeric_distributions",
    "plot_categorical_frequencies",
    "plot_temporal_density",
    "plot_daily_change_analysis",
    "plot_dimension_cost_summary",
    "plot_entity_timeseries",
    "plot_grain_persistence_comparison",
    "create_info_score_chart",
    "create_correlation_heatmap",
    # Cost Analysis - Correlation & Redundancy
    "find_correlated_pairs",
    "select_from_pairs",
    # Cost Analysis - Temporal Quality
    "temporal_quality_metrics",
    # Cost Analysis - Distribution
    "cost_distribution_metrics",
    # Cost Analysis - Entity Analysis
    "detect_entity_anomalies",
    "normalize_by_period",
    # Cost Analysis - Data Splitting
    "split_at_date",
]
