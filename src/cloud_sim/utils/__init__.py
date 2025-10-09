"""
Utility functions for cloud resource simulation.

Provides helpers for logging configuration, data processing, and notebook workflows.
"""

from .notebook_logging import (
    configure_notebook_logging,
    quiet_library_logging,
    verbose_library_logging,
)

from .eda_analysis import (
    # Schema & Information Theory
    attribute_analysis,
    comprehensive_schema_analysis,  # Deprecated alias
    daily_observation_analysis,
    numeric_column_summary,
    categorical_column_summary,
    semantic_column_analysis,
    infer_column_semantics,
    shannon_entropy,
    calculate_attribute_scores,
    cardinality_classification,
    # Temporal Normalization
    time_normalized_size,
    entity_normalized_by_day,
    # Sampling
    smart_sample,
    # Outlier Detection
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    # Visualization
    plot_numeric_distributions,
    plot_categorical_frequencies,
    create_info_score_chart,
    create_correlation_heatmap,
)

from .cost_analysis import (
    # Correlation & Redundancy
    find_correlated_pairs,
    select_from_pairs,
    # Temporal Quality
    temporal_quality_metrics,
    # Cost Distribution
    cost_distribution_metrics,
    # Entity Analysis
    detect_entity_anomalies,
    normalize_by_period,
    # Data Splitting
    split_at_date,
)

__all__ = [
    # Logging
    "configure_notebook_logging",
    "quiet_library_logging",
    "verbose_library_logging",
    # EDA - Schema & Information Theory
    "attribute_analysis",
    "comprehensive_schema_analysis",  # Deprecated
    "daily_observation_analysis",
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
    # EDA - Visualization
    "plot_numeric_distributions",
    "plot_categorical_frequencies",
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
