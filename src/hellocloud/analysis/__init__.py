"""Higher-level analysis for EDA, cost, and distributions."""

from .cost import (
    cost_distribution_metrics,
    detect_entity_anomalies,
    find_correlated_pairs,
    normalize_by_period,
    split_at_date,
    temporal_quality_metrics,
)
from .distribution import (
    compute_kl_divergence,
    compute_kl_divergences,
    compute_ks_tests,
    plot_distribution_comparison,
    plot_pdf_cdf_comparison,
    plot_statistical_tests,
    print_distribution_summary,
)
from .eda import (
    attribute_analysis,
    categorical_column_summary,
    comprehensive_schema_analysis,
    daily_observation_counts,
    numeric_column_summary,
    plot_temporal_density,
    stratified_column_filter,
)

__all__ = [
    # Cost analysis
    "cost_distribution_metrics",
    "detect_entity_anomalies",
    "find_correlated_pairs",
    "normalize_by_period",
    "split_at_date",
    "temporal_quality_metrics",
    # Distribution analysis
    "compute_kl_divergence",
    "compute_kl_divergences",
    "compute_ks_tests",
    "plot_distribution_comparison",
    "plot_pdf_cdf_comparison",
    "plot_statistical_tests",
    "print_distribution_summary",
    # EDA analysis
    "attribute_analysis",
    "categorical_column_summary",
    "comprehensive_schema_analysis",
    "daily_observation_counts",
    "numeric_column_summary",
    "plot_temporal_density",
    "stratified_column_filter",
]
