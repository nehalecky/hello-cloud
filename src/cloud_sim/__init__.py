"""
Cloud Resource Simulation Platform

Advanced synthetic data generation and ML models for cloud cost optimization.
Demonstrates cutting-edge approaches to FinOps challenges.
"""

__version__ = "0.1.0"
__author__ = "Nicholaus Halecky"

from .data_generation import (
    CloudMetricsDatasetBuilder,
    CloudMetricsSimulator,
    WorkloadPatternGenerator,
)
from .ml_models import (
    CloudResourceHierarchicalModel,
    CloudResourceTaxonomy,
)
from .utils import (
    configure_notebook_logging,
    quiet_library_logging,
    verbose_library_logging,
)

__all__ = [
    "CloudMetricsSimulator",
    "WorkloadPatternGenerator",
    "CloudMetricsDatasetBuilder",
    "CloudResourceHierarchicalModel",
    "CloudResourceTaxonomy",
    "configure_notebook_logging",
    "quiet_library_logging",
    "verbose_library_logging",
]
