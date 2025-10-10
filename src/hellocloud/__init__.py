"""
Hello Cloud - Cloud Cost Analysis and Optimization

Hands-on tools for cloud resource usage analysis and cost optimization.
"""

__version__ = "0.1.0"
__author__ = "Nicholaus Halecky"

from .data_generation import (
    CloudMetricsDatasetBuilder,
    CloudMetricsSimulator,
    WorkloadPatternGenerator,
    WorkloadType,
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
