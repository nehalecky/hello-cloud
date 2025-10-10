"""
Hello Cloud - Cloud Cost Analysis and Optimization

Hands-on tools for cloud resource usage analysis and cost optimization.
"""

__version__ = "0.1.0"
__author__ = "Nicholaus Halecky"

# Import submodules for namespace access (e.g., hc.utils.plot_temporal_density)
from . import utils

# Import key classes and functions for top-level convenience
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
    # Submodules
    "utils",
    # Data Generation
    "CloudMetricsDatasetBuilder",
    "CloudMetricsSimulator",
    "WorkloadPatternGenerator",
    "WorkloadType",
    # ML Models
    "CloudResourceHierarchicalModel",
    "CloudResourceTaxonomy",
    # Logging utilities (convenience)
    "configure_notebook_logging",
    "quiet_library_logging",
    "verbose_library_logging",
]
