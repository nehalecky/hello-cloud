"""Hello Cloud - Cloud Cost Analysis and Optimization"""

__version__ = "0.1.0"
__author__ = "Nicholaus Halecky"

# Import submodules
from . import analysis, generation, modeling, spark, transforms, utils

# Convenience imports
from .generation import (
    CloudMetricsDatasetBuilder,
    CloudMetricsSimulator,
    WorkloadPatternGenerator,
    WorkloadType,
)
from .modeling import (
    CloudResourceHierarchicalModel,
    CloudResourceTaxonomy,
)
from .utils import (
    configure_notebook_logging,
    quiet_library_logging,
    verbose_library_logging,
)

__all__ = [
    "analysis",
    "generation",
    "modeling",
    "spark",
    "transforms",
    "utils",
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
