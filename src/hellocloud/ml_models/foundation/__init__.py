"""
Foundation models for time series forecasting.

This module provides stubs for future integration with state-of-the-art
foundation models for time series forecasting. These are pretrained models
that can perform zero-shot forecasting on new time series without requiring
fine-tuning.

**CURRENT STATUS: STUB IMPLEMENTATIONS ONLY**

All models in this module are currently stubs that raise NotImplementedError.
They define the API contract for future implementations but do not contain
working model code.

Available Models (Stubs):
    - TimesFM: Google Research's decoder-only transformer for time series
      * 200M parameters
      * Context: 512 points, Horizon: 128 steps
      * GitHub: https://github.com/google-research/timesfm
      * Paper: https://arxiv.org/abs/2310.10688

    - Chronos: Amazon Science's T5-based forecasting models
      * Multiple sizes: tiny (8M) to large (710M parameters)
      * Probabilistic forecasting with quantile outputs
      * GitHub: https://github.com/amazon-science/chronos-forecasting
      * Paper: https://arxiv.org/abs/2403.07815

Usage:
    >>> from hellocloud.ml_models.foundation import TimesFMForecaster
    >>> # This will raise NotImplementedError:
    >>> model = TimesFMForecaster()
    NotImplementedError: TimesFM integration not yet implemented...

Base Class:
    All foundation models inherit from FoundationModelBase, which defines
    the standard interface:
        - predict(history, horizon, **kwargs) -> Dict[str, np.ndarray]
        - load_model(model_path) -> None
        - get_model_info() -> Dict[str, Any]

Future Development:
    To implement actual model integration:
    1. Install required packages (timesfm, chronos-forecasting)
    2. Replace NotImplementedError raises with actual model loading
    3. Implement predict() using model-specific API
    4. Add comprehensive tests in tests/ml_models/foundation/
    5. Update documentation with usage examples

See Also:
    - cloud_sim.ml_models.advanced_forecasting: Alternative forecasting methods
    - cloud_sim.ml_models.pymc_cloud_model: Bayesian hierarchical models
"""

from .base import FoundationModelBase
from .chronos import ChronosForecaster
from .timesfm import TimesFMForecaster

__all__ = [
    "FoundationModelBase",
    "TimesFMForecaster",
    "ChronosForecaster",
]

__version__ = "0.1.0-stub"
