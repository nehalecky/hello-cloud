"""TimesFM (Time Series Foundation Model) by Google Research - Stub Implementation."""

from typing import Any

import numpy as np

from .base import FoundationModelBase


class TimesFMForecaster(FoundationModelBase):
    """
    TimesFM (Time Series Foundation Model) by Google Research.

    **NOT YET IMPLEMENTED** - This is a stub for future integration.

    TimesFM is a pretrained foundation model for univariate time series
    forecasting developed by Google Research. It uses a decoder-only
    transformer architecture trained on a diverse corpus of time series data.

    Key Features (When Implemented):
        - Decoder-only transformer architecture
        - 200M parameter model
        - Context length: Up to 512 time points
        - Forecast horizons: Up to 128 steps ahead
        - Zero-shot forecasting on new time series
        - No fine-tuning required for many tasks

    References:
        - GitHub: https://github.com/google-research/timesfm
        - Paper: "A decoder-only foundation model for time-series forecasting"
          https://arxiv.org/abs/2310.10688
        - Blog: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

    Future Implementation Notes:
        Installation:
            pip install timesfm

        Basic Usage:
            import timesfm
            tfm = timesfm.TimesFm(
                context_len=512,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
            )
            tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

        Key Parameters:
            - context_len: Historical context window (default: 512)
            - horizon_len: Forecast horizon (default: 128)
            - input_patch_len: Input patch size for transformer
            - output_patch_len: Output patch size

    Example:
        >>> # This will raise NotImplementedError
        >>> forecaster = TimesFMForecaster()
        Traceback (most recent call last):
        ...
        NotImplementedError: TimesFM integration not yet implemented...
    """

    def __init__(
        self,
        model_name: str = "google/timesfm-1.0-200m",
        context_len: int = 512,
        horizon_len: int = 128,
    ) -> None:
        """
        Initialize TimesFM forecaster.

        Args:
            model_name: HuggingFace model identifier (default: google/timesfm-1.0-200m)
            context_len: Maximum historical context length (default: 512)
            horizon_len: Maximum forecast horizon (default: 128)

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError(
            "TimesFM integration not yet implemented. "
            "To use TimesFM when available, you will need to install: pip install timesfm\n"
            "See documentation: https://github.com/google-research/timesfm\n"
            "Paper: https://arxiv.org/abs/2310.10688"
        )

    def predict(self, history: np.ndarray, horizon: int, **kwargs: Any) -> dict[str, np.ndarray]:
        """
        Generate forecasts using TimesFM.

        Args:
            history: Historical time series data (n_samples,)
            horizon: Number of steps to forecast
            **kwargs: Additional parameters:
                - num_samples: Number of forecast samples (default: 1)
                - temperature: Sampling temperature (default: 1.0)

        Returns:
            Dictionary with 'mean' and optionally 'quantiles'

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("TimesFM stub - predict() not implemented")

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load TimesFM model weights.

        Args:
            model_path: Path to model checkpoint or HuggingFace repo ID

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("TimesFM stub - load_model() not implemented")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get TimesFM model metadata.

        Returns:
            Dictionary with model information

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("TimesFM stub - get_model_info() not implemented")
