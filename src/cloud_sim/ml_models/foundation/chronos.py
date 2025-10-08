"""Chronos by Amazon Science - Stub Implementation."""

from typing import Any

import numpy as np

from .base import FoundationModelBase


class ChronosForecaster(FoundationModelBase):
    """
    Chronos by Amazon Science.

    **NOT YET IMPLEMENTED** - This is a stub for future integration.

    Chronos is a family of pretrained time series forecasting models
    based on language model architectures (T5). It treats time series
    forecasting as a language modeling task by tokenizing time series
    values and training transformer models.

    Key Features (When Implemented):
        - Multiple model sizes: tiny, mini, small, base, large
        - Based on T5 (Text-to-Text Transfer Transformer) architecture
        - Supports probabilistic forecasting with quantile outputs
        - Zero-shot forecasting capabilities
        - Trained on diverse time series datasets

    Model Sizes:
        - chronos-t5-tiny: ~8M parameters
        - chronos-t5-mini: ~20M parameters
        - chronos-t5-small: ~46M parameters
        - chronos-t5-base: ~200M parameters
        - chronos-t5-large: ~710M parameters

    References:
        - GitHub: https://github.com/amazon-science/chronos-forecasting
        - Paper: "Chronos: Learning the Language of Time Series"
          https://arxiv.org/abs/2403.07815
        - HuggingFace: https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444
        - Blog: https://aws.amazon.com/blogs/machine-learning/chronos-learning-the-language-of-time-series/

    Future Implementation Notes:
        Installation:
            pip install chronos-forecasting

        Basic Usage:
            from chronos import ChronosPipeline
            import torch

            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-base",
                device_map="cpu",
                torch_dtype=torch.bfloat16,
            )

            forecast = pipeline.predict(
                context=torch.tensor(history),
                prediction_length=horizon,
                num_samples=20,
            )

        Key Parameters:
            - model_size: One of [tiny, mini, small, base, large]
            - prediction_length: Forecast horizon
            - num_samples: Number of probabilistic samples
            - temperature: Sampling temperature
            - top_k: Top-k sampling parameter
            - top_p: Nucleus sampling parameter

    Example:
        >>> # This will raise NotImplementedError
        >>> forecaster = ChronosForecaster(model_size="base")
        Traceback (most recent call last):
        ...
        NotImplementedError: Chronos integration not yet implemented...
    """

    VALID_MODEL_SIZES = ["tiny", "mini", "small", "base", "large"]

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        torch_dtype: str | None = None,
    ) -> None:
        """
        Initialize Chronos forecaster.

        Args:
            model_size: Model size - one of [tiny, mini, small, base, large] (default: base)
            device: Device to run model on - 'cpu', 'cuda', or 'mps' (default: cpu)
            torch_dtype: PyTorch dtype - 'float32', 'float16', or 'bfloat16' (default: bfloat16)

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError(
            "Chronos integration not yet implemented. "
            "To use Chronos when available, you will need to install: pip install chronos-forecasting\n"
            "See documentation: https://github.com/amazon-science/chronos-forecasting\n"
            "Paper: https://arxiv.org/abs/2403.07815\n"
            f"Valid model sizes: {', '.join(self.VALID_MODEL_SIZES)}"
        )

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """
        Generate probabilistic forecasts using Chronos.

        Args:
            history: Historical time series data (n_samples,)
            horizon: Number of steps to forecast (prediction_length)
            num_samples: Number of sample trajectories to generate (default: 20)
            temperature: Sampling temperature for controlling randomness (default: 1.0)
            top_k: Top-k sampling parameter (default: 50)
            top_p: Nucleus sampling probability threshold (default: 1.0)
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing:
                - 'mean': Mean forecast across samples (horizon,)
                - 'median': Median forecast (horizon,)
                - 'quantiles': Dict mapping quantile levels to forecasts
                  Default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                - 'samples': All forecast samples (num_samples, horizon)

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("Chronos stub - predict() not implemented")

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load Chronos model weights from HuggingFace Hub or local path.

        Args:
            model_path: HuggingFace model ID (e.g., 'amazon/chronos-t5-base')
                       or local path to model checkpoint. If None, uses
                       default model based on initialized model_size.

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("Chronos stub - load_model() not implemented")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get Chronos model metadata.

        Returns:
            Dictionary containing:
                - 'name': Model name (e.g., 'chronos-t5-base')
                - 'size': Model size tier
                - 'parameters': Approximate number of parameters
                - 'context_length': Maximum context length
                - 'max_horizon': Maximum forecast horizon
                - 'supports_probabilistic': True (Chronos supports probabilistic forecasting)
                - 'architecture': 'T5' (transformer architecture)

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("Chronos stub - get_model_info() not implemented")

    @classmethod
    def list_available_models(cls) -> list[str]:
        """
        List all available Chronos model variants.

        Returns:
            List of model identifiers on HuggingFace Hub

        Raises:
            NotImplementedError: Always raised - this is a stub implementation
        """
        raise NotImplementedError("Chronos stub - list_available_models() not implemented")
