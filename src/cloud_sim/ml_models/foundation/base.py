"""Abstract base class for time series foundation models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class FoundationModelBase(ABC):
    """
    Abstract base class for time series foundation models.

    This class defines the common interface that all foundation model
    implementations must follow. It ensures consistency across different
    model backends (TimesFM, Chronos, etc.).
    """

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int, **kwargs: Any) -> dict[str, np.ndarray]:
        """
        Generate forecasts from historical time series data.

        Args:
            history: Historical time series data with shape (n_samples,).
                     Should be a 1D array for univariate forecasting.
            horizon: Number of future time steps to forecast.
            **kwargs: Model-specific parameters such as:
                - temperature: Sampling temperature for probabilistic forecasts
                - num_samples: Number of forecast samples to generate
                - quantiles: List of quantiles to compute (e.g., [0.1, 0.5, 0.9])

        Returns:
            Dictionary containing forecast outputs:
                - 'mean': Point forecast (horizon,)
                - 'quantiles': Optional dict mapping quantile -> forecast array
                - 'samples': Optional full sample trajectories (num_samples, horizon)

        Raises:
            ValueError: If history is empty or horizon is non-positive
            NotImplementedError: If the model is not yet implemented
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str | None = None) -> None:
        """
        Load model weights from disk or HuggingFace Hub.

        Args:
            model_path: Optional path to model checkpoint. If None, loads
                       the default pretrained model for this implementation.

        Raises:
            FileNotFoundError: If model_path is provided but doesn't exist
            NotImplementedError: If the model is not yet implemented
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get metadata about the loaded model.

        Returns:
            Dictionary containing model information:
                - 'name': Model name/identifier
                - 'version': Model version
                - 'parameters': Number of model parameters
                - 'context_length': Maximum historical context length
                - 'max_horizon': Maximum forecast horizon
                - 'supports_probabilistic': Whether model supports uncertainty quantification

        Raises:
            NotImplementedError: If the model is not yet implemented
        """
        pass
