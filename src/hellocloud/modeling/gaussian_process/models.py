"""
Sparse Gaussian Process models for scalable time series learning.

This module provides variational sparse GP models that scale to large datasets
using inducing points to reduce computational complexity from O(n³) to O(nm²).

Based on:
- Sparse GPs (SVGP): Hensman et al. (2013), "Gaussian Processes for Big Data"
- Variational inference for scalability
- Learn inducing point locations during training

References:
- GPyTorch SVGP tutorial: https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
"""

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .kernels import CompositePeriodicKernel


class SparseGPModel(ApproximateGP):
    """
    Sparse Variational Gaussian Process (SVGP) with composite periodic kernel.

    Uses inducing points to achieve O(nm²) complexity instead of O(n³),
    enabling training on datasets with 100k+ samples.

    The model learns:
    1. **Inducing point locations** (optimized during training)
    2. **Variational distribution** over inducing function values
    3. **Kernel hyperparameters** (outputscale for each component)
    4. **Mean function** parameters (constant baseline)

    Example:
        ```python
        # Initialize inducing points (evenly spaced)
        M = 200
        inducing_indices = torch.linspace(0, len(X_train)-1, M, dtype=torch.long)
        inducing_points = X_train[inducing_indices].clone()

        # Create model
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.StudentTLikelihood()

        # Train (see training.py for full example)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_train))
        ```

    Args:
        inducing_points: Tensor of shape (M, D) where M is number of inducing points,
                        D is input dimensionality (typically 1 for time series)
        learn_inducing_locations: If True, optimize inducing point locations (default: True)
        slow_period: Period for slow periodic component (default: 1.0)
        fast_period: Period for fast periodic component (default: 0.2)
        rbf_lengthscale: Lengthscale for RBF component (default: 0.1)

    Attributes:
        mean_module: Constant mean function
        covar_module: CompositePeriodicKernel for multi-scale patterns
        variational_strategy: Handles inducing point approximation
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        learn_inducing_locations: bool = True,
        slow_period: float = 1.0,
        fast_period: float = 0.2,
        rbf_lengthscale: float = 0.1,
    ):
        # Initialize variational distribution (Gaussian over inducing points)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        # Initialize variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        # Mean function (constant baseline)
        self.mean_module = gpytorch.means.ConstantMean()

        # Composite periodic kernel for multi-scale patterns
        self.covar_module = CompositePeriodicKernel(
            slow_period=slow_period, fast_period=fast_period, rbf_lengthscale=rbf_lengthscale
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass through the GP model.

        Args:
            x: Input tensor of shape (n, 1) for univariate time series

        Returns:
            MultivariateNormal distribution with mean and covariance
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def num_inducing_points(self) -> int:
        """Number of inducing points used in variational approximation."""
        return self.variational_strategy.inducing_points.size(0)

    @property
    def inducing_points(self) -> torch.Tensor:
        """Current inducing point locations."""
        return self.variational_strategy.inducing_points


def initialize_inducing_points(
    X_train: torch.Tensor, num_inducing: int, method: str = "evenly_spaced"
) -> torch.Tensor:
    """
    Initialize inducing point locations for sparse GP.

    Args:
        X_train: Training inputs of shape (n, d)
        num_inducing: Number of inducing points (M)
        method: Initialization method:
            - "evenly_spaced": Evenly spaced across training range (default)
            - "random": Random subset of training points
            - "kmeans": K-means clustering (not yet implemented)

    Returns:
        Inducing points tensor of shape (M, d)

    Example:
        ```python
        inducing_points = initialize_inducing_points(
            X_train=X_train_norm,
            num_inducing=200,
            method="evenly_spaced"
        )
        ```
    """
    n_train = len(X_train)

    if method == "evenly_spaced":
        # Evenly spaced indices across training data
        indices = torch.linspace(0, n_train - 1, num_inducing, dtype=torch.long)
        return X_train[indices].clone()

    elif method == "random":
        # Random subset of training points
        indices = torch.randperm(n_train)[:num_inducing]
        return X_train[indices].clone()

    elif method == "kmeans":
        raise NotImplementedError(
            "K-means initialization not yet implemented. "
            "Use 'evenly_spaced' or 'random' instead."
        )

    else:
        raise ValueError(
            f"Unknown initialization method: {method}. "
            f"Choose from: 'evenly_spaced', 'random', 'kmeans'"
        )
