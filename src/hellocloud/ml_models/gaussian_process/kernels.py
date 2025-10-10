"""
Custom GPyTorch kernels for cloud resource time series modeling.

This module provides composite periodic kernels designed to capture
multi-scale temporal patterns observed in cloud resource metrics.

Based on empirical findings from IOPS web server analysis:
- SLOW component: Sawtooth envelope (~1250 timesteps ≈ 21 hours)
- FAST component: Sinusoidal carrier (~250 timesteps ≈ 4 hours)
- Baseline: Smooth deviations via RBF kernel

References:
- Composite kernels: Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning"
- HuggingFace Timeseries-PILE dataset: https://huggingface.co/datasets/AutonLab/Timeseries-PILE
"""

import gpytorch
import torch
from gpytorch.kernels import PeriodicKernel, RBFKernel, ScaleKernel


class CompositePeriodicKernel(gpytorch.kernels.Kernel):
    """
    Composite kernel for two-scale periodic patterns in cloud metrics.

    Architecture: k_periodic(slow) + k_periodic(fast) + k_rbf

    ADDITIVE structure (not multiplicative) for numerical stability.
    Fixed lengthscales prevent optimization instability.

    This kernel is designed to capture:
    1. **Slow periodic component**: Long-period cycles (e.g., daily patterns)
    2. **Fast periodic component**: Short-period cycles (e.g., hourly bursts)
    3. **Smooth baseline**: Non-periodic deviations via RBF kernel

    Example:
        ```python
        # For normalized timestamps in [0, 1] range
        kernel = CompositePeriodicKernel(
            slow_period=1250/X_range,  # Daily cycle
            fast_period=250/X_range,   # Hourly cycle
            rbf_lengthscale=0.1
        )
        ```

    Args:
        slow_period: Period length for slow component (in normalized units)
        fast_period: Period length for fast component (in normalized units)
        rbf_lengthscale: Lengthscale for RBF kernel (controls smoothness)
        **kwargs: Additional arguments passed to parent Kernel class

    Attributes:
        slow_periodic: Scaled periodic kernel for slow component
        fast_periodic: Scaled periodic kernel for fast component
        rbf: Scaled RBF kernel for baseline deviations
    """

    def __init__(
        self,
        slow_period: float = 1.0,
        fast_period: float = 0.2,
        rbf_lengthscale: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store configuration for model serialization
        self._slow_period = slow_period
        self._fast_period = fast_period
        self._rbf_lengthscale = rbf_lengthscale

        # SLOW: Long-period cycles (e.g., daily sawtooth envelope)
        slow_periodic_kernel = PeriodicKernel()
        slow_periodic_kernel.period_length = slow_period
        # Keep period fixed, but LEARN lengthscale for pattern smoothness
        slow_periodic_kernel.raw_period_length.requires_grad = False
        self.slow_periodic = ScaleKernel(slow_periodic_kernel)

        # FAST: Short-period cycles (e.g., hourly sinusoidal carrier)
        fast_periodic_kernel = PeriodicKernel()
        fast_periodic_kernel.period_length = fast_period
        # Keep period fixed, but LEARN lengthscale for pattern smoothness
        fast_periodic_kernel.raw_period_length.requires_grad = False
        self.fast_periodic = ScaleKernel(fast_periodic_kernel)

        # Smooth baseline deviations - use LARGER lengthscale so it doesn't dominate
        rbf_kernel = RBFKernel()
        rbf_kernel.lengthscale = rbf_lengthscale * 3.0  # 0.3 instead of 0.1 - less flexible
        # Allow RBF lengthscale to be learned but start larger
        self.rbf = ScaleKernel(rbf_kernel)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Compute covariance matrix using additive kernel combination.

        K(x1, x2) = K_slow(x1, x2) + K_fast(x1, x2) + K_rbf(x1, x2)

        Args:
            x1: First set of inputs (n1 x d)
            x2: Second set of inputs (n2 x d)
            diag: If True, return only diagonal elements
            last_dim_is_batch: If True, last dimension is treated as batch
            **params: Additional kernel parameters

        Returns:
            Covariance matrix (n1 x n2) or diagonal vector (n1,) if diag=True
        """
        # Additive combination for numerical stability (not multiplicative)
        return (
            self.slow_periodic(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
            + self.fast_periodic(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
            + self.rbf(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        )

    @property
    def is_stationary(self) -> bool:
        """Kernel is stationary (invariant to translations)."""
        return True

    @property
    def slow_period(self) -> float:
        """Period length for slow periodic component."""
        return self._slow_period

    @property
    def fast_period(self) -> float:
        """Period length for fast periodic component."""
        return self._fast_period

    @property
    def rbf_lengthscale(self) -> float:
        """Lengthscale for RBF component."""
        return self._rbf_lengthscale
