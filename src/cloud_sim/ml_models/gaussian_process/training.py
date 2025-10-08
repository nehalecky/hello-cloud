"""
Training utilities for Gaussian Process models.

Provides functions for:
- Mini-batch training with variational inference
- Model persistence (save/load)
- Numerical stability configurations
- Training loop with progress tracking

Based on GPyTorch SVGP best practices:
- Variational ELBO objective
- Mini-batch training for scalability
- Cholesky jitter for numerical stability
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import gpytorch
from gpytorch.mlls import VariationalELBO
from loguru import logger

from .models import SparseGPModel


def train_gp_model(
    model: SparseGPModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 0.01,
    cholesky_jitter: float = 1e-3,
    cholesky_max_tries: int = 10,
    verbose: bool = True,
) -> List[float]:
    """
    Train sparse GP model using mini-batch variational inference.

    Uses maximum numerical stability settings to prevent Cholesky decomposition failures.

    Args:
        model: SparseGPModel instance
        likelihood: GPyTorch likelihood (e.g., GaussianLikelihood, StudentTLikelihood)
        X_train: Training inputs of shape (n, d)
        y_train: Training outputs of shape (n,)
        n_epochs: Number of training epochs (default: 100)
        batch_size: Mini-batch size for training (default: 2048)
        learning_rate: Adam optimizer learning rate (default: 0.01)
        cholesky_jitter: Jitter added to diagonal for numerical stability (default: 1e-3)
        cholesky_max_tries: Maximum Cholesky decomposition attempts (default: 10)
        verbose: Print training progress every 10 epochs (default: True)

    Returns:
        List of average losses per epoch

    Example:
        ```python
        model = SparseGPModel(inducing_points=inducing_pts)
        likelihood = gpytorch.likelihoods.StudentTLikelihood()

        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X_train_norm,
            y_train=y_train,
            n_epochs=100,
            batch_size=2048
        )
        ```
    """
    # Put model and likelihood in training mode
    model.train()
    likelihood.train()

    # Set up optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': learning_rate},
        {'params': likelihood.parameters(), 'lr': learning_rate},
    ])

    # Set up loss function
    mll = VariationalELBO(likelihood, model, num_data=len(y_train))

    # Training configuration
    n_batches = int(torch.ceil(torch.tensor(len(X_train) / batch_size)).item())

    if verbose:
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {n_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Batches per epoch: {n_batches}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info("-" * 60)

    # Training loop with numerical stability settings
    losses = []

    with gpytorch.settings.cholesky_jitter(cholesky_jitter), \
         gpytorch.settings.cholesky_max_tries(cholesky_max_tries), \
         gpytorch.settings.cg_tolerance(1e-2):

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # Mini-batch training
            for batch_idx in range(0, len(X_train), batch_size):
                X_batch = X_train[batch_idx:batch_idx + batch_size]
                y_batch = y_train[batch_idx:batch_idx + batch_size]

                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            # Print progress
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                extra_info = ""
                if hasattr(likelihood, 'deg_free'):
                    # Student-t likelihood has degrees of freedom
                    extra_info = f" | ν: {likelihood.deg_free.item():.2f}"

                logger.info(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:10.4f}{extra_info}")

    if verbose:
        logger.info("-" * 60)
        logger.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    return losses


def save_model(
    model: SparseGPModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    save_path: str,
    losses: Optional[List[float]] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save trained GP model to disk.

    Args:
        model: Trained SparseGPModel
        likelihood: Trained likelihood
        save_path: Path to save model checkpoint
        losses: Training losses (optional)
        metadata: Additional metadata to save (optional)

    Example:
        ```python
        save_model(
            model=model,
            likelihood=likelihood,
            save_path="../models/gp_robust_model.pth",
            losses=training_losses,
            metadata={'dataset': 'IOPS', 'epochs': 100}
        )
        ```
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'inducing_points': model.inducing_points.detach().clone(),
        'num_inducing_points': model.num_inducing_points,
        # Save kernel configuration for proper model reconstruction
        'kernel_config': {
            'slow_period': model.covar_module.slow_period,
            'fast_period': model.covar_module.fast_period,
            'rbf_lengthscale': model.covar_module.rbf_lengthscale,
        },
    }

    if losses is not None:
        checkpoint['losses'] = losses

    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Save Student-t degrees of freedom if applicable
    if hasattr(likelihood, 'deg_free'):
        checkpoint['final_nu'] = likelihood.deg_free.item()

    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(
    load_path: str,
    likelihood_class: type,
    device: Optional[torch.device] = None,
    slow_period: Optional[float] = None,
    fast_period: Optional[float] = None,
    rbf_lengthscale: Optional[float] = None,
) -> Tuple[SparseGPModel, gpytorch.likelihoods.Likelihood, Dict]:
    """
    Load trained GP model from disk.

    Args:
        load_path: Path to model checkpoint
        likelihood_class: Likelihood class to instantiate (e.g., GaussianLikelihood)
        device: Device to load model onto (default: None, uses CPU)
        slow_period: Slow periodic component period (for backward compatibility with old checkpoints)
        fast_period: Fast periodic component period (for backward compatibility with old checkpoints)
        rbf_lengthscale: RBF component lengthscale (for backward compatibility with old checkpoints)

    Returns:
        Tuple of (model, likelihood, checkpoint_dict)

    Example:
        ```python
        # New checkpoints (with kernel_config saved)
        model, likelihood, checkpoint = load_model(
            load_path="../models/gp_robust_model.pth",
            likelihood_class=gpytorch.likelihoods.StudentTLikelihood,
            device=torch.device('cpu')
        )

        # Old checkpoints (backward compatibility)
        model, likelihood, checkpoint = load_model(
            load_path="../models/gp_robust_model.pth",
            likelihood_class=gpytorch.likelihoods.StudentTLikelihood,
            device=torch.device('cpu'),
            slow_period=1250/X_range,
            fast_period=250/X_range,
            rbf_lengthscale=0.1
        )

        # Access training history
        losses = checkpoint['losses']
        final_nu = checkpoint.get('final_nu')
        ```
    """
    if device is None:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)

    # Recreate model architecture
    inducing_points = checkpoint['inducing_points'].to(device)

    # Get kernel configuration (use saved config or provided parameters)
    kernel_config = checkpoint.get('kernel_config', {})
    slow_period_val = kernel_config.get('slow_period', slow_period or 1.0)
    fast_period_val = kernel_config.get('fast_period', fast_period or 0.2)
    rbf_lengthscale_val = kernel_config.get('rbf_lengthscale', rbf_lengthscale or 0.1)

    model = SparseGPModel(
        inducing_points=inducing_points,
        slow_period=slow_period_val,
        fast_period=fast_period_val,
        rbf_lengthscale=rbf_lengthscale_val
    ).to(device)

    # Recreate likelihood
    if likelihood_class == gpytorch.likelihoods.StudentTLikelihood:
        likelihood = likelihood_class(
            deg_free_prior=gpytorch.priors.NormalPrior(4.0, 1.0)
        ).to(device)
    else:
        likelihood = likelihood_class().to(device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    logger.info(f"Model loaded from {load_path}")
    # Use .get() for backward compatibility with old checkpoints
    num_inducing = checkpoint.get('num_inducing_points', inducing_points.size(0))
    logger.info(f"  Inducing points: {num_inducing}")
    logger.info(f"  Kernel config: slow={slow_period_val:.6f}, fast={fast_period_val:.6f}, rbf={rbf_lengthscale_val:.6f}")

    if 'losses' in checkpoint:
        logger.info(f"  Training epochs: {len(checkpoint['losses'])}")
        logger.info(f"  Final loss: {checkpoint['losses'][-1]:.4f}")

    if 'final_nu' in checkpoint:
        logger.info(f"  Degrees of freedom (ν): {checkpoint['final_nu']:.2f}")

    return model, likelihood, checkpoint
