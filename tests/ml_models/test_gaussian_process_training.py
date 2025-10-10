"""
Integration tests for Gaussian Process training module.

Validates training workflows including:
- Training loop convergence
- Model persistence (save/load)
- Integration with different likelihoods
- Numerical stability settings
"""

import pytest
import torch
import gpytorch
import tempfile
from pathlib import Path

from cloudlens.ml_models.gaussian_process.models import SparseGPModel, initialize_inducing_points
from cloudlens.ml_models.gaussian_process.training import (
    train_gp_model,
    save_model,
    load_model,
)


class TestGPTraining:
    """Integration tests for GP model training."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate small synthetic dataset for fast training tests."""
        n = 500
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        # Simple sine wave with noise
        y = torch.sin(2 * torch.pi * 4 * X.squeeze()) + 0.1 * torch.randn(n)
        return X, y

    @pytest.fixture
    def small_model_setup(self, synthetic_data):
        """Create small model for quick testing."""
        X, y = synthetic_data
        inducing_points = initialize_inducing_points(X, num_inducing=50)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        return model, likelihood, X, y

    def test_train_gp_model_runs_without_error(self, small_model_setup):
        """Test that training completes without errors."""
        model, likelihood, X, y = small_model_setup

        # Train for just a few epochs
        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=5,
            batch_size=128,
            verbose=False
        )

        # Check losses returned
        assert len(losses) == 5
        assert all(isinstance(loss, float) for loss in losses)

    def test_training_loss_decreases(self, small_model_setup):
        """Test that training loss generally decreases."""
        model, likelihood, X, y = small_model_setup

        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=20,
            batch_size=128,
            verbose=False
        )

        # Loss should decrease (allow some fluctuation)
        # Compare first 5 epochs to last 5 epochs
        early_loss = sum(losses[:5]) / 5
        late_loss = sum(losses[-5:]) / 5

        assert late_loss < early_loss, \
            f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"

    def test_train_with_student_t_likelihood(self, synthetic_data):
        """Test training with Student-t likelihood."""
        X, y = synthetic_data

        # Add some outliers
        y = y.clone()
        y[100:110] += 5.0  # Inject outliers

        inducing_points = initialize_inducing_points(X, num_inducing=50)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.StudentTLikelihood()

        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=10,
            batch_size=128,
            verbose=False
        )

        # Should complete without errors
        assert len(losses) == 10

        # Degrees of freedom should be learned
        nu = likelihood.deg_free.item()
        assert nu > 0, "Degrees of freedom should be positive"

    def test_batched_training_handles_full_dataset(self, small_model_setup):
        """Test that batched training processes all data."""
        model, likelihood, X, y = small_model_setup

        # Use small batch size to test batching
        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=3,
            batch_size=64,  # Small batch size
            verbose=False
        )

        # Should complete without errors
        assert len(losses) == 3

    @pytest.mark.slow
    def test_training_convergence_on_periodic_data(self):
        """Test model converges on periodic data (slow test)."""
        # Generate longer periodic dataset
        n = 2000
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        y = (
            torch.sin(2 * torch.pi * 5 * X.squeeze()) +  # Fast period
            torch.sin(2 * torch.pi * 1 * X.squeeze()) +  # Slow period
            0.1 * torch.randn(n)
        )

        # Model with composite periodic kernel
        inducing_points = initialize_inducing_points(X, num_inducing=100)
        model = SparseGPModel(
            inducing_points=inducing_points,
            slow_period=1.0,
            fast_period=0.2
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=30,
            batch_size=256,
            verbose=False
        )

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.8, \
            "Loss should decrease by at least 20%"


class TestModelPersistence:
    """Tests for model save/load functionality."""

    @pytest.fixture
    def trained_model_setup(self):
        """Create and train a small model."""
        n = 200
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        y = torch.sin(2 * torch.pi * X.squeeze()) + 0.05 * torch.randn(n)

        inducing_points = initialize_inducing_points(X, num_inducing=30)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Train briefly
        losses = train_gp_model(
            model, likelihood, X, y,
            n_epochs=5,
            batch_size=64,
            verbose=False
        )

        return model, likelihood, losses, X, y

    def test_save_model_creates_file(self, trained_model_setup):
        """Test that save_model creates a checkpoint file."""
        model, likelihood, losses, _, _ = trained_model_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pth"

            save_model(
                model=model,
                likelihood=likelihood,
                save_path=str(save_path),
                losses=losses
            )

            # File should exist
            assert save_path.exists()

    def test_save_and_load_model_roundtrip(self, trained_model_setup):
        """Test that saved model can be loaded and produces same predictions."""
        model_original, likelihood_original, losses, X, y = trained_model_setup

        # Generate predictions from original model
        model_original.eval()
        likelihood_original.eval()

        with torch.no_grad():
            pred_original = likelihood_original(model_original(X[:50]))
            mean_original = pred_original.mean.numpy()

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pth"

            save_model(
                model=model_original,
                likelihood=likelihood_original,
                save_path=str(save_path),
                losses=losses,
                metadata={'test': 'data'}
            )

            # Load model
            model_loaded, likelihood_loaded, checkpoint = load_model(
                load_path=str(save_path),
                likelihood_class=gpytorch.likelihoods.GaussianLikelihood,
                device=torch.device('cpu')
            )

            # Check metadata
            assert checkpoint['metadata']['test'] == 'data'
            assert checkpoint['losses'] == losses

            # Generate predictions from loaded model
            model_loaded.eval()
            likelihood_loaded.eval()

            with torch.no_grad():
                pred_loaded = likelihood_loaded(model_loaded(X[:50]))
                mean_loaded = pred_loaded.mean.numpy()

            # Predictions should match
            import numpy as np
            np.testing.assert_array_almost_equal(
                mean_original,
                mean_loaded,
                decimal=5,
                err_msg="Loaded model predictions don't match original"
            )

    def test_load_student_t_model(self):
        """Test loading model with Student-t likelihood."""
        # Create and save Student-t model
        n = 100
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        y = torch.randn(n)

        inducing_points = initialize_inducing_points(X, num_inducing=20)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.StudentTLikelihood(
            deg_free_prior=gpytorch.priors.NormalPrior(4.0, 1.0)
        )

        # Train briefly
        train_gp_model(
            model, likelihood, X, y,
            n_epochs=3,
            verbose=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "student_t_model.pth"

            save_model(model, likelihood, str(save_path))

            # Load with correct likelihood class
            model_loaded, likelihood_loaded, checkpoint = load_model(
                load_path=str(save_path),
                likelihood_class=gpytorch.likelihoods.StudentTLikelihood
            )

            # Check degrees of freedom saved
            assert 'final_nu' in checkpoint
            assert checkpoint['final_nu'] > 0

            # Check likelihood is Student-t
            assert isinstance(likelihood_loaded, gpytorch.likelihoods.StudentTLikelihood)

    def test_save_creates_directory_if_needed(self):
        """Test that save_model creates parent directories."""
        n = 50
        X = torch.randn(n, 1)
        y = torch.randn(n)

        inducing_points = initialize_inducing_points(X, num_inducing=10)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "model.pth"

            # Directory doesn't exist yet
            assert not save_path.parent.exists()

            save_model(model, likelihood, str(save_path))

            # Should create directories and save file
            assert save_path.exists()


class TestNumericalStability:
    """Tests for numerical stability during training."""

    def test_training_with_cholesky_jitter(self):
        """Test that Cholesky jitter prevents numerical issues."""
        # Create challenging dataset that might cause numerical issues
        n = 300
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        y = torch.randn(n) * 10.0  # Large variance

        inducing_points = initialize_inducing_points(X, num_inducing=50)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Train with explicit jitter settings
        losses = train_gp_model(
            model=model,
            likelihood=likelihood,
            X_train=X,
            y_train=y,
            n_epochs=10,
            batch_size=128,
            cholesky_jitter=1e-3,
            cholesky_max_tries=10,
            verbose=False
        )

        # Should complete without NotPSDError
        assert len(losses) == 10
        assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)

    def test_training_loss_is_finite(self, synthetic_data):
        """Test that training losses remain finite (no NaN/Inf)."""
        X, y = synthetic_data

        inducing_points = initialize_inducing_points(X, num_inducing=40)
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        losses = train_gp_model(
            model, likelihood, X, y,
            n_epochs=10,
            verbose=False
        )

        # All losses should be finite
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), \
                f"Loss at epoch {i} is not finite: {loss}"


@pytest.fixture
def synthetic_data():
    """Shared fixture for synthetic data."""
    n = 500
    X = torch.linspace(0, 1, n).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * 4 * X.squeeze()) + 0.1 * torch.randn(n)
    return X, y
