import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="Requires torch (install with: uv sync --group gpu)"
)
"""
Tests for Gaussian Process models module.

Validates SparseGPModel behavior including:
- Model initialization with inducing points
- Forward pass computation
- Variational distribution setup
- Inducing point optimization capability
- Integration with composite kernels
"""

import gpytorch

from hellocloud.modeling.gaussian_process.models import SparseGPModel, initialize_inducing_points


class TestSparseGPModel:
    """Test suite for SparseGPModel."""

    @pytest.fixture
    def inducing_points(self):
        """Create mock inducing points for testing."""
        M = 50
        return torch.linspace(0, 1, M).reshape(-1, 1)

    @pytest.fixture
    def small_training_data(self):
        """Create small synthetic training data."""
        n = 200
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        y = torch.sin(2 * torch.pi * X.squeeze()) + 0.1 * torch.randn(n)
        return X, y

    def test_model_initialization(self, inducing_points):
        """Test SVGP model initializes correctly."""
        model = SparseGPModel(inducing_points=inducing_points)

        # Verify components exist
        assert hasattr(model, "mean_module"), "Missing mean_module"
        assert hasattr(model, "covar_module"), "Missing covar_module"
        assert hasattr(model, "variational_strategy"), "Missing variational_strategy"

        # Verify it's an ApproximateGP
        assert isinstance(model, gpytorch.models.ApproximateGP)

    def test_model_num_inducing_points(self, inducing_points):
        """Test model reports correct number of inducing points."""
        M = len(inducing_points)
        model = SparseGPModel(inducing_points=inducing_points)

        assert (
            model.num_inducing_points == M
        ), f"Expected {M} inducing points, got {model.num_inducing_points}"

    def test_model_forward_pass_shape(self, inducing_points):
        """Test model forward pass returns correct distribution."""
        model = SparseGPModel(inducing_points=inducing_points)

        # Create test inputs
        n = 100
        x = torch.randn(n, 1)

        # Forward pass
        output = model(x)

        # Verify output is MultivariateNormal
        assert isinstance(
            output, gpytorch.distributions.MultivariateNormal
        ), f"Expected MultivariateNormal, got {type(output)}"

        # Verify shapes
        assert output.mean.shape == (n,), f"Expected mean shape ({n},), got {output.mean.shape}"

    def test_model_with_different_inducing_sizes(self):
        """Test model works with various inducing point counts."""
        inducing_sizes = [10, 50, 100, 200]

        for M in inducing_sizes:
            inducing_points = torch.linspace(0, 1, M).reshape(-1, 1)
            model = SparseGPModel(inducing_points=inducing_points)

            # Verify model initializes
            assert model.num_inducing_points == M

            # Verify forward pass works
            x = torch.randn(50, 1)
            output = model(x)
            assert output.mean.shape == (50,)

    def test_model_inducing_points_learnable(self, inducing_points):
        """Test inducing points are learnable parameters by default."""
        model = SparseGPModel(inducing_points=inducing_points, learn_inducing_locations=True)

        # Access inducing points
        inducing_pts = model.variational_strategy.inducing_points

        # Verify requires_grad is True
        assert (
            inducing_pts.requires_grad
        ), "Inducing points should be learnable (requires_grad=True)"

    def test_model_inducing_points_fixed(self, inducing_points):
        """Test inducing points can be fixed (not optimized)."""
        model = SparseGPModel(inducing_points=inducing_points, learn_inducing_locations=False)

        # Access inducing points
        inducing_pts = model.variational_strategy.inducing_points

        # Verify requires_grad is False
        assert (
            not inducing_pts.requires_grad
        ), "Inducing points should be fixed (requires_grad=False)"

    def test_model_with_custom_kernel_params(self):
        """Test model accepts custom kernel parameters."""
        inducing_points = torch.linspace(0, 1, 50).reshape(-1, 1)

        model = SparseGPModel(
            inducing_points=inducing_points,
            slow_period=1250 / 10000,
            fast_period=250 / 10000,
            rbf_lengthscale=0.05,
        )

        # Verify kernel exists
        assert hasattr(model.covar_module, "slow_periodic")
        assert hasattr(model.covar_module, "fast_periodic")

    def test_model_gradient_flow(self, inducing_points, small_training_data):
        """Test gradients flow through model parameters."""
        X, y = small_training_data
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Put in training mode
        model.train()
        likelihood.train()

        # Forward pass
        output = model(X)
        loss = -gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y))(output, y)

        # Backward pass
        loss.backward()

        # Check that parameters have gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                has_grads = True
                break

        assert has_grads, "No gradients found in model parameters"

    def test_model_eval_mode(self, inducing_points):
        """Test model switches to eval mode correctly."""
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        # Forward pass in eval mode
        x = torch.randn(50, 1)

        with torch.no_grad():
            output = model(x)
            pred_dist = likelihood(output)

            # Verify we can extract mean and variance
            mean = pred_dist.mean
            variance = pred_dist.variance

            assert mean.shape == (50,)
            assert variance.shape == (50,)
            assert torch.all(variance > 0), "Variance must be positive"

    def test_model_with_student_t_likelihood(self, inducing_points, small_training_data):
        """Test model works with Student-t likelihood."""
        X, y = small_training_data
        model = SparseGPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.StudentTLikelihood()

        model.train()
        likelihood.train()

        # Forward pass
        output = model(X)
        loss = -gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y))(output, y)

        # Verify loss is finite
        assert torch.isfinite(loss), "Loss should be finite"

    def test_model_device_handling(self, inducing_points):
        """Test model can be moved to different devices."""
        model = SparseGPModel(inducing_points=inducing_points)

        # Move to CPU (should already be there)
        model = model.to("cpu")
        x = torch.randn(10, 1)
        output = model(x)
        assert output.mean.device.type == "cpu"

        # Note: Skip GPU test as CI may not have GPU
        # In production, test with: model.to('cuda')


class TestInitializeInducingPoints:
    """Test suite for inducing point initialization utilities."""

    @pytest.fixture
    def training_data(self):
        """Create training data for inducing point tests."""
        n = 1000
        X = torch.linspace(0, 1, n).reshape(-1, 1)
        return X

    def test_evenly_spaced_initialization(self, training_data):
        """Test evenly spaced inducing point initialization."""
        X_train = training_data
        M = 100

        inducing_points = initialize_inducing_points(
            X_train=X_train, num_inducing=M, method="evenly_spaced"
        )

        # Verify shape
        assert inducing_points.shape == (
            M,
            1,
        ), f"Expected shape ({M}, 1), got {inducing_points.shape}"

        # Verify points are approximately evenly spaced
        # (linspace can have rounding, so check std is reasonable)
        diffs = inducing_points[1:] - inducing_points[:-1]
        mean_diff = diffs.mean()
        std_diff = diffs.std()

        # Standard deviation should be small relative to mean (allow up to 5%)
        assert (
            std_diff < mean_diff * 0.05
        ), f"Points should be approximately evenly spaced (std={std_diff:.6f}, mean={mean_diff:.6f})"

        # Verify range coverage
        assert torch.isclose(inducing_points.min(), X_train.min(), atol=1e-6)
        assert torch.isclose(inducing_points.max(), X_train.max(), atol=1e-6)

    def test_random_initialization(self, training_data):
        """Test random subset inducing point initialization."""
        X_train = training_data
        M = 100

        inducing_points = initialize_inducing_points(
            X_train=X_train, num_inducing=M, method="random"
        )

        # Verify shape
        assert inducing_points.shape == (M, 1)

        # Verify all points are from training set
        # (exact match may not work due to floating point, so check range)
        assert inducing_points.min() >= X_train.min()
        assert inducing_points.max() <= X_train.max()

    def test_kmeans_initialization_not_implemented(self, training_data):
        """Test that kmeans initialization raises NotImplementedError."""
        X_train = training_data

        with pytest.raises(NotImplementedError):
            initialize_inducing_points(X_train=X_train, num_inducing=50, method="kmeans")

    def test_invalid_method_raises_error(self, training_data):
        """Test that invalid method raises ValueError."""
        X_train = training_data

        with pytest.raises(ValueError, match="Unknown initialization method"):
            initialize_inducing_points(X_train=X_train, num_inducing=50, method="invalid_method")

    def test_initialization_with_different_sizes(self, training_data):
        """Test initialization works with various inducing point counts."""
        X_train = training_data

        sizes = [10, 50, 100, 200, 500]

        for M in sizes:
            inducing_points = initialize_inducing_points(
                X_train=X_train, num_inducing=M, method="evenly_spaced"
            )

            assert len(inducing_points) == M, f"Expected {M} points, got {len(inducing_points)}"

    def test_initialization_preserves_dtype(self, training_data):
        """Test initialization preserves tensor dtype."""
        # Float32
        X_float32 = training_data.to(torch.float32)
        inducing_float32 = initialize_inducing_points(X_float32, 50)
        assert inducing_float32.dtype == torch.float32

        # Float64
        X_float64 = training_data.to(torch.float64)
        inducing_float64 = initialize_inducing_points(X_float64, 50)
        assert inducing_float64.dtype == torch.float64

    def test_initialization_preserves_device(self):
        """Test initialization preserves tensor device."""
        X_cpu = torch.randn(100, 1)
        inducing_cpu = initialize_inducing_points(X_cpu, 20)
        assert inducing_cpu.device.type == "cpu"

        # Note: Skip GPU test for CI compatibility
        # In production: test with X_cuda = X_cpu.to('cuda')

    def test_initialization_with_multivariate_data(self):
        """Test initialization works with multi-dimensional inputs."""
        n, d = 500, 3
        X_multi = torch.randn(n, d)
        M = 50

        inducing_points = initialize_inducing_points(
            X_train=X_multi, num_inducing=M, method="evenly_spaced"
        )

        # Verify shape
        assert inducing_points.shape == (
            M,
            d,
        ), f"Expected shape ({M}, {d}), got {inducing_points.shape}"
