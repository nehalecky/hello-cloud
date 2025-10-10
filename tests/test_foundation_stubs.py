"""Tests for foundation model stubs.

These tests verify that the foundation model stubs are properly configured
and raise appropriate NotImplementedError messages when instantiated.
"""

import pytest

from hellocloud.ml_models.foundation import (
    ChronosForecaster,
    FoundationModelBase,
    TimesFMForecaster,
)


class TestFoundationModelBase:
    """Test the abstract base class for foundation models."""

    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        import inspect

        abstract_methods = {
            name
            for name, method in inspect.getmembers(FoundationModelBase)
            if getattr(method, "__isabstractmethod__", False)
        }

        required_methods = {"predict", "load_model", "get_model_info"}
        assert abstract_methods == required_methods

    def test_cannot_instantiate_directly(self):
        """Test that FoundationModelBase cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FoundationModelBase()


class TestTimesFMStub:
    """Test the TimesFM stub implementation."""

    def test_init_raises_notimplementederror(self):
        """Test that __init__ raises NotImplementedError with helpful message."""
        with pytest.raises(NotImplementedError) as exc_info:
            TimesFMForecaster()

        error_msg = str(exc_info.value).lower()
        assert "timesfm" in error_msg
        assert "not yet implemented" in error_msg
        assert "github.com" in error_msg

    def test_init_with_custom_params_raises_notimplementederror(self):
        """Test that __init__ with custom parameters also raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            TimesFMForecaster(model_name="custom-model", context_len=256, horizon_len=64)

    def test_has_comprehensive_docstring(self):
        """Test that TimesFMForecaster has a comprehensive docstring."""
        docstring = TimesFMForecaster.__doc__ or ""
        assert len(docstring) > 500
        assert "NOT YET IMPLEMENTED" in docstring
        assert "github.com" in docstring
        assert "arxiv.org" in docstring

    def test_class_defined_with_inheritance(self):
        """Test that TimesFMForecaster inherits from FoundationModelBase."""
        assert issubclass(TimesFMForecaster, FoundationModelBase)


class TestChronosStub:
    """Test the Chronos stub implementation."""

    def test_init_raises_notimplementederror(self):
        """Test that __init__ raises NotImplementedError with helpful message."""
        with pytest.raises(NotImplementedError) as exc_info:
            ChronosForecaster()

        error_msg = str(exc_info.value).lower()
        assert "chronos" in error_msg
        assert "not yet implemented" in error_msg
        assert "github.com" in error_msg

    def test_init_with_different_model_sizes_raises_notimplementederror(self):
        """Test that __init__ with different model sizes all raise NotImplementedError."""
        model_sizes = ["tiny", "mini", "small", "base", "large"]
        for size in model_sizes:
            with pytest.raises(NotImplementedError):
                ChronosForecaster(model_size=size)

    def test_has_comprehensive_docstring(self):
        """Test that ChronosForecaster has a comprehensive docstring."""
        docstring = ChronosForecaster.__doc__ or ""
        assert len(docstring) > 500
        assert "NOT YET IMPLEMENTED" in docstring
        assert "github.com" in docstring
        assert "arxiv.org" in docstring

    def test_class_defined_with_inheritance(self):
        """Test that ChronosForecaster inherits from FoundationModelBase."""
        assert issubclass(ChronosForecaster, FoundationModelBase)

    def test_valid_model_sizes_constant(self):
        """Test that VALID_MODEL_SIZES constant is properly defined."""
        expected_sizes = ["tiny", "mini", "small", "base", "large"]
        assert ChronosForecaster.VALID_MODEL_SIZES == expected_sizes


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports_available(self):
        """Test that all expected classes are exported from the module."""
        from hellocloud.ml_models import foundation

        expected_exports = [
            "FoundationModelBase",
            "TimesFMForecaster",
            "ChronosForecaster",
        ]

        for export in expected_exports:
            assert hasattr(foundation, export)

    def test_module_has_docstring(self):
        """Test that the module has a comprehensive docstring."""
        from hellocloud.ml_models import foundation

        assert foundation.__doc__ is not None
        assert len(foundation.__doc__) > 100
        assert "STUB" in foundation.__doc__ or "stub" in foundation.__doc__
