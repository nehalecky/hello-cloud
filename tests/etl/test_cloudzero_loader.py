"""Tests for CloudZero ETL stub."""

import pytest
from cloud_sim.etl import CloudZeroDataLoader, load_cloudzero_data


class TestCloudZeroDataLoaderStub:
    """Test CloudZero loader stub behavior."""

    def test_loader_raises_not_implemented(self):
        """Test that loader initialization raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="CloudZero ETL integration"):
            CloudZeroDataLoader(data_path="test.csv")

    def test_load_data_function_raises_not_implemented(self):
        """Test that convenience function raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="CloudZero data loading"):
            load_cloudzero_data("test.csv")

    def test_module_imports_work(self):
        """Test that module exports work correctly."""
        from cloud_sim.etl import CloudZeroDataLoader, load_cloudzero_data

        assert CloudZeroDataLoader is not None
        assert load_cloudzero_data is not None

    def test_class_has_docstring(self):
        """Test that stub has comprehensive documentation."""
        assert CloudZeroDataLoader.__doc__ is not None
        assert "NOT YET IMPLEMENTED" in CloudZeroDataLoader.__doc__
        assert "CloudZero" in CloudZeroDataLoader.__doc__
