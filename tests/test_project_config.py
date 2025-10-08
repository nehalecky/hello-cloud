"""
Test project configuration and setup validity.
These tests would have caught the TOML parsing errors.
"""

import pytest
import toml
import subprocess
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectConfiguration:
    """Test that project configuration files are valid and consistent."""

    def test_pyproject_toml_valid(self):
        """Test that pyproject.toml is valid TOML and parseable."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"

        # This test would have caught our TOML parsing errors
        try:
            with open(pyproject_path, "r") as f:
                config = toml.load(f)
        except toml.TomlDecodeError as e:
            pytest.fail(f"pyproject.toml is not valid TOML: {e}")

        # Verify expected sections exist
        assert "project" in config, "Missing [project] section"
        assert "dependencies" in config["project"], "Missing project.dependencies"

        # Verify dependencies is a list, not dict (would have caught our error)
        assert isinstance(config["project"]["dependencies"], list), \
            "project.dependencies must be a list, not a dict"

        # Verify no deprecated fields
        if "tool" in config and "uv" in config["tool"]:
            if "dev-dependencies" in config["tool"]["uv"]:
                pytest.fail("Deprecated tool.uv.dev-dependencies found. Use dependency-groups.dev instead")

    def test_dependencies_format(self):
        """Test that dependencies are in correct PEP 621 format."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        deps = config["project"]["dependencies"]

        for dep in deps:
            assert isinstance(dep, str), f"Dependency {dep} must be a string"
            # Basic check for valid dependency format
            assert ">" in dep or "=" in dep or "[" in dep, \
                f"Dependency {dep} doesn't look like a valid requirement specifier"

    def test_optional_dependencies_format(self):
        """Test that optional dependencies are properly formatted."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        if "optional-dependencies" in config["project"]:
            opt_deps = config["project"]["optional-dependencies"]
            assert isinstance(opt_deps, dict), "optional-dependencies must be a dict"

            for group_name, deps in opt_deps.items():
                assert isinstance(deps, list), f"{group_name} dependencies must be a list"
                for dep in deps:
                    assert isinstance(dep, str), f"Dependency {dep} in {group_name} must be a string"

    def test_uv_can_parse_pyproject(self):
        """Test that uv can successfully parse our pyproject.toml."""
        # This simulates what would happen in CI
        result = subprocess.run(
            ["uv", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            # Check if it's a TOML parsing error
            if "TOML parse error" in result.stderr:
                pytest.fail(f"uv cannot parse pyproject.toml: {result.stderr}")
            # Otherwise it might be okay (e.g., venv not activated)
            pytest.skip("uv command failed but not due to TOML parsing")

    def test_no_duplicate_keys(self):
        """Test that there are no duplicate keys in TOML sections."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"

        # Read the file as text to check for obvious duplicates
        with open(pyproject_path, "r") as f:
            content = f.read()

        # Check for duplicate section headers
        sections = []
        for line in content.split("\n"):
            if line.strip().startswith("[") and line.strip().endswith("]"):
                section = line.strip()
                if section in sections:
                    pytest.fail(f"Duplicate section found: {section}")
                sections.append(section)

    def test_dependency_groups_format(self):
        """Test that dependency-groups follow PEP 735 format."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        if "dependency-groups" in config:
            groups = config["dependency-groups"]
            assert isinstance(groups, dict), "dependency-groups must be a dict"

            for group_name, deps in groups.items():
                assert isinstance(deps, list), f"dependency-groups.{group_name} must be a list"
                for dep in deps:
                    assert isinstance(dep, str), f"Dependency {dep} in {group_name} must be a string"

    def test_python_version_consistency(self):
        """Test that Python version is consistent across configuration."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        # Check requires-python
        requires_python = config["project"].get("requires-python", "")
        assert requires_python, "requires-python not specified"

        # Check that it's compatible with our current Python
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert ">=" in requires_python, "requires-python should use >= specifier"

        min_version = requires_python.replace(">=", "").strip()
        assert min_version <= current_python, \
            f"Current Python {current_python} doesn't meet requirement {requires_python}"

    def test_package_name_valid(self):
        """Test that package name follows Python naming conventions."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        name = config["project"]["name"]

        # Check valid Python package name
        assert name.replace("-", "_").replace("_", "").isalnum(), \
            f"Package name '{name}' contains invalid characters"

        # Check it doesn't start with a number
        assert not name[0].isdigit(), f"Package name '{name}' cannot start with a number"


class TestProjectImports:
    """Test that the package can be imported without errors."""

    def test_main_package_importable(self):
        """Test that the main package can be imported."""
        try:
            import cloud_sim
        except ImportError as e:
            pytest.fail(f"Cannot import cloud_sim package: {e}")

    def test_submodules_importable(self):
        """Test that submodules can be imported."""
        submodules = [
            "cloud_sim.data_generation",
            "cloud_sim.ml_models",
        ]

        for module_name in submodules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Cannot import {module_name}: {e}")

    def test_pydantic_models_valid(self):
        """Test that our Pydantic models are valid."""
        from cloud_sim.data_generation.workload_patterns import WorkloadCharacteristics

        # This would have caught any Pydantic validation issues
        try:
            # Create a valid model instance
            wc = WorkloadCharacteristics(
                base_cpu_util=15.0,
                base_mem_util=35.0,
                cpu_variance=20.0,
                mem_variance=10.0,
                peak_multiplier=2.0,
                idle_probability=0.3,
                waste_factor=0.35,
                scaling_pattern="auto",
                seasonal_pattern=True,
                burst_probability=0.15
            )
        except Exception as e:
            pytest.fail(f"Cannot create WorkloadCharacteristics: {e}")

        # Test validation works
        with pytest.raises(ValueError):
            # This should fail - CPU util > 100
            WorkloadCharacteristics(
                base_cpu_util=150.0,  # Invalid!
                base_mem_util=35.0,
                cpu_variance=20.0,
                mem_variance=10.0,
                peak_multiplier=2.0,
                idle_probability=0.3,
                waste_factor=0.35,
                scaling_pattern="auto",
                seasonal_pattern=True,
                burst_probability=0.15
            )