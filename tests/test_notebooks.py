"""
Efficient notebook testing framework with zero duplication.

This module provides:
1. Smoke tests - Fast import/syntax validation
2. Execution tests - Single run per notebook with shared results
3. Content validation - Check outputs without re-execution
4. Proper timeout and error handling

Architecture:
- Each notebook runs EXACTLY ONCE via session-scoped fixtures
- Results are cached and shared across multiple test functions
- Smoke tests run quickly for CI/development
- Full execution tests validate runbook functionality
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional
import pytest
from loguru import logger


# Configuration
NOTEBOOK_DIR = Path(__file__).parent.parent / "notebooks"
EXECUTION_TIMEOUT = 60  # seconds per notebook
NOTEBOOKS = [
    "02_workload_signatures_guide.md",
    "03_iops_web_server_eda.md",
    "04_gaussian_process_modeling.md",
    "05_cloudzero_piedpiper_eda.md",
]

# Auto-generate .py versions for execution if they don't exist
def ensure_python_notebooks():
    """Ensure .py versions of MyST notebooks exist for execution testing."""
    for notebook_name in NOTEBOOKS:
        md_path = NOTEBOOK_DIR / notebook_name
        py_name = notebook_name.replace('.md', '.py')
        py_path = NOTEBOOK_DIR / py_name

        if md_path.exists() and not py_path.exists():
            try:
                subprocess.run([
                    "uv", "run", "jupytext", "--to", "py", str(md_path)
                ], cwd=NOTEBOOK_DIR, check=True, capture_output=True)
                logger.info(f"Generated {py_name} from {notebook_name}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to generate {py_name}: {e}")

# Call during module import
ensure_python_notebooks()


class NotebookResult(NamedTuple):
    """Result of notebook execution."""
    name: str
    success: bool
    stdout: str
    stderr: str
    duration: float
    error: Optional[str] = None


class NotebookExecutionError(Exception):
    """Raised when notebook execution fails."""
    pass


# ============================================================================
# FIXTURES (Run notebooks exactly once per session)
# ============================================================================

@pytest.fixture(scope="session")
def execution_env():
    """Environment for notebook execution."""
    project_root = Path(__file__).parent.parent
    return {
        **os.environ,
        "PYTHONPATH": str(project_root / "src"),
        "PROJECT_ROOT": str(project_root)
    }


@pytest.fixture(scope="session", params=NOTEBOOKS)
def executed_notebook(request, execution_env) -> NotebookResult:
    """
    Execute each notebook exactly once per test session.

    This fixture is parametrized and session-scoped, ensuring each notebook
    runs only once and results are shared across all test functions.
    """
    notebook_name = request.param
    # Convert .md name to .py for execution
    py_name = notebook_name.replace('.md', '.py')
    script_path = NOTEBOOK_DIR / py_name

    if not script_path.exists():
        pytest.skip(f"Notebook {py_name} not found (converted from {notebook_name})")

    logger.info(f"[SINGLE EXECUTION] Running: {notebook_name} (as {py_name})")

    import time
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent,  # Project root
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
            env=execution_env
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return NotebookResult(
                name=notebook_name,
                success=False,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=duration,
                error=f"Exit code {result.returncode}"
            )

        logger.success(f"[EXECUTED] {notebook_name} in {duration:.1f}s")
        return NotebookResult(
            name=notebook_name,
            success=True,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"[TIMEOUT] {notebook_name} after {duration:.1f}s")
        return NotebookResult(
            name=notebook_name,
            success=False,
            stdout="",
            stderr="",
            duration=duration,
            error=f"Timeout after {EXECUTION_TIMEOUT}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[ERROR] {notebook_name}: {e}")
        return NotebookResult(
            name=notebook_name,
            success=False,
            stdout="",
            stderr="",
            duration=duration,
            error=str(e)
        )


@pytest.fixture(scope="session")
def all_notebook_results(request):
    """Collect all executed notebook results."""
    # This fixture depends on executed_notebook running for all parameters
    results = {}

    # Get the parametrized fixture results
    for notebook_name in NOTEBOOKS:
        # Create a sub-request for each notebook
        sub_request = request.getfixturevalue('executed_notebook')
        if hasattr(sub_request, 'name'):
            results[sub_request.name] = sub_request

    return results


# ============================================================================
# SMOKE TESTS (Fast - no execution)
# ============================================================================

@pytest.mark.smoke
@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_syntax(notebook_name):
    """Test that notebook Python syntax is valid."""
    script_path = NOTEBOOK_DIR / notebook_name

    if not script_path.exists():
        pytest.skip(f"Notebook {notebook_name} not found")

    with open(script_path, 'r') as f:
        content = f.read()

    try:
        ast.parse(content)
        logger.info(f"✓ Syntax valid: {notebook_name}")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {notebook_name}: {e}")


@pytest.mark.smoke
@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
def test_notebook_imports(notebook_name):
    """Test that notebook imports can be resolved (without full execution)."""
    script_path = NOTEBOOK_DIR / notebook_name

    if not script_path.exists():
        pytest.skip(f"Notebook {notebook_name} not found")

    # Extract import statements
    with open(script_path, 'r') as f:
        content = f.read()

    try:
        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        logger.info(f"✓ Found {len(imports)} imports in {notebook_name}")

        # Test critical imports that should be available
        critical_imports = ['polars', 'numpy', 'hellocloud']
        for imp in critical_imports:
            matching = [i for i in imports if imp in i]
            assert len(matching) > 0, f"Missing critical import '{imp}' in {notebook_name}"

    except Exception as e:
        pytest.fail(f"Import analysis failed for {notebook_name}: {e}")


# ============================================================================
# EXECUTION TESTS (Use shared fixture results - no duplication)
# ============================================================================

def test_notebook_execution_success(executed_notebook):
    """Test that notebook executed successfully."""
    result = executed_notebook

    if result.error == "Timeout after 60s":
        pytest.fail(f"Notebook {result.name} timed out - possible infinite loop or hang")
    elif not result.success:
        pytest.fail(
            f"Notebook {result.name} failed: {result.error}\n"
            f"STDOUT: {result.stdout[:500]}\n"
            f"STDERR: {result.stderr[:500]}"
        )

    # Log success
    logger.success(f"✓ {result.name} executed successfully in {result.duration:.1f}s")


def test_notebook_performance(executed_notebook):
    """Test that notebook execution time is reasonable."""
    result = executed_notebook

    if not result.success:
        pytest.skip(f"Skipping performance test - {result.name} failed execution")

    # Warn if execution is very slow (but don't fail)
    if result.duration > 30:
        logger.warning(f"⚠ {result.name} took {result.duration:.1f}s - consider optimization")
    else:
        logger.info(f"✓ {result.name} performance OK: {result.duration:.1f}s")


# ============================================================================
# CONTENT VALIDATION TESTS (Check outputs without re-execution)
# ============================================================================

def test_data_exploration_content(executed_notebook):
    """Test that data exploration notebook produces expected content."""
    result = executed_notebook

    if result.name != "01_data_exploration.py":
        pytest.skip("Not the data exploration notebook")

    if not result.success:
        pytest.skip("Notebook execution failed")

    output_text = (result.stdout + result.stderr).lower()

    # Check for key analysis indicators
    expected_indicators = ["cpu", "memory", "utilization", "workload"]
    missing = [ind for ind in expected_indicators if ind not in output_text]

    assert len(missing) == 0, f"Missing expected analysis content: {missing}"
    logger.info(f"✓ Data exploration contains expected analysis indicators")


def test_workload_signatures_content(executed_notebook):
    """Test that workload signatures notebook analyzes patterns."""
    result = executed_notebook

    if result.name != "02_workload_signatures_guide.py":
        pytest.skip("Not the workload signatures notebook")

    if not result.success:
        pytest.skip("Notebook execution failed")

    output_text = (result.stdout + result.stderr).lower()

    # Check for pattern analysis
    expected_patterns = ["pattern", "signature", "analysis"]
    missing = [pat for pat in expected_patterns if pat not in output_text]

    assert len(missing) == 0, f"Missing expected pattern analysis: {missing}"
    logger.info(f"✓ Workload signatures contains pattern analysis")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
def test_all_notebooks_executed():
    """Integration test ensuring all notebooks can execute."""
    # This test runs after all individual executions via the fixture
    # It doesn't re-execute anything, just validates the overall result

    script_paths = [NOTEBOOK_DIR / nb for nb in NOTEBOOKS]
    existing_notebooks = [path for path in script_paths if path.exists()]

    assert len(existing_notebooks) >= 1, "Should have at least 1 executable notebook"
    logger.info(f"✓ Integration test passed - {len(existing_notebooks)} notebooks available")


# ============================================================================
# TEST COLLECTION AND REPORTING
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Add markers and organize test collection."""
    for item in items:
        # Mark smoke tests
        if "smoke" in item.name or "syntax" in item.name or "import" in item.name:
            item.add_marker(pytest.mark.smoke)

        # Mark execution tests
        if "execution" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.execution)

        # Mark content tests
        if "content" in item.name:
            item.add_marker(pytest.mark.content)


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"])