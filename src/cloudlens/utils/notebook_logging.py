"""
Logging configuration utilities for Jupyter notebooks.

Provides helpers to configure loguru for clean, readable output in notebooks
while maintaining structured logging for library operations.
"""

import sys
from loguru import logger


def configure_notebook_logging(level: str = "INFO", show_time: bool = False):
    """
    Configure loguru for clean notebook output.

    This removes the default logger and adds a notebook-friendly format
    that's less cluttered than the default timestamps/levels/locations.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
        show_time: Whether to show timestamps. Default: False (cleaner for notebooks)

    Example:
        ```python
        from cloudlens.utils import configure_notebook_logging

        # At the top of your notebook
        configure_notebook_logging(level="INFO")

        # Now library logging will show up cleanly:
        # ✓ Model loaded from ../models/gp_robust_model.pth
        #   Inducing points: 200
        #   Kernel config: slow=0.008606, fast=0.001721, rbf=0.100000
        ```
    """
    # Remove default logger
    logger.remove()

    # Add notebook-friendly format
    if show_time:
        format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    else:
        # Minimal format - just the message with level indicator
        format_str = "<level>{level: <8}</level> | {message}"

    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=True,
    )


def quiet_library_logging():
    """
    Reduce library logging to warnings and errors only.

    Use this when you want minimal logging output in notebooks,
    showing only important issues.

    Example:
        ```python
        from cloudlens.utils import quiet_library_logging

        # Suppress INFO/DEBUG messages
        quiet_library_logging()
        ```
    """
    configure_notebook_logging(level="WARNING", show_time=False)


def verbose_library_logging():
    """
    Enable detailed logging for debugging.

    Shows DEBUG-level messages with timestamps - useful when
    troubleshooting model training or data processing issues.

    Example:
        ```python
        from cloudlens.utils import verbose_library_logging

        # See all debug messages
        verbose_library_logging()
        ```
    """
    configure_notebook_logging(level="DEBUG", show_time=True)
