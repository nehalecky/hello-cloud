"""PySpark backend for cloud analytics."""

from .session import get_spark_session

__all__ = ["get_spark_session"]
