"""Tests for Spark session management."""

from pyspark.sql import SparkSession


def test_get_spark_session_returns_session():
    """Test that get_spark_session returns SparkSession."""
    from hellocloud.spark.session import get_spark_session

    session = get_spark_session()

    assert isinstance(session, SparkSession)
    # App name is "pytest" when running under pytest due to conftest fixture
    # or "hellocloud" when running standalone
    assert session.sparkContext.appName in ("hellocloud", "pytest")


def test_get_spark_session_local_mode():
    """Test local mode uses all cores."""
    from hellocloud.spark.session import get_spark_session

    session = get_spark_session(local_mode=True)

    assert session.sparkContext.master.startswith("local[")


def test_get_spark_session_is_singleton():
    """Test multiple calls return same session."""
    from hellocloud.spark.session import get_spark_session

    session1 = get_spark_session()
    session2 = get_spark_session()

    assert session1 is session2
