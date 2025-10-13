"""Spark session management for local and production."""

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "hellocloud", local_mode: bool = True) -> SparkSession:
    """
    Create or get Spark session.

    Local mode defaults:
    - local[*] master (all cores)
    - 4GB driver memory
    - 8 shuffle partitions (not 200)

    Args:
        app_name: Application name
        local_mode: Configure for local development

    Returns:
        SparkSession (singleton)

    Example:
        >>> from hellocloud.spark.session import get_spark_session
        >>> spark = get_spark_session()
        >>> df = spark.read.parquet("data.parquet")
    """
    builder = SparkSession.builder.appName(app_name)

    if local_mode:
        builder = builder.master("local[*]")
        builder = builder.config("spark.driver.memory", "4g")
        builder = builder.config("spark.sql.shuffle.partitions", "8")

    return builder.getOrCreate()
