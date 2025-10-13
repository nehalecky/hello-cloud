"""Core TimeSeries class for hierarchical time series analysis."""


from pyspark.sql import DataFrame


class TimeSeries:
    """
    Wrapper around PySpark DataFrame for hierarchical time series analysis.

    Attributes:
        df: PySpark DataFrame containing time series data
        hierarchy: Ordered list of key columns (coarsest to finest grain)
        metric_col: Name of the metric/value column
        time_col: Name of the timestamp column
    """

    def __init__(self, df: DataFrame, hierarchy: list[str], metric_col: str, time_col: str):
        """
        Initialize TimeSeries wrapper.

        Args:
            df: PySpark DataFrame with time series data
            hierarchy: Ordered key columns (e.g., ["provider", "account", "region"])
            metric_col: Name of metric column (e.g., "cost")
            time_col: Name of timestamp column (e.g., "date")
        """
        self.df = df
        self.hierarchy = hierarchy
        self.metric_col = metric_col
        self.time_col = time_col
        self._cached_stats = {}
