"""Core TimeSeries class for hierarchical time series analysis."""

import matplotlib.pyplot as plt
from loguru import logger
from pyspark.sql import DataFrame

from hellocloud.analysis import eda


class TimeSeriesError(Exception):
    """Base exception for TimeSeries operations."""

    pass


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

        Raises:
            TimeSeriesError: If required columns missing from DataFrame
        """
        self.df = df
        self.hierarchy = hierarchy
        self.metric_col = metric_col
        self.time_col = time_col
        self._cached_stats = {}

        # Validate columns exist
        self._validate_columns()

        # Warn if empty
        if df.count() == 0:
            logger.warning(
                "Creating TimeSeries from empty DataFrame. Operations will return empty results."
            )

    @classmethod
    def from_dataframe(
        cls, df: DataFrame, hierarchy: list[str], metric_col: str = "cost", time_col: str = "date"
    ) -> "TimeSeries":
        """
        Factory method to create TimeSeries from DataFrame.

        Args:
            df: PySpark DataFrame with time series data
            hierarchy: Ordered key columns (e.g., ["provider", "account"])
            metric_col: Name of metric column (default: "cost")
            time_col: Name of timestamp column (default: "date")

        Returns:
            TimeSeries instance
        """
        return cls(df=df, hierarchy=hierarchy, metric_col=metric_col, time_col=time_col)

    def _validate_columns(self) -> None:
        """Validate that required columns exist in DataFrame."""
        df_cols = set(self.df.columns)

        # Check time column
        if self.time_col not in df_cols:
            raise TimeSeriesError(
                f"time_col '{self.time_col}' not found in DataFrame columns: {list(df_cols)}"
            )

        # Check metric column
        if self.metric_col not in df_cols:
            raise TimeSeriesError(
                f"metric_col '{self.metric_col}' not found in DataFrame columns: {list(df_cols)}"
            )

        # Check hierarchy columns
        for col in self.hierarchy:
            if col not in df_cols:
                raise TimeSeriesError(
                    f"hierarchy column '{col}' not found in DataFrame columns: {list(df_cols)}"
                )

    def _resolve_grain(self, grain: list[str]) -> list[str]:
        """
        Validate grain is subset of hierarchy and return in hierarchy order.

        Args:
            grain: List of column names defining the grain

        Returns:
            Grain columns in hierarchy order

        Raises:
            TimeSeriesError: If grain contains columns not in hierarchy
        """
        grain_set = set(grain)
        hierarchy_set = set(self.hierarchy)

        # Check for invalid columns
        invalid = grain_set - hierarchy_set
        if invalid:
            raise TimeSeriesError(
                f"Invalid grain columns: {invalid}. "
                f"Must be subset of hierarchy: {self.hierarchy}"
            )

        # Return in hierarchy order
        return [col for col in self.hierarchy if col in grain_set]

    def filter(self, **entity_keys) -> "TimeSeries":
        """
        Filter to specific entity by hierarchy column values.

        Args:
            **entity_keys: Column name/value pairs to filter on
                          (must be columns in hierarchy)

        Returns:
            New TimeSeries with filtered DataFrame

        Raises:
            TimeSeriesError: If filter column not in hierarchy

        Example:
            ts.filter(provider="AWS", account="acc1")
        """
        from pyspark.sql import functions as F

        # Validate all filter columns are in hierarchy
        invalid = set(entity_keys.keys()) - set(self.hierarchy)
        if invalid:
            raise TimeSeriesError(
                f"Invalid filter column(s): {invalid}. "
                f"Must be columns in hierarchy: {self.hierarchy}"
            )

        # Apply filters
        filtered_df = self.df
        for col, value in entity_keys.items():
            filtered_df = filtered_df.filter(F.col(col) == value)

        # Return new TimeSeries with filtered data
        return TimeSeries(
            df=filtered_df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def sample(self, grain: list[str], n: int = 1) -> "TimeSeries":
        """
        Sample n random entities at specified grain level.

        Args:
            grain: Column names defining the grain (must be subset of hierarchy)
            n: Number of entities to sample (default: 1)

        Returns:
            New TimeSeries with sampled entities

        Example:
            ts.sample(grain=["account", "region"], n=10)
        """
        from pyspark.sql import functions as F

        # Validate and resolve grain
        grain_cols = self._resolve_grain(grain)

        # Get unique entities at grain
        entities_df = self.df.select(*grain_cols).distinct()
        total_entities = entities_df.count()

        # Warn if requesting more than available
        if n > total_entities:
            logger.warning(
                f"Requested {n} entities but only {total_entities} exist at grain {grain}. "
                f"Returning all {total_entities}."
            )
            n = total_entities

        # Sample entities randomly
        sampled_entities = entities_df.orderBy(F.rand()).limit(n)

        # Join back to get full time series for sampled entities
        sampled_df = self.df.join(sampled_entities, on=grain_cols, how="inner")

        return TimeSeries(
            df=sampled_df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def aggregate(self, grain: list[str]) -> "TimeSeries":
        """
        Aggregate metric to coarser grain level.

        Sums metric values across entities, grouping by grain + time.

        Args:
            grain: Column names defining target grain (must be subset of hierarchy)

        Returns:
            New TimeSeries aggregated to specified grain

        Example:
            # Aggregate from account+region to just account
            ts.aggregate(grain=["account"])
        """
        from pyspark.sql import functions as F

        # Validate and resolve grain
        grain_cols = self._resolve_grain(grain)

        # Check if already at requested grain
        current_grain = [col for col in self.hierarchy if col in self.df.columns]
        if set(grain_cols) == set(current_grain):
            logger.info(f"Data already at grain {grain}. Returning copy.")
            return TimeSeries(
                df=self.df,
                hierarchy=self.hierarchy,
                metric_col=self.metric_col,
                time_col=self.time_col,
            )

        # Group by grain + time, sum metric
        group_cols = grain_cols + [self.time_col]
        agg_df = self.df.groupBy(*group_cols).agg(F.sum(self.metric_col).alias(self.metric_col))

        # New hierarchy only includes grain columns (in original hierarchy order)
        new_hierarchy = grain_cols

        return TimeSeries(
            df=agg_df,
            hierarchy=new_hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def summary_stats(self, grain: list[str] | None = None) -> DataFrame:
        """
        Compute summary statistics for the time series.

        Args:
            grain: Optional grain to aggregate to before computing stats.
                  If None, uses current grain of the data.

        Returns:
            PySpark DataFrame with entity keys and summary statistics
            (count, mean, std, min, max)

        Example:
            stats = ts.summary_stats()  # Stats at current grain
            stats = ts.summary_stats(grain=["account"])  # Aggregate first
        """
        from pyspark.sql import functions as F

        # Aggregate to target grain if specified
        if grain is not None:
            ts_for_stats = self.aggregate(grain)
        else:
            ts_for_stats = self

        # Identify entity columns (hierarchy columns present in data)
        entity_cols = [col for col in ts_for_stats.hierarchy if col in ts_for_stats.df.columns]

        # Group by entity and compute stats on metric over time
        stats_df = ts_for_stats.df.groupBy(*entity_cols).agg(
            F.count(self.metric_col).alias("count"),
            F.mean(self.metric_col).alias("mean"),
            F.stddev(self.metric_col).alias("std"),
            F.min(self.metric_col).alias("min"),
            F.max(self.metric_col).alias("max"),
        )

        return stats_df

    def plot_temporal_density(
        self,
        log_scale: bool = False,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 5),
        **kwargs,
    ) -> plt.Figure:
        """
        Plot temporal observation density at current grain.

        Shows record count per timestamp to inspect observation consistency across time.
        Uses ConciseDateFormatter for adaptive date labeling that adjusts to the time range.
        Automatically generates subtitle with grain context and entity count.

        Args:
            log_scale: Use logarithmic y-axis scale
            title: Plot title (None = auto-generate with grain context)
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to eda.plot_temporal_density()

        Returns:
            Matplotlib Figure object

        Example:
            >>> ts = PiedPiperLoader.load(df)
            >>> ts.filter(account="123").plot_temporal_density(log_scale=True)
        """
        # Identify current grain (hierarchy columns present in data)
        current_grain = [col for col in self.hierarchy if col in self.df.columns]

        # Count entities at current grain
        if current_grain:
            entity_count = self.df.select(*current_grain).distinct().count()
        else:
            entity_count = 1  # No hierarchy columns = single entity

        # Generate subtitle with grain context
        if current_grain:
            grain_str = "-".join(current_grain)
            subtitle = f"Grain: {grain_str} ({entity_count:,} entities)"
        else:
            subtitle = f"Aggregated view ({entity_count:,} entity)"

        # Delegate to eda.plot_temporal_density with subtitle
        return eda.plot_temporal_density(
            df=self.df,
            date_col=self.time_col,
            metric_col=None,  # Count records, not aggregate metric
            subtitle=subtitle,
            log_scale=log_scale,
            title=title,
            figsize=figsize,
            **kwargs,
        )

    def with_df(self, df: DataFrame) -> "TimeSeries":
        """
        Create new TimeSeries with different DataFrame, preserving metadata.

        Useful for applying transformations while keeping hierarchy/metric/time column info.

        Args:
            df: New DataFrame to wrap

        Returns:
            New TimeSeries instance with same metadata

        Example:
            >>> # Filter and create new instance
            >>> filtered_df = ts.df.filter(F.col('cost') > 100)
            >>> ts_filtered = ts.with_df(filtered_df)
        """
        return TimeSeries(
            df=df,
            hierarchy=self.hierarchy,
            metric_col=self.metric_col,
            time_col=self.time_col,
        )

    def plot_density_by_grain(
        self,
        grains: list[str],
        log_scale: bool = True,
        show_pct_change: bool = False,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Plot temporal record density for multiple aggregation grains on a single figure.

        For each grain, aggregates to that level and shows records per day over time.
        Optionally includes day-over-day percent change subplot below.

        Args:
            grains: List of dimension names to plot (e.g., ['region', 'product', 'usage'])
            log_scale: Use logarithmic y-axis scale (default True)
            show_pct_change: If True, add day-over-day percent change subplot below
            title: Plot title (None = auto-generate)
            figsize: Figure size (width, height)

        Returns:
            Matplotlib Figure object

        Example:
            >>> # Compare temporal density across multiple aggregation grains
            >>> ts.plot_density_by_grain(['region', 'product', 'usage', 'provider'])
            >>> # With percent change subplot
            >>> ts.plot_density_by_grain(['region', 'product'], show_pct_change=True)
        """
        import matplotlib.dates as mdates
        from pyspark.sql import functions as F

        # Create figure with optional pct_change subplot
        if show_pct_change:
            fig, (ax, ax_pct) = plt.subplots(
                2,
                1,
                figsize=(figsize[0], figsize[1] * 1.4),
                height_ratios=[2, 1],
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_pct = None

        plt.close(fig)  # Prevent double display

        # Plot each grain
        for grain in grains:
            # Aggregate to grain and count records per day
            grain_ts = self.aggregate([grain])
            daily = (
                grain_ts.df.groupBy(self.time_col)
                .agg(F.count("*").alias("count"))
                .orderBy(self.time_col)
                .toPandas()
            )

            # Plot density line
            ax.plot(
                daily[self.time_col],
                daily["count"],
                marker="o",
                markersize=3,
                label=f"{grain} grain",
                linewidth=2,
                alpha=0.8,
            )

            # Plot percent change if requested
            if ax_pct is not None:
                daily["pct_change"] = daily["count"].pct_change()
                daily_clean = daily.dropna(subset=["pct_change"])

                ax_pct.plot(
                    daily_clean[self.time_col],
                    daily_clean["pct_change"],
                    marker="o",
                    markersize=2,
                    label=f"{grain} grain",
                    linewidth=1.5,
                    alpha=0.7,
                )

        # Styling for main plot
        ax.set_ylabel("Records per Day", fontsize=12, fontweight="bold")
        ax.set_title(
            title or "Temporal Density by Aggregation Grain",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        ax.set_ylim(bottom=1 if log_scale else 0)

        if log_scale:
            ax.set_yscale("log")

        # Styling for pct_change subplot
        if ax_pct is not None:
            ax_pct.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
            ax_pct.set_ylabel("Day-over-Day\nChange", fontsize=10, fontweight="bold")
            ax_pct.set_xlabel("Date", fontsize=12, fontweight="bold")
            ax_pct.grid(True, alpha=0.3, linestyle="--", axis="y")
            ax_pct.legend(loc="best", fontsize=8)

            # Format y-axis as percentage
            from matplotlib.ticker import PercentFormatter

            ax_pct.yaxis.set_major_formatter(PercentFormatter(1.0))
        else:
            ax.set_xlabel("Date", fontsize=12, fontweight="bold")

        # Smart date formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        if ax_pct is not None:
            ax_pct.xaxis.set_major_locator(locator)
            ax_pct.xaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        fig.tight_layout()
        return fig

    def plot_cost_treemap(
        self,
        hierarchy: list[str],
        top_n: int = 30,
        title: str | None = None,
        width: int = 1200,
        height: int = 700,
    ):
        """
        Plot hierarchical cost treemap showing cost distribution across dimensions.

        Creates nested rectangular tiles sized by total cost, with proper hierarchical grouping.
        All children of a parent are grouped together spatially (e.g., all AWS regions grouped).

        Args:
            hierarchy: Hierarchy levels to display (e.g., ['provider', 'region'])
            top_n: Show only top N leaf entities by total cost (default 30)
            title: Plot title (None = auto-generate)
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly Figure object (displays automatically in Jupyter)

        Example:
            >>> # Cost breakdown by provider and region (nested)
            >>> ts.plot_cost_treemap(['provider', 'region'], top_n=20)
            >>> # Deep hierarchy with grouping
            >>> ts.plot_cost_treemap(['provider', 'region', 'product'], top_n=50)
        """
        import plotly.express as px
        from pyspark.sql import functions as F

        # Aggregate to specified hierarchy and sum cost
        agg_ts = self.aggregate(hierarchy)
        cost_data = (
            agg_ts.df.groupBy(*hierarchy)
            .agg(F.sum(self.metric_col).alias("total_cost"))
            .orderBy(F.desc("total_cost"))
            .limit(top_n)
            .toPandas()
        )

        # Plotly treemap requires a 'path' column with full hierarchy
        # Format: [top_level, level_2, ..., leaf]
        cost_data["path"] = cost_data[hierarchy].apply(
            lambda row: [str(row[col]) for col in hierarchy], axis=1
        )

        # Create hierarchical treemap with plotly
        fig = px.treemap(
            cost_data,
            path=[px.Constant("Total")] + hierarchy,  # Add root node
            values="total_cost",
            color=hierarchy[0],  # Color by top-level hierarchy
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=title or f"Cost Distribution: {' > '.join(hierarchy)}",
            width=width,
            height=height,
        )

        # Update layout for better appearance
        fig.update_traces(
            textinfo="label+value+percent parent",
            texttemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percentParent}",
            marker=dict(line=dict(width=2, color="white")),
            textfont=dict(size=12, family="Arial, sans-serif"),
        )

        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            font=dict(size=14, family="Arial, sans-serif"),
        )

        return fig

    def cost_summary_by_grain(
        self,
        grain: list[str],
        sort_by: str = "total",
    ):
        """
        Compute summary statistics for cost at specified grain.

        For each entity at the grain, computes:
        - total_cost: Sum across all time
        - mean_cost: Average daily cost
        - median_cost: Median daily cost
        - std_cost: Standard deviation (volatility)
        - min_cost: Minimum daily cost
        - max_cost: Maximum daily cost
        - days: Number of days with data

        Args:
            grain: Dimension(s) to analyze (e.g., ['region'] or ['provider', 'region'])
            sort_by: Sort by 'total', 'mean', 'volatility' (std), or 'median'

        Returns:
            PySpark DataFrame with summary statistics, sorted descending

        Example:
            >>> # Top regions by total cost with volatility stats
            >>> stats = ts.cost_summary_by_grain(['region'])
            >>> stats.show(10)
        """
        from pyspark.sql import functions as F

        # Aggregate to grain
        grain_ts = self.aggregate(grain)

        # Compute summary stats per entity
        stats_df = grain_ts.df.groupBy(*grain).agg(
            F.sum(self.metric_col).alias("total_cost"),
            F.mean(self.metric_col).alias("mean_cost"),
            F.expr(f"percentile_approx({self.metric_col}, 0.5)").alias("median_cost"),
            F.stddev(self.metric_col).alias("std_cost"),
            F.min(self.metric_col).alias("min_cost"),
            F.max(self.metric_col).alias("max_cost"),
            F.count(self.metric_col).alias("days"),
        )

        # Sort by requested metric
        sort_col_map = {
            "total": "total_cost",
            "mean": "mean_cost",
            "median": "median_cost",
            "volatility": "std_cost",
        }
        sort_col = sort_col_map.get(sort_by, "total_cost")
        stats_df = stats_df.orderBy(F.desc(sort_col))

        return stats_df

    def plot_cost_distribution(
        self,
        grain: list[str],
        top_n: int = 10,
        sort_by: str = "total",
        min_cost: float = 0.0,
        log_scale: bool = False,
        group_by_parent: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot daily cost distribution for entities at specified grain.

        Shows box plot with one box per entity, displaying:
        - Median (line in box)
        - 25th-75th percentiles (box)
        - Whiskers (1.5 * IQR)
        - Outliers (dots)

        Args:
            grain: Dimension(s) to analyze (e.g., ['region'])
            top_n: Show top N entities by sort metric
            sort_by: Sort by 'total', 'mean', 'volatility', or 'median'
            min_cost: Filter out daily cost values below this threshold (default: 0.0)
            log_scale: Use logarithmic y-axis scale
            group_by_parent: If True, color boxes by parent hierarchy level (e.g., provider)
            title: Plot title (None = auto-generate)
            figsize: Figure size (width, height)

        Returns:
            Matplotlib Figure object

        Example:
            >>> # Top 10 regions grouped/colored by provider
            >>> ts.plot_cost_distribution(['region'], top_n=10, group_by_parent=True)
            >>> # Most volatile products
            >>> ts.plot_cost_distribution(['product'], top_n=5, sort_by='volatility')
        """
        import seaborn as sns
        from pyspark.sql import functions as F

        # Determine if we need parent hierarchy for grouping
        parent_col = None
        if group_by_parent and len(grain) == 1:
            # Find parent in hierarchy
            grain_col = grain[0]
            if grain_col in self.hierarchy:
                grain_idx = self.hierarchy.index(grain_col)
                if grain_idx > 0:
                    parent_col = self.hierarchy[grain_idx - 1]

        # Get top entities (include parent if grouping)
        analysis_grain = grain if parent_col is None else [parent_col] + grain
        stats_df = self.cost_summary_by_grain(analysis_grain, sort_by=sort_by)
        top_entities = stats_df.limit(top_n).toPandas()

        # Create entity identifier
        top_entities["entity"] = top_entities[grain].apply(
            lambda row: " > ".join(str(row[col]) for col in grain), axis=1
        )

        # Get daily cost data for these entities
        grain_ts = self.aggregate(analysis_grain)

        # Filter to top entities
        entity_list = [tuple(row[analysis_grain].tolist()) for _, row in top_entities.iterrows()]

        # Build filter condition (one condition per entity)
        filter_conds = []
        for entity in entity_list:
            # Create AND condition for all grain columns matching this entity
            entity_cond = F.col(analysis_grain[0]) == entity[0]
            for i in range(1, len(analysis_grain)):
                entity_cond = entity_cond & (F.col(analysis_grain[i]) == entity[i])
            filter_conds.append(entity_cond)

        # Combine with OR
        filter_expr = filter_conds[0]
        for cond in filter_conds[1:]:
            filter_expr = filter_expr | cond

        daily_data = grain_ts.df.filter(filter_expr).toPandas()

        # Filter out small cost outliers
        if min_cost > 0:
            daily_data = daily_data[daily_data[self.metric_col] >= min_cost]

        daily_data["entity"] = daily_data[grain].apply(
            lambda row: " > ".join(str(row[col]) for col in grain), axis=1
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        plt.close(fig)

        # Create box plot with optional grouping by parent
        if parent_col and parent_col in daily_data.columns:
            sns.boxplot(
                data=daily_data,
                x="entity",
                y=self.metric_col,
                hue=parent_col,
                ax=ax,
                palette="Set2",
            )
            ax.legend(title=parent_col.title(), loc="upper right", framealpha=0.9)
        else:
            sns.boxplot(
                data=daily_data,
                x="entity",
                y=self.metric_col,
                ax=ax,
                palette="Set2",
            )

        # Styling
        ax.set_xlabel(" > ".join(grain).title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Daily Cost ($)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or f"Daily Cost Distribution by {' > '.join(grain).title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if log_scale:
            ax.set_yscale("log")

        fig.tight_layout()
        return fig

    def plot_cost_trends(
        self,
        grain: list[str],
        top_n: int = 5,
        sort_by: str = "total",
        show_total: bool = True,
        log_scale: bool = False,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot cost trends over time for top entities at specified grain.

        Shows time series with one line per entity, optionally including aggregate total.

        Args:
            grain: Dimension(s) to analyze (e.g., ['region'])
            top_n: Show top N entities by sort metric
            sort_by: Sort by 'total', 'mean', 'volatility', or 'median'
            show_total: If True, add line showing total across all entities
            log_scale: Use logarithmic y-axis scale
            title: Plot title (None = auto-generate)
            figsize: Figure size (width, height)

        Returns:
            Matplotlib Figure object

        Example:
            >>> # Top 5 regions with trends and total
            >>> ts.plot_cost_trends(['region'], top_n=5, show_total=True)
            >>> # Most volatile products without total
            >>> ts.plot_cost_trends(['product'], top_n=3, sort_by='volatility', show_total=False)
        """
        import matplotlib.dates as mdates
        from pyspark.sql import functions as F

        # Get top entities
        stats_df = self.cost_summary_by_grain(grain, sort_by=sort_by)
        top_entities = stats_df.limit(top_n).toPandas()

        # Create entity identifier
        top_entities["entity"] = top_entities[grain].apply(
            lambda row: " > ".join(str(row[col]) for col in grain), axis=1
        )

        # Get daily cost data for these entities
        grain_ts = self.aggregate(grain)

        # Filter to top entities
        entity_list = [tuple(row[grain].tolist()) for _, row in top_entities.iterrows()]

        # Build filter condition (one condition per entity)
        filter_conds = []
        for entity in entity_list:
            # Create AND condition for all grain columns matching this entity
            entity_cond = F.col(grain[0]) == entity[0]
            for i in range(1, len(grain)):
                entity_cond = entity_cond & (F.col(grain[i]) == entity[i])
            filter_conds.append(entity_cond)

        # Combine with OR
        filter_expr = filter_conds[0]
        for cond in filter_conds[1:]:
            filter_expr = filter_expr | cond

        daily_data = grain_ts.df.filter(filter_expr).toPandas()
        daily_data["entity"] = daily_data[grain].apply(
            lambda row: " > ".join(str(row[col]) for col in grain), axis=1
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        plt.close(fig)

        # Plot each entity
        for entity in top_entities["entity"]:
            entity_data = daily_data[daily_data["entity"] == entity].sort_values(self.time_col)
            ax.plot(
                entity_data[self.time_col],
                entity_data[self.metric_col],
                marker="o",
                markersize=3,
                label=entity,
                linewidth=2,
                alpha=0.8,
            )

        # Add total line if requested
        if show_total:
            total_daily = (
                self.df.groupBy(self.time_col)
                .agg(F.sum(self.metric_col).alias("total"))
                .orderBy(self.time_col)
                .toPandas()
            )
            ax.plot(
                total_daily[self.time_col],
                total_daily["total"],
                marker="",
                linewidth=3,
                label="Total (All Entities)",
                linestyle="--",
                color="black",
                alpha=0.7,
            )

        # Styling
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Daily Cost ($)", fontsize=12, fontweight="bold")
        ax.set_title(
            title or f"Cost Trends by {' > '.join(grain).title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", framealpha=0.9)

        if log_scale:
            ax.set_yscale("log")

        # Smart date formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        fig.tight_layout()
        return fig

    def filter_time(
        self,
        start: str | None = None,
        end: str | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> "TimeSeries":
        """
        Filter time series to specified time range.

        Supports multiple filtering styles:
        - Range filtering: start/end (inclusive start, exclusive end)
        - Single-sided: before (exclusive) or after (inclusive)

        Args:
            start: Start time (inclusive), format: 'YYYY-MM-DD' or datetime-compatible string
            end: End time (exclusive), format: 'YYYY-MM-DD' or datetime-compatible string
            before: Filter to times before this value (exclusive), alternative to end
            after: Filter to times after this value (inclusive), alternative to start

        Returns:
            New TimeSeries with filtered data

        Example:
            >>> # Filter to specific range
            >>> ts_filtered = ts.filter_time(start='2024-01-01', end='2024-12-31')
            >>> # Filter before a date
            >>> ts_clean = ts.filter_time(before='2025-10-05')
            >>> # Filter after a date
            >>> ts_recent = ts.filter_time(after='2024-01-01')
        """
        from pyspark.sql import functions as F

        filtered_df = self.df

        # Apply filters based on what's provided
        if start is not None:
            filtered_df = filtered_df.filter(F.col(self.time_col) >= start)
        if after is not None:
            filtered_df = filtered_df.filter(F.col(self.time_col) >= after)
        if end is not None:
            filtered_df = filtered_df.filter(F.col(self.time_col) < end)
        if before is not None:
            filtered_df = filtered_df.filter(F.col(self.time_col) < before)

        return self.with_df(filtered_df)
