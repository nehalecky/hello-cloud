"""
Tests for EDA analysis and visualization functions.

Focuses on testing plot_temporal_density and related visualization utilities.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pyspark.sql import functions as F

from hellocloud.analysis.eda import plot_temporal_density
from hellocloud.spark import get_spark_session


@pytest.fixture
def spark():
    """Get or create Spark session for tests."""
    return get_spark_session(app_name="test_eda_analysis")


@pytest.fixture
def sample_cost_data(spark):
    """
    Create sample cost data for testing temporal visualizations.

    Returns DataFrame with:
    - date: Daily timestamps
    - entity_id: Resource identifier
    - cost: Cost values with some variation
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D").tolist()
    data = pd.DataFrame(
        {
            "date": dates * 3,  # 3 entities
            "entity_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "cost": [
                # Entity A: varying costs
                100,
                120,
                110,
                130,
                125,
                # Entity B: different pattern
                50,
                55,
                60,
                58,
                62,
                # Entity C: higher variance
                200,
                180,
                220,
                190,
                210,
            ],
        }
    )
    return spark.createDataFrame(data)


class TestPlotTemporalDensity:
    """Test temporal density plotting function."""

    def test_basic_aggregation_mode(self, sample_cost_data):
        """Test default aggregation mode (line plot)."""
        fig = plot_temporal_density(sample_cost_data, date_col="date", metric_col="cost")

        # Verify it returns a matplotlib Figure
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_distribution_mode_violin(self, sample_cost_data):
        """Test distribution mode with violin plot."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_distribution=True,
            dist_type="violin",
        )

        # Verify it returns a matplotlib Figure
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_distribution_mode_box(self, sample_cost_data):
        """Test distribution mode with box plot."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_distribution=True,
            dist_type="box",
        )

        # Verify it returns a matplotlib Figure
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_invalid_dist_type_raises_error(self, sample_cost_data):
        """Test that invalid dist_type raises ValueError."""
        with pytest.raises(ValueError, match="dist_type must be 'violin' or 'box'"):
            plot_temporal_density(
                sample_cost_data,
                date_col="date",
                metric_col="cost",
                show_distribution=True,
                dist_type="invalid",
            )

    def test_count_mode(self, sample_cost_data):
        """Test count mode (metric_col=None)."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col=None,  # Count records instead
        )

        # Verify it returns a matplotlib Figure
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_log_scale(self, sample_cost_data):
        """Test log scale mode."""
        fig = plot_temporal_density(
            sample_cost_data, date_col="date", metric_col="cost", log_scale=True
        )

        # Verify it returns a matplotlib Figure
        assert isinstance(fig, plt.Figure)

        # Verify log scale was applied
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

        # Clean up
        plt.close(fig)

    def test_custom_title(self, sample_cost_data):
        """Test custom title."""
        custom_title = "My Custom Title"
        fig = plot_temporal_density(
            sample_cost_data, date_col="date", metric_col="cost", title=custom_title
        )

        # Verify title was set
        ax = fig.axes[0]
        assert ax.get_title() == custom_title

        # Clean up
        plt.close(fig)

    def test_custom_figsize(self, sample_cost_data):
        """Test custom figure size."""
        figsize = (20, 10)
        fig = plot_temporal_density(
            sample_cost_data, date_col="date", metric_col="cost", figsize=figsize
        )

        # Verify figure size
        assert fig.get_figwidth() == figsize[0]
        assert fig.get_figheight() == figsize[1]

        # Clean up
        plt.close(fig)

    def test_distribution_without_metric_col(self, sample_cost_data):
        """Test that distribution mode requires metric_col."""
        # When metric_col is None, show_distribution should be ignored
        # and fall back to aggregation mode
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col=None,
            show_distribution=True,  # Should be ignored
        )

        # Should still work (falls back to aggregation)
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_marginal_box_plot(self, sample_cost_data):
        """Test marginal box plot."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_marginal=True,
            marginal_type="box",
        )

        # Should have 2 axes (main + marginal)
        assert len(fig.axes) == 2
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_marginal_histogram(self, sample_cost_data):
        """Test marginal histogram."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_marginal=True,
            marginal_type="hist",
        )

        # Should have 2 axes (main + marginal)
        assert len(fig.axes) == 2
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_marginal_kde(self, sample_cost_data):
        """Test marginal KDE plot."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_marginal=True,
            marginal_type="kde",
        )

        # Should have 2 axes (main + marginal)
        assert len(fig.axes) == 2
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_marginal_with_count_mode(self, sample_cost_data):
        """Test marginal plot with count mode (metric_col=None)."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col=None,  # Count mode
            show_marginal=True,
            marginal_type="box",
        )

        # Should have 2 axes
        assert len(fig.axes) == 2
        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_invalid_marginal_type_raises_error(self, sample_cost_data):
        """Test that invalid marginal_type raises ValueError."""
        with pytest.raises(ValueError, match="marginal_type must be 'box', 'hist', or 'kde'"):
            plot_temporal_density(
                sample_cost_data,
                date_col="date",
                metric_col="cost",
                show_marginal=True,
                marginal_type="invalid",
            )

    def test_marginal_with_log_scale(self, sample_cost_data):
        """Test marginal plot with log scale."""
        fig = plot_temporal_density(
            sample_cost_data,
            date_col="date",
            metric_col="cost",
            show_marginal=True,
            marginal_type="box",
            log_scale=True,
        )

        # Verify both axes have log scale
        assert fig.axes[0].get_yscale() == "log"
        assert fig.axes[1].get_yscale() == "log"

        # Clean up
        plt.close(fig)
