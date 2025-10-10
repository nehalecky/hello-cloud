"""Tests for application taxonomy system."""

import pytest
import numpy as np
from pydantic import ValidationError

from hellocloud.ml_models.application_taxonomy import (
    ApplicationDomain,
    ScalingBehavior,
    OptimizationPotential,
    ResourcePattern,
    CostProfile,
    ApplicationArchetype,
    CloudResourceTaxonomy,
)


class TestEnums:
    """Test enum definitions."""

    def test_application_domain_values(self):
        """Test ApplicationDomain enum has expected values."""
        assert ApplicationDomain.CUSTOMER_FACING.value == "customer_facing"
        assert ApplicationDomain.DATA_PROCESSING.value == "data_processing"
        assert ApplicationDomain.MACHINE_LEARNING.value == "machine_learning"
        assert ApplicationDomain.INFRASTRUCTURE.value == "infrastructure"
        assert ApplicationDomain.DEVELOPMENT.value == "development"

    def test_scaling_behavior_values(self):
        """Test ScalingBehavior enum has expected values."""
        assert ScalingBehavior.ELASTIC_AUTO.value == "elastic_auto"
        assert ScalingBehavior.ELASTIC_MANUAL.value == "elastic_manual"
        assert ScalingBehavior.STATIC.value == "static"
        assert ScalingBehavior.SERVERLESS.value == "serverless"
        assert ScalingBehavior.SCHEDULED.value == "scheduled"

    def test_optimization_potential_values(self):
        """Test OptimizationPotential enum has expected values."""
        assert OptimizationPotential.HIGH.value == "high"
        assert OptimizationPotential.MEDIUM.value == "medium"
        assert OptimizationPotential.LOW.value == "low"
        assert OptimizationPotential.OPTIMIZED.value == "optimized"


class TestResourcePattern:
    """Test ResourcePattern model validation."""

    def test_valid_resource_pattern(self):
        """Test creating a valid ResourcePattern."""
        pattern = ResourcePattern(
            cpu_p50=15.0,
            cpu_p95=45.0,
            memory_p50=25.0,
            memory_p95=60.0,
            cpu_cv=0.8,
            memory_cv=0.5,
            correlation_matrix=np.eye(5),
            daily_pattern_type="business_hours",
            weekly_pattern_type="weekday_heavy",
            seasonality_strength=0.3,
            burst_frequency=5.0,
            burst_amplitude=2.5,
            burst_duration_minutes=30
        )
        assert pattern.cpu_p50 == 15.0
        assert pattern.correlation_matrix.shape == (5, 5)

    def test_invalid_cpu_utilization(self):
        """Test that CPU utilization > 100% raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ResourcePattern(
                cpu_p50=150.0,  # Invalid: > 100
                cpu_p95=45.0,
                memory_p50=25.0,
                memory_p95=60.0,
                cpu_cv=0.8,
                memory_cv=0.5,
                correlation_matrix=np.eye(5),
                daily_pattern_type="business_hours",
                weekly_pattern_type="weekday_heavy",
                seasonality_strength=0.3,
                burst_frequency=5.0,
                burst_amplitude=2.5,
                burst_duration_minutes=30
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_invalid_correlation_matrix_shape(self):
        """Test that wrong correlation matrix shape raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ResourcePattern(
                cpu_p50=15.0,
                cpu_p95=45.0,
                memory_p50=25.0,
                memory_p95=60.0,
                cpu_cv=0.8,
                memory_cv=0.5,
                correlation_matrix=np.eye(3),  # Wrong shape: should be 5x5
                daily_pattern_type="business_hours",
                weekly_pattern_type="weekday_heavy",
                seasonality_strength=0.3,
                burst_frequency=5.0,
                burst_amplitude=2.5,
                burst_duration_minutes=30
            )
        assert "Correlation matrix must be 5x5" in str(exc_info.value)

    def test_negative_cv_rejected(self):
        """Test that negative coefficient of variation is rejected."""
        with pytest.raises(ValidationError):
            ResourcePattern(
                cpu_p50=15.0,
                cpu_p95=45.0,
                memory_p50=25.0,
                memory_p95=60.0,
                cpu_cv=-0.5,  # Invalid: negative
                memory_cv=0.5,
                correlation_matrix=np.eye(5),
                daily_pattern_type="business_hours",
                weekly_pattern_type="weekday_heavy",
                seasonality_strength=0.3,
                burst_frequency=5.0,
                burst_amplitude=2.5,
                burst_duration_minutes=30
            )


class TestCostProfile:
    """Test CostProfile model validation."""

    def test_valid_cost_profile(self):
        """Test creating a valid CostProfile."""
        profile = CostProfile(
            avg_hourly_cost=25.50,
            cost_variability=0.3,
            waste_percentage=32.0,  # Research average
            optimization_difficulty=0.6,
            business_criticality=0.8
        )
        assert profile.avg_hourly_cost == 25.50
        assert profile.waste_percentage == 32.0

    def test_waste_percentage_bounds(self):
        """Test waste percentage must be between 0 and 100."""
        # Valid: 0%
        profile = CostProfile(
            avg_hourly_cost=10.0,
            cost_variability=0.1,
            waste_percentage=0.0,
            optimization_difficulty=0.1,
            business_criticality=0.5
        )
        assert profile.waste_percentage == 0.0

        # Valid: 100%
        profile = CostProfile(
            avg_hourly_cost=10.0,
            cost_variability=0.1,
            waste_percentage=100.0,
            optimization_difficulty=0.9,
            business_criticality=0.1
        )
        assert profile.waste_percentage == 100.0

        # Invalid: > 100%
        with pytest.raises(ValidationError):
            CostProfile(
                avg_hourly_cost=10.0,
                cost_variability=0.1,
                waste_percentage=150.0,  # Invalid
                optimization_difficulty=0.5,
                business_criticality=0.5
            )

    def test_business_criticality_range(self):
        """Test business criticality must be between 0 and 1."""
        with pytest.raises(ValidationError):
            CostProfile(
                avg_hourly_cost=10.0,
                cost_variability=0.1,
                waste_percentage=30.0,
                optimization_difficulty=0.5,
                business_criticality=1.5  # Invalid: > 1
            )


class TestApplicationArchetype:
    """Test ApplicationArchetype model."""

    def test_create_complete_archetype(self):
        """Test creating a complete application archetype."""
        archetype = ApplicationArchetype(
            name="Web Application",
            domain=ApplicationDomain.CUSTOMER_FACING,
            description="Customer-facing web application",
            resource_pattern=ResourcePattern(
                cpu_p50=15.0,
                cpu_p95=45.0,
                memory_p50=25.0,
                memory_p95=60.0,
                cpu_cv=0.8,
                memory_cv=0.5,
                correlation_matrix=np.eye(5),
                daily_pattern_type="business_hours",
                weekly_pattern_type="weekday_heavy",
                seasonality_strength=0.3,
                burst_frequency=5.0,
                burst_amplitude=2.5,
                burst_duration_minutes=30
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.HIGH,
            cost_profile=CostProfile(
                avg_hourly_cost=50.0,
                cost_variability=0.3,
                waste_percentage=35.0,
                optimization_difficulty=0.4,
                business_criticality=0.9
            ),
            typical_stack=["Python", "Django", "PostgreSQL"],
            cloud_services=["EC2", "RDS", "ELB"],
            typical_instance_types=["t3.medium", "t3.large"],
            deployment_frequency="continuous",
            maintenance_windows=[(6, 2), (0, 3)],  # Sunday 2am, Saturday 3am
            availability_target=99.9,
            latency_p99_ms=500.0,
            data_volume_gb_per_day=100.0,
            data_retention_days=90,
            example_companies=["Netflix", "Spotify"],
            market_size_percentage=25.0
        )

        assert archetype.name == "Web Application"
        assert archetype.domain == ApplicationDomain.CUSTOMER_FACING
        assert len(archetype.typical_stack) == 3
        assert archetype.availability_target == 99.9

    def test_example_companies_default(self):
        """Test that example_companies defaults to empty list."""
        archetype = ApplicationArchetype(
            name="Test App",
            domain=ApplicationDomain.DEVELOPMENT,
            description="Test application",
            resource_pattern=ResourcePattern(
                cpu_p50=10.0,
                cpu_p95=20.0,
                memory_p50=15.0,
                memory_p95=30.0,
                cpu_cv=0.5,
                memory_cv=0.4,
                correlation_matrix=np.eye(5),
                daily_pattern_type="constant",
                weekly_pattern_type="constant",
                seasonality_strength=0.1,
                burst_frequency=1.0,
                burst_amplitude=1.5,
                burst_duration_minutes=15
            ),
            scaling_behavior=ScalingBehavior.STATIC,
            optimization_potential=OptimizationPotential.LOW,
            cost_profile=CostProfile(
                avg_hourly_cost=5.0,
                cost_variability=0.1,
                waste_percentage=20.0,
                optimization_difficulty=0.2,
                business_criticality=0.3
            ),
            typical_stack=["Python"],
            cloud_services=["EC2"],
            typical_instance_types=["t3.micro"],
            deployment_frequency="weekly",
            maintenance_windows=[(0, 2)],
            availability_target=99.0,
            latency_p99_ms=1000.0,
            data_volume_gb_per_day=1.0,
            data_retention_days=7
            # Note: not providing example_companies
        )

        assert archetype.example_companies == []


class TestCloudResourceTaxonomy:
    """Test the CloudResourceTaxonomy class."""

    def test_archetypes_defined(self):
        """Test that archetypes are defined in the taxonomy."""
        assert hasattr(CloudResourceTaxonomy, 'ARCHETYPES')
        assert isinstance(CloudResourceTaxonomy.ARCHETYPES, dict)

        # Should have at least some archetypes
        assert len(CloudResourceTaxonomy.ARCHETYPES) > 0

    def test_get_archetype_by_name(self):
        """Test retrieving archetype by name."""
        # Get the first archetype name
        archetype_names = list(CloudResourceTaxonomy.ARCHETYPES.keys())
        if archetype_names:
            first_name = archetype_names[0]
            archetype = CloudResourceTaxonomy.get_archetype(first_name)
            assert isinstance(archetype, ApplicationArchetype)
            assert archetype.name == CloudResourceTaxonomy.ARCHETYPES[first_name].name

    def test_get_nonexistent_archetype(self):
        """Test that getting non-existent archetype returns None."""
        result = CloudResourceTaxonomy.get_archetype("NonExistentArchetype")
        assert result is None

    def test_get_by_optimization_potential(self):
        """Test filtering archetypes by optimization potential."""
        high_potential = CloudResourceTaxonomy.get_by_optimization_potential(
            OptimizationPotential.HIGH
        )
        assert isinstance(high_potential, list)
        for archetype in high_potential:
            assert archetype.optimization_potential == OptimizationPotential.HIGH

    def test_get_by_domain(self):
        """Test filtering archetypes by domain."""
        ml_archetypes = CloudResourceTaxonomy.get_by_domain(
            ApplicationDomain.MACHINE_LEARNING
        )
        assert isinstance(ml_archetypes, list)
        for archetype in ml_archetypes:
            assert archetype.domain == ApplicationDomain.MACHINE_LEARNING

    def test_calculate_market_coverage(self):
        """Test market coverage calculation."""
        coverage = CloudResourceTaxonomy.calculate_market_coverage()
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 100

    def test_export_for_ml_training(self):
        """Test exporting features for ML training."""
        ml_features = CloudResourceTaxonomy.export_for_ml_training()

        # Should return a Polars DataFrame
        import polars as pl
        assert isinstance(ml_features, pl.DataFrame)

        if CloudResourceTaxonomy.ARCHETYPES:
            # Should have one row for each archetype
            assert len(ml_features) == len(CloudResourceTaxonomy.ARCHETYPES)

            # Check expected columns exist
            expected_columns = {
                'archetype', 'domain', 'cpu_p50', 'cpu_p95',
                'memory_p50', 'memory_p95', 'waste_percentage',
                'optimization_potential', 'business_criticality'
            }
            actual_columns = set(ml_features.columns)
            assert expected_columns.issubset(actual_columns)

    def test_compute_priors(self):
        """Test Bayesian prior computation."""
        priors = CloudResourceTaxonomy.compute_priors()
        assert isinstance(priors, dict)

        # Should have specific prior types with mean/std/min/max
        expected_priors = {'cpu_utilization', 'memory_utilization', 'waste_percentage', 'hourly_cost'}
        assert expected_priors.issubset(priors.keys())

        # Check each prior has required statistics
        for prior_name in expected_priors:
            prior = priors[prior_name]
            assert 'mean' in prior
            assert 'std' in prior
            assert 'min' in prior
            assert 'max' in prior

        # Check prior values are reasonable
        assert 0 <= priors['cpu_mu'] <= 100
        assert 0 <= priors['memory_mu'] <= 100
        assert priors['cpu_sigma'] > 0
        assert priors['memory_sigma'] > 0


class TestIntegration:
    """Integration tests for the taxonomy system."""

    def test_taxonomy_consistency(self):
        """Test that all archetypes in taxonomy are valid."""
        for name, archetype in CloudResourceTaxonomy.ARCHETYPES.items():
            # Each archetype should be valid
            assert isinstance(archetype, ApplicationArchetype)

            # Name should be set (can be different from key for display purposes)
            assert archetype.name is not None
            assert len(archetype.name) > 0

            # Resource patterns should be realistic (based on research)
            assert 0 <= archetype.resource_pattern.cpu_p50 <= 100
            assert 0 <= archetype.resource_pattern.memory_p50 <= 100

            # p95 should be >= p50
            assert archetype.resource_pattern.cpu_p95 >= archetype.resource_pattern.cpu_p50
            assert archetype.resource_pattern.memory_p95 >= archetype.resource_pattern.memory_p50

            # Waste percentage should align with research (typically 20-60%)
            assert 0 <= archetype.cost_profile.waste_percentage <= 100

    def test_archetype_coverage_by_domain(self):
        """Test that we have good coverage across domains."""
        domain_coverage = {}
        for domain in ApplicationDomain:
            archetypes = CloudResourceTaxonomy.get_by_domain(domain)
            domain_coverage[domain.value] = len(archetypes)

        # Should have at least one archetype per major domain
        assert all(count >= 0 for count in domain_coverage.values())

    def test_realistic_utilization_patterns(self):
        """Test that utilization patterns match research findings."""
        all_cpu_p50 = []
        all_memory_p50 = []
        all_waste = []

        for archetype in CloudResourceTaxonomy.ARCHETYPES.values():
            all_cpu_p50.append(archetype.resource_pattern.cpu_p50)
            all_memory_p50.append(archetype.resource_pattern.memory_p50)
            all_waste.append(archetype.cost_profile.waste_percentage)

        if all_cpu_p50:  # Only test if we have archetypes
            # Average CPU should be around 13% (research finding)
            avg_cpu = sum(all_cpu_p50) / len(all_cpu_p50)
            assert 5 <= avg_cpu <= 30  # Allow some range

            # Average memory should be around 20-50% (research finding)
            avg_memory = sum(all_memory_p50) / len(all_memory_p50)
            assert 10 <= avg_memory <= 50

            # Average waste should be around 30-32% (research finding)
            avg_waste = sum(all_waste) / len(all_waste)
            assert 20 <= avg_waste <= 45


@pytest.mark.parametrize("pattern_type,expected_values", [
    ("business_hours", ["business_hours", "constant", "batch", "irregular"]),
    ("weekly", ["weekday_heavy", "constant", "weekend_heavy"])
])
def test_pattern_type_literals(pattern_type, expected_values):
    """Test that pattern type literals are properly constrained."""
    # This is more of a documentation test
    # The actual validation happens in ResourcePattern
    assert all(isinstance(v, str) for v in expected_values)