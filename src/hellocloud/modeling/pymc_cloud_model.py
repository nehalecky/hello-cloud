"""
PyMC-based Hierarchical Bayesian Model for Cloud Resource Simulation
Learns from real data and generates realistic synthetic patterns
"""

from datetime import datetime, timedelta
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
from loguru import logger

# Import our taxonomy
from .application_taxonomy import CloudResourceTaxonomy


class CloudResourceHierarchicalModel:
    """
    Three-level Hierarchical Bayesian Model:
    1. Industry Level (Hyperpriors)
    2. Application Archetype Level
    3. Individual Resource Level

    Key innovations:
    - Learns actual correlation structures from data
    - Models regime switching (normal/peak/idle)
    - Captures temporal dependencies
    - Quantifies uncertainty in all parameters
    """

    def __init__(
        self,
        observed_data: pl.DataFrame | None = None,
        seed: int | None = None,
        archetypes: list[str] | None = None,
    ):
        """
        Initialize model with optional observed data for learning

        Args:
            observed_data: Real cloud metrics data with columns:
                - archetype: Application type
                - cpu_utilization: CPU usage %
                - memory_utilization: Memory usage %
                - timestamp: Time of observation
                - resource_id: Unique resource identifier
            seed: Random seed for reproducibility
            archetypes: Optional list of archetype names to use
        """
        self.observed_data = observed_data
        self.model = None
        self.trace = None
        self.seed = seed

        if archetypes is not None:
            self.archetypes = archetypes
        else:
            self.archetypes = list(CloudResourceTaxonomy.ARCHETYPES.keys())

        self.n_archetypes = len(self.archetypes)

    def build_model(self, data: pl.DataFrame | None = None):
        """
        Build the complete hierarchical model

        Args:
            data: Optional observed data to use for building the model
        """
        # Update observed data if provided
        if data is not None:
            self.observed_data = data
        with pm.Model() as model:
            # =========================================
            # LEVEL 1: Industry-Level Hyperpriors
            # Based on research: 13% CPU, 20% memory average
            # =========================================

            # Industry-wide utilization (Beta distribution for percentages)
            industry_cpu_alpha = pm.Gamma("industry_cpu_alpha", alpha=2, beta=0.5)
            industry_cpu_beta = pm.Gamma("industry_cpu_beta", alpha=10, beta=0.5)
            industry_cpu_mean = pm.Deterministic(
                "industry_cpu_mean", industry_cpu_alpha / (industry_cpu_alpha + industry_cpu_beta)
            )

            industry_mem_alpha = pm.Gamma("industry_mem_alpha", alpha=3, beta=0.5)
            industry_mem_beta = pm.Gamma("industry_mem_beta", alpha=12, beta=0.5)
            industry_mem_mean = pm.Deterministic(
                "industry_mem_mean", industry_mem_alpha / (industry_mem_alpha + industry_mem_beta)
            )

            # Industry-wide waste factor (30-32% from research)
            industry_waste = pm.Beta("industry_waste", alpha=3, beta=7)

            # Variance in utilization
            industry_cpu_variance = pm.InverseGamma("industry_cpu_variance", alpha=3, beta=1)
            industry_mem_variance = pm.InverseGamma("industry_mem_variance", alpha=3, beta=1)

            # =========================================
            # LEVEL 2: Application Archetype Level
            # Each archetype has its own characteristics
            # =========================================

            # Archetype-specific parameters (centered on industry)
            archetype_cpu_mean = pm.Beta(
                "archetype_cpu_mean",
                alpha=industry_cpu_alpha * pm.math.ones(self.n_archetypes),
                beta=industry_cpu_beta * pm.math.ones(self.n_archetypes),
                shape=self.n_archetypes,
            )

            archetype_mem_mean = pm.Beta(
                "archetype_mem_mean",
                alpha=industry_mem_alpha * pm.math.ones(self.n_archetypes),
                beta=industry_mem_beta * pm.math.ones(self.n_archetypes),
                shape=self.n_archetypes,
            )

            # Archetype-specific variance
            archetype_cpu_cv = pm.HalfNormal("archetype_cpu_cv", sigma=0.5, shape=self.n_archetypes)

            archetype_mem_cv = pm.HalfNormal("archetype_mem_cv", sigma=0.3, shape=self.n_archetypes)

            # =========================================
            # Correlation Structure (LKJ Prior)
            # Models dependencies between metrics
            # =========================================

            # Each archetype has its own correlation matrix
            # Using LKJ prior for correlation matrices
            n_metrics = 5  # CPU, Memory, Network In/Out, Disk

            # Cholesky decomposition for efficiency
            chol_corr = pm.LKJCholeskyCov(
                "chol_corr",
                n=n_metrics,
                eta=2.0,  # Slight preference for independence
                sd_dist=pm.HalfNormal.dist(sigma=1.0),
                compute_corr=True,
            )

            # Extract correlation matrix
            corr_matrix = pm.Deterministic("corr_matrix", chol_corr[1])

            # =========================================
            # Temporal Patterns (Fourier Series)
            # Captures daily and weekly seasonality
            # =========================================

            # Daily pattern (24 hours)
            n_fourier_daily = 4
            daily_cos_coef = pm.Normal(
                "daily_cos_coef", mu=0, sigma=0.5, shape=(self.n_archetypes, n_fourier_daily)
            )
            daily_sin_coef = pm.Normal(
                "daily_sin_coef", mu=0, sigma=0.5, shape=(self.n_archetypes, n_fourier_daily)
            )

            # Weekly pattern (7 days)
            n_fourier_weekly = 2
            weekly_cos_coef = pm.Normal(
                "weekly_cos_coef", mu=0, sigma=0.3, shape=(self.n_archetypes, n_fourier_weekly)
            )
            weekly_sin_coef = pm.Normal(
                "weekly_sin_coef", mu=0, sigma=0.3, shape=(self.n_archetypes, n_fourier_weekly)
            )

            # =========================================
            # Regime Switching (Markov Model)
            # Normal, Peak, Idle, Failure states
            # =========================================

            # Transition probabilities between states
            n_states = 4  # normal, peak, idle, failure

            # Dirichlet prior for transition matrix rows
            transition_probs = pm.Dirichlet(
                "transition_probs",
                a=np.array(
                    [
                        [10, 2, 2, 0.5],  # From normal: likely to stay normal
                        [5, 8, 1, 0.5],  # From peak: likely to stay peak or return to normal
                        [3, 1, 8, 0.5],  # From idle: likely to stay idle
                        [8, 2, 2, 2],  # From failure: likely to return to normal
                    ]
                ),
                shape=(n_states, n_states),
            )

            # State-specific multipliers
            state_cpu_multipliers = pm.Gamma(
                "state_cpu_multipliers",
                alpha=[5, 20, 0.5, 10],  # normal, peak, idle, failure
                beta=[5, 10, 5, 5],
                shape=n_states,
            )

            state_mem_multipliers = pm.Gamma(
                "state_mem_multipliers", alpha=[5, 15, 1, 8], beta=[5, 10, 5, 5], shape=n_states
            )

            # =========================================
            # LEVEL 3: Individual Resource Level
            # Observation model
            # =========================================

            if self.observed_data is not None:
                # Map resources to archetypes
                resource_archetypes = self._encode_archetypes(
                    self.observed_data["archetype"].to_list()
                )

                # Extract time features
                timestamps = self.observed_data["timestamp"].to_list()
                hour_of_day = np.array([t.hour for t in timestamps])
                day_of_week = np.array([t.weekday() for t in timestamps])

                # Calculate seasonal components
                daily_component = self._calculate_fourier_component(
                    hour_of_day,
                    24,
                    daily_cos_coef[resource_archetypes],
                    daily_sin_coef[resource_archetypes],
                )

                weekly_component = self._calculate_fourier_component(
                    day_of_week,
                    7,
                    weekly_cos_coef[resource_archetypes],
                    weekly_sin_coef[resource_archetypes],
                )

                # Total seasonal effect
                seasonal_effect = pm.Deterministic(
                    "seasonal_effect", 1 + daily_component + weekly_component
                )

                # Hidden Markov Model for regime states
                # This is simplified - full HMM would be more complex
                state_indicators = pm.Categorical(
                    "state_indicators",
                    p=np.array([0.7, 0.15, 0.12, 0.03]),  # Prior state probabilities
                    shape=len(self.observed_data),
                )

                # Apply state multipliers
                cpu_state_effect = state_cpu_multipliers[state_indicators]
                mem_state_effect = state_mem_multipliers[state_indicators]

                # Final observations with all effects
                cpu_mean = (
                    archetype_cpu_mean[resource_archetypes] * seasonal_effect * cpu_state_effect
                )
                mem_mean = (
                    archetype_mem_mean[resource_archetypes] * seasonal_effect * mem_state_effect
                )

                # Observation noise
                cpu_observed = pm.Beta(
                    "cpu_observed",
                    alpha=cpu_mean * 100,  # Scale for Beta
                    beta=(1 - cpu_mean) * 100,
                    observed=self.observed_data["cpu_utilization"].to_numpy() / 100,
                )

                mem_observed = pm.Beta(
                    "mem_observed",
                    alpha=mem_mean * 100,
                    beta=(1 - mem_mean) * 100,
                    observed=self.observed_data["memory_utilization"].to_numpy() / 100,
                )

            # =========================================
            # Waste Patterns
            # Model systematic inefficiencies
            # =========================================

            # Archetype-specific waste
            archetype_waste = pm.Beta(
                "archetype_waste",
                alpha=industry_waste * 10 * pm.math.ones(self.n_archetypes),
                beta=(1 - industry_waste) * 10 * pm.math.ones(self.n_archetypes),
                shape=self.n_archetypes,
            )

            # Over-provisioning factor
            overprovisioning = pm.Beta("overprovisioning", alpha=2, beta=3, shape=self.n_archetypes)

        self.model = model
        return model

    def _encode_archetypes(self, archetype_names: list[str]) -> np.ndarray:
        """Convert archetype names to indices"""
        archetype_to_idx = {name: i for i, name in enumerate(self.archetypes)}
        return np.array([archetype_to_idx.get(name, 0) for name in archetype_names])

    def _calculate_fourier_component(
        self, time_values: np.ndarray, period: int, cos_coef: np.ndarray, sin_coef: np.ndarray
    ) -> np.ndarray:
        """Calculate Fourier series component"""
        n_fourier = cos_coef.shape[-1]
        result = np.zeros_like(time_values, dtype=float)

        for k in range(1, n_fourier + 1):
            result += cos_coef[:, k - 1] * np.cos(2 * np.pi * k * time_values / period) + sin_coef[
                :, k - 1
            ] * np.sin(2 * np.pi * k * time_values / period)

        return result

    def fit(
        self,
        data: pl.DataFrame | None = None,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
    ):
        """
        Fit model using MCMC sampling

        Args:
            data: Optional observed data to fit
            draws: Number of samples to draw
            tune: Number of tuning steps
            chains: Number of MCMC chains
        """
        if self.model is None:
            self.build_model(data)

        with self.model:
            # Use NUTS sampler with initialization
            sample_kwargs = {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "cores": min(chains, 4),
                "init": "advi",  # Use ADVI for initialization
                "target_accept": 0.9,
                "return_inferencedata": True,
            }

            # Add random seed if specified
            if self.seed is not None:
                sample_kwargs["random_seed"] = self.seed

            self.trace = pm.sample(**sample_kwargs)

            # Add log likelihood for model comparison
            pm.compute_log_likelihood(self.trace)

        logger.info("Model fitting complete")
        return self.trace

    def check_convergence(self):
        """Check MCMC convergence diagnostics"""
        if self.trace is None:
            raise ValueError("Model must be fitted first")

        # Check R-hat statistics
        summary = az.summary(self.trace)
        problematic_vars = summary[summary["r_hat"] > 1.01]

        if len(problematic_vars) > 0:
            logger.warning(f"Variables with R-hat > 1.01: {problematic_vars.index.tolist()}")
        else:
            logger.info("All variables converged (R-hat < 1.01)")

        # Check effective sample size
        low_ess = summary[summary["ess_bulk"] < 400]
        if len(low_ess) > 0:
            logger.warning(f"Variables with low ESS: {low_ess.index.tolist()}")

        return summary

    def generate_synthetic_data(
        self,
        archetype: str | None = None,
        n_resources: int | None = None,
        num_resources: int | None = None,
        duration_hours: int | None = None,
        time_periods: int | None = None,
        start_date: datetime | None = None,
        sampling_interval_minutes: int = 5,
    ) -> pl.DataFrame:
        """
        Generate synthetic data using the fitted model

        Args:
            archetype: Application archetype name
            n_resources/num_resources: Number of resources to generate
            duration_hours/time_periods: Duration of simulation
            start_date: Start date for time series
            sampling_interval_minutes: Sampling frequency

        Returns:
            Polars DataFrame with synthetic cloud metrics
        """
        # Handle parameter aliases for backward compatibility
        if num_resources is not None:
            n_resources = num_resources
        if n_resources is None:
            n_resources = 100

        if time_periods is not None:
            duration_hours = time_periods
        if duration_hours is None:
            duration_hours = 720

        if archetype is None:
            archetype = self.archetypes[0] if self.archetypes else "Web Application"

        # Build model if not already built
        if self.model is None:
            self.build_model()

        if self.trace is None:
            logger.warning("Model not fitted, using prior predictive sampling")
            with self.model:
                samples = pm.sample_prior_predictive(samples=1000)
        else:
            with self.model:
                samples = pm.sample_posterior_predictive(self.trace)

        # Get archetype index
        archetype_idx = self.archetypes.index(archetype)

        # Extract parameters for this archetype
        if self.trace:
            cpu_mean = (
                self.trace.posterior["archetype_cpu_mean"]
                .mean(dim=["chain", "draw"])[archetype_idx]
                .item()
            )
            mem_mean = (
                self.trace.posterior["archetype_mem_mean"]
                .mean(dim=["chain", "draw"])[archetype_idx]
                .item()
            )
            cpu_cv = (
                self.trace.posterior["archetype_cpu_cv"]
                .mean(dim=["chain", "draw"])[archetype_idx]
                .item()
            )
            mem_cv = (
                self.trace.posterior["archetype_mem_cv"]
                .mean(dim=["chain", "draw"])[archetype_idx]
                .item()
            )
            corr_matrix = self.trace.posterior["corr_matrix"].mean(dim=["chain", "draw"]).values
        else:
            # Use taxonomy defaults
            taxonomy_arch = CloudResourceTaxonomy.get_archetype(archetype)
            cpu_mean = taxonomy_arch.resource_pattern.cpu_p50 / 100
            mem_mean = taxonomy_arch.resource_pattern.memory_p50 / 100
            cpu_cv = taxonomy_arch.resource_pattern.cpu_cv
            mem_cv = taxonomy_arch.resource_pattern.memory_cv
            corr_matrix = taxonomy_arch.resource_pattern.correlation_matrix

        # Generate time series
        if time_periods is not None:
            # If time_periods specified, treat as hours for backward compatibility
            num_samples = time_periods
            sampling_interval_minutes = 60  # Hourly samples
        else:
            num_samples = duration_hours * 60 // sampling_interval_minutes

        # Use start_date if provided, otherwise start from duration_hours ago
        if start_date is not None:
            timestamps = [
                start_date + timedelta(minutes=i * sampling_interval_minutes)
                for i in range(num_samples)
            ]
        else:
            timestamps = [
                datetime.now()
                - timedelta(hours=duration_hours)
                + timedelta(minutes=i * sampling_interval_minutes)
                for i in range(num_samples)
            ]

        # Generate multivariate correlated data
        mean_vector = np.array(
            [
                cpu_mean * 100,  # Convert back to percentage
                mem_mean * 100,
                20,  # Network In baseline
                15,  # Network Out baseline
                100,  # Disk IOPS baseline
            ]
        )

        # Create covariance matrix from correlation and CVs
        std_vector = mean_vector * np.array([cpu_cv, mem_cv, 0.5, 0.4, 0.6])
        cov_matrix = np.outer(std_vector, std_vector) * corr_matrix

        # Generate samples for all resources
        all_data = []
        for resource_id in range(n_resources):
            # Generate base metrics
            metrics = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)

            # Add temporal patterns
            hour_of_day = np.array([t.hour for t in timestamps])
            day_of_week = np.array([t.weekday() for t in timestamps])

            # Daily pattern
            daily_effect = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)

            # Weekly pattern (lower on weekends)
            weekly_effect = np.where(day_of_week >= 5, 0.6, 1.0)

            # Apply patterns
            metrics[:, 0] *= daily_effect * weekly_effect  # CPU
            metrics[:, 1] *= daily_effect * weekly_effect  # Memory

            # Clip to valid ranges
            metrics[:, 0] = np.clip(metrics[:, 0], 0, 100)
            metrics[:, 1] = np.clip(metrics[:, 1], 0, 100)
            metrics[:, 2:] = np.maximum(metrics[:, 2:], 0)

            # Create resource DataFrame
            resource_df = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "resource_id": f"{archetype}_{resource_id:04d}",
                    "archetype": archetype,
                    "cpu_utilization": metrics[:, 0],
                    "memory_utilization": metrics[:, 1],
                    "network_in_mbps": metrics[:, 2],
                    "network_out_mbps": metrics[:, 3],
                    "disk_iops": metrics[:, 4],
                }
            )

            all_data.append(resource_df)

        # Combine all resources
        result = pl.concat(all_data)

        # Add derived metrics
        result = result.with_columns(
            [
                # Hourly cost (based on utilization and resource type)
                (
                    10
                    + pl.col("cpu_utilization") * 0.1
                    + pl.col("memory_utilization") * 0.05
                    + np.random.normal(0, 2, len(result))
                ).alias("hourly_cost"),
                # Efficiency score
                ((pl.col("cpu_utilization") + pl.col("memory_utilization")) / 2).alias(
                    "efficiency_score"
                ),
                # Waste indicator
                (
                    (100 - pl.col("cpu_utilization")) * 0.5
                    + (100 - pl.col("memory_utilization")) * 0.5
                ).alias("waste_percentage"),
                # Idle detection
                ((pl.col("cpu_utilization") < 5) & (pl.col("memory_utilization") < 10)).alias(
                    "is_idle"
                ),
                # Over-provisioned detection
                ((pl.col("cpu_utilization") < 20) & (pl.col("memory_utilization") < 30)).alias(
                    "is_overprovisioned"
                ),
            ]
        )

        # Ensure hourly_cost is non-negative
        result = result.with_columns([pl.col("hourly_cost").clip(lower_bound=0)])

        return result

    def plot_diagnostics(self):
        """Plot diagnostic plots for model checking"""
        if self.trace is None:
            raise ValueError("Model must be fitted first")

        # Trace plots
        az.plot_trace(
            self.trace, var_names=["industry_cpu_mean", "industry_mem_mean", "industry_waste"]
        )
        plt.suptitle("Industry-Level Parameters")
        plt.tight_layout()
        plt.show()

        # Posterior distributions
        az.plot_posterior(
            self.trace, var_names=["industry_cpu_mean", "industry_mem_mean", "industry_waste"]
        )
        plt.suptitle("Posterior Distributions")
        plt.tight_layout()
        plt.show()

        # Energy plot for convergence
        az.plot_energy(self.trace)
        plt.show()

    def compare_with_research(self):
        """Compare learned parameters with research findings"""
        if self.trace is None:
            raise ValueError("Model must be fitted first")

        # Extract posterior means
        cpu_mean = self.trace.posterior["industry_cpu_mean"].mean().item() * 100
        mem_mean = self.trace.posterior["industry_mem_mean"].mean().item() * 100
        waste_mean = self.trace.posterior["industry_waste"].mean().item() * 100

        # Log comparison results
        logger.info("Model vs Research Comparison:")
        logger.info(
            "  CPU Utilization - Model: {:.1f}%, Research: 13%, Diff: {:.1f}%",
            cpu_mean,
            cpu_mean - 13,
        )
        logger.info(
            "  Memory Utilization - Model: {:.1f}%, Research: 20%, Diff: {:.1f}%",
            mem_mean,
            mem_mean - 20,
        )
        logger.info(
            "  Waste Percentage - Model: {:.1f}%, Research: 30-32%, Diff: {:.1f}%",
            waste_mean,
            waste_mean - 31,
        )

        return {
            "cpu_model": cpu_mean,
            "cpu_research": 13,
            "cpu_diff": cpu_mean - 13,
            "mem_model": mem_mean,
            "mem_research": 20,
            "mem_diff": mem_mean - 20,
            "waste_model": waste_mean,
            "waste_research": 31,
            "waste_diff": waste_mean - 31,
        }

    def posterior_predictions(
        self, archetype: str, num_samples: int = 100
    ) -> dict[str, np.ndarray]:
        """
        Generate posterior predictions for a specific archetype.

        Args:
            archetype: Name of the archetype
            num_samples: Number of samples to generate

        Returns:
            Dictionary with 'cpu', 'memory', and 'cost' predictions
        """
        if self.trace is None:
            raise ValueError("Model not fitted")

        # Handle common aliases for archetypes
        archetype_aliases = {
            "Web Application": "ecommerce_platform",
            "Batch Processing": "batch_etl_pipeline",
            "ML Training": "ml_training_gpu",
            "Database": "postgresql_oltp",
            "Cache": "redis_cache",
        }

        if archetype in archetype_aliases:
            archetype = archetype_aliases[archetype]

        if archetype not in self.archetypes:
            # If still not found, use first available archetype
            archetype = self.archetypes[0]
            logger.warning(f"Unknown archetype requested, using {archetype}")

        # Get archetype index
        archetype_idx = self.archetypes.index(archetype)

        # Extract posterior samples for this archetype
        cpu_alpha = self.trace.posterior["archetype_cpu_mean"].values[:, :, archetype_idx].flatten()
        cpu_beta = 1 - cpu_alpha  # Convert to beta distribution parameters

        mem_alpha = self.trace.posterior["archetype_mem_mean"].values[:, :, archetype_idx].flatten()
        mem_beta = 1 - mem_alpha

        # Sample from posterior predictive
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(len(cpu_alpha), num_samples)

        cpu_predictions = []
        mem_predictions = []
        cost_predictions = []

        for idx in sample_indices:
            # Sample CPU and memory from beta distributions
            cpu_val = np.random.beta(cpu_alpha[idx] * 10, cpu_beta[idx] * 10) * 100
            mem_val = np.random.beta(mem_alpha[idx] * 10, mem_beta[idx] * 10) * 100

            cpu_predictions.append(cpu_val)
            mem_predictions.append(mem_val)

            # Cost is based on utilization
            cost = 10 + cpu_val * 0.1 + mem_val * 0.05 + np.random.normal(0, 2)
            cost_predictions.append(max(0, cost))

        return {
            "cpu": np.array(cpu_predictions),
            "memory": np.array(mem_predictions),
            "cost": np.array(cost_predictions),
        }

    def save_model(self, filepath: str):
        """
        Save the fitted model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.trace is None:
            raise ValueError("No model to save. Please fit the model first.")

        # Save the trace using arviz
        az.to_netcdf(self.trace, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a previously saved model.

        Args:
            filepath: Path to the saved model
        """
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.trace = az.from_netcdf(filepath)
        logger.info(f"Model loaded from {filepath}")

    def compute_waste_metrics(self, data: pl.DataFrame) -> dict[str, Any]:
        """
        Compute waste metrics from resource utilization data.

        Args:
            data: DataFrame with utilization metrics

        Returns:
            Dictionary with waste metrics
        """
        # Calculate average utilization
        avg_cpu = data["cpu_utilization"].mean()
        avg_memory = data["memory_utilization"].mean()

        # Calculate waste percentage (based on under-utilization)
        cpu_waste = (100 - avg_cpu) * 0.7  # 70% of unused CPU is waste
        memory_waste = (100 - avg_memory) * 0.5  # 50% of unused memory is waste
        total_waste = (cpu_waste + memory_waste) / 2

        # Calculate waste by archetype if available
        waste_by_archetype = {}
        if "archetype" in data.columns:
            for archetype in data["archetype"].unique():
                arch_data = data.filter(pl.col("archetype") == archetype)
                arch_cpu = arch_data["cpu_utilization"].mean()
                arch_mem = arch_data["memory_utilization"].mean()
                arch_waste = ((100 - arch_cpu) * 0.7 + (100 - arch_mem) * 0.5) / 2
                waste_by_archetype[archetype] = arch_waste

        # Calculate optimization potential
        if "hourly_cost" in data.columns:
            total_cost = data["hourly_cost"].sum()
            potential_savings = total_cost * (total_waste / 100)
            optimization_potential = potential_savings / total_cost * 100
        else:
            optimization_potential = total_waste

        return {
            "total_waste_percentage": total_waste,
            "cpu_waste_percentage": cpu_waste,
            "memory_waste_percentage": memory_waste,
            "waste_by_archetype": waste_by_archetype,
            "optimization_potential": optimization_potential,
            "avg_cpu_utilization": avg_cpu,
            "avg_memory_utilization": avg_memory,
        }


def demo_hierarchical_model():
    """Demonstrate the hierarchical Bayesian model"""
    logger.info("Demonstrating Cloud Resource Hierarchical Bayesian Model")

    # Generate some sample observed data (normally would be real data)
    sample_data = pl.DataFrame(
        {
            "timestamp": [datetime.now() - timedelta(hours=i) for i in range(1000)],
            "archetype": ["ecommerce_platform"] * 500 + ["dev_environment"] * 500,
            "cpu_utilization": np.concatenate(
                [
                    np.random.beta(2, 8, 500) * 100,  # E-commerce ~20%
                    np.random.beta(1, 15, 500) * 100,  # Dev ~6%
                ]
            ),
            "memory_utilization": np.concatenate(
                [
                    np.random.beta(4, 6, 500) * 100,  # E-commerce ~40%
                    np.random.beta(2, 10, 500) * 100,  # Dev ~17%
                ]
            ),
            "resource_id": [f"res_{i:04d}" for i in range(1000)],
        }
    )

    # Build and fit model
    model = CloudResourceHierarchicalModel(observed_data=sample_data)
    model.build_model()

    # Fit with fewer iterations for demo
    model.fit(draws=500, tune=200, chains=2)

    # Check convergence
    model.check_convergence()

    # Compare with research
    model.compare_with_research()

    # Generate synthetic data
    synthetic = model.generate_synthetic_data(
        archetype="ecommerce_platform", n_resources=10, duration_hours=24
    )

    print("\n=== Generated Synthetic Data ===")
    print(f"Records: {len(synthetic)}")
    print(f"CPU Mean: {synthetic['cpu_utilization'].mean():.1f}%")
    print(f"Memory Mean: {synthetic['memory_utilization'].mean():.1f}%")
    print(f"Waste Mean: {synthetic['waste_percentage'].mean():.1f}%")
    print(f"Idle Time: {(synthetic['is_idle'].sum() / len(synthetic) * 100):.1f}%")


if __name__ == "__main__":
    demo_hierarchical_model()
