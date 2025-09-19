"""
Cloud Metrics Simulator for CloudZero AI Platform
Generates realistic cloud infrastructure metrics with patterns matching real-world usage
"""

import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from loguru import logger

@dataclass
class CloudResource:
    """Represents a cloud resource with usage patterns"""
    resource_id: str
    resource_type: str
    cloud_provider: str
    region: str
    tags: Dict[str, str]
    base_cost_per_hour: float
    cpu_cores: int
    memory_gb: int

class CloudMetricsSimulator:
    """Simulates cloud infrastructure metrics for multiple providers"""

    RESOURCE_TYPES = {
        'compute': ['t3.micro', 't3.small', 't3.medium', 't3.large', 'm5.large', 'm5.xlarge', 'c5.2xlarge'],
        'storage': ['gp3', 'io1', 'st1', 'sc1'],
        'database': ['db.t3.micro', 'db.m5.large', 'db.r5.xlarge'],
        'container': ['fargate-small', 'fargate-medium', 'fargate-large'],
        'ai_ml': ['ml.p3.2xlarge', 'ml.g4dn.xlarge', 'ml.c5.4xlarge']
    }

    CLOUD_PROVIDERS = ['AWS', 'Azure', 'GCP']
    REGIONS = {
        'AWS': ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
        'Azure': ['eastus', 'westus', 'northeurope', 'southeastasia'],
        'GCP': ['us-central1', 'us-east1', 'europe-west1', 'asia-east1']
    }

    def __init__(self,
                 start_date: datetime = None,
                 end_date: datetime = None,
                 num_resources: int = 100,
                 sampling_interval_minutes: int = 60):
        """
        Initialize the simulator

        Args:
            start_date: Start of simulation period
            end_date: End of simulation period
            num_resources: Number of resources to simulate
            sampling_interval_minutes: Data sampling frequency
        """
        self.start_date = start_date or datetime.now() - timedelta(days=30)
        self.end_date = end_date or datetime.now()
        self.num_resources = num_resources
        self.sampling_interval = timedelta(minutes=sampling_interval_minutes)
        self.resources = self._generate_resources()

        logger.info(f"Initialized simulator with {num_resources} resources from {self.start_date} to {self.end_date}")

    def _generate_resources(self) -> List[CloudResource]:
        """Generate a diverse set of cloud resources"""
        resources = []

        for i in range(self.num_resources):
            provider = random.choice(self.CLOUD_PROVIDERS)
            resource_category = random.choice(list(self.RESOURCE_TYPES.keys()))
            resource_type = random.choice(self.RESOURCE_TYPES[resource_category])

            # Generate resource metadata
            resource = CloudResource(
                resource_id=f"{provider.lower()}-{resource_category}-{i:04d}",
                resource_type=resource_type,
                cloud_provider=provider,
                region=random.choice(self.REGIONS[provider]),
                tags=self._generate_tags(resource_category),
                base_cost_per_hour=self._calculate_base_cost(resource_type, provider),
                cpu_cores=self._get_cpu_cores(resource_type),
                memory_gb=self._get_memory_gb(resource_type)
            )
            resources.append(resource)

        return resources

    def _generate_tags(self, resource_category: str) -> Dict[str, str]:
        """Generate realistic resource tags with some missing (simulating real-world tagging issues)"""
        teams = ['platform', 'data', 'ml', 'api', 'frontend', 'backend']
        environments = ['prod', 'staging', 'dev', 'test']
        products = ['core', 'analytics', 'reporting', 'customer-portal', 'admin']
        customers = [f"customer-{i:03d}" for i in range(1, 21)]

        # Simulate imperfect tagging (30% chance of missing tags)
        tags = {}

        if random.random() > 0.3:
            tags['team'] = random.choice(teams)
        if random.random() > 0.3:
            tags['environment'] = random.choice(environments)
        if random.random() > 0.3:
            tags['product'] = random.choice(products)
        if resource_category in ['compute', 'container'] and random.random() > 0.5:
            tags['customer_id'] = random.choice(customers)

        tags['cost_center'] = f"cc-{random.randint(100, 999)}"

        return tags

    def _calculate_base_cost(self, resource_type: str, provider: str) -> float:
        """Calculate base hourly cost for a resource type"""
        # Simplified cost model (real costs would come from pricing APIs)
        base_costs = {
            't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
            't3.large': 0.0832, 'm5.large': 0.096, 'm5.xlarge': 0.192,
            'c5.2xlarge': 0.34, 'gp3': 0.08, 'io1': 0.125,
            'db.t3.micro': 0.017, 'db.m5.large': 0.171, 'db.r5.xlarge': 0.476,
            'fargate-small': 0.04, 'fargate-medium': 0.08, 'fargate-large': 0.16,
            'ml.p3.2xlarge': 3.06, 'ml.g4dn.xlarge': 0.526, 'ml.c5.4xlarge': 0.68
        }

        # Add provider pricing variations
        provider_multipliers = {'AWS': 1.0, 'Azure': 1.05, 'GCP': 0.95}

        return base_costs.get(resource_type, 0.1) * provider_multipliers[provider]

    def _get_cpu_cores(self, resource_type: str) -> int:
        """Get CPU core count for resource type"""
        cpu_counts = {
            't3.micro': 2, 't3.small': 2, 't3.medium': 2, 't3.large': 2,
            'm5.large': 2, 'm5.xlarge': 4, 'c5.2xlarge': 8,
            'db.t3.micro': 2, 'db.m5.large': 2, 'db.r5.xlarge': 4,
            'fargate-small': 0.25, 'fargate-medium': 0.5, 'fargate-large': 1,
            'ml.p3.2xlarge': 8, 'ml.g4dn.xlarge': 4, 'ml.c5.4xlarge': 16
        }
        return cpu_counts.get(resource_type, 2)

    def _get_memory_gb(self, resource_type: str) -> int:
        """Get memory in GB for resource type"""
        memory_gb = {
            't3.micro': 1, 't3.small': 2, 't3.medium': 4, 't3.large': 8,
            'm5.large': 8, 'm5.xlarge': 16, 'c5.2xlarge': 16,
            'db.t3.micro': 1, 'db.m5.large': 8, 'db.r5.xlarge': 32,
            'fargate-small': 0.5, 'fargate-medium': 1, 'fargate-large': 2,
            'ml.p3.2xlarge': 61, 'ml.g4dn.xlarge': 16, 'ml.c5.4xlarge': 32
        }
        return memory_gb.get(resource_type, 4)

    def generate_usage_patterns(self, resource: CloudResource, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic usage patterns for a resource"""
        num_points = len(timestamps)

        # Base utilization patterns
        if resource.resource_type.startswith('db'):
            # Database: steady with periodic spikes
            base_cpu = 30 + 10 * np.random.randn(num_points)
            base_memory = 60 + 5 * np.random.randn(num_points)
        elif resource.resource_type.startswith('ml'):
            # ML workloads: batch processing patterns
            base_cpu = 20 + 60 * (np.sin(np.arange(num_points) * 0.1) > 0.5)
            base_memory = 40 + 40 * (np.sin(np.arange(num_points) * 0.1) > 0.5)
        elif 'fargate' in resource.resource_type:
            # Container: variable with scaling
            base_cpu = 45 + 15 * np.sin(np.arange(num_points) * 0.05)
            base_memory = 50 + 10 * np.sin(np.arange(num_points) * 0.05)
        else:
            # Compute: daily patterns with business hours
            hour_of_day = np.array([t.hour for t in timestamps])
            business_hours = ((hour_of_day >= 8) & (hour_of_day <= 18)).astype(float)
            base_cpu = 20 + 40 * business_hours + 10 * np.random.randn(num_points)
            base_memory = 40 + 20 * business_hours + 5 * np.random.randn(num_points)

        # Add weekly patterns (lower on weekends)
        day_of_week = np.array([t.weekday() for t in timestamps])
        weekend_factor = 0.6 * (day_of_week >= 5).astype(float)
        base_cpu *= (1 - weekend_factor)
        base_memory *= (1 - weekend_factor)

        # Add noise and ensure bounds
        cpu_utilization = np.clip(base_cpu + 5 * np.random.randn(num_points), 0, 100)
        memory_utilization = np.clip(base_memory + 3 * np.random.randn(num_points), 0, 100)

        # Calculate costs with utilization-based pricing (for burstable instances)
        effective_cost = resource.base_cost_per_hour * (1 + 0.2 * (cpu_utilization > 80).astype(float))

        # Simulate network and storage
        network_in_gb = np.abs(np.random.gamma(2, 2, num_points))
        network_out_gb = np.abs(np.random.gamma(2, 1, num_points))
        storage_gb = 100 + 10 * np.cumsum(np.random.randn(num_points) * 0.1)

        return pd.DataFrame({
            'timestamp': timestamps,
            'resource_id': resource.resource_id,
            'resource_type': resource.resource_type,
            'cloud_provider': resource.cloud_provider,
            'region': resource.region,
            'cpu_utilization': cpu_utilization,
            'memory_utilization': memory_utilization,
            'cpu_cores': resource.cpu_cores,
            'memory_gb': resource.memory_gb,
            'network_in_gb': network_in_gb,
            'network_out_gb': network_out_gb,
            'storage_gb': storage_gb,
            'hourly_cost': effective_cost,
            **{f'tag_{k}': v for k, v in resource.tags.items()}
        })

    def inject_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.02) -> pd.DataFrame:
        """Inject realistic anomalies into the data"""
        df = df.copy()
        num_points = len(df)
        num_anomalies = int(num_points * anomaly_rate)

        anomaly_indices = random.sample(range(num_points), num_anomalies)

        for idx in anomaly_indices:
            anomaly_type = random.choice(['spike', 'drop', 'gradual', 'resource_leak'])

            if anomaly_type == 'spike':
                # Sudden cost spike (2-5x normal)
                df.loc[idx:min(idx+3, num_points-1), 'hourly_cost'] *= random.uniform(2, 5)
                df.loc[idx:min(idx+3, num_points-1), 'cpu_utilization'] = 95 + 5 * random.random()
            elif anomaly_type == 'drop':
                # Sudden drop (potential waste)
                df.loc[idx:min(idx+5, num_points-1), 'cpu_utilization'] *= 0.1
                df.loc[idx:min(idx+5, num_points-1), 'memory_utilization'] *= 0.1
            elif anomaly_type == 'gradual':
                # Gradual increase (memory leak pattern)
                if idx < num_points - 20:
                    increase = np.linspace(1, 2, 20)
                    df.loc[idx:idx+19, 'memory_utilization'] *= increase
                    df.loc[idx:idx+19, 'hourly_cost'] *= increase
            else:  # resource_leak
                # Storage continuously growing
                if idx < num_points - 10:
                    df.loc[idx:idx+9, 'storage_gb'] *= np.linspace(1, 3, 10)

        # Mark anomalies
        df['is_anomaly'] = False
        df.loc[anomaly_indices, 'is_anomaly'] = True

        return df

    def generate_dataset(self, include_anomalies: bool = True) -> pl.DataFrame:
        """Generate complete dataset with all resources"""
        timestamps = pd.date_range(self.start_date, self.end_date, freq=self.sampling_interval)
        all_data = []

        for resource in self.resources:
            resource_data = self.generate_usage_patterns(resource, timestamps)
            if include_anomalies:
                resource_data = self.inject_anomalies(resource_data)
            all_data.append(resource_data)

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Add derived metrics
        combined_df['cost_per_cpu'] = combined_df['hourly_cost'] / (combined_df['cpu_cores'] + 0.001)
        combined_df['cost_per_gb'] = combined_df['hourly_cost'] / (combined_df['memory_gb'] + 0.001)
        combined_df['efficiency_score'] = (combined_df['cpu_utilization'] + combined_df['memory_utilization']) / 2

        # Convert to Polars for better performance
        return pl.from_pandas(combined_df)

    def calculate_unit_economics(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Calculate unit economics metrics (cost per customer, feature, etc.)"""
        results = {}

        # Cost per customer
        if 'tag_customer_id' in df.columns:
            customer_costs = df.group_by(['tag_customer_id', 'timestamp']).agg([
                pl.col('hourly_cost').sum().alias('total_cost'),
                pl.col('resource_id').n_unique().alias('resource_count'),
                pl.col('cpu_utilization').mean().alias('avg_cpu_utilization'),
                pl.col('memory_utilization').mean().alias('avg_memory_utilization')
            ])
            results['cost_per_customer'] = customer_costs

        # Cost per team
        if 'tag_team' in df.columns:
            team_costs = df.group_by(['tag_team', 'timestamp']).agg([
                pl.col('hourly_cost').sum().alias('total_cost'),
                pl.col('resource_id').n_unique().alias('resource_count')
            ])
            results['cost_per_team'] = team_costs

        # Cost per environment
        if 'tag_environment' in df.columns:
            env_costs = df.group_by(['tag_environment', 'timestamp']).agg([
                pl.col('hourly_cost').sum().alias('total_cost'),
                pl.col('efficiency_score').mean().alias('avg_efficiency')
            ])
            results['cost_per_environment'] = env_costs

        # Waste analysis (low utilization with high cost)
        waste_threshold_cpu = 20
        waste_threshold_mem = 30
        waste_df = df.filter(
            (pl.col('cpu_utilization') < waste_threshold_cpu) &
            (pl.col('memory_utilization') < waste_threshold_mem)
        )

        waste_summary = waste_df.group_by('resource_type').agg([
            pl.col('hourly_cost').sum().alias('wasted_cost'),
            pl.col('resource_id').n_unique().alias('wasted_resources')
        ])
        results['waste_analysis'] = waste_summary

        return results

def main():
    """Generate sample cloud metrics data"""
    logger.info("Starting cloud metrics simulation...")

    # Initialize simulator
    simulator = CloudMetricsSimulator(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        num_resources=200,
        sampling_interval_minutes=60
    )

    # Generate dataset
    df = simulator.generate_dataset(include_anomalies=True)

    # Save to parquet for efficient storage
    output_path = "data/cloud_metrics.parquet"
    df.write_parquet(output_path)
    logger.info(f"Saved {len(df)} records to {output_path}")

    # Calculate unit economics
    unit_economics = simulator.calculate_unit_economics(df)

    # Save unit economics
    for metric_name, metric_df in unit_economics.items():
        metric_path = f"data/unit_economics_{metric_name}.parquet"
        metric_df.write_parquet(metric_path)
        logger.info(f"Saved {metric_name} to {metric_path}")

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique resources: {df['resource_id'].n_unique()}")
    print(f"Total cost: ${df['hourly_cost'].sum():,.2f}")
    print(f"Anomalies injected: {df['is_anomaly'].sum()}")

    # Print waste analysis
    if 'waste_analysis' in unit_economics:
        waste_df = unit_economics['waste_analysis']
        total_waste = waste_df['wasted_cost'].sum()
        print(f"\n=== Waste Analysis ===")
        print(f"Total wasted cost: ${total_waste:,.2f}")
        print(f"Waste percentage: {100 * total_waste / df['hourly_cost'].sum():.1f}%")

if __name__ == "__main__":
    main()