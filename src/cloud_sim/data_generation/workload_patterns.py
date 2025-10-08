"""
Realistic Cloud Workload Pattern Generator
Based on research showing actual cloud utilization statistics:
- Average CPU utilization: 13% (shocking!)
- Average memory utilization: 20%
- 30-32% of cloud spending is wasted
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import random
from loguru import logger

class WorkloadType(Enum):
    """Different application workload patterns based on research"""
    WEB_APP = "web_application"
    MICROSERVICE = "microservice"
    BATCH_PROCESSING = "batch_processing"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    DATABASE_OLTP = "database_oltp"
    DATABASE_OLAP = "database_olap"
    STREAMING = "streaming_pipeline"
    SERVERLESS = "serverless_function"
    CACHE = "cache_layer"
    QUEUE = "message_queue"
    DEVELOPMENT = "development_environment"

class WorkloadCharacteristics(BaseModel):
    """Characteristics based on real-world data"""
    base_cpu_util: float = Field(..., ge=0, le=100, description="Average CPU utilization (%)")
    base_mem_util: float = Field(..., ge=0, le=100, description="Average memory utilization (%)")
    cpu_variance: float = Field(..., ge=0, description="Variance in CPU usage")
    mem_variance: float = Field(..., ge=0, description="Variance in memory usage")
    peak_multiplier: float = Field(..., gt=1, description="Peak load multiplier")
    idle_probability: float = Field(..., ge=0, le=1, description="Probability of being idle")
    waste_factor: float = Field(..., ge=0, le=1, description="Resource waste percentage")
    scaling_pattern: Literal["auto", "manual", "none"] = Field(..., description="Scaling strategy")
    seasonal_pattern: bool = Field(..., description="Has seasonal patterns")
    burst_probability: float = Field(..., ge=0, le=1, description="Probability of bursts")

    @field_validator('base_cpu_util', 'base_mem_util')
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError(f"Utilization must be between 0 and 100, got {v}")
        return v

    @field_validator('idle_probability', 'waste_factor', 'burst_probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {v}")
        return v

class WorkloadPatternGenerator:
    """Generate realistic cloud workload patterns based on research data"""

    # Based on research: actual utilization is shockingly low
    WORKLOAD_PROFILES = {
        WorkloadType.WEB_APP: WorkloadCharacteristics(
            base_cpu_util=15,  # Research shows 13% average
            base_mem_util=35,
            cpu_variance=20,
            mem_variance=10,
            peak_multiplier=3.5,
            idle_probability=0.3,  # Often idle at night
            waste_factor=0.35,
            scaling_pattern="auto",
            seasonal_pattern=True,
            burst_probability=0.15
        ),
        WorkloadType.MICROSERVICE: WorkloadCharacteristics(
            base_cpu_util=12,
            base_mem_util=25,
            cpu_variance=25,
            mem_variance=15,
            peak_multiplier=5.0,
            idle_probability=0.4,
            waste_factor=0.45,  # High waste in microservices
            scaling_pattern="auto",
            seasonal_pattern=True,
            burst_probability=0.2
        ),
        WorkloadType.BATCH_PROCESSING: WorkloadCharacteristics(
            base_cpu_util=8,  # Very low when not running
            base_mem_util=15,
            cpu_variance=40,
            mem_variance=30,
            peak_multiplier=10.0,  # Huge spikes during batch runs
            idle_probability=0.7,  # Idle most of the time
            waste_factor=0.6,  # Major waste source
            scaling_pattern="manual",
            seasonal_pattern=False,
            burst_probability=0.05
        ),
        WorkloadType.ML_TRAINING: WorkloadCharacteristics(
            base_cpu_util=25,  # Better utilized but still low
            base_mem_util=40,
            cpu_variance=30,
            mem_variance=20,
            peak_multiplier=4.0,
            idle_probability=0.5,  # Idle between training runs
            waste_factor=0.4,
            scaling_pattern="manual",
            seasonal_pattern=False,
            burst_probability=0.1
        ),
        WorkloadType.ML_INFERENCE: WorkloadCharacteristics(
            base_cpu_util=30,
            base_mem_util=45,
            cpu_variance=15,
            mem_variance=10,
            peak_multiplier=2.0,
            idle_probability=0.2,
            waste_factor=0.25,
            scaling_pattern="auto",
            seasonal_pattern=True,
            burst_probability=0.15
        ),
        WorkloadType.DATABASE_OLTP: WorkloadCharacteristics(
            base_cpu_util=20,
            base_mem_util=60,  # Memory heavy for caching
            cpu_variance=15,
            mem_variance=5,
            peak_multiplier=2.5,
            idle_probability=0.1,
            waste_factor=0.3,
            scaling_pattern="none",
            seasonal_pattern=True,
            burst_probability=0.1
        ),
        WorkloadType.DATABASE_OLAP: WorkloadCharacteristics(
            base_cpu_util=10,
            base_mem_util=30,
            cpu_variance=35,
            mem_variance=25,
            peak_multiplier=8.0,
            idle_probability=0.6,
            waste_factor=0.5,
            scaling_pattern="manual",
            seasonal_pattern=False,
            burst_probability=0.05
        ),
        WorkloadType.STREAMING: WorkloadCharacteristics(
            base_cpu_util=35,
            base_mem_util=40,
            cpu_variance=10,
            mem_variance=8,
            peak_multiplier=1.5,
            idle_probability=0.05,
            waste_factor=0.2,
            scaling_pattern="auto",
            seasonal_pattern=False,
            burst_probability=0.2
        ),
        WorkloadType.SERVERLESS: WorkloadCharacteristics(
            base_cpu_util=5,  # Very low average
            base_mem_util=10,
            cpu_variance=45,
            mem_variance=40,
            peak_multiplier=20.0,  # Extreme spikes
            idle_probability=0.85,  # Mostly idle
            waste_factor=0.1,  # Low waste due to pay-per-use
            scaling_pattern="auto",
            seasonal_pattern=True,
            burst_probability=0.3
        ),
        WorkloadType.CACHE: WorkloadCharacteristics(
            base_cpu_util=10,
            base_mem_util=75,  # Memory intensive
            cpu_variance=5,
            mem_variance=10,
            peak_multiplier=1.2,
            idle_probability=0.0,
            waste_factor=0.25,
            scaling_pattern="none",
            seasonal_pattern=False,
            burst_probability=0.05
        ),
        WorkloadType.QUEUE: WorkloadCharacteristics(
            base_cpu_util=8,
            base_mem_util=20,
            cpu_variance=15,
            mem_variance=10,
            peak_multiplier=3.0,
            idle_probability=0.3,
            waste_factor=0.35,
            scaling_pattern="auto",
            seasonal_pattern=False,
            burst_probability=0.25
        ),
        WorkloadType.DEVELOPMENT: WorkloadCharacteristics(
            base_cpu_util=5,
            base_mem_util=15,
            cpu_variance=30,
            mem_variance=20,
            peak_multiplier=3.0,
            idle_probability=0.8,  # Idle nights/weekends
            waste_factor=0.7,  # Major waste source
            scaling_pattern="none",
            seasonal_pattern=True,
            burst_probability=0.1
        ),
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility"""
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def generate(
        self,
        num_samples: int = 1000,
        workload_type: Optional[WorkloadType] = None,
        **kwargs
    ) -> pl.DataFrame:
        """Generate synthetic data with specified number of samples.

        This is a wrapper for generate_time_series for backward compatibility.
        """
        if workload_type is None:
            workload_type = WorkloadType.WEB_APP

        # Calculate time range based on number of samples
        interval_minutes = kwargs.get('interval_minutes', 5)
        end_time = datetime.now()
        total_minutes = num_samples * interval_minutes
        start_time = end_time - timedelta(minutes=total_minutes)

        return self.generate_time_series(
            workload_type=workload_type,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes
        )

    def generate_time_series(
        self,
        workload_type: WorkloadType,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 5
    ) -> pl.DataFrame:
        """Generate time series data for a specific workload type"""

        profile = self.WORKLOAD_PROFILES[workload_type]

        # Create timestamp array
        # Calculate exact number of points (excluding end_time)
        total_minutes = (end_time - start_time).total_seconds() / 60
        num_points = int(total_minutes / interval_minutes)

        timestamps = []
        current = start_time
        for i in range(num_points):
            timestamps.append(current)
            current += timedelta(minutes=interval_minutes)

        # Generate base patterns
        cpu_utilization = self._generate_utilization_pattern(
            num_points, timestamps, profile, "cpu"
        )
        memory_utilization = self._generate_utilization_pattern(
            num_points, timestamps, profile, "memory"
        )

        # Add correlated patterns
        cpu_utilization, memory_utilization = self._add_correlation(
            cpu_utilization, memory_utilization, workload_type
        )

        # Add waste patterns (over-provisioning)
        waste_indicators = self._generate_waste_patterns(
            cpu_utilization, memory_utilization, profile
        )

        # Generate additional metrics
        network_in_mbps = self._generate_network_pattern(cpu_utilization, "in")
        network_out_mbps = self._generate_network_pattern(cpu_utilization, "out")
        disk_iops = self._generate_disk_pattern(cpu_utilization, workload_type)

        # Calculate efficiency scores
        efficiency_score = self._calculate_efficiency(
            cpu_utilization, memory_utilization, profile
        )

        # Create Polars DataFrame
        return pl.DataFrame({
            "timestamp": timestamps,
            "workload_type": [workload_type.value] * num_points,
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "network_in_mbps": network_in_mbps,
            "network_out_mbps": network_out_mbps,
            "disk_iops": disk_iops,
            "efficiency_score": efficiency_score,
            "is_idle": waste_indicators["is_idle"],
            "is_overprovisioned": waste_indicators["is_overprovisioned"],
            "waste_percentage": waste_indicators["waste_percentage"]
        })

    def _generate_utilization_pattern(
        self,
        num_points: int,
        timestamps: List[datetime],
        profile: WorkloadCharacteristics,
        metric_type: str
    ) -> np.ndarray:
        """Generate utilization pattern based on profile"""

        base_util = profile.base_cpu_util if metric_type == "cpu" else profile.base_mem_util
        variance = profile.cpu_variance if metric_type == "cpu" else profile.mem_variance

        # Start with base utilization
        utilization = np.ones(num_points) * base_util

        # Add daily patterns (business hours)
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            weekday = ts.weekday()

            # Business hours effect (9-5 on weekdays)
            if weekday < 5:  # Monday-Friday
                if 9 <= hour <= 17:
                    utilization[i] *= 1.5
                elif 6 <= hour <= 9 or 17 <= hour <= 20:
                    utilization[i] *= 1.2
                else:
                    utilization[i] *= 0.3  # Night time
            else:  # Weekend
                utilization[i] *= 0.2

        # Add seasonal patterns
        if profile.seasonal_pattern:
            seasonal_wave = np.sin(np.linspace(0, 4*np.pi, num_points))
            utilization += seasonal_wave * variance * 0.5

        # Add random variance
        noise = np.random.normal(0, variance/4, num_points)
        utilization += noise

        # Add burst patterns
        if random.random() < profile.burst_probability:
            burst_locations = random.sample(
                range(num_points),
                k=int(num_points * profile.burst_probability)
            )
            for loc in burst_locations:
                burst_duration = random.randint(5, 20)
                burst_end = min(loc + burst_duration, num_points)
                utilization[loc:burst_end] *= profile.peak_multiplier

        # Add idle periods
        if profile.idle_probability > 0:
            idle_mask = np.random.random(num_points) < profile.idle_probability
            utilization[idle_mask] = np.random.uniform(0, 5, np.sum(idle_mask))

        # Clip to valid range
        return np.clip(utilization, 0, 100)

    def _add_correlation(
        self,
        cpu: np.ndarray,
        memory: np.ndarray,
        workload_type: WorkloadType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add realistic correlation between CPU and memory"""

        # Different workloads have different correlation patterns
        if workload_type in [WorkloadType.ML_TRAINING, WorkloadType.ML_INFERENCE]:
            # Strong correlation for ML workloads
            memory = 0.7 * memory + 0.3 * cpu
        elif workload_type in [WorkloadType.CACHE, WorkloadType.DATABASE_OLTP]:
            # Memory-heavy, less CPU correlation
            cpu = 0.3 * cpu + 0.1 * memory
        elif workload_type == WorkloadType.BATCH_PROCESSING:
            # Alternating high CPU then high memory
            shift = len(cpu) // 10
            memory = np.roll(cpu, shift) * 0.8 + memory * 0.2

        return cpu, np.clip(memory, 0, 100)

    def _generate_waste_patterns(
        self,
        cpu: np.ndarray,
        memory: np.ndarray,
        profile: WorkloadCharacteristics
    ) -> Dict[str, np.ndarray]:
        """Generate waste indicators based on research"""

        # Idle detection (CPU < 5% AND memory < 10%)
        is_idle = (cpu < 5) & (memory < 10)

        # Over-provisioning detection (consistently low usage)
        window = 20  # 20 samples window
        cpu_smooth = np.convolve(cpu, np.ones(window)/window, mode='same')
        mem_smooth = np.convolve(memory, np.ones(window)/window, mode='same')
        is_overprovisioned = (cpu_smooth < 20) & (mem_smooth < 30)

        # Calculate waste percentage
        optimal_cpu = np.maximum(cpu * 1.2, 30)  # 20% headroom or 30% minimum
        optimal_memory = np.maximum(memory * 1.2, 40)  # 20% headroom or 40% minimum

        waste_cpu = np.maximum(0, (100 - optimal_cpu) / 100)
        waste_memory = np.maximum(0, (100 - optimal_memory) / 100)
        waste_percentage = (waste_cpu + waste_memory) / 2 * 100

        return {
            "is_idle": is_idle,
            "is_overprovisioned": is_overprovisioned,
            "waste_percentage": waste_percentage
        }

    def _generate_network_pattern(
        self,
        cpu: np.ndarray,
        direction: str
    ) -> np.ndarray:
        """Generate network traffic patterns"""

        base = 10 if direction == "in" else 5
        # Network correlates with CPU but with some lag
        network = base + cpu * 0.3 + np.random.exponential(2, len(cpu))

        # Add some spikes
        spike_locations = np.random.choice(len(cpu), size=int(len(cpu)*0.05))
        network[spike_locations] *= np.random.uniform(2, 5, len(spike_locations))

        return np.clip(network, 0, 1000)  # Cap at 1Gbps

    def _generate_disk_pattern(
        self,
        cpu: np.ndarray,
        workload_type: WorkloadType
    ) -> np.ndarray:
        """Generate disk I/O patterns"""

        if workload_type in [WorkloadType.DATABASE_OLTP, WorkloadType.DATABASE_OLAP]:
            base_iops = 1000
            variance = 500
        elif workload_type == WorkloadType.BATCH_PROCESSING:
            base_iops = 2000
            variance = 1000
        else:
            base_iops = 100
            variance = 50

        iops = base_iops + cpu * 10 + np.random.normal(0, variance, len(cpu))
        return np.clip(iops, 0, 10000)

    def _calculate_efficiency(
        self,
        cpu: np.ndarray,
        memory: np.ndarray,
        profile: WorkloadCharacteristics
    ) -> np.ndarray:
        """Calculate efficiency score (0-100)"""

        # Efficiency based on how close to optimal utilization
        optimal_cpu = 70  # Target 70% CPU utilization
        optimal_memory = 80  # Target 80% memory utilization

        cpu_efficiency = 100 - np.abs(cpu - optimal_cpu)
        memory_efficiency = 100 - np.abs(memory - optimal_memory)

        # Penalize waste
        waste_penalty = profile.waste_factor * 100

        efficiency = (cpu_efficiency + memory_efficiency) / 2 - waste_penalty
        return np.clip(efficiency, 0, 100)

    def generate_anomalies(
        self,
        df: pl.DataFrame,
        anomaly_types: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Inject realistic anomalies into the data"""

        if anomaly_types is None:
            anomaly_types = ["memory_leak", "cpu_spike", "idle_waste", "scaling_failure"]

        # Work with numpy arrays for easier manipulation
        cpu_values = df["cpu_utilization"].to_numpy().copy()
        mem_values = df["memory_utilization"].to_numpy().copy()
        eff_values = df["efficiency_score"].to_numpy().copy() if "efficiency_score" in df.columns else None

        num_points = len(df)

        for anomaly_type in anomaly_types:
            if anomaly_type == "memory_leak" and num_points > 100:
                # Gradual memory increase over time
                start_idx = random.randint(0, num_points - 100)
                end_idx = min(start_idx + 100, num_points)
                leak_pattern = np.linspace(0, 50, end_idx - start_idx)
                mem_values[start_idx:end_idx] = np.clip(
                    mem_values[start_idx:end_idx] + leak_pattern, 0, 100
                )

            elif anomaly_type == "cpu_spike" and num_points > 10:
                # Sudden CPU spike
                spike_idx = random.randint(0, num_points - 10)
                cpu_values[spike_idx:spike_idx+10] = 95 + np.random.rand(10) * 5

            elif anomaly_type == "idle_waste" and num_points > 50:
                # Extended idle period (common waste pattern)
                idle_start = random.randint(0, num_points - 50)
                idle_end = min(idle_start + 50, num_points)
                cpu_values[idle_start:idle_end] = np.random.rand(idle_end - idle_start) * 5
                mem_values[idle_start:idle_end] = np.random.rand(idle_end - idle_start) * 10

            elif anomaly_type == "scaling_failure" and num_points > 30:
                # Auto-scaling failure pattern
                failure_idx = random.randint(0, num_points - 30)
                # CPU maxes out but memory stays low (scaling didn't trigger)
                cpu_values[failure_idx:failure_idx+30] = 100
                if eff_values is not None:
                    eff_values[failure_idx:failure_idx+30] = 0

        # Create new DataFrame with modified values
        result_df = df.clone()
        result_df = result_df.with_columns([
            pl.Series("cpu_utilization", cpu_values),
            pl.Series("memory_utilization", mem_values),
        ])

        if eff_values is not None:
            result_df = result_df.with_columns([
                pl.Series("efficiency_score", eff_values),
            ])

        return result_df

def create_multi_workload_dataset(
    start_time: datetime,
    end_time: datetime,
    workload_distribution: Optional[Dict[WorkloadType, int]] = None
) -> pl.DataFrame:
    """Create a dataset with multiple workload types"""

    if workload_distribution is None:
        # Default realistic distribution
        workload_distribution = {
            WorkloadType.WEB_APP: 30,
            WorkloadType.MICROSERVICE: 40,
            WorkloadType.DATABASE_OLTP: 10,
            WorkloadType.BATCH_PROCESSING: 5,
            WorkloadType.ML_INFERENCE: 5,
            WorkloadType.CACHE: 5,
            WorkloadType.DEVELOPMENT: 5
        }

    generator = WorkloadPatternGenerator()
    all_workloads = []

    for workload_type, count in workload_distribution.items():
        for i in range(count):
            df = generator.generate_time_series(
                workload_type=workload_type,
                start_time=start_time,
                end_time=end_time,
                interval_minutes=5
            )

            # Add resource identifier
            df = df.with_columns([
                pl.lit(f"{workload_type.value}_{i:03d}").alias("resource_id")
            ])

            # Add anomalies to some resources (20% chance)
            if random.random() < 0.2:
                df = generator.generate_anomalies(df)

            all_workloads.append(df)

    # Combine all workloads
    combined_df = pl.concat(all_workloads)

    # Sort by timestamp and resource
    combined_df = combined_df.sort(["timestamp", "resource_id"])

    return combined_df

def main():
    """Generate sample workload data"""
    logger.info("Generating realistic cloud workload patterns...")

    # Generate 30 days of data
    start_time = datetime.now() - timedelta(days=30)
    end_time = datetime.now()

    # Create multi-workload dataset
    df = create_multi_workload_dataset(start_time, end_time)

    # Save to parquet
    output_path = "data/workload_patterns.parquet"
    df.write_parquet(output_path)

    logger.info(f"Generated {len(df)} records for {df['resource_id'].n_unique()} resources")

    # Calculate statistics
    stats = df.group_by("workload_type").agg([
        pl.col("cpu_utilization").mean().alias("avg_cpu"),
        pl.col("memory_utilization").mean().alias("avg_memory"),
        pl.col("waste_percentage").mean().alias("avg_waste"),
        pl.col("efficiency_score").mean().alias("avg_efficiency"),
        pl.col("is_idle").sum().alias("idle_count"),
        pl.col("is_overprovisioned").sum().alias("overprovisioned_count")
    ])

    print("\n=== Workload Statistics (Matching Real-World Research) ===")
    print(stats)

    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Average CPU Utilization: {df['cpu_utilization'].mean():.1f}% (Research: 13%)")
    print(f"Average Memory Utilization: {df['memory_utilization'].mean():.1f}% (Research: 20%)")
    print(f"Average Waste: {df['waste_percentage'].mean():.1f}% (Research: 30-32%)")
    print(f"Idle Time: {(df['is_idle'].sum() / len(df) * 100):.1f}%")
    print(f"Over-provisioned Time: {(df['is_overprovisioned'].sum() / len(df) * 100):.1f}%")

if __name__ == "__main__":
    main()