"""
CloudZero Application Taxonomy System
A comprehensive, hierarchical classification of cloud workloads
based on resource consumption patterns, business criticality, and optimization potential

This could become the industry standard for workload classification
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pymc as pm
import polars as pl
from datetime import datetime, timedelta

class ApplicationDomain(Enum):
    """Top-level business domains"""
    CUSTOMER_FACING = "customer_facing"      # Revenue generating
    DATA_PROCESSING = "data_processing"      # Analytics & ETL
    MACHINE_LEARNING = "machine_learning"    # AI/ML workloads
    INFRASTRUCTURE = "infrastructure"        # Supporting services
    DEVELOPMENT = "development"              # Non-production

class ScalingBehavior(Enum):
    """How the application scales"""
    ELASTIC_AUTO = "elastic_auto"            # Auto-scaling
    ELASTIC_MANUAL = "elastic_manual"        # Manual scaling
    STATIC = "static"                         # Fixed resources
    SERVERLESS = "serverless"                # Function-based
    SCHEDULED = "scheduled"                  # Time-based

class OptimizationPotential(Enum):
    """Potential for cost optimization"""
    HIGH = "high"           # >40% potential savings
    MEDIUM = "medium"       # 20-40% potential savings
    LOW = "low"             # <20% potential savings
    OPTIMIZED = "optimized" # Already optimized

@dataclass
class ResourcePattern:
    """Empirically-derived resource consumption pattern"""
    # Base utilization (from research: 13% CPU, 20% memory average)
    cpu_p50: float          # Median CPU utilization
    cpu_p95: float          # 95th percentile CPU
    memory_p50: float       # Median memory utilization
    memory_p95: float       # 95th percentile memory

    # Variability metrics
    cpu_cv: float           # Coefficient of variation for CPU
    memory_cv: float        # Coefficient of variation for memory

    # Correlation matrix (5x5: CPU, Memory, Network In/Out, Disk)
    correlation_matrix: np.ndarray

    # Temporal patterns
    daily_pattern_type: str     # 'business_hours', 'constant', 'batch', 'irregular'
    weekly_pattern_type: str    # 'weekday_heavy', 'constant', 'weekend_heavy'
    seasonality_strength: float # 0-1, strength of seasonal patterns

    # Burst characteristics
    burst_frequency: float       # Bursts per day
    burst_amplitude: float       # Multiplier during bursts
    burst_duration_minutes: int  # Typical burst duration

@dataclass
class CostProfile:
    """Financial characteristics"""
    avg_hourly_cost: float           # Average hourly cost
    cost_variability: float          # CV of cost
    waste_percentage: float          # From research: 30-32% average
    optimization_difficulty: float   # 0-1, how hard to optimize
    business_criticality: float     # 0-1, impacts over-provisioning

@dataclass
class ApplicationArchetype:
    """Complete application profile"""
    # Identity
    name: str
    domain: ApplicationDomain
    description: str

    # Resource behavior
    resource_pattern: ResourcePattern
    scaling_behavior: ScalingBehavior
    optimization_potential: OptimizationPotential

    # Financial
    cost_profile: CostProfile

    # Technical characteristics
    typical_stack: List[str]           # Technologies used
    cloud_services: List[str]          # AWS/Azure/GCP services
    typical_instance_types: List[str]  # Common instance types

    # Operational patterns
    deployment_frequency: str          # 'continuous', 'daily', 'weekly'
    maintenance_windows: List[Tuple[int, int]]  # [(day, hour)]

    # SLA requirements
    availability_target: float         # 99.9%, 99.99%, etc.
    latency_p99_ms: float              # 99th percentile latency

    # Data characteristics
    data_volume_gb_per_day: float
    data_retention_days: int

    # Example companies/use cases
    example_companies: List[str] = field(default_factory=list)
    market_size_percentage: float = 0.0  # % of cloud market

class CloudZeroTaxonomy:
    """
    The complete CloudZero Application Taxonomy
    Based on empirical research and real-world patterns
    """

    ARCHETYPES = {
        # === CUSTOMER-FACING APPLICATIONS ===

        "ecommerce_platform": ApplicationArchetype(
            name="E-Commerce Platform",
            domain=ApplicationDomain.CUSTOMER_FACING,
            description="Online retail platforms with shopping cart, payment processing",
            resource_pattern=ResourcePattern(
                cpu_p50=18.0,  # Slightly above average
                cpu_p95=75.0,  # Black Friday spikes
                memory_p50=45.0,
                memory_p95=70.0,
                cpu_cv=0.65,  # High variability
                memory_cv=0.35,
                correlation_matrix=np.array([
                    [1.00, 0.65, 0.78, 0.82, 0.45],  # CPU correlates with network
                    [0.65, 1.00, 0.42, 0.38, 0.35],  # Memory for session state
                    [0.78, 0.42, 1.00, 0.72, 0.30],  # Network In (browsing)
                    [0.82, 0.38, 0.72, 1.00, 0.32],  # Network Out (images)
                    [0.45, 0.35, 0.30, 0.32, 1.00],  # Disk (logs, cache)
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',
                seasonality_strength=0.8,  # Holiday shopping
                burst_frequency=5.0,
                burst_amplitude=3.5,
                burst_duration_minutes=30
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.HIGH,
            cost_profile=CostProfile(
                avg_hourly_cost=450.0,
                cost_variability=0.6,
                waste_percentage=35.0,  # Over-provisioned for peaks
                optimization_difficulty=0.4,
                business_criticality=1.0  # Revenue critical
            ),
            typical_stack=["React", "Node.js", "PostgreSQL", "Redis", "Elasticsearch"],
            cloud_services=["EC2", "RDS", "ElastiCache", "CloudFront", "S3"],
            typical_instance_types=["m5.xlarge", "m5.2xlarge", "r5.xlarge"],
            deployment_frequency='continuous',
            maintenance_windows=[(6, 3), (0, 3)],  # Sunday 3am
            availability_target=99.95,
            latency_p99_ms=200.0,
            data_volume_gb_per_day=500.0,
            data_retention_days=90,
            example_companies=["Shopify", "Amazon", "eBay", "Etsy"],
            market_size_percentage=15.0
        ),

        "video_streaming": ApplicationArchetype(
            name="Video Streaming Service",
            domain=ApplicationDomain.CUSTOMER_FACING,
            description="Video content delivery and streaming platforms",
            resource_pattern=ResourcePattern(
                cpu_p50=35.0,  # Encoding/transcoding
                cpu_p95=85.0,
                memory_p50=60.0,  # Buffering
                memory_p95=80.0,
                cpu_cv=0.45,
                memory_cv=0.25,
                correlation_matrix=np.array([
                    [1.00, 0.70, 0.85, 0.92, 0.35],  # CPU drives network out
                    [0.70, 1.00, 0.55, 0.60, 0.30],
                    [0.85, 0.55, 1.00, 0.78, 0.25],
                    [0.92, 0.60, 0.78, 1.00, 0.28],  # Heavy network out
                    [0.35, 0.30, 0.25, 0.28, 1.00],
                ]),
                daily_pattern_type='constant',  # 24/7 viewing
                weekly_pattern_type='weekend_heavy',
                seasonality_strength=0.3,
                burst_frequency=2.0,  # New releases
                burst_amplitude=2.5,
                burst_duration_minutes=120
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.MEDIUM,
            cost_profile=CostProfile(
                avg_hourly_cost=2800.0,  # CDN costs
                cost_variability=0.4,
                waste_percentage=25.0,
                optimization_difficulty=0.6,  # CDN optimization is complex
                business_criticality=0.95
            ),
            typical_stack=["FFmpeg", "HLS", "DASH", "Kubernetes", "Kafka"],
            cloud_services=["CloudFront", "S3", "MediaConvert", "EC2 GPU"],
            typical_instance_types=["g4dn.xlarge", "c5n.large", "m5.xlarge"],
            deployment_frequency='daily',
            maintenance_windows=[(2, 4)],  # Tuesday 4am
            availability_target=99.99,
            latency_p99_ms=100.0,  # Buffer time
            data_volume_gb_per_day=50000.0,  # Massive
            data_retention_days=365,
            example_companies=["Netflix", "YouTube", "Disney+", "Twitch"],
            market_size_percentage=8.0
        ),

        # === DATA PROCESSING APPLICATIONS ===

        "batch_etl_pipeline": ApplicationArchetype(
            name="Batch ETL Pipeline",
            domain=ApplicationDomain.DATA_PROCESSING,
            description="Scheduled data extraction, transformation, and loading jobs",
            resource_pattern=ResourcePattern(
                cpu_p50=8.0,   # Mostly idle (shocking but true)
                cpu_p95=92.0,  # Maxed during batch runs
                memory_p50=15.0,
                memory_p95=85.0,
                cpu_cv=1.2,   # Extreme variability
                memory_cv=0.9,
                correlation_matrix=np.array([
                    [1.00, 0.85, 0.70, 0.65, 0.90],  # CPU and disk correlation
                    [0.85, 1.00, 0.60, 0.55, 0.80],
                    [0.70, 0.60, 1.00, 0.88, 0.65],
                    [0.65, 0.55, 0.88, 1.00, 0.60],
                    [0.90, 0.80, 0.65, 0.60, 1.00],  # Heavy disk I/O
                ]),
                daily_pattern_type='batch',  # Scheduled runs
                weekly_pattern_type='constant',
                seasonality_strength=0.1,
                burst_frequency=4.0,  # 4 runs per day
                burst_amplitude=10.0,  # 10x spike
                burst_duration_minutes=45
            ),
            scaling_behavior=ScalingBehavior.SCHEDULED,
            optimization_potential=OptimizationPotential.HIGH,  # Major waste
            cost_profile=CostProfile(
                avg_hourly_cost=180.0,
                cost_variability=1.1,
                waste_percentage=60.0,  # Worst offender!
                optimization_difficulty=0.2,  # Easy to fix
                business_criticality=0.6
            ),
            typical_stack=["Apache Spark", "Airflow", "Python", "Parquet"],
            cloud_services=["EMR", "Glue", "S3", "Step Functions"],
            typical_instance_types=["r5.4xlarge", "m5.2xlarge"],
            deployment_frequency='weekly',
            maintenance_windows=[(0, 2)],  # Sunday 2am
            availability_target=99.0,
            latency_p99_ms=60000.0,  # Minutes acceptable
            data_volume_gb_per_day=5000.0,
            data_retention_days=30,
            example_companies=["Most enterprises"],
            market_size_percentage=12.0
        ),

        "realtime_streaming": ApplicationArchetype(
            name="Real-time Stream Processing",
            domain=ApplicationDomain.DATA_PROCESSING,
            description="Continuous data stream processing (Kafka, Kinesis)",
            resource_pattern=ResourcePattern(
                cpu_p50=45.0,  # Steady state processing
                cpu_p95=70.0,
                memory_p50=55.0,
                memory_p95=75.0,
                cpu_cv=0.3,  # Low variability
                memory_cv=0.25,
                correlation_matrix=np.array([
                    [1.00, 0.75, 0.88, 0.85, 0.50],
                    [0.75, 1.00, 0.65, 0.62, 0.45],
                    [0.88, 0.65, 1.00, 0.90, 0.40],  # Network heavy
                    [0.85, 0.62, 0.90, 1.00, 0.38],
                    [0.50, 0.45, 0.40, 0.38, 1.00],
                ]),
                daily_pattern_type='constant',
                weekly_pattern_type='constant',
                seasonality_strength=0.2,
                burst_frequency=8.0,  # Periodic spikes
                burst_amplitude=1.5,
                burst_duration_minutes=10
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.MEDIUM,
            cost_profile=CostProfile(
                avg_hourly_cost=320.0,
                cost_variability=0.3,
                waste_percentage=28.0,
                optimization_difficulty=0.5,
                business_criticality=0.8
            ),
            typical_stack=["Kafka", "Flink", "Storm", "Cassandra"],
            cloud_services=["Kinesis", "MSK", "Lambda", "DynamoDB"],
            typical_instance_types=["m5.xlarge", "c5.2xlarge"],
            deployment_frequency='continuous',
            maintenance_windows=[],  # No maintenance
            availability_target=99.95,
            latency_p99_ms=50.0,
            data_volume_gb_per_day=2000.0,
            data_retention_days=7,
            example_companies=["Uber", "LinkedIn", "Twitter"],
            market_size_percentage=6.0
        ),

        # === MACHINE LEARNING WORKLOADS ===

        "ml_training_gpu": ApplicationArchetype(
            name="ML Model Training (GPU)",
            domain=ApplicationDomain.MACHINE_LEARNING,
            description="Deep learning model training on GPU clusters",
            resource_pattern=ResourcePattern(
                cpu_p50=25.0,  # Better utilized than most
                cpu_p95=95.0,
                memory_p50=70.0,  # Model parameters
                memory_p95=92.0,
                cpu_cv=0.8,
                memory_cv=0.4,
                correlation_matrix=np.array([
                    [1.00, 0.95, 0.50, 0.45, 0.88],  # GPU/CPU/Memory move together
                    [0.95, 1.00, 0.48, 0.42, 0.85],
                    [0.50, 0.48, 1.00, 0.75, 0.55],  # Data loading
                    [0.45, 0.42, 0.75, 1.00, 0.50],
                    [0.88, 0.85, 0.55, 0.50, 1.00],  # Checkpointing
                ]),
                daily_pattern_type='batch',
                weekly_pattern_type='constant',
                seasonality_strength=0.1,
                burst_frequency=2.0,  # Training runs
                burst_amplitude=4.0,
                burst_duration_minutes=360  # Hours of training
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_MANUAL,
            optimization_potential=OptimizationPotential.HIGH,  # Idle GPUs
            cost_profile=CostProfile(
                avg_hourly_cost=850.0,  # GPU expensive
                cost_variability=0.9,
                waste_percentage=45.0,  # Idle between experiments
                optimization_difficulty=0.4,
                business_criticality=0.5
            ),
            typical_stack=["PyTorch", "TensorFlow", "CUDA", "Horovod"],
            cloud_services=["SageMaker", "EC2 P3", "EFS", "S3"],
            typical_instance_types=["p3.2xlarge", "p3.8xlarge", "g4dn.12xlarge"],
            deployment_frequency='daily',
            maintenance_windows=[],
            availability_target=99.0,
            latency_p99_ms=0.0,  # Not applicable
            data_volume_gb_per_day=1000.0,
            data_retention_days=90,
            example_companies=["OpenAI", "Anthropic", "Meta AI"],
            market_size_percentage=4.0
        ),

        "ml_inference_realtime": ApplicationArchetype(
            name="ML Model Inference (Real-time)",
            domain=ApplicationDomain.MACHINE_LEARNING,
            description="Low-latency model serving for production predictions",
            resource_pattern=ResourcePattern(
                cpu_p50=40.0,  # Better utilized
                cpu_p95=75.0,
                memory_p50=60.0,  # Model in memory
                memory_p95=70.0,
                cpu_cv=0.35,
                memory_cv=0.15,  # Stable memory
                correlation_matrix=np.array([
                    [1.00, 0.60, 0.82, 0.85, 0.30],
                    [0.60, 1.00, 0.45, 0.42, 0.25],
                    [0.82, 0.45, 1.00, 0.88, 0.20],
                    [0.85, 0.42, 0.88, 1.00, 0.22],
                    [0.30, 0.25, 0.20, 0.22, 1.00],
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',
                seasonality_strength=0.3,
                burst_frequency=10.0,
                burst_amplitude=2.0,
                burst_duration_minutes=5
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.MEDIUM,
            cost_profile=CostProfile(
                avg_hourly_cost=280.0,
                cost_variability=0.4,
                waste_percentage=30.0,
                optimization_difficulty=0.5,
                business_criticality=0.85
            ),
            typical_stack=["TensorFlow Serving", "TorchServe", "Triton", "Redis"],
            cloud_services=["SageMaker Endpoints", "EC2", "ELB", "CloudWatch"],
            typical_instance_types=["g4dn.xlarge", "c5.4xlarge", "inf1.xlarge"],
            deployment_frequency='weekly',
            maintenance_windows=[(0, 4)],
            availability_target=99.9,
            latency_p99_ms=100.0,
            data_volume_gb_per_day=100.0,
            data_retention_days=30,
            example_companies=["Google", "Facebook", "Amazon"],
            market_size_percentage=3.0
        ),

        # === INFRASTRUCTURE SERVICES ===

        "postgresql_oltp": ApplicationArchetype(
            name="PostgreSQL OLTP Database",
            domain=ApplicationDomain.INFRASTRUCTURE,
            description="Transactional database for operational data",
            resource_pattern=ResourcePattern(
                cpu_p50=22.0,  # Research validated
                cpu_p95=65.0,
                memory_p50=70.0,  # Buffer pool
                memory_p95=85.0,
                cpu_cv=0.4,
                memory_cv=0.15,  # Memory stable
                correlation_matrix=np.array([
                    [1.00, 0.35, 0.65, 0.68, 0.92],  # CPU correlates with IOPS
                    [0.35, 1.00, 0.25, 0.22, 0.30],  # Memory independent
                    [0.65, 0.25, 1.00, 0.85, 0.60],
                    [0.68, 0.22, 0.85, 1.00, 0.62],
                    [0.92, 0.30, 0.60, 0.62, 1.00],  # Disk I/O drives CPU
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',
                seasonality_strength=0.4,
                burst_frequency=20.0,  # Query spikes
                burst_amplitude=3.0,
                burst_duration_minutes=2
            ),
            scaling_behavior=ScalingBehavior.STATIC,  # Vertical scaling
            optimization_potential=OptimizationPotential.MEDIUM,
            cost_profile=CostProfile(
                avg_hourly_cost=380.0,
                cost_variability=0.3,
                waste_percentage=32.0,
                optimization_difficulty=0.7,  # Hard to right-size
                business_criticality=0.95
            ),
            typical_stack=["PostgreSQL", "PgBouncer", "Patroni", "WAL-E"],
            cloud_services=["RDS", "EBS", "Aurora PostgreSQL"],
            typical_instance_types=["db.r5.2xlarge", "db.m5.4xlarge"],
            deployment_frequency='monthly',
            maintenance_windows=[(0, 3)],
            availability_target=99.95,
            latency_p99_ms=50.0,
            data_volume_gb_per_day=200.0,
            data_retention_days=365,
            example_companies=["Every company"],
            market_size_percentage=10.0
        ),

        "redis_cache": ApplicationArchetype(
            name="Redis Cache Layer",
            domain=ApplicationDomain.INFRASTRUCTURE,
            description="In-memory caching for application performance",
            resource_pattern=ResourcePattern(
                cpu_p50=12.0,  # CPU light
                cpu_p95=35.0,
                memory_p50=78.0,  # Memory heavy
                memory_p95=88.0,
                cpu_cv=0.5,
                memory_cv=0.1,  # Very stable memory
                correlation_matrix=np.array([
                    [1.00, 0.20, 0.75, 0.78, 0.15],  # CPU for serialization
                    [0.20, 1.00, 0.18, 0.15, 0.10],  # Memory independent
                    [0.75, 0.18, 1.00, 0.92, 0.12],
                    [0.78, 0.15, 0.92, 1.00, 0.10],
                    [0.15, 0.10, 0.12, 0.10, 1.00],  # Minimal disk
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',
                seasonality_strength=0.3,
                burst_frequency=30.0,  # Cache misses
                burst_amplitude=2.5,
                burst_duration_minutes=1
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_MANUAL,
            optimization_potential=OptimizationPotential.LOW,  # Already efficient
            cost_profile=CostProfile(
                avg_hourly_cost=150.0,
                cost_variability=0.2,
                waste_percentage=20.0,  # Less waste
                optimization_difficulty=0.3,
                business_criticality=0.9
            ),
            typical_stack=["Redis", "Redis Sentinel", "Redis Cluster"],
            cloud_services=["ElastiCache", "Redis Enterprise"],
            typical_instance_types=["cache.r6g.xlarge", "cache.m5.large"],
            deployment_frequency='weekly',
            maintenance_windows=[(0, 4)],
            availability_target=99.9,
            latency_p99_ms=5.0,
            data_volume_gb_per_day=50.0,
            data_retention_days=1,
            example_companies=["Twitter", "GitHub", "Stack Overflow"],
            market_size_percentage=5.0
        ),

        # === DEVELOPMENT ENVIRONMENTS ===

        "dev_environment": ApplicationArchetype(
            name="Development Environment",
            domain=ApplicationDomain.DEVELOPMENT,
            description="Non-production development and testing environments",
            resource_pattern=ResourcePattern(
                cpu_p50=5.0,   # Shocking waste!
                cpu_p95=40.0,
                memory_p50=18.0,
                memory_p95=45.0,
                cpu_cv=1.5,   # Extreme variability
                memory_cv=0.8,
                correlation_matrix=np.array([
                    [1.00, 0.65, 0.55, 0.52, 0.70],
                    [0.65, 1.00, 0.40, 0.38, 0.60],
                    [0.55, 0.40, 1.00, 0.85, 0.45],
                    [0.52, 0.38, 0.85, 1.00, 0.42],
                    [0.70, 0.60, 0.45, 0.42, 1.00],
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',  # Idle weekends
                seasonality_strength=0.2,
                burst_frequency=5.0,
                burst_amplitude=8.0,  # Build/test spikes
                burst_duration_minutes=15
            ),
            scaling_behavior=ScalingBehavior.STATIC,  # Never adjusted
            optimization_potential=OptimizationPotential.HIGH,  # Massive waste
            cost_profile=CostProfile(
                avg_hourly_cost=280.0,
                cost_variability=1.2,
                waste_percentage=70.0,  # Worst waste!
                optimization_difficulty=0.1,  # Easy to fix
                business_criticality=0.3
            ),
            typical_stack=["Docker", "Kubernetes", "Jenkins", "Git"],
            cloud_services=["EC2", "EKS", "RDS Dev", "S3"],
            typical_instance_types=["t3.large", "m5.large", "t3.xlarge"],
            deployment_frequency='continuous',
            maintenance_windows=[],  # Anytime
            availability_target=95.0,
            latency_p99_ms=500.0,
            data_volume_gb_per_day=20.0,
            data_retention_days=7,
            example_companies=["Every company"],
            market_size_percentage=8.0  # Significant hidden cost
        ),

        "ci_cd_pipeline": ApplicationArchetype(
            name="CI/CD Pipeline",
            domain=ApplicationDomain.DEVELOPMENT,
            description="Continuous integration and deployment infrastructure",
            resource_pattern=ResourcePattern(
                cpu_p50=10.0,  # Mostly idle
                cpu_p95=88.0,  # Build spikes
                memory_p50=25.0,
                memory_p95=70.0,
                cpu_cv=1.3,
                memory_cv=0.7,
                correlation_matrix=np.array([
                    [1.00, 0.80, 0.60, 0.55, 0.75],
                    [0.80, 1.00, 0.50, 0.45, 0.70],
                    [0.60, 0.50, 1.00, 0.82, 0.55],
                    [0.55, 0.45, 0.82, 1.00, 0.50],
                    [0.75, 0.70, 0.55, 0.50, 1.00],  # Artifact storage
                ]),
                daily_pattern_type='business_hours',
                weekly_pattern_type='weekday_heavy',
                seasonality_strength=0.1,
                burst_frequency=50.0,  # Many builds/day
                burst_amplitude=9.0,
                burst_duration_minutes=5
            ),
            scaling_behavior=ScalingBehavior.ELASTIC_AUTO,
            optimization_potential=OptimizationPotential.HIGH,
            cost_profile=CostProfile(
                avg_hourly_cost=120.0,
                cost_variability=1.0,
                waste_percentage=55.0,
                optimization_difficulty=0.3,
                business_criticality=0.7
            ),
            typical_stack=["Jenkins", "GitLab CI", "GitHub Actions", "Docker"],
            cloud_services=["CodeBuild", "CodePipeline", "ECR", "S3"],
            typical_instance_types=["c5.xlarge", "m5.large"],
            deployment_frequency='continuous',
            maintenance_windows=[(0, 2)],
            availability_target=99.0,
            latency_p99_ms=0.0,
            data_volume_gb_per_day=100.0,
            data_retention_days=30,
            example_companies=["Every tech company"],
            market_size_percentage=3.0
        ),

        # Add more archetypes as needed...
    }

    @classmethod
    def get_archetype(cls, name: str) -> ApplicationArchetype:
        """Get archetype by name"""
        return cls.ARCHETYPES.get(name)

    @classmethod
    def get_by_domain(cls, domain: ApplicationDomain) -> List[ApplicationArchetype]:
        """Get all archetypes in a domain"""
        return [a for a in cls.ARCHETYPES.values() if a.domain == domain]

    @classmethod
    def get_by_optimization_potential(cls, potential: OptimizationPotential) -> List[ApplicationArchetype]:
        """Get archetypes by optimization potential"""
        return [a for a in cls.ARCHETYPES.values() if a.optimization_potential == potential]

    @classmethod
    def calculate_market_coverage(cls) -> float:
        """Calculate total market coverage"""
        return sum(a.market_size_percentage for a in cls.ARCHETYPES.values())

    @classmethod
    def get_correlation_matrix(cls, archetype_name: str) -> np.ndarray:
        """Get empirical correlation matrix for an archetype"""
        archetype = cls.get_archetype(archetype_name)
        if archetype:
            return archetype.resource_pattern.correlation_matrix
        return np.eye(5)  # Default to identity

    @classmethod
    def export_for_ml_training(cls) -> pl.DataFrame:
        """Export taxonomy as features for ML model training"""
        data = []
        for name, archetype in cls.ARCHETYPES.items():
            data.append({
                'archetype': name,
                'domain': archetype.domain.value,
                'cpu_p50': archetype.resource_pattern.cpu_p50,
                'cpu_p95': archetype.resource_pattern.cpu_p95,
                'memory_p50': archetype.resource_pattern.memory_p50,
                'memory_p95': archetype.resource_pattern.memory_p95,
                'waste_percentage': archetype.cost_profile.waste_percentage,
                'optimization_potential': archetype.optimization_potential.value,
                'business_criticality': archetype.cost_profile.business_criticality,
                'avg_hourly_cost': archetype.cost_profile.avg_hourly_cost,
                'market_size_percentage': archetype.market_size_percentage
            })

        return pl.DataFrame(data)

def generate_synthetic_workload(
    archetype_name: str,
    duration_hours: int = 720,
    sampling_interval_minutes: int = 5
) -> pl.DataFrame:
    """
    Generate synthetic workload data for a specific archetype
    This is where PyMC model would generate the actual time series
    """
    archetype = CloudZeroTaxonomy.get_archetype(archetype_name)
    if not archetype:
        raise ValueError(f"Unknown archetype: {archetype_name}")

    # This would connect to the PyMC model for generation
    # For now, simplified generation
    num_samples = duration_hours * 60 // sampling_interval_minutes

    # Generate correlated metrics using the correlation matrix
    mean = [
        archetype.resource_pattern.cpu_p50,
        archetype.resource_pattern.memory_p50,
        20.0,  # Network In baseline
        15.0,  # Network Out baseline
        100.0  # Disk IOPS baseline
    ]

    # Scale correlation matrix to covariance
    std_devs = np.array([
        archetype.resource_pattern.cpu_p50 * archetype.resource_pattern.cpu_cv,
        archetype.resource_pattern.memory_p50 * archetype.resource_pattern.memory_cv,
        10.0, 8.0, 50.0
    ])

    cov_matrix = np.outer(std_devs, std_devs) * archetype.resource_pattern.correlation_matrix

    # Generate multivariate normal samples
    samples = np.random.multivariate_normal(mean, cov_matrix, num_samples)

    # Clip to valid ranges
    samples[:, 0] = np.clip(samples[:, 0], 0, 100)  # CPU
    samples[:, 1] = np.clip(samples[:, 1], 0, 100)  # Memory
    samples[:, 2:] = np.maximum(samples[:, 2:], 0)  # Network and disk (non-negative)

    # Create timestamps
    start_time = datetime.now() - timedelta(hours=duration_hours)
    timestamps = [start_time + timedelta(minutes=i*sampling_interval_minutes)
                 for i in range(num_samples)]

    # Build DataFrame
    df = pl.DataFrame({
        'timestamp': timestamps,
        'archetype': archetype_name,
        'cpu_utilization': samples[:, 0],
        'memory_utilization': samples[:, 1],
        'network_in_mbps': samples[:, 2],
        'network_out_mbps': samples[:, 3],
        'disk_iops': samples[:, 4],
        'hourly_cost': archetype.cost_profile.avg_hourly_cost * (1 + np.random.normal(0, 0.2, num_samples)),
        'waste_percentage': archetype.cost_profile.waste_percentage * (1 + np.random.normal(0, 0.1, num_samples))
    })

    return df

def main():
    """Demonstrate the taxonomy system"""
    print("=== CloudZero Application Taxonomy ===\n")

    # Show market coverage
    coverage = CloudZeroTaxonomy.calculate_market_coverage()
    print(f"Total market coverage: {coverage:.1f}%\n")

    # Show archetypes by optimization potential
    high_potential = CloudZeroTaxonomy.get_by_optimization_potential(OptimizationPotential.HIGH)
    print(f"High optimization potential ({len(high_potential)} archetypes):")
    for arch in high_potential:
        print(f"  - {arch.name}: {arch.cost_profile.waste_percentage:.0f}% waste")

    # Export for ML training
    ml_features = CloudZeroTaxonomy.export_for_ml_training()
    print(f"\nExported {len(ml_features)} archetypes for ML training")

    # Generate sample data
    sample_data = generate_synthetic_workload("ecommerce_platform", duration_hours=24)
    print(f"\nGenerated {len(sample_data)} samples for e-commerce platform")
    print(f"Average CPU: {sample_data['cpu_utilization'].mean():.1f}%")
    print(f"Average Memory: {sample_data['memory_utilization'].mean():.1f}%")

if __name__ == "__main__":
    main()