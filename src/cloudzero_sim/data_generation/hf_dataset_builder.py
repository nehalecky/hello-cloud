"""
HuggingFace Dataset Builder for Cloud Metrics
Converts our Polars-based cloud metrics into HuggingFace datasets format
for use with foundation models and easy sharing
"""

import polars as pl
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from loguru import logger
from huggingface_hub import HfApi, create_repo
import pyarrow as pa

class CloudMetricsDatasetBuilder:
    """Build and manage HuggingFace datasets for cloud metrics"""

    FEATURE_SCHEMA = Features({
        # Time series identifiers
        "resource_id": Value("string"),
        "timestamp": Value("timestamp[ns]"),
        "workload_type": ClassLabel(names=[
            "web_application", "microservice", "batch_processing",
            "ml_training", "ml_inference", "database_oltp", "database_olap",
            "streaming_pipeline", "serverless_function", "cache_layer",
            "message_queue", "development_environment"
        ]),

        # Resource metrics
        "cpu_utilization": Value("float32"),
        "memory_utilization": Value("float32"),
        "network_in_mbps": Value("float32"),
        "network_out_mbps": Value("float32"),
        "disk_iops": Value("float32"),

        # Cost metrics
        "hourly_cost": Value("float32"),
        "efficiency_score": Value("float32"),
        "waste_percentage": Value("float32"),

        # Anomaly indicators
        "is_idle": Value("bool"),
        "is_overprovisioned": Value("bool"),
        "is_anomaly": Value("bool"),

        # Cloud provider info
        "cloud_provider": ClassLabel(names=["AWS", "Azure", "GCP"]),
        "region": Value("string"),
        "instance_type": Value("string"),

        # Business context (may be null)
        "customer_id": Value("string"),
        "team": Value("string"),
        "environment": ClassLabel(names=["prod", "staging", "dev", "test"]),
        "cost_center": Value("string"),
    })

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset builder"""
        self.cache_dir = cache_dir or "./data/hf_cache"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.api = HfApi()

    def polars_to_dataset(
        self,
        df: pl.DataFrame,
        split_name: str = "train"
    ) -> Dataset:
        """Convert Polars DataFrame to HuggingFace Dataset"""

        # Ensure all required columns exist
        required_cols = [
            "resource_id", "timestamp", "cpu_utilization",
            "memory_utilization", "hourly_cost"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Add missing columns with defaults if needed
        if "workload_type" not in df.columns:
            df = df.with_columns(pl.lit("web_application").alias("workload_type"))

        if "cloud_provider" not in df.columns:
            df = df.with_columns(pl.lit("AWS").alias("cloud_provider"))

        if "is_anomaly" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("is_anomaly"))

        # Convert to PyArrow table (efficient for HF datasets)
        arrow_table = df.to_arrow()

        # Create HuggingFace dataset
        dataset = Dataset(arrow_table)

        logger.info(f"Created dataset with {len(dataset)} samples for split '{split_name}'")

        return dataset

    def create_time_series_sequences(
        self,
        dataset: Dataset,
        sequence_length: int = 168,  # 1 week of hourly data
        stride: int = 24,  # 1 day stride
        target_column: str = "hourly_cost"
    ) -> Dataset:
        """Create sequences for time series models"""

        sequences = []
        targets = []
        metadata = []

        # Group by resource_id for per-resource sequences
        df = pl.from_arrow(dataset.data.table)
        resource_ids = df["resource_id"].unique().to_list()

        for resource_id in resource_ids:
            resource_data = df.filter(pl.col("resource_id") == resource_id).sort("timestamp")

            if len(resource_data) < sequence_length + 1:
                continue

            # Extract values
            values = resource_data[target_column].to_numpy()

            # Create sliding windows
            for i in range(0, len(values) - sequence_length, stride):
                # Input sequence
                seq = values[i:i+sequence_length]
                # Target (next value)
                target = values[i+sequence_length]

                sequences.append(seq.tolist())
                targets.append(target)
                metadata.append({
                    "resource_id": resource_id,
                    "start_time": resource_data["timestamp"][i],
                    "end_time": resource_data["timestamp"][i+sequence_length-1]
                })

        # Create new dataset with sequences
        sequence_dataset = Dataset.from_dict({
            "input_sequence": sequences,
            "target": targets,
            "resource_id": [m["resource_id"] for m in metadata],
            "start_time": [m["start_time"] for m in metadata],
            "end_time": [m["end_time"] for m in metadata]
        })

        logger.info(f"Created {len(sequence_dataset)} sequences of length {sequence_length}")

        return sequence_dataset

    def create_anomaly_detection_dataset(
        self,
        df: pl.DataFrame,
        window_size: int = 24,
        contamination_rate: float = 0.1
    ) -> DatasetDict:
        """Create dataset specifically for anomaly detection"""

        # Separate normal and anomalous data
        normal_df = df.filter(~pl.col("is_anomaly"))
        anomaly_df = df.filter(pl.col("is_anomaly"))

        logger.info(f"Normal samples: {len(normal_df)}, Anomalies: {len(anomaly_df)}")

        # Create sliding windows for both
        def create_windows(data: pl.DataFrame, label: int) -> List[Dict]:
            windows = []
            resource_ids = data["resource_id"].unique().to_list()

            for resource_id in resource_ids:
                resource_data = data.filter(
                    pl.col("resource_id") == resource_id
                ).sort("timestamp")

                if len(resource_data) < window_size:
                    continue

                # Extract features
                cpu = resource_data["cpu_utilization"].to_numpy()
                memory = resource_data["memory_utilization"].to_numpy()
                cost = resource_data["hourly_cost"].to_numpy()

                for i in range(len(resource_data) - window_size + 1):
                    window = {
                        "cpu_sequence": cpu[i:i+window_size].tolist(),
                        "memory_sequence": memory[i:i+window_size].tolist(),
                        "cost_sequence": cost[i:i+window_size].tolist(),
                        "label": label,
                        "resource_id": resource_id,
                        "timestamp": resource_data["timestamp"][i]
                    }
                    windows.append(window)

            return windows

        # Create windows
        normal_windows = create_windows(normal_df, label=0)
        anomaly_windows = create_windows(anomaly_df, label=1)

        # Balance dataset if needed
        if len(anomaly_windows) < len(normal_windows) * contamination_rate:
            # Oversample anomalies
            import random
            while len(anomaly_windows) < len(normal_windows) * contamination_rate:
                anomaly_windows.append(random.choice(anomaly_windows))

        # Combine and shuffle
        all_windows = normal_windows + anomaly_windows
        import random
        random.shuffle(all_windows)

        # Split into train/test
        split_idx = int(0.8 * len(all_windows))
        train_data = all_windows[:split_idx]
        test_data = all_windows[split_idx:]

        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)

        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

    def create_cost_optimization_dataset(
        self,
        df: pl.DataFrame
    ) -> Dataset:
        """Create dataset for cost optimization recommendations"""

        # Calculate optimization opportunities
        optimization_data = []

        resource_groups = df.group_by("resource_id").agg([
            pl.col("cpu_utilization").mean().alias("avg_cpu"),
            pl.col("memory_utilization").mean().alias("avg_memory"),
            pl.col("hourly_cost").mean().alias("avg_cost"),
            pl.col("waste_percentage").mean().alias("avg_waste"),
            pl.col("efficiency_score").mean().alias("avg_efficiency"),
            pl.col("is_idle").sum().alias("idle_hours"),
            pl.col("is_overprovisioned").sum().alias("overprovisioned_hours"),
            pl.count().alias("total_hours")
        ])

        for row in resource_groups.iter_rows(named=True):
            # Calculate optimization recommendations
            recommendations = []
            potential_savings = 0

            # Check for idle resources
            idle_ratio = row["idle_hours"] / row["total_hours"]
            if idle_ratio > 0.3:
                recommendations.append("shutdown_when_idle")
                potential_savings += row["avg_cost"] * idle_ratio

            # Check for over-provisioning
            if row["avg_cpu"] < 20 and row["avg_memory"] < 30:
                recommendations.append("rightsize_instance")
                potential_savings += row["avg_cost"] * 0.3

            # Check for inefficiency
            if row["avg_efficiency"] < 40:
                recommendations.append("optimize_workload")
                potential_savings += row["avg_cost"] * 0.2

            optimization_data.append({
                "resource_id": row["resource_id"],
                "avg_cpu": row["avg_cpu"],
                "avg_memory": row["avg_memory"],
                "avg_cost": row["avg_cost"],
                "avg_waste": row["avg_waste"],
                "avg_efficiency": row["avg_efficiency"],
                "idle_ratio": idle_ratio,
                "recommendations": recommendations,
                "potential_savings": potential_savings,
                "optimization_score": min(100, potential_savings / row["avg_cost"] * 100)
            })

        # Create dataset
        optimization_dataset = Dataset.from_list(optimization_data)

        logger.info(f"Created optimization dataset with {len(optimization_dataset)} resources")

        return optimization_dataset

    def push_to_hub(
        self,
        dataset: Dataset,
        repo_name: str,
        private: bool = True,
        token: Optional[str] = None
    ):
        """Push dataset to HuggingFace Hub"""

        try:
            # Create repository if it doesn't exist
            create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=private,
                token=token,
                exist_ok=True
            )

            # Push dataset
            dataset.push_to_hub(
                repo_id=repo_name,
                private=private,
                token=token
            )

            logger.info(f"Successfully pushed dataset to hub: {repo_name}")

        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")

    def load_from_hub(
        self,
        repo_name: str,
        split: Optional[str] = None,
        token: Optional[str] = None
    ) -> Dataset:
        """Load dataset from HuggingFace Hub"""

        from datasets import load_dataset

        dataset = load_dataset(
            repo_name,
            split=split,
            token=token,
            cache_dir=self.cache_dir
        )

        logger.info(f"Loaded dataset from hub: {repo_name}")

        return dataset

def create_comprehensive_cloud_dataset():
    """Create a comprehensive cloud metrics dataset"""

    logger.info("Creating comprehensive cloud metrics dataset...")

    # Import our data generators
    from workload_patterns import create_multi_workload_dataset

    # Generate 30 days of data
    start_time = datetime.now() - timedelta(days=30)
    end_time = datetime.now()

    # Create realistic workload data
    workload_df = create_multi_workload_dataset(start_time, end_time)

    # Add cloud provider information
    providers = ["AWS", "Azure", "GCP"]
    workload_df = workload_df.with_columns([
        pl.lit(None).alias("customer_id"),  # Will be filled for some resources
        pl.lit(None).alias("team"),
        pl.lit(None).alias("environment"),
        pl.lit(None).alias("cost_center"),
        pl.lit("m5.large").alias("instance_type"),
        pl.Series("cloud_provider", np.random.choice(providers, len(workload_df))),
        pl.lit("us-east-1").alias("region")
    ])

    # Add some business context randomly
    teams = ["platform", "data", "ml", "api", "frontend"]
    environments = ["prod", "staging", "dev", "test"]

    # Randomly assign teams and environments to resources
    resource_ids = workload_df["resource_id"].unique()
    resource_metadata = {}

    for rid in resource_ids:
        if np.random.random() > 0.3:  # 70% have metadata
            resource_metadata[rid] = {
                "team": np.random.choice(teams),
                "environment": np.random.choice(environments),
                "cost_center": f"cc-{np.random.randint(100, 999)}",
                "customer_id": f"cust-{np.random.randint(1, 100):03d}" if np.random.random() > 0.5 else None
            }

    # Apply metadata
    for rid, metadata in resource_metadata.items():
        mask = workload_df["resource_id"] == rid
        for key, value in metadata.items():
            workload_df = workload_df.with_columns(
                pl.when(mask).then(pl.lit(value)).otherwise(pl.col(key)).alias(key)
            )

    # Initialize dataset builder
    builder = CloudMetricsDatasetBuilder()

    # Create base dataset
    base_dataset = builder.polars_to_dataset(workload_df, split_name="train")

    # Create specialized datasets
    sequence_dataset = builder.create_time_series_sequences(
        base_dataset,
        sequence_length=168,  # 1 week
        stride=24  # 1 day
    )

    anomaly_dataset = builder.create_anomaly_detection_dataset(
        workload_df,
        window_size=24
    )

    optimization_dataset = builder.create_cost_optimization_dataset(workload_df)

    # Create comprehensive dataset dictionary
    comprehensive_dataset = DatasetDict({
        "raw": base_dataset,
        "sequences": sequence_dataset,
        "anomaly_detection_train": anomaly_dataset["train"],
        "anomaly_detection_test": anomaly_dataset["test"],
        "optimization": optimization_dataset
    })

    # Save locally
    comprehensive_dataset.save_to_disk("./data/cloud_metrics_dataset")

    logger.info("Dataset creation complete!")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    for split_name, split_data in comprehensive_dataset.items():
        print(f"{split_name}: {len(split_data)} samples")

    return comprehensive_dataset

if __name__ == "__main__":
    dataset = create_comprehensive_cloud_dataset()