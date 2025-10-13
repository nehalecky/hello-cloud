# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: python3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understanding Cloud Workload Signatures: A Comprehensive Guide
#
# This notebook explores **why** different cloud workload types have distinct resource utilization patterns. We'll examine the underlying technical and business reasons that create these signatures, grounded in empirical research.

# %%
# Auto-reload: Picks up library changes without kernel restart
# %load_ext autoreload
# %autoreload 2

# %%
from datetime import datetime, timedelta

import altair as alt
import numpy as np
import polars as pl

# Configure Altair for interactive visualizations
alt.data_transformers.enable("default")  # Use default transformer for now
alt.theme.enable("quartz")  # Clean, professional theme

# Import our simulation framework
from hellocloud.generation import WorkloadPatternGenerator, WorkloadType

# %% [markdown]
# ## Part 1: Foundations - Why Do Workload Signatures Exist?
#
# Before diving into specific patterns, let's understand the fundamental forces that create distinct workload signatures.

# %%
# Create a conceptual diagram showing the forces that shape workload signatures
forces_data = pl.DataFrame(
    {
        "category": [
            "Hardware",
            "Hardware",
            "Architecture",
            "Architecture",
            "Business",
            "Business",
            "Optimization",
            "Optimization",
        ],
        "factor": [
            "CPU-Memory Bus",
            "I/O Latency",
            "Request Model",
            "State Management",
            "User Behavior",
            "Business Hours",
            "Auto-scaling",
            "Caching",
        ],
        "impact_level": [7, 8, 9, 8, 10, 9, 6, 7],
        "description": [
            "Physical constraints on data movement",
            "Storage and network access delays",
            "Sync vs async, batch vs stream",
            "Stateful vs stateless design",
            "Human activity patterns",
            "Work schedules and timezones",
            "Dynamic resource adjustment",
            "Memory-speed tradeoffs",
        ],
    }
)

forces_chart = (
    alt.Chart(forces_data.to_pandas())
    .mark_bar()
    .encode(
        x=alt.X("impact_level:Q", title="Impact on Signature", scale=alt.Scale(domain=[0, 10])),
        y=alt.Y("factor:N", title="Contributing Factor", sort="-x"),
        color=alt.Color("category:N", title="Category", scale=alt.Scale(scheme="tableau10")),
        tooltip=["factor", "description", "impact_level"],
    )
    .properties(width=600, height=400, title="Forces That Shape Workload Signatures")
    .interactive()
)

forces_chart

# %% [markdown]
# ### The Physics of Computing
#
# Resource utilization patterns emerge from fundamental computing constraints:
#
# 1. **CPU-Memory Bandwidth**: Data must move between CPU and memory, creating correlations
# 2. **I/O Wait States**: CPUs idle while waiting for disk/network operations
# 3. **Cache Hierarchies**: L1/L2/L3 caches create step functions in performance
# 4. **Thermal Limits**: Sustained high utilization triggers throttling

# %%
# Visualize the relationship between I/O wait and CPU utilization
np.random.seed(42)  # Set seed for reproducible results
io_wait_data = {
    "time": list(range(100)),
    "cpu_active": [
        max(0, min(100, 20 + 10 * np.sin(i / 10) + np.random.normal(0, 3))) for i in range(100)
    ],
    "io_wait": [
        max(0, min(100, 15 + 8 * np.sin(i / 10 + np.pi) + np.random.normal(0, 2)))
        for i in range(100)
    ],
}

# Calculate idle time and create DataFrame directly in pandas for Altair
import pandas as pd

io_wait_pandas = pd.DataFrame(io_wait_data)
io_wait_pandas["idle"] = 100 - io_wait_pandas["cpu_active"] - io_wait_pandas["io_wait"]

# Ensure proper data types for Altair
io_wait_pandas = io_wait_pandas.astype(
    {"time": "int64", "cpu_active": "float64", "io_wait": "float64", "idle": "float64"}
)

# Melt the DataFrame before passing to Altair (avoids transform_fold type issues)
io_wait_melted = io_wait_pandas.melt(
    id_vars=["time"],
    value_vars=["cpu_active", "io_wait", "idle"],
    var_name="component",
    value_name="percentage",
)

io_chart = (
    alt.Chart(io_wait_melted)
    .mark_area()
    .encode(
        x=alt.X("time:Q", title="Time"),
        y=alt.Y("percentage:Q", stack="normalize", title="CPU State (%)"),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(
                domain=["cpu_active", "io_wait", "idle"], range=["#2ca02c", "#ff7f0e", "#d62728"]
            ),
            title="CPU State",
        ),
        tooltip=["time", "component", "percentage"],
    )
    .properties(width=700, height=300, title="Why CPUs Show Low Utilization: I/O Wait States")
    .interactive()
)

io_chart

# %% [markdown]
# ## Part 2: Deep Dive - Understanding Each Workload Type
#
# Now let's explore WHY each workload type has its unique signature, backed by research data.

# %%
# Generate sample data for all workload types
generator = WorkloadPatternGenerator(seed=42)
workload_samples = {}

for workload_type in WorkloadType:
    df = generator.generate_time_series(
        workload_type=workload_type,
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
        interval_minutes=60,
    )
    workload_samples[workload_type.value] = df

print(f"Generated samples for {len(workload_samples)} workload types")

# %% [markdown]
# ### 2.1 Web Applications: Why 15% CPU and Business Hours Pattern?
#
# Web applications show low CPU utilization because they spend most time waiting for I/O operations.
#
# **Key Reasons:**
# - **Request/Response Model**: Each request triggers database queries, API calls
# - **Network Latency**: Waiting for client requests and responses
# - **Connection Pooling**: Maintaining idle connections for quick response
# - **Human Users**: Activity follows work schedules and timezones

# %%
# Analyze web application patterns
web_app_data = workload_samples["web_application"]

# Calculate hourly averages to show business hours pattern
hourly_stats = web_app_data.group_by(web_app_data["timestamp"].dt.hour().alias("hour")).agg(
    [
        pl.col("cpu_utilization").mean().alias("cpu_mean"),
        pl.col("memory_utilization").mean().alias("memory_mean"),
        pl.col("waste_percentage").mean().alias("waste_mean"),
    ]
)

# Add explanation for each hour
hourly_stats = hourly_stats.with_columns(
    [
        pl.when(pl.col("hour").is_between(0, 5))
        .then(pl.lit("Night - Minimal activity"))
        .when(pl.col("hour").is_between(6, 8))
        .then(pl.lit("Morning ramp-up"))
        .when(pl.col("hour").is_between(9, 11))
        .then(pl.lit("Peak morning activity"))
        .when(pl.col("hour").is_between(12, 13))
        .then(pl.lit("Lunch dip"))
        .when(pl.col("hour").is_between(14, 16))
        .then(pl.lit("Afternoon peak"))
        .when(pl.col("hour").is_between(17, 18))
        .then(pl.lit("End of day wind-down"))
        .otherwise(pl.lit("Evening - Reduced activity"))
        .alias("period_explanation")
    ]
)

web_app_chart = (
    alt.Chart(hourly_stats.to_pandas())
    .mark_line(point=True)
    .encode(
        x=alt.X("hour:O", title="Hour of Day"),
        y=alt.Y("cpu_mean:Q", title="Average CPU Utilization (%)"),
        color=alt.value("#1f77b4"),
        tooltip=["hour", "cpu_mean", "period_explanation"],
    )
    .properties(width=700, height=350, title="Web Applications: Why CPU Follows Business Hours")
)

memory_layer = (
    alt.Chart(hourly_stats.to_pandas())
    .mark_line(point=True, strokeDash=[5, 5])
    .encode(
        x="hour:O",
        y=alt.Y("memory_mean:Q", title="Average Memory Utilization (%)"),
        color=alt.value("#ff7f0e"),
        tooltip=["hour", "memory_mean", "period_explanation"],
    )
)

(web_app_chart + memory_layer).interactive()

# %% [markdown]
# ### 2.2 Batch Processing: Why 70% Idle with 10x Peaks?
#
# Batch processing shows extreme waste because resources are reserved for scheduled jobs that run infrequently.
#
# **Key Reasons:**
# - **Schedule-Driven**: Jobs run at specific times (nightly, weekly)
# - **Resource Reservation**: Capacity kept available for batch windows
# - **Sequential Processing**: Cannot easily parallelize across time
# - **Data Dependencies**: Must wait for data to be ready

# %%
# Visualize batch processing patterns
batch_data = workload_samples["batch_processing"]

# Create a view showing idle periods and spike patterns
batch_sample = batch_data.head(168)  # One week

batch_chart = (
    alt.Chart(batch_sample.to_pandas())
    .mark_area(
        line={"color": "darkblue"},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="lightblue", offset=0),
                alt.GradientStop(color="darkblue", offset=1),
            ],
            x1=0,
            y1=0,
            x2=0,
            y2=1,
        ),
    )
    .encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("cpu_utilization:Q", title="CPU Utilization (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["timestamp", "cpu_utilization", "is_idle"],
    )
    .properties(
        width=800, height=300, title="Batch Processing: Extreme Waste from Scheduled Execution"
    )
)

# Add annotations for batch windows
annotations_df = pl.DataFrame(
    {
        "timestamp": [
            batch_sample["timestamp"][20],
            batch_sample["timestamp"][68],
            batch_sample["timestamp"][116],
            batch_sample["timestamp"][164],
        ],
        "cpu_utilization": [80, 85, 78, 82],
        "label": ["Nightly ETL", "Report Generation", "Data Backup", "Weekly Analytics"],
    }
)

annotations = (
    alt.Chart(annotations_df.to_pandas())
    .mark_text(align="center", baseline="bottom", fontSize=10)
    .encode(x="timestamp:T", y="cpu_utilization:Q", text="label:N")
)

(batch_chart + annotations).interactive()

# %% [markdown]
# ### 2.3 Machine Learning: Different Patterns for Training vs Inference
#
# ML workloads show distinct patterns based on their phase and hardware utilization.
#
# **Training (25% CPU, 40% Memory):**
# - **Batch Processing**: Loading data batches into memory
# - **GPU Offloading**: CPU coordinates, GPU computes
# - **Experimentation Gaps**: Idle between hyperparameter runs
# - **Checkpointing**: Periodic saves create I/O spikes
#
# **Inference (30% CPU, 45% Memory):**
# - **Model in Memory**: Loaded model consumes constant memory
# - **Request Serving**: More consistent than training
# - **Lower Variance**: Predictable computation per request

# %%
# Compare ML Training vs Inference patterns
ml_training = workload_samples["ml_training"]
ml_inference = workload_samples["ml_inference"]

# Sample 48 hours for detailed comparison
comparison_hours = 48
ml_comparison = pl.DataFrame(
    {
        "hour": list(range(comparison_hours)),
        "training_cpu": ml_training.head(comparison_hours)["cpu_utilization"].to_list(),
        "training_memory": ml_training.head(comparison_hours)["memory_utilization"].to_list(),
        "inference_cpu": ml_inference.head(comparison_hours)["cpu_utilization"].to_list(),
        "inference_memory": ml_inference.head(comparison_hours)["memory_utilization"].to_list(),
    }
)

# Create side-by-side comparison
ml_comparison_long = ml_comparison.unpivot(
    index=["hour"], on=["training_cpu", "training_memory", "inference_cpu", "inference_memory"]
).with_columns(
    [
        pl.when(pl.col("variable").str.contains("training"))
        .then(pl.lit("Training"))
        .otherwise(pl.lit("Inference"))
        .alias("workload_type"),
        pl.when(pl.col("variable").str.contains("cpu"))
        .then(pl.lit("CPU"))
        .otherwise(pl.lit("Memory"))
        .alias("resource_type"),
    ]
)

ml_chart = (
    alt.Chart(ml_comparison_long.to_pandas())
    .mark_line()
    .encode(
        x=alt.X("hour:Q", title="Hour"),
        y=alt.Y("value:Q", title="Utilization (%)"),
        color=alt.Color("resource_type:N", title="Resource"),
        strokeDash=alt.StrokeDash("workload_type:N", title="ML Phase"),
        tooltip=["hour", "value", "workload_type", "resource_type"],
    )
    .properties(width=700, height=350, title="ML Workloads: Why Training and Inference Differ")
    .interactive()
)

ml_chart

# %% [markdown]
# ### 2.4 Databases: Why Memory-Heavy with Different OLTP vs OLAP Patterns?
#
# Databases prioritize memory for performance, but OLTP and OLAP have very different access patterns.
#
# **OLTP (20% CPU, 60% Memory):**
# - **Buffer Pool Cache**: Keep hot data in memory
# - **Connection Pools**: Each connection consumes memory
# - **Index Structures**: B-trees and hash indexes in RAM
# - **Transaction Logs**: Write-ahead logging for durability
#
# **OLAP (10% CPU, 30% Memory):**
# - **Columnar Storage**: Different memory access patterns
# - **Batch Queries**: Periodic analytical workloads
# - **Result Caching**: Store query results for reuse
# - **Compression**: CPU/memory tradeoff for storage

# %%
# Analyze database patterns
db_oltp = workload_samples["database_oltp"]
db_olap = workload_samples["database_olap"]

# Show correlation between queries and resource usage
db_comparison = pl.DataFrame(
    {
        "workload": ["OLTP"] * 24 + ["OLAP"] * 24,
        "hour": list(range(24)) * 2,
        "cpu": (
            db_oltp.head(24)["cpu_utilization"].to_list()
            + db_olap.head(24)["cpu_utilization"].to_list()
        ),
        "memory": (
            db_oltp.head(24)["memory_utilization"].to_list()
            + db_olap.head(24)["memory_utilization"].to_list()
        ),
        "pattern_type": (["Transactional"] * 24 + ["Analytical"] * 24),
    }
)

db_scatter = (
    alt.Chart(db_comparison.to_pandas())
    .mark_circle(size=100)
    .encode(
        x=alt.X("cpu:Q", title="CPU Utilization (%)", scale=alt.Scale(domain=[0, 50])),
        y=alt.Y("memory:Q", title="Memory Utilization (%)", scale=alt.Scale(domain=[0, 80])),
        color=alt.Color("workload:N", title="Database Type", scale=alt.Scale(scheme="dark2")),
        tooltip=["workload", "cpu", "memory", "pattern_type"],
    )
    .properties(width=600, height=400, title="Database Resource Patterns: OLTP vs OLAP")
)

# Add annotation regions
regions = pl.DataFrame(
    {
        "region": ["OLTP Zone", "OLAP Zone"],
        "cpu_center": [20, 10],
        "memory_center": [60, 30],
        "cpu_range": [10, 15],
        "memory_range": [15, 20],
    }
)

db_scatter.interactive()

# %% [markdown]
# ### 2.5 Development Environments: Why 70% Waste?
#
# Development environments are the worst offenders for waste, and there are clear reasons why.
#
# **Root Causes of Waste:**
# - **24/7 Provisioning**: Resources allocated continuously
# - **8/5 Usage**: Only used during work hours on weekdays
# - **Overprovisioning**: "Just in case" resource allocation
# - **Forgotten Resources**: Developers forget to shut down

# %%
# Analyze development environment waste patterns
dev_env = workload_samples["development_environment"]

# Calculate waste by day of week and hour
dev_analysis = dev_env.with_columns(
    [
        dev_env["timestamp"].dt.weekday().alias("weekday"),
        dev_env["timestamp"].dt.hour().alias("hour"),
    ]
)

weekly_pattern = (
    dev_analysis.group_by("weekday")
    .agg(
        [
            pl.col("cpu_utilization").mean().alias("cpu_mean"),
            pl.col("waste_percentage").mean().alias("waste_mean"),
        ]
    )
    .with_columns(
        [
            pl.when(pl.col("weekday") == 0)
            .then(pl.lit("Monday"))
            .when(pl.col("weekday") == 1)
            .then(pl.lit("Tuesday"))
            .when(pl.col("weekday") == 2)
            .then(pl.lit("Wednesday"))
            .when(pl.col("weekday") == 3)
            .then(pl.lit("Thursday"))
            .when(pl.col("weekday") == 4)
            .then(pl.lit("Friday"))
            .when(pl.col("weekday") == 5)
            .then(pl.lit("Saturday"))
            .otherwise(pl.lit("Sunday"))
            .alias("day_name")
        ]
    )
)

dev_waste_chart = (
    alt.Chart(weekly_pattern.to_pandas())
    .mark_bar()
    .encode(
        x=alt.X(
            "day_name:N",
            title="Day of Week",
            sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ),
        y=alt.Y("waste_mean:Q", title="Average Waste (%)"),
        color=alt.Color("waste_mean:Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["day_name", "cpu_mean", "waste_mean"],
    )
    .properties(width=600, height=350, title="Development Environments: Why 70% Waste Occurs")
)

# Add text annotations
text = (
    alt.Chart(weekly_pattern.to_pandas())
    .mark_text(dy=-10)
    .encode(
        x=alt.X(
            "day_name:N",
            sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ),
        y=alt.Y("waste_mean:Q"),
        text=alt.Text("waste_mean:Q", format=".1f"),
    )
)

(dev_waste_chart + text).interactive()

# %% [markdown]
# ### 2.6 Serverless: Why Extreme Variance with Low Waste?
#
# Serverless shows unique patterns due to its pay-per-use model.
#
# **Distinctive Characteristics:**
# - **Cold Starts**: Initial invocations have high latency
# - **Auto-scaling**: Instant scale from 0 to thousands
# - **Micro-billing**: Pay only for actual execution time
# - **Stateless Design**: No persistent resource allocation

# %%
# Analyze serverless patterns
serverless = workload_samples["serverless_function"]

# Show the extreme variance
serverless_sample = serverless.head(48)

serverless_chart = (
    alt.Chart(serverless_sample.to_pandas())
    .mark_area(
        line={"color": "purple"},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="white", offset=0),
                alt.GradientStop(color="purple", offset=1),
            ],
            x1=0,
            y1=0,
            x2=0,
            y2=1,
        ),
        opacity=0.7,
    )
    .encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("cpu_utilization:Q", title="CPU Utilization (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=["timestamp", "cpu_utilization", "waste_percentage"],
    )
    .properties(
        width=800, height=300, title="Serverless: Extreme Variance but Low Waste (Pay-per-Use)"
    )
)

# Add cold start indicators
cold_starts = serverless_sample.filter(pl.col("cpu_utilization") > 80)
cold_start_markers = (
    alt.Chart(cold_starts.to_pandas())
    .mark_circle(color="red", size=100)
    .encode(
        x="timestamp:T",
        y="cpu_utilization:Q",
        tooltip=[alt.Tooltip("timestamp:T", title="Cold Start At")],
    )
)

(serverless_chart + cold_start_markers).interactive()

# %% [markdown]
# ## Part 3: Understanding Correlation Patterns
#
# Different workloads show distinct correlations between resource metrics, and understanding why helps with optimization.

# %%
# Calculate correlations for each workload type
correlations = {}

for workload_name, df in workload_samples.items():
    cpu = df["cpu_utilization"].to_numpy()
    memory = df["memory_utilization"].to_numpy()
    network_in = df["network_in_mbps"].to_numpy()
    disk = df["disk_iops"].to_numpy()

    correlations[workload_name] = {
        "cpu_memory": np.corrcoef(cpu, memory)[0, 1],
        "cpu_network": np.corrcoef(cpu, network_in)[0, 1],
        "cpu_disk": np.corrcoef(cpu, disk)[0, 1],
        "memory_network": np.corrcoef(memory, network_in)[0, 1],
    }

# Create correlation heatmap data
corr_data = []
for workload, corr_values in correlations.items():
    for metric_pair, correlation in corr_values.items():
        corr_data.append(
            {
                "workload": workload.replace("_", " ").title(),
                "metric_pair": metric_pair.replace("_", " ").title(),
                "correlation": correlation,
            }
        )

corr_df = pl.DataFrame(corr_data)

# Create heatmap
correlation_heatmap = (
    alt.Chart(corr_df.to_pandas())
    .mark_rect()
    .encode(
        x=alt.X("metric_pair:N", title="Metric Pair"),
        y=alt.Y("workload:N", title="Workload Type"),
        color=alt.Color(
            "correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Correlation"
        ),
        tooltip=["workload", "metric_pair", alt.Tooltip("correlation:Q", format=".3f")],
    )
    .properties(width=500, height=400, title="Why Different Workloads Show Different Correlations")
    .interactive()
)

correlation_heatmap

# %% [markdown]
# ### Explaining Correlation Patterns
#
# **Strong CPU-Memory Correlation (>0.7):**
# - **ML Training**: Loading batches requires both compute and memory
# - **Streaming**: Processing data streams uses both proportionally
# - **Why**: Data must be in memory to be processed
#
# **Weak CPU-Memory Correlation (<0.3):**
# - **Databases**: Memory for caching, CPU for queries (independent)
# - **Cache Layers**: High memory, low CPU consistently
# - **Why**: Memory serves different purpose than computation
#
# **CPU-Network Correlation:**
# - **Web Apps**: High correlation - requests drive processing
# - **Batch**: Low correlation - network for data transfer, CPU for processing
# - **Why**: Depends on whether network I/O drives computation
#
# ## Part 4: Temporal Patterns and Autocorrelation
#
# Understanding why patterns persist over time helps with forecasting and capacity planning.

# %%
# Calculate autocorrelation for different workloads

autocorr_results = {}
lags = list(range(1, 25))  # Check up to 24 hours

for workload_name in ["web_application", "batch_processing", "streaming_pipeline"]:
    df = workload_samples[workload_name]
    cpu_series = df["cpu_utilization"].to_numpy()

    # Calculate autocorrelation
    autocorr = []
    for lag in lags:
        if lag < len(cpu_series):
            corr = np.corrcoef(cpu_series[:-lag], cpu_series[lag:])[0, 1]
            autocorr.append(corr)
        else:
            autocorr.append(0)

    autocorr_results[workload_name] = autocorr

# Create visualization
autocorr_data = []
for workload, values in autocorr_results.items():
    for lag, corr in zip(lags, values):
        autocorr_data.append(
            {
                "workload": workload.replace("_", " ").title(),
                "lag_hours": lag,
                "autocorrelation": corr,
            }
        )

autocorr_df = pl.DataFrame(autocorr_data)

autocorr_chart = (
    alt.Chart(autocorr_df.to_pandas())
    .mark_line(point=True)
    .encode(
        x=alt.X("lag_hours:Q", title="Lag (hours)"),
        y=alt.Y("autocorrelation:Q", title="Autocorrelation", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("workload:N", title="Workload Type"),
        tooltip=["workload", "lag_hours", alt.Tooltip("autocorrelation:Q", format=".3f")],
    )
    .properties(width=700, height=400, title="Temporal Autocorrelation: Why Patterns Persist")
    .interactive()
)

autocorr_chart

# %% [markdown]
# ### Why Autocorrelation Matters
#
# **High Autocorrelation (Web Apps, Streaming):**
# - **Gradual Changes**: User activity changes slowly
# - **Predictability**: Future similar to recent past
# - **Optimization**: Can forecast and pre-scale
#
# **Low Autocorrelation (Batch Processing):**
# - **Discrete Events**: Jobs start and stop abruptly
# - **Less Predictable**: Harder to forecast
# - **Optimization**: Need event-driven scaling
#
# ## Part 5: Cost Implications and Optimization Opportunities
#
# Understanding signatures enables targeted optimization strategies.

# %%
# Calculate potential savings by workload type
savings_analysis = []

for workload_name, df in workload_samples.items():
    avg_cpu = df["cpu_utilization"].mean()
    avg_memory = df["memory_utilization"].mean()
    avg_waste = df["waste_percentage"].mean()

    # Calculate optimization potential
    if avg_cpu < 20:  # Low utilization
        optimization_strategy = "Aggressive auto-scaling or serverless"
        potential_savings = avg_waste * 0.7  # Can eliminate 70% of waste
    elif avg_waste > 50:  # High waste
        optimization_strategy = "Right-sizing and scheduling"
        potential_savings = avg_waste * 0.6
    elif "database" in workload_name:  # Databases
        optimization_strategy = "Reserved instances and caching"
        potential_savings = avg_waste * 0.4
    else:
        optimization_strategy = "Standard auto-scaling"
        potential_savings = avg_waste * 0.5

    savings_analysis.append(
        {
            "workload": workload_name.replace("_", " ").title(),
            "current_waste": avg_waste,
            "potential_savings": potential_savings,
            "optimization_strategy": optimization_strategy,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
        }
    )

savings_df = pl.DataFrame(savings_analysis)

# Create savings opportunity chart
savings_chart = (
    alt.Chart(savings_df.to_pandas())
    .mark_bar()
    .encode(
        x=alt.X("potential_savings:Q", title="Potential Savings (%)"),
        y=alt.Y(
            "workload:N",
            title="Workload Type",
            sort=alt.EncodingSortField(field="potential_savings", order="descending"),
        ),
        color=alt.Color("optimization_strategy:N", title="Recommended Strategy"),
        tooltip=["workload", "current_waste", "potential_savings", "optimization_strategy"],
    )
    .properties(
        width=700, height=400, title="Optimization Opportunities by Understanding Signatures"
    )
    .interactive()
)

savings_chart

# %% [markdown]
# ## Key Takeaways: Why Signatures Matter
#
# Understanding **why** different workloads have distinct signatures enables:
#
# 1. **Right-sizing**: Match resources to actual needs, not peaks
# 2. **Scheduling**: Run batch jobs during web app quiet times
# 3. **Architecture Decisions**: Choose serverless for variable loads
# 4. **Cost Optimization**: Target biggest waste sources first
# 5. **Capacity Planning**: Predict future needs from patterns
#
# ### The Math Behind the Patterns
#
# For those interested in the statistical foundations:

# %%
# Show the mathematical relationships
math_explanation = """
### Statistical Properties of Workload Signatures

**1. Mean Utilization (μ)**: Baseline resource consumption
   - Web Apps: μ_cpu ≈ 15%, μ_mem ≈ 35%
   - Batch: μ_cpu ≈ 8%, μ_mem ≈ 15%

**2. Variance (σ²)**: Variability in resource usage
   - High variance → Unpredictable (Serverless: σ² > 40)
   - Low variance → Stable (Streaming: σ² < 10)

**3. Correlation (ρ)**: Relationship between metrics
   ρ(CPU, Memory) = Cov(CPU, Memory) / (σ_cpu * σ_mem)
   - ML Training: ρ ≈ 0.8 (strong positive)
   - Database: ρ ≈ 0.2 (weak)

**4. Autocorrelation (r_k)**: Temporal dependency
   r_k = Cov(X_t, X_{t+k}) / Var(X_t)
   - Web Apps: r_1 ≈ 0.8 (strong persistence)
   - Batch: r_1 ≈ 0.3 (weak persistence)

**5. Waste Function**: W = (Provisioned - Used) / Provisioned
   - Development: W ≈ 0.7 (70% waste)
   - Streaming: W ≈ 0.2 (20% waste)
"""

from IPython.display import Markdown

Markdown(math_explanation)

# %% [markdown]
# ## Conclusion: From Understanding to Action
#
# This guide has explored the fundamental reasons why different cloud workloads exhibit distinct resource utilization signatures. By understanding these patterns, we can:
#
# 1. **Predict** future resource needs with greater accuracy
# 2. **Optimize** resource allocation to reduce waste
# 3. **Design** better architectures that match workload characteristics
# 4. **Save** significant costs through targeted interventions
#
# The shocking reality of 13% average CPU utilization and 30-32% waste isn't just a statistic—it's an opportunity. By understanding the "why" behind these patterns, we can build more efficient cloud infrastructure.
#
# ### Next Steps
#
# To apply these insights:
# 1. Profile your own workloads to identify their signatures
# 2. Compare against these patterns to find optimization opportunities
# 3. Implement targeted strategies based on workload type
# 4. Monitor and iterate to continuously improve efficiency
#
# Remember: Every workload is unique, but understanding these fundamental patterns provides a foundation for optimization.
