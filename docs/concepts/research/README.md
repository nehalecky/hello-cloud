# Research Foundation

This directory contains the empirical research that forms the foundation of hello cloud understand of cloud workloads.

## Core Research Documents

### 1. [Cloud Resource Patterns Research](cloud-resource-patterns-research.md)
- Comprehensive analysis of real-world cloud utilization statistics
- Key findings: 13% CPU utilization, 30-32% waste
- Workload-specific patterns and inefficiencies
- Industry benchmarks and best practices

### 2. [Cloud Resource Correlations Report](cloud-resource-correlations-report.md)
- Empirical correlation matrices between resource metrics
- Time-lagged correlations and dependencies
- Application-specific correlation patterns
- Statistical validation methods

### 3. [Time Series Anomaly Datasets Review](timeseries-anomaly-datasets-review.md)
- **HuggingFace dataset catalog for cloud resource anomaly detection**
- Top recommendations: AutonLab/Timeseries-PILE, Lemma-RCA-NEC/Cloud_Computing
- Energy domain datasets as cloud infrastructure analogs
- **Essential resource for Gaussian process modeling**

### 4. [Alibaba Trace Analysis](alibaba-trace-analysis.md) _(Historical Reference)_
- Comprehensive analysis of Alibaba 2018 cluster traces
- ⚠️ **Datacenter-wide aggregations (NOT machine-level data)**
- Empirical findings: 38% CPU, 88% memory utilization
- Valid uses: Temporal pattern analysis, benchmarking
- Invalid uses: Machine-level optimization, co-location studies
- 28 research citations
- **Note**: ETL functionality has been removed; this document preserved for research insights only

### 5. [OpenTSLM Foundation Model Evaluation](opentslm-foundation-model-evaluation.md)
- Evaluation of Stanford's OpenTSLM timeseries foundation model
- Key finding: Not suitable for cloud anomaly detection
- No pre-trained weights, medical domain focus
- Recommendations for alternative approaches

## Quick Access by Use Case

### For Gaussian Process Modeling (Next Priority)
→ **[Time Series Anomaly Datasets Review](timeseries-anomaly-datasets-review.md)** - HuggingFace datasets catalog

### For Temporal Pattern Analysis
→ [Alibaba Trace Analysis](alibaba-trace-analysis.md) + [Cloud Resource Patterns](cloud-resource-patterns-research.md)

### For Multivariate Modeling
→ [Cloud Resource Correlations Report](cloud-resource-correlations-report.md)

## Key Findings Summary

### Utilization Statistics
- **CPU**: Average 13% utilization (industry-wide)
- **Memory**: Average 20% utilization
- **Waste**: 30-32% of cloud spending
- **Alibaba Dataset**: 38% CPU, 88% memory (datacenter-wide averages)

### Correlation Patterns
- **Strong temporal autocorrelation**: 0.7-0.8 for first 10 time lags
- **CPU-Memory**: Varies by workload (0.2-0.95 correlation)
- **Network-CPU**: High correlation in web apps (0.7-0.8)

### Temporal Patterns
- **Daily**: 40-60% increase during business hours
- **Weekly**: 60-80% drop on weekends
- **Seasonal**: 300-500% spikes for retail during holidays

## Application to Simulation

These research findings directly inform:
1. **Base utilization rates** in our models
2. **Correlation matrices** for multivariate generation
3. **Temporal patterns** (daily, weekly, seasonal)
4. **Waste factors** by application type
5. **Anomaly patterns** and failure modes

## References

All findings are backed by:
- Academic research papers (35+ citations)
- Industry reports (Gartner, FinOps Foundation)
- Cloud provider documentation (AWS, Azure, GCP)
- Real-world case studies (Netflix, Uber, etc.)
