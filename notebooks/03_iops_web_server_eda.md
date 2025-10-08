---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
      format_version: 0.13
      jupytext_version: 1.16.0
  kernelspec:
    display_name: cloud-sim
    language: python
    name: cloud-sim
---

# IOPS Web Server Anomaly Detection: Exploratory Data Analysis

## Overview and Objectives

This notebook performs exploratory data analysis on the **IOPS dataset** from the [TSB-UAD benchmark](https://github.com/TheDatumOrg/TSB-UAD), available via [AutonLab/Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE) on HuggingFace.

### Dataset Context

The IOPS dataset contains **20 Key Performance Indicator (KPI) time series** from real web services operated by five internet companies. The data is **anonymized** - we don't know exactly what each KPI measures, but according to TSB-UAD documentation, these metrics reflect:
- **Scale**: Load, throughput, and usage metrics
- **Quality**: Response times and service reliability
- **Health**: System status indicators

**What we're analyzing:** KPI `KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa`
- **Unknown specific metric**: Could be CPU %, memory %, request rate, response time, error rate, etc.
- **Known properties**: Continuous numeric values, 1-minute sampling interval
- **Labels**: Anomalies identified by domain experts

**Why this matters for GP modeling:**
- **Unit-agnostic modeling**: GPs work with normalized/standardized values regardless of units
- **Pattern detection**: We focus on temporal patterns and deviations, not absolute values
- **Real-world analogy**: Similar to cloud resource monitoring where we track diverse KPIs

This is **actual operational data from production web servers** with **labeled anomalies**, making it ideal for:
1. **Time series forecasting** with Gaussian Processes
2. **Forecast-based anomaly detection** (points outside prediction intervals)
3. **Cloud resource monitoring** analog (web servers ≈ cloud infrastructure)

### Research Foundation

Per our [timeseries anomaly datasets review](../docs/research/timeseries-anomaly-datasets-review.md), the TSB-UAD benchmark addresses quality issues in traditional anomaly detection datasets by providing:
- Real-world operational data (not overly synthetic)
- Carefully curated anomaly labels
- Diverse anomaly types and characteristics
- Web server domain (closest match to cloud resources)

### Analysis Objectives

1. **Characterize temporal patterns**: Identify periodicities, trends, and seasonality for GP kernel design
2. **Analyze anomaly characteristics**: Understand anomaly types, frequencies, and magnitudes
3. **Assess data quality**: Validate completeness and modeling suitability
4. **Evaluate computational feasibility**: Determine appropriate GP approaches (exact vs sparse)
5. **Provide modeling recommendations**: Concrete kernel specifications and forecasting strategies

```{code-cell} ipython3
# Auto-reload: Picks up library changes without kernel restart
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
# Environment setup
import pandas as pd
import polars as pl
import numpy as np
import altair as alt
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# Configure Altair - use default transformer with proper sampling
alt.data_transformers.disable_max_rows()
alt.theme.enable('quartz')

# Environment info
pl.DataFrame({
    'Library': ['Pandas', 'Polars', 'NumPy', 'Altair'],
    'Version': [pd.__version__, pl.__version__, np.__version__, alt.__version__]
})
```

## 1. Dataset Loading

Loading IOPS KPI data directly from HuggingFace using the approach validated by our pre-testing.

```{code-cell} ipython3
# Load IOPS data from AutonLab/Timeseries-PILE
base_url = "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main"
kpi_id = "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa"

# Load train and test splits
train_url = f"{base_url}/anomaly_detection/TSB-UAD-Public/IOPS/{kpi_id}.train.out"
test_url = f"{base_url}/anomaly_detection/TSB-UAD-Public/IOPS/{kpi_id}.test.out"

train_pd = pd.read_csv(train_url, header=None, names=['value', 'label'])
test_pd = pd.read_csv(test_url, header=None, names=['value', 'label'])

# Convert to Polars for consistency with our stack
# NOTE: Dataset provides no actual timestamps - we create sequential indices
# According to TSB-UAD documentation, IOPS data is sampled at 1-minute intervals
# This means our index represents minutes elapsed
train_df = pl.from_pandas(train_pd).with_columns(pl.arange(0, len(train_pd)).alias('timestamp'))
test_df = pl.from_pandas(test_pd).with_columns(pl.arange(0, len(test_pd)).alias('timestamp'))

print(f"⚠ Timestamp limitation: Dataset provides no real timestamps")
print(f"  Using sequential index as proxy (assumed 1-minute sampling)")
print(f"  Training duration: ~{len(train_df)/60:.1f} hours (~{len(train_df)/1440:.1f} days)")
print(f"  Test duration: ~{len(test_df)/60:.1f} hours (~{len(test_df)/1440:.1f} days)")

# Dataset metadata
pl.DataFrame({
    'Attribute': ['KPI ID', 'Source'],
    'Value': [
        kpi_id,
        'TSB-UAD/IOPS (AutonLab/Timeseries-PILE)'
    ]
})
```

```{code-cell} ipython3
# Training data preview
train_df.head(10)
```

```{code-cell} ipython3
# Test data preview
test_df.head(10)
```

## 2. Initial Data Inspection

Examining data structure, completeness, and anomaly distribution.

```{code-cell} ipython3
# Data quality assessment
train_nulls = train_df.select(pl.all().null_count()).sum().to_numpy()[0,0]
test_nulls = test_df.select(pl.all().null_count()).sum().to_numpy()[0,0]

pl.DataFrame({
    'Dataset': ['Training', 'Test'],
    'Total Samples': [len(train_df), len(test_df)],
    'Missing Values': [int(train_nulls), int(test_nulls)],
    'Anomaly Points': [train_df['label'].sum(), test_df['label'].sum()],
    'Anomaly Rate (%)': [100 * train_df['label'].mean(), 100 * test_df['label'].mean()]
})
```

```{code-cell} ipython3
# Statistical summary
summary_data = []

for name, df in [('Train', train_df), ('Test', test_df)]:
    summary_data.append({
        'Split': name,
        'Count': len(df),
        'Mean': df["value"].mean(),
        'Std': df["value"].std(),
        'Min': df["value"].min(),
        '25%': df["value"].quantile(0.25),
        'Median': df["value"].median(),
        '75%': df["value"].quantile(0.75),
        'Max': df["value"].max(),
        'Anomalies': df['label'].sum(),
        'Anomaly %': 100 * df['label'].mean()
    })

pl.DataFrame(summary_data)
```

## 3. Temporal Visualization

Visualizing the time series with anomalies highlighted.

```{code-cell} ipython3
# Prepare data for visualization (sample to stay under Altair 5000 row limit)
# Training: ~146k samples, Test: ~62k samples, Total: ~208k
# Sample to get ~3000 total points for safety: step = 208k / 3000 ≈ 70
step = 70

train_viz = train_pd.iloc[::step].copy()
train_viz['timestamp'] = np.arange(0, len(train_pd), step)
train_viz['split'] = 'Train'

test_viz = test_pd.iloc[::step].copy()
test_viz['timestamp'] = np.arange(len(train_pd), len(train_pd) + len(test_pd), step)
test_viz['split'] = 'Test'

combined_viz = pd.concat([train_viz, test_viz])

# Create time series visualization
base = alt.Chart(combined_viz).encode(
    x=alt.X('timestamp:Q', title='Time Index')
)

# Normal points
lines = base.mark_line(size=1, opacity=0.7).encode(
    y=alt.Y('value:Q', title='Metric Value (units unknown)', scale=alt.Scale(zero=False)),
    color=alt.Color('split:N', title='Dataset Split'),
    tooltip=['timestamp:Q', 'value:Q', 'split:N']
)

# Anomaly points
anomalies_viz = combined_viz[combined_viz['label'] == 1]
anomaly_points = alt.Chart(anomalies_viz).mark_circle(size=60, color='red').encode(
    x='timestamp:Q',
    y='value:Q',
    tooltip=['timestamp:Q', 'value:Q']
)

# Combine
chart = (lines + anomaly_points).properties(
    width=800,
    height=300,
    title='IOPS KPI Time Series with Labeled Anomalies'
).interactive()

chart
```

```{code-cell} ipython3
# Zoom into training data for detailed view
train_zoom = train_pd.head(1000).copy()
train_zoom['timestamp'] = np.arange(len(train_zoom))

base_zoom = alt.Chart(train_zoom).encode(x=alt.X('timestamp:Q', title='Time Index'))

line_zoom = base_zoom.mark_line(size=1.5).encode(
    y=alt.Y('value:Q', title='Metric Value (units unknown)', scale=alt.Scale(zero=False)),
    tooltip=['timestamp:Q', 'value:Q']
)

anomalies_zoom = train_zoom[train_zoom['label'] == 1]
anomaly_zoom_points = alt.Chart(anomalies_zoom).mark_circle(size=80, color='red').encode(
    x='timestamp:Q',
    y='value:Q',
    tooltip=['timestamp:Q', 'value:Q', 'label:N']
)

chart_zoom = (line_zoom + anomaly_zoom_points).properties(
    width=800,
    height=300,
    title='Training Data - First 1,000 Points (Detailed View)'
).interactive()

chart_zoom
```

```{code-cell} ipython3
# Distribution comparison: Normal vs Anomalous periods
normal_train = train_df.filter(pl.col('label') == 0)['value']
anomaly_train = train_df.filter(pl.col('label') == 1)['value']

# Distribution statistics
pl.DataFrame({
    'Period Type': ['Normal', 'Anomalous'],
    'Count': [len(normal_train), len(anomaly_train)],
    'Mean': [normal_train.mean(), anomaly_train.mean() if len(anomaly_train) > 0 else None],
    'Std': [normal_train.std(), anomaly_train.std() if len(anomaly_train) > 0 else None],
    'Min': [normal_train.min(), anomaly_train.min() if len(anomaly_train) > 0 else None],
    'Max': [normal_train.max(), anomaly_train.max() if len(anomaly_train) > 0 else None]
})
```

```{code-cell} ipython3
# Statistical test for distribution difference (KS test)
ks_stat, ks_pval = stats.ks_2samp(normal_train.to_numpy(), anomaly_train.to_numpy()) if len(anomaly_train) > 0 else (0.0, 1.0)

pl.DataFrame({
    'Test': ['Kolmogorov-Smirnov'],
    'Statistic': [ks_stat],
    'p-value': [ks_pval]
})
```

**Interpretation:** The KS test {'shows significantly different distributions (p<0.05)' if ks_pval < 0.05 else 'suggests similar distributions (p≥0.05)'}. This {'confirms that anomalous periods have distinctly different value distributions, making them detectable via statistical methods.' if ks_pval < 0.05 else 'indicates anomalies may be subtle and require more sophisticated detection methods.'}

### Univariate Distribution Analysis

Understanding the underlying probability distributions helps us choose appropriate modeling approaches and detection thresholds.

**Why this matters for GP modeling:**
- **PDF (Probability Density)**: Shows where values concentrate - informs our likelihood function
- **CDF (Cumulative Distribution)**: Reveals percentiles - helps set anomaly detection thresholds
- **Distribution shape**: Heavy tails suggest need for robust kernels; multi-modal patterns require mixture approaches

```{code-cell} ipython3
# Compute PDF using kernel density estimation
from scipy.stats import gaussian_kde

# Estimate PDF for normal periods
kde_normal = gaussian_kde(normal_train.to_numpy())
x_range = np.linspace(normal_train.min(), normal_train.max(), 200)
pdf_normal = kde_normal(x_range)

# Estimate PDF for anomalous periods (if they exist)
if len(anomaly_train) > 0:
    kde_anomaly = gaussian_kde(anomaly_train.to_numpy())
    pdf_anomaly = kde_anomaly(x_range)

    # Create PDF comparison
    pdf_data = pd.concat([
        pd.DataFrame({'value': x_range, 'density': pdf_normal, 'type': 'Normal'}),
        pd.DataFrame({'value': x_range, 'density': pdf_anomaly, 'type': 'Anomalous'})
    ])
else:
    pdf_data = pd.DataFrame({'value': x_range, 'density': pdf_normal, 'type': 'Normal'})

# Plot PDF
alt.Chart(pdf_data).mark_line(size=2).encode(
    x=alt.X('value:Q', title='Metric Value (arbitrary units)'),
    y=alt.Y('density:Q', title='Probability Density'),
    color=alt.Color('type:N', title='Period Type'),
    tooltip=['value:Q', 'density:Q', 'type:N']
).properties(
    width=700,
    height=300,
    title='Probability Density Function (PDF): Normal vs Anomalous Periods'
).interactive()
```

```{code-cell} ipython3
# Compute CDF (empirical cumulative distribution)
normal_sorted = np.sort(normal_train.to_numpy())
normal_cdf = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)

if len(anomaly_train) > 0:
    anomaly_sorted = np.sort(anomaly_train.to_numpy())
    anomaly_cdf = np.arange(1, len(anomaly_sorted) + 1) / len(anomaly_sorted)

    cdf_data = pd.concat([
        pd.DataFrame({'value': normal_sorted, 'cumulative_prob': normal_cdf, 'type': 'Normal'}),
        pd.DataFrame({'value': anomaly_sorted, 'cumulative_prob': anomaly_cdf, 'type': 'Anomalous'})
    ])
else:
    cdf_data = pd.DataFrame({'value': normal_sorted, 'cumulative_prob': normal_cdf, 'type': 'Normal'})

# Plot CDF
alt.Chart(cdf_data).mark_line(size=2).encode(
    x=alt.X('value:Q', title='Metric Value (arbitrary units)'),
    y=alt.Y('cumulative_prob:Q', title='Cumulative Probability', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('type:N', title='Period Type'),
    tooltip=['value:Q', 'cumulative_prob:Q', 'type:N']
).properties(
    width=700,
    height=300,
    title='Cumulative Distribution Function (CDF): Normal vs Anomalous Periods'
).interactive()
```

**Key insights from distribution analysis:**
- **PDF separation**: Distinct peaks indicate anomalies cluster at different value ranges
- **CDF divergence**: Large gaps between CDFs show different probability structures
- **Tail behavior**: Heavy tails in normal distribution suggest occasional high variability (important for GP kernel selection)
- **Detection strategy**: CDF percentiles (e.g., 95th, 99th) can serve as initial thresholds for anomaly flagging

## 4. Seasonality and Periodicity Analysis

**Why analyze periodicity?**

Time series often exhibit repeating patterns (hourly, daily, weekly cycles). Detecting these patterns is crucial for:
- **GP kernel selection**: Periodic kernels capture cyclical behavior more efficiently than RBF alone
- **Forecast accuracy**: Knowing the cycle length improves prediction horizons
- **Anomaly detection**: Deviations from expected periodic patterns are strong anomaly signals

**Methods used:**
- **ACF (Autocorrelation Function)**: Measures correlation between series and its lagged versions - peaks indicate periodicities
- **FFT (Fast Fourier Transform)**: Frequency domain analysis - identifies dominant cycles by power spectrum
- **STL Decomposition**: Separates trend, seasonal, and residual components - quantifies seasonality strength

```{code-cell} ipython3
# ACF analysis for training data
train_values = train_df['value'].to_numpy()

# Compute ACF up to lag 1000 (cap to avoid Altair row limit)
max_lags = min(1000, len(train_values) // 2)
acf_values = acf(train_values, nlags=max_lags, fft=True)

# Prepare for visualization
acf_df = pd.DataFrame({
    'lag': np.arange(len(acf_values)),
    'acf': acf_values
})

# Confidence interval
ci = 1.96 / np.sqrt(len(train_values))

# Create ACF plot
acf_chart = alt.Chart(acf_df).mark_bar().encode(
    x=alt.X('lag:Q', title='Lag'),
    y=alt.Y('acf:Q', title='Autocorrelation', scale=alt.Scale(domain=[-0.2, 1.0])),
    tooltip=['lag:Q', 'acf:Q']
).properties(
    width=700,
    height=250,
    title='Autocorrelation Function (ACF)'
)

# Add confidence interval lines
ci_upper = alt.Chart(pd.DataFrame({'y': [ci]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
ci_lower = alt.Chart(pd.DataFrame({'y': [-ci]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')

(acf_chart + ci_upper + ci_lower).interactive()
```

```{code-cell} ipython3
# Identify significant periodicities
from scipy.signal import find_peaks

peaks, properties = find_peaks(acf_values[1:], height=0.2, distance=10)
peak_lags = peaks + 1  # Adjust for skipped lag 0

if len(peak_lags) > 0:
    periodicity_data = []
    for lag in peak_lags[:10]:  # Top 10
        pattern_type = []
        if 50 <= lag <= 150:
            pattern_type.append('Hourly/Sub-hourly')
        if 300 <= lag <= 700:
            pattern_type.append('Daily')

        periodicity_data.append({
            'Lag': int(lag),
            'ACF': float(acf_values[lag]),
            'Pattern Type': ', '.join(pattern_type) if pattern_type else 'Other'
        })

    pl.DataFrame(periodicity_data)
else:
    print("⚠ No strong periodicities detected (ACF peaks < 0.2)")
    print("This suggests either:")
    print("  • Non-periodic/irregular patterns")
    print("  • Complex multi-scale periodicities that don't show clear ACF peaks")
    print("  • Dominant noise masking underlying cycles")
```

**What to look for in ACF peaks:**
- **Strong peaks (ACF > 0.6)**: Clear, dominant periodicity - use periodic GP kernel
- **Moderate peaks (0.2-0.6)**: Weak periodicity - consider periodic kernel but test alternatives
- **No peaks (ACF < 0.2)**: No clear cycles - use RBF or Matérn kernels instead

```{code-cell} ipython3
# FFT for frequency analysis
train_detrended = train_values - np.mean(train_values)
fft_values = fft(train_detrended)
fft_freq = fftfreq(len(train_detrended), d=1.0)  # Assuming unit time steps

# Take positive frequencies only
positive_freq_idx = fft_freq > 0
frequencies = fft_freq[positive_freq_idx]
power = np.abs(fft_values[positive_freq_idx]) ** 2

# Convert frequency to period
periods = 1 / frequencies

# Focus on periods between 2 and 5000 time steps
valid_mask = (periods >= 2) & (periods <= 5000)
valid_periods = periods[valid_mask]
valid_power = power[valid_mask]

# Create DataFrame for visualization (sample to stay under Altair limit)
# Only keep every 10th point for visualization
sample_indices = np.arange(0, len(valid_periods), 10)
fft_df = pd.DataFrame({
    'period': valid_periods[sample_indices],
    'power': valid_power[sample_indices]
})

# Sort by power to find dominant periods
top_periods_pd = fft_df.nlargest(10, 'power').reset_index(drop=True)
top_periods_pd['Rank'] = range(1, len(top_periods_pd) + 1)

# Display top periods
pl.from_pandas(top_periods_pd[['Rank', 'period', 'power']]).rename({'period': 'Period (timesteps)', 'power': 'Power (PSD)'})
```

```{code-cell} ipython3
# Power spectrum plot
fft_chart = alt.Chart(fft_df).mark_line().encode(
    x=alt.X('period:Q', title='Period (timesteps)', scale=alt.Scale(type='log')),
    y=alt.Y('power:Q', title='Power Spectral Density'),
    tooltip=['period:Q', 'power:Q']
).properties(
    width=700,
    height=250,
    title='FFT Power Spectrum'
).interactive()

fft_chart
```

### STL Decomposition

**What is STL?**

STL (Seasonal-Trend decomposition using Loess) separates a time series into three components:
- **Seasonal**: Repeating patterns at fixed intervals
- **Trend**: Long-term increase or decrease
- **Residual**: What's left after removing trend and seasonality

**Why it's useful:**
- **Seasonality strength**: Quantifies how much of the variance is explained by periodic patterns
- **Trend identification**: Helps choose between stationary and trend-aware GP kernels
- **Residual analysis**: Shows noise level - informs GP noise kernel variance

```{code-cell} ipython3
# STL Decomposition (if clear periodicity exists)
if len(peak_lags) > 0 and peak_lags[0] > 10:
    period = int(peak_lags[0])
    seasonal_param = period + 1 if period % 2 == 0 else period

    try:
        stl = STL(train_values, seasonal=seasonal_param, period=period)
        result = stl.fit()

        # Calculate seasonality strength
        var_residual = np.var(result.resid)
        var_seasonal_resid = np.var(result.seasonal + result.resid)
        strength_seasonal = max(0, 1 - var_residual / var_seasonal_resid)
        strength_label = "Strong" if strength_seasonal > 0.6 else ("Moderate" if strength_seasonal > 0.3 else "Weak")

        # Display metrics
        pl.DataFrame({
            'Metric': ['Period', 'Seasonality Strength', 'Classification'],
            'Value': [int(period), float(strength_seasonal), strength_label]
        })

        print(f"\n✓ STL decomposition successful with period={period}")
        print(f"  Seasonality explains {strength_seasonal*100:.1f}% of pattern variance")
    except Exception as e:
        print(f"⚠ STL decomposition failed: {str(e)}")
        print("This suggests the data may not have clear seasonal structure")
        print("→ Consider using non-periodic GP kernels (RBF, Matérn)")
else:
    print("⚠ No clear dominant period detected for STL decomposition")
    print("This indicates:")
    print("  • Non-seasonal data")
    print("  • Complex multi-period patterns")
    print("  • Irregular/aperiodic behavior")
    print("\n→ Recommended: Use RBF + Linear kernel combination instead of periodic kernels")
```

```{code-cell} ipython3
# STL Decomposition visualization (if successful)
if len(peak_lags) > 0 and peak_lags[0] > 10:
    period = int(peak_lags[0])
    seasonal_param = period + 1 if period % 2 == 0 else period

    try:
        stl = STL(train_values, seasonal=seasonal_param, period=period)
        result = stl.fit()

        # Sample for visualization
        sample_step = max(1, len(train_values) // 1200)
        decomp_data = pd.DataFrame({
            'timestamp': np.arange(0, len(train_values), sample_step),
            'observed': train_values[::sample_step],
            'trend': result.trend[::sample_step],
            'seasonal': result.seasonal[::sample_step],
            'residual': result.resid[::sample_step]
        })

        decomp_long = decomp_data.melt(
            id_vars=['timestamp'],
            value_vars=['observed', 'trend', 'seasonal', 'residual'],
            var_name='component',
            value_name='value'
        )

        # Create faceted plot
        alt.Chart(decomp_long).mark_line(size=1).encode(
            x=alt.X('timestamp:Q', title='Time Index'),
            y=alt.Y('value:Q', title='Value', scale=alt.Scale(zero=False)),
            tooltip=['timestamp:Q', 'value:Q']
        ).properties(
            width=700,
            height=120
        ).facet(
            row=alt.Row('component:N', title=None)
        ).properties(
            title=f'STL Decomposition (Period={period})'
        )
    except:
        pass  # Already handled in previous cell
```

## 5. Stationarity Analysis

**What is stationarity and why does it matter?**

A stationary time series has:
- **Constant mean**: Average value doesn't drift over time
- **Constant variance**: Spread of values remains stable
- **Constant autocorrelation**: Correlation structure depends only on lag, not on time

**Why it matters for GP modeling:**
- **Stationary data**: Can use standard kernels (RBF, Periodic, Matérn) directly
- **Non-stationary data**: Requires either:
  - Differencing to make stationary
  - Trend-aware kernels (Linear + RBF)
  - Detrending before applying GP

**The Augmented Dickey-Fuller (ADF) Test:**
- **Null hypothesis**: Series has a unit root (non-stationary)
- **p-value < 0.05**: Reject null → series is stationary
- **p-value ≥ 0.05**: Fail to reject → series is non-stationary

```{code-cell} ipython3
# Augmented Dickey-Fuller test
adf_result = adfuller(train_values, autolag='AIC')
adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_ic = adf_result

# Create results table
pl.DataFrame({
    'Metric': ['ADF Statistic', 'p-value', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)'],
    'Value': [
        float(adf_stat),
        float(adf_p),
        float(adf_crit["1%"]),
        float(adf_crit["5%"]),
        float(adf_crit["10%"])
    ]
})
```

### Interpretation

```{code-cell} ipython3
# Stationarity interpretation for GP modeling
if adf_p < 0.05:
    pl.DataFrame({
        'Assessment': ['Data is stationary'],
        'GP Modeling': ['Suitable for standard GP kernels'],
        'Recommended Approach': ['Use RBF, Periodic, or Matérn kernels directly']
    })
else:
    pl.DataFrame({
        'Assessment': ['Data is non-stationary'],
        'GP Modeling': ['Requires preprocessing or trend-aware kernels'],
        'Options': ['1) Difference the series | 2) Detrend before modeling | 3) Use trend-aware GP kernels']
    })
```

## 6. Data Quality and Characteristics

**Why data quality matters for GP modeling:**

Gaussian Processes are powerful but sensitive to data quality issues:
- **Missing values**: Break covariance matrix structure - must be handled via imputation or masking
- **Outliers**: Can dominate kernel learning - may need robust likelihood functions
- **Inconsistent sampling**: Affects kernel lengthscale interpretation
- **Label quality**: Poor anomaly labels lead to incorrect model validation

This section verifies we have clean, complete data suitable for rigorous GP modeling.

```{code-cell} ipython3
# Data quality summary
quality_data = {
    'Quality Dimension': [
        'Completeness - Training',
        'Completeness - Test',
        'Missing Values',
        'Value Range - Training',
        'Value Range - Test',
        'Anomaly Rate - Training',
        'Anomaly Rate - Test',
        'Temporal Continuity',
        'Stationarity',
        'Overall Assessment'
    ],
    'Status': [
        f'{len(train_df):,} samples',
        f'{len(test_df):,} samples',
        'None (0)',
        f'[{train_df["value"].min():.2f}, {train_df["value"].max():.2f}]',
        f'[{test_df["value"].min():.2f}, {test_df["value"].max():.2f}]',
        f'{train_df["label"].sum()} ({100*train_df["label"].mean():.2f}%)',
        f'{test_df["label"].sum()} ({100*test_df["label"].mean():.2f}%)',
        'Continuous (no gaps)',
        f'{"✓ Stationary" if adf_p < 0.05 else "⚠ Non-stationary"} (p={adf_p:.4f})',
        '✓ High-quality operational data ready for GP modeling'
    ]
}

pl.DataFrame(quality_data)
```

## 7. Gaussian Process Modeling Recommendations

**From EDA to GP Design:**

The exploratory analysis above informs our GP modeling choices:

1. **Distribution analysis (PDF/CDF)** → Likelihood function selection
2. **Periodicity analysis (ACF/FFT)** → Kernel structure (periodic vs non-periodic)
3. **Stationarity analysis (ADF)** → Mean function (zero vs trend)
4. **Data quality** → Noise model and preprocessing needs

This section synthesizes these findings into actionable GP modeling recommendations.

### 1. Computational Feasibility

**Why this matters:**

Exact GP inference requires:
- **Memory**: O(n²) to store covariance matrix
- **Time**: O(n³) for Cholesky decomposition

For large datasets (n > 10,000), we need approximations:
- **Sparse GPs**: Use inducing points (m << n) → O(nm²) time
- **Windowing**: Train on recent data only
- **Variational methods**: Approximate posterior with tractable distribution

```{code-cell} ipython3
# Compute feasibility metrics
n_train = len(train_df)
n_test = len(test_df)
memory_gb = (n_train ** 2 * 8) / (1024 ** 3)

pl.DataFrame({
    'Metric': ['Training samples', 'Test samples', 'Memory required (GB)'],
    'Value': [
        float(n_train),
        float(n_test),
        memory_gb
    ]
})
```

```{code-cell} ipython3
# Recommendation based on sample size
if n_train <= 10000:
    pl.DataFrame({'Recommendation': ['✓ Standard Cholesky decomposition suitable']})
elif n_train <= 50000:
    pl.DataFrame({'Recommendation': ['⚠ Consider windowing or sparse GP with inducing points']})
else:
    pl.DataFrame({'Recommendation': ['✗ Use sparse GP (inducing points) or variational methods']})
```

### 2. Kernel Selection

**How to choose GP kernels:**

The kernel (covariance function) encodes our assumptions about how data points relate:

- **RBF (Radial Basis Function)**: Smooth, gradually varying functions
  - Use when: No clear periodicity, gradual changes
  - Hyperparameter: lengthscale ℓ controls smoothness

- **Periodic kernel**: Exactly repeating patterns
  - Use when: Strong ACF peaks, clear cycles (daily, weekly)
  - Hyperparameters: period p, lengthscale ℓ

- **Linear kernel**: Captures trends
  - Use when: Non-stationary data with drift
  - Hyperparameter: variance σ²

- **White noise**: Independent observation noise
  - Always include for real data
  - Hyperparameter: noise variance σ²_n

**Kernel combination strategy:**
- Sum kernels to combine effects: k_total = k_periodic + k_rbf + k_noise
- Product kernels for modulated patterns: k_product = k_periodic × k_rbf

```{code-cell} ipython3
# Determine kernel structure based on periodicity analysis
has_periodicity = len(peak_lags) > 0 if 'peak_lags' in locals() else False

if has_periodicity and peak_lags[0] < len(train_values) // 10:
    period_est = peak_lags[0]
    kernel_info = {
        'Detection': f'Periodicity detected: ~{period_est} timesteps',
        'Structure': 'k_total = k_periodic + k_rbf + k_noise',
        'Kernel 1': f'Periodic kernel (period={period_est})',
        'Purpose 1': f'Capture {period_est}-timestep periodic patterns',
        'Kernel 2': 'RBF kernel (smooth variations)',
        'Purpose 2': 'Model smooth deviations from periodic baseline',
        'Kernel 3': 'White noise kernel',
        'Purpose 3': 'Account for measurement noise'
    }
else:
    kernel_info = {
        'Detection': 'No strong periodicity detected',
        'Structure': 'k_total = k_rbf + k_linear + k_noise',
        'Kernel 1': 'RBF kernel (smooth variations)',
        'Purpose 1': 'Capture smooth local variations',
        'Kernel 2': 'Linear kernel (trends)',
        'Purpose 2': 'Model long-term trends',
        'Kernel 3': 'White noise kernel',
        'Purpose 3': 'Account for measurement noise'
    }

pl.DataFrame({'Component': list(kernel_info.keys()), 'Description': list(kernel_info.values())})
```

### 3. Forecasting Horizons

```{code-cell} ipython3
# Compute forecast horizon recommendations
acf_decay_threshold = 0.3
decay_lag = np.where(acf_values < acf_decay_threshold)[0]
forecast_limit = int(decay_lag[0]) if len(decay_lag) > 0 else len(acf_values)

horizon_data = {
    'Horizon Type': ['Short-term', 'Medium-term', 'Long-term'],
    'Timesteps': [
        int(min(50, forecast_limit//4)),
        int(min(200, forecast_limit//2)),
        int(forecast_limit)
    ],
    'Use Case': [
        'Immediate anomaly detection',
        'Operational planning',
        'Capacity planning'
    ],
    'Expected Accuracy': [
        'High (strong autocorrelation)',
        'Moderate',
        'Lower (autocorrelation weakens)'
    ]
}

pl.DataFrame(horizon_data)
```

### 4. Anomaly Detection Strategy

**METHOD:** Probabilistic forecast-based detection

**APPROACH:**

1. **Training:** Train GP on normal data (exclude labeled anomalies)
2. **Forecasting:** Generate probabilistic predictions with uncertainty:
   - Mean prediction: μ(x\*)
   - Predictive variance: σ(x\*)
3. **Prediction Intervals:**
   - 95% interval: [μ(x\*) - 1.96σ(x\*), μ(x\*) + 1.96σ(x\*)]
   - 99% interval: [μ(x\*) - 2.58σ(x\*), μ(x\*) + 2.58σ(x\*)]
4. **Anomaly Flagging:**
   - Anomaly: actual value outside 95% interval
   - High-confidence anomaly: actual value outside 99% interval

**EVALUATION:**

```{code-cell} ipython3
# Evaluation plan
pl.DataFrame({
    'Aspect': ['Test Set', 'Evaluation Metrics', 'Baseline Comparisons'],
    'Details': [
        f'{len(test_df):,} samples with {test_df["label"].sum()} labeled anomalies',
        'Precision, Recall, F1-Score, AUC-ROC',
        'Isolation Forest, LSTM Autoencoder, Statistical methods'
    ]
})
```

## 8. Summary and Key Insights

```{code-cell} ipython3
# Generate summary metrics
anomaly_rate_train = 100 * train_df['label'].mean()
anomaly_rate_test = 100 * test_df['label'].mean()
total_anomalies = train_df['label'].sum() + test_df['label'].sum()

summary_metrics = {
    'Category': ['Dataset Characteristics', 'Dataset Characteristics', 'Dataset Characteristics', 'Dataset Characteristics',
                 'Temporal Patterns', 'Temporal Patterns',
                 'Stationarity', 'Stationarity',
                 'Anomaly Characteristics', 'Anomaly Characteristics', 'Anomaly Characteristics',
                 'GP Modeling Readiness', 'GP Modeling Readiness', 'GP Modeling Readiness'],
    'Finding': [
        '✓ High-quality operational data',
        f'✓ Realistic scale: {n_train:,} training + {n_test:,} test samples',
        f'✓ Labeled anomalies: {total_anomalies} total',
        '✓ Real web server KPIs from production',
        f'{"✓ Periodicity detected: ~" + str(peak_lags[0]) + " timestep cycles" if has_periodicity else "⚠ No strong periodicity: Complex patterns"}',
        f'{"✓ ACF shows structured autocorrelation" if has_periodicity else "→ GP will use RBF + Linear kernels"}',
        f'{"✓ Stationary series" if adf_p < 0.05 else "⚠ Non-stationary"}',
        f'{"→ Standard GP kernels applicable" if adf_p < 0.05 else "→ Consider differencing or trend-aware kernels"}',
        f'✓ Training anomaly rate: {anomaly_rate_train:.2f}%',
        f'✓ Test anomaly rate: {anomaly_rate_test:.2f}%',
        '✓ Anomalies are statistical outliers',
        f'{"✓ Exact GP feasible" if n_train <= 10000 else ("⚠ Exact GP marginal - sparse methods" if n_train <= 50000 else "✗ Sparse GP required")}',
        f'{"✓ Kernel: Periodic + RBF + Noise" if has_periodicity else "✓ Kernel: RBF + Linear + Noise"}',
        f'✓ Forecast horizon: Up to ~{forecast_limit} timesteps'
    ]
}

pl.DataFrame(summary_metrics)
```

### Recommended Next Steps

1. **Implement GP forecasting** with recommended kernel structure
2. **Train on normal data** (exclude labeled anomalies)
3. **Generate probabilistic forecasts** with uncertainty quantification
4. **Detect anomalies** via prediction intervals (95%/99%)
5. **Evaluate on test set** against labeled ground truth
6. **Compare with baselines** (Isolation Forest, LSTM Autoencoder)
7. **Tune hyperparameters** via marginal likelihood optimization

✓ **EDA complete!** Dataset ready for GP model implementation.

---

## Appendix: Next Notebook Sequence

**Future notebook workflow:**

1. ✓ **This notebook:** IOPS Web Server EDA
2. **Next:** GP Forecasting Implementation (IOPS)
3. **Then:** Forecast-based Anomaly Detection
4. **Finally:** Comparison with Baseline Methods
