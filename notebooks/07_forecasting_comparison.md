---
jupytext:
  formats: notebooks//md:myst,notebooks/_build//ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: .venv
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.12.12
---

# IOPS Time Series Forecasting: Baseline Models & Future Approaches

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nehalecky/hello-cloud/blob/master/notebooks/published/07_forecasting_comparison.ipynb)

**Objective:** Build end-to-end forecasting workflow with working baselines and placeholders for sophisticated models.

**Dataset:** IOPS KPI from TSB-UAD benchmark (295K samples, 205 days at 1-minute intervals)

**Approach:** Start simple, validate workflow, then iterate toward foundation models.

---

## 0. Auto-Reload Configuration

**Hot Reload**: Enable automatic reloading of library code (src/) without kernel restart.

```{code-cell} ipython3
# Auto-reload: Picks up library changes without kernel restart
%load_ext autoreload
%autoreload 2
```

---

## 1. Data Loading

Loading IOPS KPI data from HuggingFace, reusing the approach from notebook 03 (EDA).

**Dataset Details:**
- **KPI ID:** `KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa`
- **Source:** TSB-UAD/IOPS via AutonLab/Timeseries-PILE
- **Size:** 146K training, 149K test samples
- **Sampling:** 1-minute intervals (synthetic timestamps)

```{code-cell} ipython3
# Environment Setup
# Local: Uses installed hellocloud
# Colab: Installs from GitHub
try:
    import hellocloud
except ImportError:
    !pip install -q git+https://github.com/nehalecky/hello-cloud.git
    import hellocloud
```

```{code-cell} ipython3
# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Forecasting library
from hellocloud.modeling.forecasting import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    mae, rmse, mape, mase,
    compute_all_metrics
)

# Visualization
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Random seed for reproducibility
np.random.seed(42)
```

```{code-cell} ipython3
# Load IOPS data from HuggingFace (same as EDA notebook)
base_url = "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main"
kpi_id = "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa"

# Load train and test splits
train_url = f"{base_url}/anomaly_detection/TSB-UAD-Public/IOPS/{kpi_id}.train.out"
test_url = f"{base_url}/anomaly_detection/TSB-UAD-Public/IOPS/{kpi_id}.test.out"

print("Downloading IOPS data from HuggingFace...")
train_df = pd.read_csv(train_url, header=None, names=['value', 'label'])
test_df = pd.read_csv(test_url, header=None, names=['value', 'label'])

# Add sequential timestamps (1-minute intervals)
train_df['timestamp'] = np.arange(len(train_df))
test_df['timestamp'] = np.arange(len(test_df))

print(f"✓ Data loaded successfully")
print(f"  Training: {len(train_df):,} samples (~{len(train_df)/1440:.1f} days)")
print(f"  Test: {len(test_df):,} samples (~{len(test_df)/1440:.1f} days)")
print(f"  Total: {len(train_df) + len(test_df):,} samples (~{(len(train_df) + len(test_df))/1440:.1f} days)")
```

```{code-cell} ipython3
# Extract arrays for modeling
y_train_full = train_df['value'].values
y_test_full = test_df['value'].values

print(f"Training data: {len(y_train_full):,} samples")
print(f"Test data: {len(y_test_full):,} samples")
print(f"Value range: [{y_train_full.min():.2f}, {y_train_full.max():.2f}]")
```

---

## 2. Data Preprocessing: Subsampling for Computational Efficiency

**Why subsample?**

1. **Computation:** 146K training samples is expensive for iterative model development
2. **TimesFM context limits:** Foundation models have fixed context windows (1024 for TimesFM)
3. **Baseline validation:** Start fast, validate workflow, then scale up

**Subsampling strategy:**
- **20x subsampling:** 146K → 7.3K points (manageable for baselines)
- **Preserves temporal structure:** From notebook 03 EDA, we know periodicities are ~250 and ~1250 timesteps
- **Nyquist-aware:** 20x sampling still captures slow period (1250/20 = 62.5 samples/cycle)

```{code-cell} ipython3
# Subsample training data (20x reduction for baselines)
subsample_factor = 20
subsample_indices = np.arange(0, len(y_train_full), subsample_factor)

y_train = y_train_full[subsample_indices]

print(f"Subsampling training data:")
print(f"  Original: {len(y_train_full):,} samples")
print(f"  Subsampled: {len(y_train):,} samples (factor={subsample_factor})")
print(f"  Reduction: {100*(1-len(y_train)/len(y_train_full)):.1f}%")
```

```{code-cell} ipython3
# Create train/val/test splits from subsampled data
# Train: 80% | Val: 10% | Test: 10%
n_train = int(0.8 * len(y_train))
n_val = int(0.1 * len(y_train))

y_train_split = y_train[:n_train]
y_val_split = y_train[n_train:n_train+n_val]
y_test_split = y_train[n_train+n_val:]

print(f"Data splits (from subsampled training data):")
print(f"  Train: {len(y_train_split):,} samples")
print(f"  Val: {len(y_val_split):,} samples")
print(f"  Test: {len(y_test_split):,} samples")
```

```{code-cell} ipython3
# Visualize subsampled data vs full data (zoomed view)
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: First 5000 timesteps - Full data vs subsampled
n_viz = 5000
axes[0].plot(np.arange(n_viz), y_train_full[:n_viz], 'k-', linewidth=0.5, alpha=0.5, label='Full data')
axes[0].scatter(subsample_indices[:n_viz//subsample_factor],
                y_train[:n_viz//subsample_factor],
                c='red', s=20, alpha=0.7, zorder=5,
                label=f'Subsampled (every {subsample_factor}th)')
axes[0].set_title(f'Subsampling Validation: First {n_viz} Timesteps', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Original Timestep')
axes[0].set_ylabel('IOPS Value')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Distribution comparison
axes[1].hist(y_train_full, bins=50, alpha=0.5, density=True, color='black', label='Full data')
axes[1].hist(y_train, bins=50, alpha=0.5, density=True, color='red', label='Subsampled')
axes[1].set_title('Value Distribution: Full vs Subsampled', fontsize=14, fontweight='bold')
axes[1].set_xlabel('IOPS Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Subsampling preserves distribution and visual patterns")
```

---

## 3. Baseline Forecasting (WORKING)

**Goal:** Establish performance floor - any sophisticated model must beat these baselines.

### 3.1 Naive Baseline

**Method:** Forecast = last observed value (persistence model)

**When it works:** Strong autocorrelation, short-term forecasts

```{code-cell} ipython3
# Initialize Naive forecaster
naive_forecaster = NaiveForecaster()

# Fit on training data
naive_forecaster.fit(y_train_split)

# Forecast multiple horizons
horizons = {
    '12 steps (~12 minutes)': 12,
    '62 steps (~1 hour)': 62,
    '250 steps (~4 hours)': 250
}

naive_results = {}

for horizon_name, horizon in horizons.items():
    # Forecast
    forecast = naive_forecaster.forecast(horizon=horizon)

    # Compute metrics using validation set
    if len(y_val_split) >= horizon:
        y_true = y_val_split[:horizon]

        metrics = {
            'MAE': mae(y_true, forecast),
            'RMSE': rmse(y_true, forecast),
            'MAPE': mape(y_true, forecast),
            'MASE': mase(y_true, forecast, y_train_split)
        }

        naive_results[horizon_name] = {
            'forecast': forecast,
            'y_true': y_true,
            'metrics': metrics
        }

        print(f"\nNaive Forecast - {horizon_name}")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAPE: {metrics['MAPE']:.3f}")
        print(f"  MASE: {metrics['MASE']:.3f}")
```

```{code-cell} ipython3
# Visualize Naive forecasts
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for idx, (horizon_name, result) in enumerate(naive_results.items()):
    ax = axes[idx]

    # Plot last 100 training points for context
    context_window = 100
    ax.plot(np.arange(-context_window, 0), y_train_split[-context_window:],
            'k-', linewidth=1.5, label='Training data', alpha=0.7)

    # Plot forecast vs actual
    forecast_steps = np.arange(len(result['forecast']))
    ax.plot(forecast_steps, result['y_true'], 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(forecast_steps, result['forecast'], 'r--', linewidth=2, label='Naive forecast', alpha=0.8)

    # Add vertical line at forecast start
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)

    ax.set_title(f'Naive Forecast - {horizon_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (relative to forecast start)')
    ax.set_ylabel('IOPS Value')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 Seasonal Naive Baseline

**Method:** Forecast = value from `period` steps ago (seasonal persistence)

**Periodicities from EDA (notebook 03):**
- **Fast period:** ~250 timesteps (≈4 hours at full sampling)
- **Slow period:** ~1250 timesteps (≈21 hours at full sampling)
- **At 20x subsampling:** Fast = 13 steps, Slow = 62 steps

**Test both periods:**

```{code-cell} ipython3
# Test both periodicities from EDA
periods = {
    'Fast period (13 steps)': 13,   # 250 / 20 = 12.5 ≈ 13
    'Slow period (62 steps)': 62    # 1250 / 20 = 62.5 ≈ 62
}

seasonal_results = {}

for period_name, period in periods.items():
    # Initialize Seasonal Naive forecaster
    sn_forecaster = SeasonalNaiveForecaster(period=period)
    sn_forecaster.fit(y_train_split)

    # Forecast at 62-step horizon (1 slow period)
    horizon = 62
    forecast = sn_forecaster.forecast(horizon=horizon)

    # Compute metrics
    if len(y_val_split) >= horizon:
        y_true = y_val_split[:horizon]

        metrics = {
            'MAE': mae(y_true, forecast),
            'RMSE': rmse(y_true, forecast),
            'MAPE': mape(y_true, forecast),
            'MASE': mase(y_true, forecast, y_train_split)
        }

        seasonal_results[period_name] = {
            'period': period,
            'forecast': forecast,
            'y_true': y_true,
            'metrics': metrics
        }

        print(f"\nSeasonal Naive - {period_name}")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAPE: {metrics['MAPE']:.3f}")
        print(f"  MASE: {metrics['MASE']:.3f}")
```

```{code-cell} ipython3
# Visualize Seasonal Naive forecasts
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for idx, (period_name, result) in enumerate(seasonal_results.items()):
    ax = axes[idx]

    # Plot last 150 training points for context
    context_window = 150
    ax.plot(np.arange(-context_window, 0), y_train_split[-context_window:],
            'k-', linewidth=1.5, label='Training data', alpha=0.7)

    # Highlight the seasonal reference window
    period = result['period']
    ax.axvspan(-period, 0, alpha=0.1, color='orange', label=f'Seasonal reference (period={period})')

    # Plot forecast vs actual
    forecast_steps = np.arange(len(result['forecast']))
    ax.plot(forecast_steps, result['y_true'], 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(forecast_steps, result['forecast'], 'g--', linewidth=2, label='Seasonal Naive', alpha=0.8)

    # Add vertical line at forecast start
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)

    ax.set_title(f'Seasonal Naive - {period_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (relative to forecast start)')
    ax.set_ylabel('IOPS Value')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# Compare seasonal naive performance
print("\n" + "="*70)
print("SEASONAL NAIVE COMPARISON: Which period performs better?")
print("="*70)

best_period = None
best_mase = float('inf')

for period_name, result in seasonal_results.items():
    metrics = result['metrics']
    print(f"\n{period_name}:")
    print(f"  MASE: {metrics['MASE']:.3f}")

    if metrics['MASE'] < best_mase:
        best_mase = metrics['MASE']
        best_period = period_name

print(f"\n→ Best period: {best_period} (MASE = {best_mase:.3f})")
print("="*70)
```

### 3.3 Moving Average Baseline

**Method:** Forecast = average of last `window` values

**Window sizes to test:** 25, 50, 100 (at subsampled scale)

```{code-cell} ipython3
# Test multiple window sizes
window_sizes = [25, 50, 100]
ma_results = {}

for window_size in window_sizes:
    # Initialize Moving Average forecaster
    ma_forecaster = MovingAverageForecaster(window=window_size)
    ma_forecaster.fit(y_train_split)

    # Forecast at 62-step horizon
    horizon = 62
    forecast = ma_forecaster.forecast(horizon=horizon)

    # Compute metrics
    if len(y_val_split) >= horizon:
        y_true = y_val_split[:horizon]

        metrics = {
            'MAE': mae(y_true, forecast),
            'RMSE': rmse(y_true, forecast),
            'MAPE': mape(y_true, forecast),
            'MASE': mase(y_true, forecast, y_train_split)
        }

        ma_results[f'Window={window_size}'] = {
            'window': window_size,
            'forecast': forecast,
            'y_true': y_true,
            'metrics': metrics
        }

        print(f"\nMoving Average - Window={window_size}")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAPE: {metrics['MAPE']:.3f}")
        print(f"  MASE: {metrics['MASE']:.3f}")
```

```{code-cell} ipython3
# Visualize Moving Average forecasts
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for idx, (window_name, result) in enumerate(ma_results.items()):
    ax = axes[idx]

    # Plot last 150 training points for context
    context_window = 150
    ax.plot(np.arange(-context_window, 0), y_train_split[-context_window:],
            'k-', linewidth=1.5, label='Training data', alpha=0.7)

    # Highlight the moving average window
    window = result['window']
    ax.axvspan(-window, 0, alpha=0.1, color='purple', label=f'MA window (size={window})')

    # Plot forecast vs actual
    forecast_steps = np.arange(len(result['forecast']))
    ax.plot(forecast_steps, result['y_true'], 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax.plot(forecast_steps, result['forecast'], 'm--', linewidth=2, label='MA forecast', alpha=0.8)

    # Add vertical line at forecast start
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)

    ax.set_title(f'Moving Average - {window_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (relative to forecast start)')
    ax.set_ylabel('IOPS Value')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.4 Baseline Comparison Table

**Performance floor:** Any sophisticated model must beat these baselines.

```{code-cell} ipython3
# Compile all baseline results into comparison table
comparison_data = []

# Naive baseline (use 62-step horizon for consistency)
if '62 steps (~1 hour)' in naive_results:
    result = naive_results['62 steps (~1 hour)']
    comparison_data.append({
        'Model': 'Naive',
        'Configuration': 'last_value',
        'MAE': result['metrics']['MAE'],
        'RMSE': result['metrics']['RMSE'],
        'MAPE': result['metrics']['MAPE'],
        'MASE': result['metrics']['MASE']
    })

# Seasonal Naive baselines
for period_name, result in seasonal_results.items():
    comparison_data.append({
        'Model': 'Seasonal Naive',
        'Configuration': f"period={result['period']}",
        'MAE': result['metrics']['MAE'],
        'RMSE': result['metrics']['RMSE'],
        'MAPE': result['metrics']['MAPE'],
        'MASE': result['metrics']['MASE']
    })

# Moving Average baselines
for window_name, result in ma_results.items():
    comparison_data.append({
        'Model': 'Moving Average',
        'Configuration': f"window={result['window']}",
        'MAE': result['metrics']['MAE'],
        'RMSE': result['metrics']['RMSE'],
        'MAPE': result['metrics']['MAPE'],
        'MASE': result['metrics']['MASE']
    })

# Create DataFrame and sort by MASE
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('MASE').reset_index(drop=True)

print("\n" + "="*80)
print("BASELINE MODEL COMPARISON (62-step forecast horizon)")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Identify best baseline
best_baseline = comparison_df.iloc[0]
print(f"\n→ Best Baseline: {best_baseline['Model']} ({best_baseline['Configuration']})")
print(f"   MASE = {best_baseline['MASE']:.3f} (performance floor to beat)")
print("="*80)
```

---

## 4. ARIMA Forecasting (PLACEHOLDER)

**To Be Implemented:** ARIMA wrapper using statsmodels

**What will be tested:**
1. **Auto-select (p,d,q) with `auto_arima`:**
   - Automated order selection via AIC/BIC
   - Seasonal vs non-seasonal based on EDA findings

2. **Compare to baselines:**
   - Does ARIMA beat best baseline (MASE < {best_baseline['MASE']:.3f})?
   - How much better?

3. **Training time analysis:**
   - Computational cost vs baseline methods
   - Scalability to full dataset

**Implementation plan:**
```python
# TODO: Create ARIMAForecaster class in hellocloud.modeling.forecasting.arima
# TODO: Implement fit(y_train, seasonal=True, period=62) method
# TODO: Add forecast(horizon, return_conf_int=True) method
# TODO: Integrate with compute_all_metrics() for consistent evaluation
```

**Expected workflow:**
```python
# from hellocloud.modeling.forecasting import ARIMAForecaster
#
# arima = ARIMAForecaster(seasonal=True, period=62)  # Use slow period from EDA
# arima.fit(y_train_split)
# forecast, lower, upper = arima.forecast(horizon=62, return_conf_int=True)
#
# metrics = compute_all_metrics(y_val_split[:62], forecast, y_train_split)
# print(f"ARIMA MASE: {metrics['MASE']:.3f} vs Best Baseline: {best_baseline['MASE']:.3f}")
```

**Key questions to answer:**
- Does seasonality improve ARIMA performance?
- What (p,d,q) orders does auto-selection choose?
- Is training time acceptable for real-time retraining?

---

## 5. TimesFM Foundation Model (PLACEHOLDER)

**To Be Implemented:** TimesFM zero-shot forecasting

**What is TimesFM?**
- Google's pre-trained time series foundation model
- 200M parameters trained on diverse time series corpora
- Zero-shot forecasting (no training needed)
- Context window: 1024 timesteps

**Subsampling adjustment for TimesFM:**
- **Current:** 20x subsampled = 7.3K points
- **TimesFM context limit:** 1024 timesteps
- **Required:** 140x subsampling (146K / 1024 ≈ 143)
- **Trade-off:** Lose more temporal resolution but gain foundation model capabilities

**What will be tested:**
1. **Zero-shot forecasting:**
   - No hyperparameter tuning
   - Direct forecasting using pre-trained weights

2. **Point + quantile forecasts:**
   - Median forecast (point estimate)
   - 10th, 90th percentiles (prediction intervals)

3. **Compare to baselines:**
   - Does foundation model beat statistical baselines without training?
   - How do prediction intervals compare to ARIMA?

**Implementation plan:**
```python
# TODO: Install TimesFM: pip install timesfm
# TODO: Create TimesFMForecaster wrapper in hellocloud.modeling.forecasting.foundation
# TODO: Implement 140x subsampling for context window compatibility
# TODO: Add quantile forecast support (10th, 50th, 90th percentiles)
```

**Expected workflow:**
```python
# from hellocloud.modeling.forecasting import TimesFMForecaster
#
# # Subsample further for TimesFM context window
# subsample_timesfm = 140
# y_train_timesfm = y_train_full[::subsample_timesfm]
#
# # Initialize TimesFM
# timesfm = TimesFMForecaster(model_name="google/timesfm-2.5-200m-pytorch")
# timesfm.fit(y_train_timesfm)  # Just loads pre-trained model
#
# # Forecast with quantiles
# forecast_dict = timesfm.forecast(horizon=10, quantiles=[0.1, 0.5, 0.9])
#
# # Extract predictions
# y_pred = forecast_dict['median']
# lower = forecast_dict['q10']
# upper = forecast_dict['q90']
#
# # Evaluate
# metrics = compute_all_metrics(y_val_timesfm[:10], y_pred, y_train_timesfm)
# print(f"TimesFM MASE: {metrics['MASE']:.3f}")
```

**Key questions to answer:**
- Does zero-shot TimesFM beat baselines without any training?
- How accurate are prediction intervals?
- Is inference time acceptable for production use?

**References:**
- [TimesFM Paper](https://arxiv.org/abs/2310.10688)
- [HuggingFace Model](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [TimesFM GitHub](https://github.com/google-research/timesfm)

---

## 6. Gaussian Process Forecasting (FUTURE WORK)

**Deferred to Phase 2:** GP forecasting using existing SparseGPModel from notebook 04

**Why defer?**
- Notebook 04 already demonstrates GP modeling for this dataset
- Current GP implementation focuses on anomaly detection (prediction intervals)
- Forecasting requires adding `forecast()` method to trained models

**What would be added:**
1. **Load trained GP model from notebook 04:**
   ```python
   from hellocloud.modeling.gaussian_process import load_model
   model, likelihood, checkpoint = load_model('models/gp_robust_model.pth')
   ```

2. **Add forecast method:**
   ```python
   def forecast_gp(model, likelihood, X_train, y_train, horizon):
       # Use GP predictive distribution
       # Sample from posterior or use mean
       pass
   ```

3. **Compare uncertainty quantification:**
   - GP prediction intervals vs ARIMA confidence intervals
   - GP vs TimesFM quantile forecasts

**Phase 2 implementation notes:**
- GP is expensive for large datasets (already using sparse variational approximation)
- Main value is uncertainty quantification, not point forecasts
- Best for anomaly detection (existing notebook 04 use case)

---

## 7. Discussion & Next Steps

### What's Working

**Baseline Forecasting (Section 3):**
- All three baseline methods implemented and tested
- Metrics computed: MAE, RMSE, MAPE, MASE
- Best baseline identified: {best_baseline['Model']} (MASE = {best_baseline['MASE']:.3f})
- Performance floor established for future models

**Data Pipeline:**
- Subsampling workflow validated
- Train/val/test splits created
- Visualization confirms temporal structure preserved

### Implementation Roadmap

**Phase 1 (Next):** ARIMA
- Implement `ARIMAForecaster` wrapper
- Test seasonal vs non-seasonal variants
- Compare to baselines

**Phase 2:** TimesFM
- Install TimesFM package
- Create 140x subsampled dataset for context window
- Evaluate zero-shot forecasting performance
- Compare prediction intervals to ARIMA

**Phase 3:** GP Forecasting
- Add forecast method to SparseGPModel
- Compare uncertainty quantification across models
- Integrate with anomaly detection workflow

### Key Questions to Answer

**Forecast Horizons:**
- What horizons are most useful for operational use cases?
- Short-term (12 steps) vs medium-term (62 steps) vs long-term (250 steps)?
- Trade-off between accuracy and planning horizon?

**Retraining Frequency:**
- How often should models be retrained (daily, weekly)?
- Walk-forward validation vs fixed train/test split?
- Online learning vs batch retraining?

**Anomaly Detection Integration:**
- Use forecast errors as anomaly signals?
- Prediction interval violations as anomaly threshold?
- Combine with existing GP-based detection from notebook 04?

**Production Deployment:**
- Which model offers best accuracy/speed trade-off?
- Real-time inference requirements?
- Model serving infrastructure (FastAPI, Docker)?

### Success Criteria

**Any sophisticated model must:**
1. Beat best baseline: MASE < {best_baseline['MASE']:.3f}
2. Provide prediction intervals (not just point forecasts)
3. Scale to full dataset (146K samples) or justify subsampling
4. Inference time < 1 second for production use

---

## References

**Baseline Methods:**
- Hyndman & Athanasopoulos. (2021). Forecasting: Principles and Practice (3rd ed.)
- Naive/Seasonal Naive: Classical benchmark methods

**Statistical Models (To Be Implemented):**
- ARIMA: Box & Jenkins (1970)
- Auto-ARIMA: Hyndman & Khandakar (2008)

**Foundation Models (To Be Implemented):**
- TimesFM: Das et al. (2023) - [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- Chronos: Ansari et al. (2024) - [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)

**Evaluation Metrics:**
- MASE: Hyndman & Koehler (2006)
- Coverage/Sharpness: Gneiting & Raftery (2007)

```{code-cell} ipython3

```

```{code-cell} ipython3

```
