# Enhanced Time Series EDA Workflow

## Overview

The `03_iops_web_server_eda.md` notebook now provides a **comprehensive, reusable time series EDA workflow** suitable for ANY time series data - not just the IOPS dataset.

This workflow combines robust statistical methods with clear visualizations to provide **data-driven modeling recommendations** rather than assumptions.

---

## Key Enhancements

### 1. Welch's Power Spectral Density Analysis (Section 4)

**What Changed:**
- Replaced basic FFT with **Welch's method** (overlapping windows, averaging)
- Added **automatic peak detection** using `scipy.signal.find_peaks`
- Prominence thresholds identify **significant periodicities**
- Comprehensive PSD visualization with annotated peaks

**Why It Matters:**
- Welch's method is **more robust to noise** than raw FFT
- Automatic peak detection is **objective** (no visual guessing!)
- Prominence scoring quantifies **how significant** each period is
- Log-log plots reveal structure across **multiple frequency scales**

**Output:**
- Table of detected periodicities with prominence scores
- Annotated PSD plot showing significant frequencies
- Data-driven verdict: "Periodic" vs "Non-periodic"

### 2. Data-Driven STL Decomposition (Section 4.3)

**What Changed:**
- Uses **detected spectral peaks** (not assumed periods) for decomposition
- Calculates **seasonality strength** (variance explained by periodic components)
- Provides clear interpretation of results

**Why It Matters:**
- No more guessing at period lengths!
- Seasonality strength quantifies **how periodic** the data really is
- Differentiates between strong/moderate/weak periodicities

**Output:**
- Seasonality strength metric
- Interpretation: "Suitable for periodic kernels" vs "Consider alternatives"

### 3. Subsampling Validation Workflow (Section 4.5)

**What Changed:**
- Added **complete subsampling validation** workflow
- Validates statistical preservation, ACF maintenance, distribution match
- **Nyquist-aware** subsampling factor selection

**Why It Matters:**
- Large datasets (n>100k) often need subsampling for computational feasibility
- This workflow **proves** subsampling doesn't destroy signal characteristics
- Reusable for any dataset requiring downsampling

**Output:**
- Statistical comparison table
- ACF preservation check
- Visual overlay and distribution comparison plots
- Clear verdict on subsampling suitability

### 4. Evidence-Based Modeling Recommendations (Section 7)

**What Changed:**
- Kernel selection now **driven by PSD findings** (not assumptions)
- Multi-kernel recommendations for multi-scale periodicities
- Warnings when GP may not be optimal
- Alternative model suggestions (ARIMA, Prophet, LOESS)

**Why It Matters:**
- No more "try periodic kernel and see what happens"
- Recommendations are **justified by spectral evidence**
- Prevents wasted time on inappropriate models
- Provides fallback options

**Output:**
- Kernel structure table with justifications
- Specific hyperparameter recommendations
- Alternative model suggestions if GP unsuitable

---

## How to Use This Workflow

### For IOPS Dataset (Current)

1. Run sections 1-3 (data loading, basic stats, distributions)
2. Run section 4 (periodicity analysis) - **key section!**
3. Review PSD plot and detected peaks table
4. Run section 4.5 if subsampling needed
5. Run sections 5-6 (stationarity, anomaly analysis)
6. Review section 7 for modeling recommendations

### For New Time Series (CloudZero, etc.)

**Step 1: Load Your Data**
```python
# Replace section 1 with your data source
df = pl.read_csv("your_timeseries.csv")
# Ensure columns: timestamp, value, [optional: is_anomaly]
```

**Step 2: Run Sections 2-4 Unchanged**
- Distribution analysis (section 2-3)
- **Periodicity analysis (section 4)** - unchanged!
- PSD will detect periodicities regardless of domain

**Step 3: Interpret PSD Results**
- Check detected peaks table
- Review prominence scores
- Note recommended periods

**Step 4: Subsampling (if n > 50k)**
- Run section 4.5
- Adjust `subsample_factor` based on Nyquist constraint
- Validate preservation

**Step 5: Get Modeling Recommendations**
- Section 7 automatically generates kernel recommendations
- Based on YOUR data's spectral characteristics
- No manual tuning needed!

---

## Workflow Outputs

### Diagnostic Files (Optional)

You can save diagnostic plots for reference:

```python
# In PSD visualization cell, add:
plt.savefig('../figures/psd_analysis.png', dpi=150, bbox_inches='tight')

# In subsampling validation cell, add:
plt.savefig('../figures/subsampling_validation.png', dpi=150, bbox_inches='tight')
```

### Expected Decisions

**If PSD shows clear peaks:**
→ Periodic kernels recommended
→ GP suitable
→ Use detected periods as kernel hyperparameters

**If PSD is flat/broadband:**
→ Non-periodic kernels (RBF + Linear)
→ Consider alternatives: ARIMA, Prophet, LOESS
→ GP may underperform vs simpler models

---

## Technical Details

### Welch's PSD Parameters

```python
frequencies, psd = signal.welch(
    data,
    fs=1.0,                          # Sampling rate (1 sample/timestep)
    nperseg=min(2048, len(data)//4), # Window size for averaging
    scaling='density'                # PSD (not power spectrum)
)
```

**Window size choice:**
- Larger windows → better frequency resolution, more noise
- Smaller windows → worse resolution, smoother estimate
- Rule of thumb: `nperseg = len(data)//4` balances both

### Peak Detection Parameters

```python
peak_indices, properties = signal.find_peaks(
    psd_normalized,
    prominence=0.05,  # Minimum 5% prominence above baseline
    distance=20       # Peaks separated by ≥20 frequency bins
)
```

**Prominence threshold:**
- 0.05: Catches weak periodicities (may have false positives)
- 0.10: Moderate threshold (balanced)
- 0.15+: Only very strong periodicities (may miss real patterns)

### Subsampling Nyquist Constraint

For a signal with highest frequency `f_max`:
- **Nyquist rate**: Sample at ≥ `2 * f_max`
- **Safe subsampling**: `factor ≤ 1 / (2 * f_max)`

Example: If `f_max = 0.004` (period=250):
- Nyquist safe: `factor ≤ 1 / (2 * 0.004) = 125`
- Conservative: `factor = 125 / 2 = 62` (2× safety margin)

---

## Validation Checklist

When applying this workflow to new data, verify:

- ✅ **Data loaded correctly**: Check `len(df)`, `df.describe()`
- ✅ **PSD computed**: No NaN/Inf in frequencies or PSD
- ✅ **Peaks detected**: Check `len(peak_indices) > 0` if expected
- ✅ **Periods realistic**: Detected periods make sense for your domain
- ✅ **Subsampling safe**: Factor respects Nyquist constraint
- ✅ **Recommendations make sense**: Kernel structure matches your understanding

---

## Common Issues and Solutions

### Issue: No peaks detected, but you know data is periodic

**Causes:**
- Prominence threshold too high
- High noise masking peaks
- Non-stationary periods (changing over time)

**Solutions:**
1. Lower prominence threshold to 0.03
2. Detrend data before PSD computation
3. Use STL decomposition to extract seasonal component first

### Issue: Too many peaks detected

**Causes:**
- Prominence threshold too low
- Broadband noise with random fluctuations

**Solutions:**
1. Increase prominence to 0.10 or 0.15
2. Increase `distance` parameter to avoid nearby peaks
3. Only use top 3 peaks by power for modeling

### Issue: Subsampling destroys pattern

**Symptoms:**
- ACF preservation check fails
- Distribution shapes differ significantly
- Visual overlay shows sampling misses key features

**Solutions:**
1. Reduce subsampling factor (less aggressive)
2. Use stratified sampling (sample proportionally from each period)
3. Use full dataset with sparse GP (inducing points)

---

## Next Steps After EDA

### If Periodic Structure Detected:

1. **Try exact GP first** (if n < 10k)
   - Implement recommended kernel structure
   - Use detected periods as initial hyperparameters
   - Optimize via marginal likelihood

2. **Use sparse GP** (if n > 10k)
   - Subsample validation (section 4.5)
   - Inducing points: M ≈ n/100 to n/50
   - Monitor kernel outputscales (check balance)

3. **Validate results**:
   - R² > 0.7 expected for good periodic fit
   - Prediction intervals should be well-calibrated
   - Visual inspection: does GP capture the pattern?

### If No Periodic Structure:

1. **Try ARIMA** (classical time series)
   - Auto-ARIMA for order selection
   - Good for irregular patterns

2. **Try Prophet** (Facebook's tool)
   - Handles trend + noise well
   - Less sensitive to periodicity assumptions

3. **Try local regression** (LOESS, LOWESS)
   - Non-parametric, adaptive
   - Good for smooth but irregular trends

---

## References

- **Welch's method**: Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- **Peak detection**: SciPy documentation on `signal.find_peaks`
- **STL decomposition**: Cleveland et al. (1990). "STL: A seasonal-trend decomposition procedure based on loess"
- **Nyquist-Shannon theorem**: Shannon (1949). "Communication in the presence of noise"

---

**Last Updated:** 2025-10-08
**Status:** Production-ready for any time series EDA
**Tested On:** IOPS web server dataset (146k samples, quasi-periodic)
