---
title: "Rebase Execution Plan"
created: 2025-10-08
status: Ready to Execute
---

# Rebase Execution Strategy

## Approach: Soft Reset + Manual Recommit

**Why this approach:**
- 89 commits to squash - interactive rebase would be extremely tedious
- Want full control over narrative and commit messages
- Minimize conflict risk
- Can reference original commits for guidance

## Key Milestone Commits to Preserve (as reference)

| SHA     | Description | Action |
|---------|-------------|--------|
| d0d1e1f | Project vision | **KEEP INTACT** |
| fe20c8e | Research foundation | **KEEP INTACT** |
| f5709bb | Architecture | **KEEP INTACT** |
| 91cf3e3 | Core engine | **KEEP INTACT** |
| 3ff7b8f | Workload signatures guide | Reference for new commit |
| 97342c0 | EDA notebooks added | Reference for new commit |
| a6e6648 | IOPS EDA initial | Reference for new commit |
| 3ba2011 | GP modeling notebook | Reference for new commit |
| 15c1a67 | GP library extraction | **CHERRY-PICK** |
| 3c30138 | GP runbook conversion | **CHERRY-PICK** |
| 794d945 | CloudZero PiedPiper EDA | Reference for new commit |
| a4aa885 | Cleanup (just committed) | **KEEP** |

## Execution Steps

### Step 1: Foundation (Already Done)
Commits d0d1e1f → 91cf3e3 are good. Keep as-is.

### Step 2: Reset to Foundation
```bash
git reset --soft 91cf3e3
```
This keeps all changes since 91cf3e3 as staged.

### Step 3: Unstage Everything
```bash
git reset HEAD
```
Now all changes are unstaged. We can selectively build history.

### Step 4: Build Clean Commits

**Commit 5: ML Foundation**
```bash
git add src/cloud_sim/ml_models/ pyproject.toml
git add src/cloud_sim/data_generation/
git add tests/
git commit -m "feat: add ML models and workload taxonomy foundation

Establish machine learning capabilities:
- PyMC hierarchical Bayesian model for cloud resources
- Workload taxonomy with 12+ application archetypes
- HuggingFace dataset builder integration
- Pydantic-based data validation

Add development infrastructure:
- GitHub Actions CI with uv
- Comprehensive test suite (70%+ coverage)
- Black + Ruff code quality tooling

Based on empirical research showing 12-15% CPU utilization and
25-35% waste in cloud infrastructure.
"
```

**Commit 6: Workload Signatures Guide**
```bash
git add notebooks/02_workload_signatures_guide.md
git add docs/research/cloud-resource-correlations-report.md
git commit -m "docs: add comprehensive workload signatures guide

Educational notebook demonstrating 12 empirical workload patterns:
- Web applications, batch processing, ML training
- Temporal autocorrelation (0.7-0.8)
- Multi-variate correlation structures
- Altair visualizations for pattern analysis

Based on peer-reviewed research on cloud waste patterns.
"
```

**Commit 7: Dataset Exploration**
```bash
git add notebooks/01_data_exploration.md
git add src/cloud_sim/etl/
git commit -m "feat: explore public time series datasets for cloud workload analysis

Evaluate multiple public datasets for cloud pattern research:
- Alibaba Cluster Trace 2018 (explored, not retained)
- Google Cluster Traces (explored, not retained)
- IOPS webserver dataset (TSB-UAD) ← Selected for analysis

Add ETL infrastructure for HuggingFace dataset integration.
CloudZero production data loader (stub for future use).
"
```

**Commit 8: IOPS Webserver EDA**
```bash
git add notebooks/03_iops_web_server_eda.md
git commit -m "feat: add IOPS webserver anomaly detection EDA

Comprehensive exploratory data analysis of IOPS dataset from TSB-UAD:
- Statistical characterization (distributions, stationarity)
- Seasonality and periodicity analysis (ACF, FFT, STL)
- Univariate distribution analysis (PDF, CDF)
- GP kernel selection methodology

Key findings:
- Clear periodic patterns detected (useful for GP modeling)
- Anomalies show distinct distributional characteristics
- Dataset suitable for time series forecasting experiments

Source: AutonLab/Timeseries-PILE on HuggingFace
"
```

**Commit 9: GP Modeling Notebook**
```bash
git add notebooks/04_gaussian_process_modeling.md
git commit -m "feat: add Gaussian Process modeling notebook for cloud time series

Develop GP-based approach for cloud workload forecasting:
- Composite periodic kernel (multi-scale patterns)
- Student-t likelihood for robustness to outliers
- Sparse variational GP for scalability
- Comprehensive evaluation (RMSE, calibration, coverage)

Comparison: Robust (Student-t) vs Traditional (Gaussian) likelihood
- Robust model maintains calibration in presence of anomalies
- Traditional model overconfident on outliers

Based on IOPS dataset analysis - demonstrates methodology for
cloud workload time series.
"
```

**Commit 10: Library Extraction** (Cherry-pick)
```bash
git cherry-pick 15c1a67
# If conflicts, resolve and continue
```

**Commit 11: Runbook Conversion** (Cherry-pick)
```bash
git cherry-pick 3c30138
# If conflicts, resolve and continue
```

**Commit 12: Foundation Model Stubs**
```bash
git add src/cloud_sim/ml_models/foundation/
git add pyproject.toml  # Optional dependencies
git commit -m "feat: add foundation model stubs for future integration

Add stub implementations for time series foundation models:
- TimesFM (Google Research) - Decoder-only transformer
- Chronos (Amazon) - Probabilistic forecasting

Not yet implemented - placeholders for future work.

Also add optional dependency groups in pyproject.toml:
- [gpu] for GPyTorch training
- [foundation] for foundation model integration
"
```

**Commit 13: Repository Cleanup**
```bash
git add .gitignore docs/
git commit -m "chore: comprehensive repository cleanup and documentation

- Enhance .gitignore for notebook artifacts
- Reorganize documentation structure
- Remove redundant test files
- Update README with current architecture
- Add notebook workflow documentation
"
```

**Commit 14: CloudZero PiedPiper EDA**
```bash
git add notebooks/05_cloudzero_piedpiper_eda.md
git add src/cloud_sim/utils/eda_analysis.py
git add docs/eda-workflow-summary.md
git commit -m "feat: add CloudZero PiedPiper EDA with rigorous analysis framework

Comprehensive EDA notebook for CloudZero production data:
- Reusable EDA analysis utilities module
- Statistical foundations for cloud cost patterns
- Integration with workflow automation
- Hot reload pattern for iterative development

Demonstrates library-first architecture: notebooks import
cloud_sim utilities rather than embedding implementation.
"
```

**Commit 15: Final Cleanup** (Already exists)
This is commit a4aa885 we just made.

### Step 5: Restore WIP Changes
```bash
git stash pop
```

## Rollback Plan

If anything goes wrong:
```bash
git reset --hard backup/pre-cleanup-rebase-20251008-1905
git stash pop  # Restore WIP changes
```

## Success Criteria

- [ ] 15 commits total (down from 95)
- [ ] All commits build successfully
- [ ] Tests pass at HEAD
- [ ] Clean narrative visible in `git log --oneline`
- [ ] Foundation commits (first 4) unchanged
- [ ] WIP changes restored after rebase

