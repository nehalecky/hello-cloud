# Repository Cleanup Plan

**Date**: 2025-10-09
**Purpose**: Remove old artifacts and maintenance docs while preserving active work

---

## Files to Remove

### Category 1: Old Cleanup/Maintenance Reports

These are reports from **completed** maintenance work - no longer needed:

```bash
# Remove old cleanup reports (5 files)
rm HISTORY_CLEANUP_COMPLETE.md          # Report: Git history cleanup (95→14 commits)
rm REPOSITORY_CLEANUP_REPORT.md         # Report: Previous cleanup effort
rm docs/design/history-cleanup-strategy.md  # Strategy for completed cleanup
rm docs/design/rebase-execution-plan.md     # Execution plan for completed rebase
rm .claude-project-memory.md            # Old project memory file (822 bytes)
```

**Justification**: These document completed work. Historical value is in git history, not working tree.

### Category 2: Workflow Planning Artifacts (Optional - Review First)

Design docs for consolidating notebooks 05+06 - may be obsolete if consolidation is complete:

```bash
# Review these first, then remove if consolidation is done:
rm docs/eda-workflow-consolidated.md    # Design: Consolidate notebooks 05+06
rm docs/eda-workflow-visual-summary.md  # Visual summary of consolidation
```

**Decision Point**:
- Remove if notebooks 05+06 consolidation is **complete**
- Keep if you're still **actively using** these as reference

---

## Files to Keep

### Active Work - PiedPiper Analysis

**DO NOT REMOVE** - These are active analysis notebooks:

- ✅ `notebooks/05_cloudzero_piedpiper_eda.md` - Active EDA
- ✅ `notebooks/05_cloudzero_piedpiper_analysis.md` - Active analysis
- ✅ `notebooks/06_piedpiper_statistical_analysis.md` - Active statistical work
- ✅ `analyze_frequency.py` - Active frequency analysis script
- ✅ `entity_cost_analysis.py` - Active entity analysis script
- ✅ `FREQUENCY_ANALYSIS_REPORT.md` - Active findings report
- ✅ `frequency_and_entity_analysis.md` - Active analysis report
- ✅ `data/piedpiper_optimized_daily.parquet` - Active dataset (981 MB)
- ✅ `src/cloud_sim/utils/cost_analysis.py` - Active utility module

### Core Framework

**KEEP** - These are the shareable simulation framework:

- ✅ `notebooks/01_data_exploration.md` - Synthetic data exploration
- ✅ `notebooks/02_workload_signatures_guide.md` - Workload archetypes
- ✅ `notebooks/03_iops_web_server_eda.md` - IOPS analysis
- ✅ `notebooks/04_gaussian_process_modeling.md` - GP modeling
- ✅ `src/cloud_sim/` - All library code
- ✅ `tests/` - All tests
- ✅ `docs/research/` - Research documentation
- ✅ `docs/modeling/` - Modeling documentation

---

## Execution Plan

### Step 1: Safe Removals (No Review Needed)

```bash
cd /Users/nehalecky/Projects/cloudzero/cloud-resource-simulator

# Remove old maintenance reports
rm HISTORY_CLEANUP_COMPLETE.md
rm REPOSITORY_CLEANUP_REPORT.md
rm .claude-project-memory.md
rm docs/design/history-cleanup-strategy.md
rm docs/design/rebase-execution-plan.md

echo "✓ Removed 5 old maintenance reports"
```

### Step 2: Optional Removals (Review First)

```bash
# Review these files first:
cat docs/eda-workflow-consolidated.md      # Still useful?
cat docs/eda-workflow-visual-summary.md    # Still useful?

# If no longer needed:
rm docs/eda-workflow-consolidated.md
rm docs/eda-workflow-visual-summary.md

echo "✓ Removed 2 workflow planning docs"
```

### Step 3: Verification

```bash
# Check for any remaining cleanup artifacts
ls -la *.md | grep -i "cleanup\|history\|report"

# Verify active work is intact
ls -la notebooks/05*.md notebooks/06*.md
ls -la analyze_frequency.py entity_cost_analysis.py
ls -lh data/piedpiper_optimized_daily.parquet

echo "✓ Verification complete"
```

---

## Size Impact

**Will remove**:
- 5 maintenance reports (~32 KB)
- 2 workflow docs (~18 KB, optional)
- **Total**: ~50 KB

**Will keep**:
- PiedPiper active work: ~7,500 lines + 981 MB data
- Core framework: ~4,000 lines of notebooks + all library code

---

## Notes for Future Sharing

When ready to share the repository publicly (post-PiedPiper analysis):

1. **Create private branch** with PiedPiper work
2. **Public branch** removes:
   - All PiedPiper notebooks (05, 06)
   - Analysis scripts (analyze_frequency.py, entity_cost_analysis.py)
   - Reports (FREQUENCY_ANALYSIS_REPORT.md, frequency_and_entity_analysis.md)
   - Data file (piedpiper_optimized_daily.parquet - 981 MB)
   - cost_analysis.py utility module

3. **Keep for public**:
   - Core notebooks 01-04 (synthetic data only)
   - All library code (except cost_analysis.py)
   - Research and modeling docs
   - Tests

This cleanup plan focuses on **old artifacts**, not **active work**.
