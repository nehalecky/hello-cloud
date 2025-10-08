---
title: "Repository History Cleanup Strategy"
version: 1.0.0
status: Ready for Execution
author: Nicholaus Halecky
created: 2025-10-08
updated: 2025-10-08
tags: [refactoring, git-history, cleanup, rebase]
---

# Comprehensive History Cleanup Strategy

## Executive Summary

**Current State:** 94 commits with significant concept churn, abandoned dataset explorations, and temporary diagnostic artifacts.

**Desired State:** Clean, linear history telling the story of a focused research project exploring cloud workload characterization, culminating in a library-first GP modeling implementation.

**Key Principle:** Preserve genuine intellectual progression while eliminating false starts and implementation churn.

---

## Phase 1: Pre-Rebase Cleanup (File & Code Level)

### 1.1 Remove Temp Artifacts
**Root directory diagnostics:**
```bash
rm diagnose_gp_results.py
rm gp_diagnostics.txt gp_diagnostics_plots.png
rm subsampling_aliasing_analysis.txt subsampling_psd_analysis.png
rm subsampling_validation.txt subsampling_visual_validation.png
```

**Untracked documentation:**
```bash
# Keep: docs/eda-workflow-summary.md (currently in active use)
rm docs/logging-strategy.md
rm docs/modeling/gp-inducing-points-analysis.md
rm docs/modeling/gp-initialization-fix.md
```

### 1.2 Remove Alibaba/Google Dataset References
**Files to delete:**
```bash
rm docs/research/alibaba-trace-analysis.md
```

**Files to edit:**
- `tests/test_notebooks.py`: Remove `"alibaba_trace_analysis.md"` from skip list (line ~35)

**Files to KEEP (legitimate references):**
- `src/cloud_sim/ml_models/foundation/timesfm.py` - Google's TimesFM model
- `src/cloud_sim/ml_models/foundation/__init__.py` - TimesFM documentation
- `src/cloud_sim/ml_models/application_taxonomy.py` - "Google" in example_companies list

### 1.3 Commit Cleanup Changes
```bash
git add -A
git commit -m "chore: remove diagnostic artifacts and abandoned dataset documentation

Remove temporary diagnostic files from GP experimentation:
- diagnose_gp_results.py
- gp_diagnostics.txt, gp_diagnostics_plots.png
- subsampling analysis artifacts

Remove untracked documentation from exploratory phases:
- logging-strategy.md
- gp modeling iteration docs (inducing points, initialization fixes)

Keep active documentation:
- eda-workflow-summary.md (currently in use)

Remove Alibaba trace dataset documentation (dataset exploration abandoned):
- docs/research/alibaba-trace-analysis.md
- Test reference in test_notebooks.py

Retain legitimate Google references (TimesFM foundation model).
"
```

---

## Phase 2: Design Clean History Narrative

### 2.1 Current History Analysis (94 Commits)

**Foundation (Keep as-is):**
- `d0d1e1f` - feat: establish project vision (initial commit)
- `fe20c8e` - docs: establish empirical research foundation
- `f5709bb` - feat: define simulation architecture
- `91cf3e3` - feat: implement core simulation engine

**Middle Churn (Squash heavily):**
- Commits `f59f45d` through `97342c0` (30+ commits):
  - Initial ML models
  - Workload taxonomy iterations
  - Alibaba/Google dataset explorations (abandoned)
  - Multiple documentation rewrites
  - Test infrastructure setup

**Recent Mature Work (Preserve structure, squash fixes):**
- `97342c0` - docs: add comprehensive EDA notebooks
- `a6e6648` - feat: add web server anomaly detection EDA (IOPS)
- `3ba2011` - feat: add Gaussian Process modeling notebook
- `04c055f` - refactor: remove Alibaba/Google datasets
- `15c1a67` - refactor: extract GP implementation to library (92% coverage)
- `3c30138` - refactor: convert GP notebook to runbook
- Recent commits - CloudZero EDA work

### 2.2 Target History Structure (15-20 Commits)

```
FOUNDATION PHASE (Keep original commits):
1. d0d1e1f - feat: establish project vision for cloud cost optimization
2. fe20c8e - docs: establish empirical research foundation
3. f5709bb - feat: define simulation architecture and modeling framework
4. 91cf3e3 - feat: implement core cloud resource simulation engine

RESEARCH SPIKE PHASE (Squash into 3-4 commits):
5. NEW - feat: add workload pattern taxonomy and ML modeling foundation
   [Squash: f59f45d + rebranding + Pydantic migration + CI setup]

6. NEW - docs: add workload signatures guide with empirical patterns
   [Squash: 3ff7b8f + correlation reports + documentation improvements]

7. NEW - feat: establish notebook infrastructure for research workflows
   [Squash: notebook setup + MyST conversion + test infrastructure]

DATASET EXPLORATION PHASE (Squash into 2 commits):
8. NEW - feat: explore public time series datasets for cloud workload analysis
   [Squash: Initial dataset explorations, HF integration, preliminary analysis]
   [Narrative: "Evaluated multiple datasets (Alibaba, Google traces, IOPS) for cloud workload patterns"]

9. NEW - feat: add IOPS web server EDA - anomaly detection analysis
   [Squash: a6e6648 + refinements + print statement removal + citation fixes]
   [Keep: Final IOPS notebook as canonical example]

GAUSSIAN PROCESS PHASE (Squash into 3-4 commits):
10. NEW - feat: add Gaussian Process modeling notebook for cloud time series
    [Squash: 3ba2011 + GP fixes + array handling + defensive programming]

11. NEW - refactor: extract GP implementation to library module (92% test coverage)
    [Keep: 15c1a67 - this is a major milestone]

12. NEW - refactor: convert GP notebook to runbook using library
    [Keep: 3c30138 - demonstrates library-first architecture]

13. NEW - feat: add foundation model stubs (Chronos, TimesFM) and CloudZero ETL
    [Squash: b216f58 + related dependency updates]

POLISH PHASE (Squash into 2-3 commits):
14. NEW - chore: comprehensive repository cleanup and documentation
    [Squash: Multiple cleanup commits + .gitignore updates + redundancy removal]

15. NEW - feat: add CloudZero PiedPiper EDA with rigorous analysis framework
    [Squash: 794d945 + eda utilities + workflow updates]

16. NEW - chore: remove diagnostic artifacts and abandoned dataset documentation
    [This is the cleanup we're doing now]
```

### 2.3 Narrative Arc

**The Story We Tell:**
> "In September 2025, I started this project to tackle cloud waste optimization. After establishing the architecture and core simulation engine, I spiked into multiple public datasets (Alibaba traces, Google traces, IOPS anomaly detection data) to understand real-world cloud workload patterns. The IOPS webserver dataset proved most valuable for anomaly detection analysis.
>
> I then developed a Gaussian Process modeling approach for cloud time series, iterated on the implementation in a notebook, and ultimately extracted it into a production-ready library with 92% test coverage. This library-first approach enables both research (notebooks) and production use (importable modules).
>
> Most recently, I've been applying this framework to CloudZero's PiedPiper production data, creating rigorous EDA workflows."

---

## Phase 3: Interactive Rebase Execution Plan

### 3.1 Safety Preparations

```bash
# Create backup branch
git branch backup/pre-cleanup-rebase-$(date +%Y%m%d)

# Create rebase script
cat > .git/rebase-plan.txt << 'EOF'
# Keep foundation commits as-is
pick d0d1e1f feat: establish project vision
pick fe20c8e docs: establish empirical research foundation
pick f5709bb feat: define simulation architecture
pick 91cf3e3 feat: implement core simulation engine

# Squash ML foundation commits
pick f59f45d feat: add ML models and forecasting capabilities
squash 6b881af refactor: rebrand as Cloud Resource Simulator
squash de8a404 docs: reorganize documentation structure
squash 5b92385 refactor: migrate to Pydantic BaseModel
squash 40af580 fix: correct TOML parsing errors
squash b662202 ci: add GitHub Actions workflow
# ... (continue for all related commits)

# New commit: Workload signatures guide
pick 3ff7b8f feat: add workload signatures guide
squash 8c40494 docs: update correlation report
squash 8e080f5 chore: add coverage.xml to .gitignore

# Continue pattern for each phase...
EOF
```

### 3.2 Rebase Strategy

**Option A: Interactive Rebase (Recommended)**
```bash
git rebase -i d0d1e1f~1
# Edit rebase-todo list following the plan above
# Resolve conflicts as needed
# Reword commit messages to match target narrative
```

**Option B: Soft Reset + Recommit (Nuclear Option)**
```bash
git checkout -b clean-history d0d1e1f
# Manually cherry-pick foundation commits
git cherry-pick fe20c8e f5709bb 91cf3e3

# Create new squashed commits manually
# (More control, but more work)
```

**Recommendation:** Start with Option A. If rebase becomes too complex, fall back to Option B for problematic sections.

### 3.3 Commit Message Template for Squashed Commits

```
<type>(<scope>): <concise description>

Narrative: <Why this work was done>

Key Changes:
- <Bullet point 1>
- <Bullet point 2>

Research Context: <Any empirical findings or decisions>

Coverage: <If applicable, test coverage %>

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Phase 4: Post-Rebase Verification

### 4.1 Validation Checklist

```bash
# Verify all commits build
git rebase --exec "uv sync" <target-branch>

# Verify tests pass at each commit
git rebase --exec "uv run pytest tests/ -v" <target-branch>

# Verify notebooks are valid
git rebase --exec "uv run pytest tests/test_notebooks.py -m smoke" <target-branch>

# Check commit count
git log --oneline | wc -l  # Should be ~15-20

# Verify no broken references
rg "alibaba_trace_analysis" --type py
rg "GoogleTraceLoader" --type py
```

### 4.2 Documentation Updates

After rebase, update:
- `.claude-project-memory.md` - Remove references to abandoned explorations
- `docs/design/library-first-refactoring-plan.md` - Mark Phase 3 complete
- `README.md` - Ensure history aligns with cleaned narrative
- `CHANGELOG.md` - Create clean changelog from new history

---

## Phase 5: Risk Mitigation

### 5.1 Rollback Plan

**If rebase fails:**
```bash
git rebase --abort
git reset --hard backup/pre-cleanup-rebase-YYYYMMDD
```

**If rebase succeeds but breaks something:**
```bash
git reflog  # Find pre-rebase HEAD
git reset --hard HEAD@{N}  # Where N is pre-rebase state
```

### 5.2 Testing Before Force Push

```bash
# Run full test suite
uv run pytest tests/ -v --cov=src/cloud_sim --cov-report=term-missing

# Test notebook execution
uv run pytest tests/test_notebooks.py -v

# Verify build
uv build

# Check for missing files
git ls-files | wc -l
```

### 5.3 Force Push Strategy

**DO NOT force push to main immediately!**

```bash
# Push to temporary branch first
git push origin HEAD:temp/clean-history-verification

# Test in GitHub UI
# - Browse files
# - Check Actions run
# - Review commit history

# If satisfied, force push to main
git push --force-with-lease origin main
```

---

## Phase 6: Execution Order (Step-by-Step)

### Day 1: Preparation
1. ✅ Review this strategy document
2. ✅ Get user approval on narrative and approach
3. ✅ Create backup branch
4. ✅ Run full test suite (baseline)
5. ✅ Execute Phase 1 cleanup (remove artifacts)

### Day 2: Rebase Execution
6. ⏳ Start interactive rebase from d0d1e1f
7. ⏳ Squash commits following target structure
8. ⏳ Reword commit messages with narrative
9. ⏳ Resolve any conflicts

### Day 3: Verification
10. ⏳ Run verification checklist
11. ⏳ Push to temporary branch
12. ⏳ Review in GitHub UI
13. ⏳ Get final approval

### Day 4: Finalization
14. ⏳ Force push to main with lease
15. ⏳ Update documentation
16. ⏳ Delete backup branch (after confirmation)
17. ⏳ Create summary audio (work-completion-summary)

---

## Decision Points

### Q1: Should we preserve the Alibaba/Google dataset exploration commits?
**A:** No. The code was removed in commit 04c055f, and these were false starts. The narrative should mention "evaluated multiple datasets" in a single squashed commit, but not preserve the full iteration history.

### Q2: How much of the GP iteration history should we keep?
**A:** Keep 3 commits:
1. Initial GP notebook implementation
2. Library extraction (92% coverage) - **MILESTONE**
3. Notebook conversion to runbook

Squash all the fixes, array handling, defensive programming iterations.

### Q3: Should we preserve CI/CD and test infrastructure commits?
**A:** Squash into "Establish notebook infrastructure" commit. The final state is what matters, not the iteration history.

### Q4: What about the CloudZero-specific work?
**A:** Keep as recent commits (794d945, 4f4d57e, etc). This is current work and shouldn't be squashed yet. May refine later when sharing externally.

### Q5: Should we use git filter-branch instead of rebase?
**A:** No. filter-branch is for removing sensitive data or large files. Interactive rebase gives us fine-grained control over commit messages and narrative.

---

## Success Metrics

✅ **Commit Count:** 15-20 commits (down from 94)
✅ **All Tests Pass:** Every commit buildable and testable
✅ **Clean Narrative:** History tells coherent research story
✅ **No Broken References:** Removed datasets don't appear in code/tests
✅ **Documentation Aligned:** Docs reflect cleaned history
✅ **Shareable:** Repository ready for public/external sharing

---

## References

- Original Plan: `docs/design/library-first-refactoring-plan.md`
- Git Rebase Documentation: https://git-scm.com/docs/git-rebase
- Conventional Commits: https://www.conventionalcommits.org/
- Pro Git Chapter 7.6: Rewriting History: https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History

---

## Appendix: Commit Mapping Table

| Old Commits | New Commit | Rationale |
|-------------|------------|-----------|
| d0d1e1f | Keep as-is | Foundation commit |
| fe20c8e | Keep as-is | Research foundation |
| f5709bb | Keep as-is | Architecture definition |
| 91cf3e3 | Keep as-is | Core engine |
| f59f45d + 6b881af + de8a404 + 5b92385 + 40af580 + b662202 | Squash into "feat: add workload taxonomy and ML foundation" | Multiple iterations of same work |
| 3ff7b8f + 8c40494 + b8965b2 + 76b5aa3 | Squash into "docs: add workload signatures guide" | Documentation and refinement |
| 8c203bb + 69a7c64 + d557b52 + (Google commits) | Squash into "feat: explore datasets" | Dataset exploration spike |
| a6e6648 + d423b24 + fixes | Squash into "feat: IOPS EDA analysis" | IOPS notebook work |
| 3ba2011 + 6bf9581 + d20b046 + 00d56c3 | Squash into "feat: GP modeling notebook" | GP notebook iterations |
| 15c1a67 | Keep as-is | **MILESTONE:** Library extraction |
| 3c30138 | Keep as-is | Runbook conversion |
| 04c055f + 78320f2 + 41a1f3b + 7374d24 | Squash into "chore: repository cleanup" | Cleanup iterations |
| 794d945 + 4f4d57e + 3d59a6a + 9e8c7b3 | Keep structure, maybe minor squash | Recent CloudZero work |

---

*End of Strategy Document*
