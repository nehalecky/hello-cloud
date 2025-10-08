---
title: "Library-First Refactoring Plan"
version: 1.0.0
status: Draft
author: Nicholaus Halecky
created: 2025-10-07
updated: 2025-10-07
tags: [refactoring, architecture, git-history, notebooks]
---

# Comprehensive Plan: Library-First Refactoring + History Cleanup

## Context Analysis

**Current State:**
- 75 commits with messy iteration history
- Library code exists: `src/cloud_sim/{data_generation,ml_models}`
- Notebooks mix implementation AND educational content
- GP model code lives IN notebook (04_gaussian_process_modeling.md)
- No TimesFM/Chronos integration yet
- ETL module empty after Alibaba/Google cleanup

**Execution Environments:**
1. **Local**: Development, light inference, visualization (M4 CPU, no GPU)
2. **Google Colab**: GP training, TimesFM, large ETL (Free GPU, Pro $10/mo)
3. **HuggingFace Inference API**: Production inference (pay-per-use)

---

## Phase 1: Extract Implementation from Notebooks to Library

### 1.1 Create GP Module Structure
**New files to create:**
```
src/cloud_sim/ml_models/gaussian_process/
├── __init__.py
├── kernels.py        # CompositePeriodicKernel, custom kernels
├── models.py         # SparseGPModel, base classes
├── training.py       # train_gp, save_model, load_model utilities
└── evaluation.py     # compute_metrics, compute_anomaly_metrics
```

**Extract from:** `notebooks/04_gaussian_process_modeling.md` lines 259-937
- CompositePeriodicKernel class → kernels.py
- SparseGPModel class → models.py
- Training logic → training.py (with batched prediction fixes)
- compute_metrics, compute_anomaly_metrics → evaluation.py

### 1.2 Add Foundation Model Module
**New files to create:**
```
src/cloud_sim/ml_models/foundation/
├── __init__.py
├── timesfm.py        # TimesFM wrapper (Google)
├── chronos.py        # Chronos wrapper (Amazon)
└── base.py           # Abstract base class for foundation models
```

**Capabilities:**
- Load pre-trained models from HF
- Inference API support
- Fine-tuning hooks (Colab-only)

### 1.3 Add CloudZero ETL Module
**New files to create:**
```
src/cloud_sim/etl/
├── __init__.py
└── cloudzero_loader.py   # Production data loader
```

**Design:**
- Load CloudZero production data samples
- Convert to standardized format (Polars DataFrames)
- Integrate with HuggingFace dataset publishing

### 1.4 Update pyproject.toml Dependencies
**Add optional dependency groups:**
```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",
    "gpytorch>=1.11.0",
]

foundation = [
    "timesfm @ git+https://github.com/google-research/timesfm",
    "chronos-forecasting>=1.0.0",
    "transformers>=4.35.0",
]

all = [
    "cloud-resource-simulator[gpu,foundation]",
]
```

---

## Phase 2: Convert Notebooks to Runbooks

### 2.1 Restructure Notebooks Directory
```
notebooks/
├── README.md                        # Update with new structure
├── quickstart/
│   ├── 01_installation.md          # Setup instructions
│   ├── 02_basic_usage.md           # Simple library usage
│   └── 03_data_generation.md       # WorkloadPatternGenerator demo
├── guides/
│   ├── workload_patterns.md        # Educational (keep current 02)
│   ├── gp_inference.md             # Use pre-trained GP models
│   └── foundation_models.md        # TimesFM/Chronos inference
├── eda/
│   ├── iops_web_server_analysis.md # Keep current 03 (research)
│   └── cloudzero_data_exploration.md # NEW: CloudZero data
└── training/ (Colab-specific)
    ├── README.md                    # Colab setup instructions
    ├── train_gp_colab.ipynb        # GPU training workflow
    ├── train_timesfm_colab.ipynb   # Foundation model fine-tuning
    └── etl_pipeline_colab.ipynb    # Large-scale data processing
```

### 2.2 Refactor Existing Notebooks

**01_data_exploration.md → quickstart/03_data_generation.md**
- Remove implementation details
- Focus on library usage: `from cloud_sim.data_generation import WorkloadPatternGenerator`
- Show basic examples

**02_workload_signatures_guide.md → guides/workload_patterns.md**
- Keep as-is (already educational)
- No implementation code to extract

**03_iops_web_server_eda.md → eda/iops_web_server_analysis.md**
- Keep as-is (research notebook)
- Uses HuggingFace dataset (good runbook example)

**04_gaussian_process_modeling.md → Refactor:**
1. Extract implementation → `src/cloud_sim/ml_models/gaussian_process/`
2. Create `guides/gp_inference.md` - Load pre-trained models, run inference
3. Create `training/train_gp_colab.ipynb` - Full training workflow for Colab

### 2.3 Create New Runbooks

**quickstart/01_installation.md:**
```markdown
# Installation

## Local Development
`uv pip install -e ".[dev]"`

## GPU/ML Work (Colab)
`!pip install cloud-resource-simulator[gpu,foundation]`

## CloudZero Integration
Requires production data access...
```

**guides/gp_inference.md:**
```python
from cloud_sim.ml_models.gaussian_process import SparseGPModel, load_model

# Load pre-trained model
model = load_model("models/gp_robust_model.pth")

# Run inference
predictions = model.predict(X_test)
```

**training/train_gp_colab.ipynb:**
```python
# GPU-enabled training in Colab
# 1. Mount Drive
# 2. Install with GPU support
# 3. Train on full dataset
# 4. Save to Drive/HF Hub
```

---

## Phase 3: Interactive Rebase to Clean History

### 3.1 Rebase Strategy

**Goal:** Squash 75 commits into ~15 clean commits

**Target structure:**
```
1. feat: establish project foundation (d0d1e1f - keep as-is)
2. docs: establish research foundation (fe20c8e - keep as-is)
3. feat: define simulation architecture (f5709bb - keep as-is)
4. feat: implement core library (91cf3e3 + squash 10 related commits)
5. feat: add PyMC Bayesian models (f59f45d + squash related)
6. feat: add workload taxonomy (squash commits)
7. docs: add research notebooks (squash all notebook commits)
8. feat: add GP modeling support (squash GP commits)
9. test: add comprehensive test suite (squash test commits)
10. ci: configure GitHub Actions (squash CI commits)
11. refactor: remove Alibaba/Google datasets (04c055f - keep as-is)
12. refactor: extract GP implementation to library (NEW)
13. refactor: convert notebooks to runbooks (NEW)
14. feat: add foundation model support (NEW)
15. feat: add CloudZero ETL (NEW)
```

### 3.2 Rebase Commands

**Option A: Interactive rebase from foundation**
```bash
git rebase -i d0d1e1f~1  # Interactive from project start
# Mark commits as 'squash' or 'fixup' to consolidate
```

**Option B: Reset and recommit (nuclear option)**
```bash
git checkout -b clean-history d0d1e1f  # Branch from foundation
git cherry-pick fe20c8e f5709bb 91cf3e3  # Keep foundation commits
# Then create new clean commits for Phase 1 & 2 work
```

**Recommendation:** Option A (interactive rebase) - Less destructive, preserves authorship

---

## Phase 4: Execution Environment Support

### 4.1 Local Development (M4 Mac)
**Supported:**
- Library development
- Runbook execution (inference only)
- Small-scale testing
- Visualization

**Not Supported:**
- GP training (CPU too slow, MPS incompatibility)
- Large ETL pipelines (memory limits)
- TimesFM experiments (requires GPU)

### 4.2 Google Colab (Recommended for Training)
**Setup:**
```python
# In Colab notebook
!pip install cloud-resource-simulator[gpu,foundation]

# Mount Drive for model persistence
from google.colab import drive
drive.mount('/content/drive')
```

**Workflow:**
1. Train GP models in `training/train_gp_colab.ipynb`
2. Save trained models to Google Drive
3. Download to `models/` directory for local inference
4. OR upload to HuggingFace Model Hub

**Cost:** Free tier (12hr session, T4 GPU) or Pro ($10/mo, better GPUs)

### 4.3 HuggingFace Inference API (Production)
**For:**
- Published trained models
- Public inference endpoints
- Sharing with community

**Workflow:**
1. Train in Colab
2. Push to HF Model Hub: `model.push_to_hub("username/cloud-gp-model")`
3. Enable Inference API
4. Use in runbooks: `from huggingface_hub import InferenceClient`

---

## Phase 5: Testing & Validation

### 5.1 Update Test Suite
- Test extracted GP modules: `tests/test_gp_models.py`
- Test foundation model wrappers: `tests/test_foundation_models.py`
- Test CloudZero loader: `tests/test_cloudzero_etl.py`
- Runbook execution tests: Update `test_notebooks.py` for new structure

### 5.2 CI/CD Updates
- Run tests on GPU runner for GP tests (GitHub Actions with GPU)
- OR skip GPU tests in CI, document manual testing in Colab
- Add notebook linting for runbooks

---

## Decision Points & Recommendations

### Q1: Where should GP models be trained?
**A:** Google Colab Pro ($10/mo) for GPU access
- Local M4 too slow + MPS compatibility issues
- Colab provides T4/V100 GPUs, persistent storage via Drive
- Can save models to HF Hub for sharing

### Q2: How to handle model persistence?
**A:** Three-tier approach:
1. **Development**: Save to `models/` directory (gitignored, local only)
2. **Sharing**: Push to HuggingFace Model Hub (public/private)
3. **Production**: HF Inference API or Colab-hosted

### Q3: Should ETL run locally or in cloud?
**A:** Hybrid approach:
- **Sample data**: Local testing with CloudZero production samples
- **Full pipelines**: Colab for large-scale processing
- Document both workflows in runbooks

### Q4: TimesFM vs Chronos vs GP?
**A:** Use cases differ:
- **GP**: Anomaly detection, uncertainty quantification, small-scale
- **TimesFM**: Zero-shot forecasting, Google's latest, requires GPU
- **Chronos**: Probabilistic forecasting, Amazon's model, lighter weight

Support all three, let runbooks demonstrate each

### Q5: Rebase timing?
**A:** Do Phase 1-2 first, THEN rebase:
1. Extract implementation to library (Phase 1)
2. Convert notebooks to runbooks (Phase 2)
3. Test everything works
4. Interactive rebase to clean history (Phase 3)
5. Force push to clean remote history

**Why:** Easier to debug if rebase goes wrong, can always revert

---

## Implementation Order

### Week 1: Library Extraction
1. Create GP module structure
2. Extract code from notebook 04
3. Add training utilities
4. Update tests
5. Verify notebook still works with imports

### Week 2: Foundation Models & ETL
1. Add foundation model wrappers (stub implementations first)
2. Add CloudZero ETL module
3. Update pyproject.toml dependencies
4. Create Colab training notebooks

### Week 3: Runbook Conversion
1. Restructure notebooks/ directory
2. Refactor existing notebooks
3. Create quickstart guides
4. Create Colab-specific training notebooks
5. Update documentation

### Week 4: History Cleanup & Polish
1. Interactive rebase to squash commits
2. Force push clean history
3. Update README/docs to reflect new structure
4. Verify CI still passes
5. Create release tag

---

## Risk Mitigation

**Risk 1: Rebase breaks something**
- Mitigation: Create backup branch before rebase
- Recovery: `git reflog` to restore pre-rebase state

**Risk 2: Model training costs escalate**
- Mitigation: Start with Colab free tier, monitor usage
- Recovery: Can train locally (slow but works)

**Risk 3: HF Inference API limits**
- Mitigation: Document rate limits, provide local inference fallback
- Recovery: Self-host models on Colab

**Risk 4: CloudZero data structure changes**
- Mitigation: Abstract ETL interface, version loaders
- Recovery: Create adapters for different schemas

---

## Success Metrics

✅ **Library Quality:**
- All implementation code in `src/cloud_sim/`
- No implementation in runbooks (only imports)
- >70% test coverage maintained

✅ **Runbook Quality:**
- Can execute all runbooks locally (inference only)
- Colab notebooks have clear GPU setup
- Educational content preserved

✅ **History Quality:**
- Clean commit history (<20 commits)
- Each commit buildable and testable
- Clear commit messages

✅ **Documentation Quality:**
- README reflects new structure
- Installation instructions for local vs Colab
- Clear separation of concerns

---

## Next Steps After Approval

1. Create feature branch: `git checkout -b refactor/library-first`
2. Start Phase 1: Extract GP module
3. Commit incrementally with tests
4. When stable, proceed to Phase 2-4
5. Final interactive rebase on feature branch
6. Review before force-pushing to main

**Estimated Effort:** 15-20 hours over 2-4 weeks

---

## Agent Coordination Strategy

### Available Specialized Agents

**Primary Agents for This Work:**
1. **ai-modeling-developer**: 🎯 **LEAD IMPLEMENTATION AGENT** - Test-strategy-first, enforces 70% coverage, research-grounded code
2. **workflow-orchestrator**: Coordinates all phases, manages agent handoffs
3. **repository-manager**: Git operations, commits, branch management, interactive rebase
4. **workflow-designer**: Creates implementation workflows, process documentation
5. **huggingface-hub**: HuggingFace integration, model/dataset operations (when ready)

**Supporting Agents:**
6. **professional-document-architect**: Update READMEs, documentation
7. **llm-ai-agents-and-eng-research**: Research latest AI/ML techniques
8. **work-completion-summary**: Audio summaries after each phase completion

### Agent Execution Plan by Phase

#### Phase 1: Library Extraction (Week 1)
**Lead Agent:** `ai-modeling-developer`
- Proposes test strategy BEFORE extraction
- Extracts code with parallel test development
- Validates 70% coverage before any commits
- Coordinates with repository-manager for commits

**Execution:**
```
ai-modeling-developer:
  # For each module extraction:
  1. Analyze code to be extracted (GP models, foundation stubs, ETL)
  2. Propose test strategy:
     - Unit tests for kernels, models, training utilities
     - Integration tests for end-to-end GP workflow
     - Coverage target: >70% for new modules
  3. Get user approval on test strategy
  4. Extract code + write comprehensive tests in parallel
  5. Run coverage validation: uv run pytest --cov=src/cloud_sim
  6. If coverage ≥70%: Delegate to repository-manager for commit
  7. If coverage <70%: Add tests until threshold met

workflow-orchestrator:
  - Coordinates sequential extraction (GP → Foundation → ETL)
  - Manages handoffs between ai-modeling-developer and repository-manager

repository-manager:
  - Commit after coverage validation
  - Format: "refactor(ml_models): extract GP implementation from notebook (72% coverage)"
```

#### Phase 2: Runbook Conversion (Week 2-3)
**Lead Agent:** `workflow-designer`
- Designs new notebook structure
- Creates quickstart guides
- Documents Colab workflows

**Execution:**
```
workflow-designer:
  - Create notebook restructuring plan
  - Design Colab training workflow documentation
  - Create installation guides

workflow-orchestrator:
  - Coordinate notebook file moves
  - Update notebook references
  - Create new runbook templates

repository-manager:
  - Commit notebook restructuring: "refactor(notebooks): convert to runbook architecture"
  - Commit new guides: "docs(notebooks): add quickstart and training guides"
```

#### Phase 3: History Cleanup (Week 4)
**Lead Agent:** `repository-manager`
- Execute interactive rebase
- Handle merge conflicts
- Verify commit integrity

**Execution:**
```
repository-manager:
  - Create backup branch: backup/pre-rebase
  - Interactive rebase: git rebase -i d0d1e1f~1
  - Squash 75 commits → ~15 clean commits
  - Verify each commit builds: git rebase --exec "uv run pytest tests/"
  - Force push after verification
```

#### Phase 4: Documentation Updates
**Lead Agent:** `professional-document-architect`
- Rewrite project history
- Update README files
- Create migration guide

**Execution:**
```
professional-document-architect:
  - Update .claude-project-memory.md with new architecture
  - Rewrite main README.md with new structure
  - Create MIGRATION.md for existing users
  - Update CLAUDE.md with new module structure

repository-manager:
  - Commit: "docs: update documentation for library-first architecture"
```

#### Phase 5: Completion
**Lead Agent:** `work-completion-summary`
- Generate audio summary of changes
- Highlight key decisions and next steps

### Parallel Execution Opportunities

**Week 1: Can run in parallel:**
- GP extraction + Foundation model stubs + CloudZero ETL
- Use single workflow-orchestrator invocation with multiple subtasks

**Week 2-3: Sequential required:**
- Notebook restructuring must follow library extraction (imports need to work)
- But can parallelize: quickstart guides + Colab notebooks + EDA moves

**Week 4: Must be sequential:**
- Interactive rebase is atomic operation
- Documentation updates must follow rebase (commit SHAs change)

### Agent Invocation Pattern

**For each phase:**
```python
# Phase 1 Example
workflow_orchestrator.execute({
    "phase": "Library Extraction",
    "tasks": [
        {
            "name": "Extract GP module",
            "files": ["notebooks/04_gaussian_process_modeling.md"],
            "output": ["src/cloud_sim/ml_models/gaussian_process/"],
            "agent": "general-purpose"  # For code extraction
        },
        {
            "name": "Commit extraction",
            "agent": "repository-manager",
            "commit_message": "refactor(ml_models): extract GP implementation from notebook"
        }
    ]
})
```

### Success Criteria per Agent

**ai-modeling-developer (CRITICAL):**
- ✅ Test strategy proposed and approved BEFORE implementation
- ✅ All new code has ≥70% test coverage
- ✅ Tests include research-validated parameters
- ✅ No commits without coverage validation
- ✅ All tests passing before handoff to repository-manager
- ✅ Polars-only (no pandas) enforced
- ✅ No print statements in notebooks

**workflow-orchestrator:**
- ✅ All subtasks completed
- ✅ Context properly passed between agents
- ✅ TodoWrite tracking maintained

**repository-manager:**
- ✅ All commits follow conventional format
- ✅ Each commit builds and passes tests
- ✅ SSH signing verified on all commits
- ✅ Interactive rebase completes without conflicts
- ✅ Commit messages include coverage percentage

**workflow-designer:**
- ✅ Clear workflow documentation created
- ✅ Colab setup instructions are executable
- ✅ Runbook structure is intuitive

**professional-document-architect:**
- ✅ Documentation reflects new architecture
- ✅ Migration path clearly explained
- ✅ No broken links or outdated references

### Error Recovery Plan

**If workflow-orchestrator fails:**
- Review TodoWrite progress
- Complete failed subtask manually
- Resume from next incomplete task

**If repository-manager rebase fails:**
- `git rebase --abort`
- Restore from backup: `git reset --hard backup/pre-rebase`
- Review conflict manually
- Retry with adjusted strategy

**If documentation updates incomplete:**
- Use professional-document-architect for final pass
- Verify all cross-references
- Run link checker

---

## Related Documentation to Update

After implementation:
- [ ] Update `.claude-project-memory.md` with new architecture
- [ ] Rewrite project history in documentation
- [ ] Update main README.md with new structure
- [ ] Create migration guide for existing users
- [ ] Update agent coordination documentation with lessons learned
