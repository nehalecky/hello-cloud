# Rename Plan: cloudlens → hellocloud

## Complete Rename Checklist

### 1. Repository Directory
```bash
cd /Users/nehalecky/Projects/cloudzero
mv cloud-resource-simulator hello-cloud
cd hello-cloud
```

### 2. README.md (Alternative Tone - Direct, Implicit)
```markdown
# Hello Cloud

Hands-on exploration of cloud resource usage and cost optimization.

Workload characterization • Cost analysis • Time series forecasting • Anomaly detection

Ibis+DuckDB (local) • PySpark (scale)

## Installation
[keep current structure, update imports to hellocloud]

## Usage
[keep current examples, update imports to hellocloud]

## Stack
[keep current]

## Development
[keep current, update paths to hellocloud]
```

### 3. pyproject.toml Updates
```toml
name = "hellocloud"
description = "Hands-on exploration of cloud resource usage and cost optimization"
keywords = ["cloud", "cost-optimization", "workload-analysis", "time-series", "forecasting"]

[project.urls]
Homepage = "https://github.com/nehalecky/hello-cloud"
Repository = "https://github.com/nehalecky/hello-cloud"
Issues = "https://github.com/nehalecky/hello-cloud/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/hellocloud"]

[tool.pytest.ini_options]
addopts = ["--cov=src/hellocloud", ...]

[tool.coverage.run]
source = ["src/hellocloud"]
```

### 4. Source Directory Rename
```bash
mv src/cloudlens src/hellocloud
```

### 5. Update All Imports
Replace in all files:
- `from cloudlens.` → `from hellocloud.`
- `import cloudlens` → `import hellocloud`

**Files to update:**
- `src/hellocloud/**/*.py`
- `tests/**/*.py`
- `notebooks/**/*.md`

```bash
# Batch update commands
find src/hellocloud -name "*.py" -type f | xargs sed -i '' 's/from cloudlens\./from hellocloud./g; s/import cloudlens/import hellocloud/g'
find tests -name "*.py" -type f | xargs sed -i '' 's/from cloudlens\./from hellocloud./g; s/import cloudlens/import hellocloud/g'
find notebooks -name "*.md" -type f | xargs sed -i '' 's/from cloudlens\./from hellocloud./g; s/import cloudlens/import hellocloud/g'
```

### 6. CLAUDE.md Updates
```bash
sed -i '' 's/cloudlens/hellocloud/g' CLAUDE.md
sed -i '' 's/cloud-resource-simulator/hello-cloud/g' CLAUDE.md
```

### 7. .gitignore Update
```bash
sed -i '' 's/src\/cloudlens\/data_generation/src\/hellocloud\/data_generation/g' .gitignore
```

### 8. Keep "PiedPiper" References
**NO CHANGES** - PiedPiper is fictional (Silicon Valley TV show reference)

### 9. GitHub Remote Update
```bash
# Option A: Update remote URL
git remote set-url origin git@github.com:nehalecky/hello-cloud.git

# Option B: If repo doesn't exist yet, create it
gh repo create hello-cloud --public --source=. --remote=origin

# Then push
git push -u origin master
```

### 10. Stage and Commit Changes
```bash
git add -A
git commit -m "refactor: rename project to Hello Cloud (hellocloud)

- Rename package: cloudlens → hellocloud
- Update README with direct, implicit tone
- Update all imports across source, tests, notebooks
- Update pyproject.toml metadata and URLs
- Update CLAUDE.md references

Rationale: Personal exploration project for understanding cloud
resource usage and cost optimization. Name reflects learning intent
without explicit marketing language."
```

## Tone Principles Applied
- **Implicit learning intent**: No explicit "like Hello World" - let the name speak
- **Direct, factual**: No marketing language
- **Personal exploration**: "Hands-on" conveys active learning
- **Technical rigor**: Stack and tools front and center

## Current Status (Pre-Execution)
- ✅ cloudlens → ready to rename to hellocloud
- ✅ All imports currently use cloudlens
- ✅ Directory: cloud-resource-simulator → needs rename to hello-cloud
- ⚠️ GitHub repo needs creation or update
- ⚠️ data_generation already unversioned (good)

## Files Modified (Will Be)
- `README.md` - complete rewrite with new tone
- `pyproject.toml` - name, description, URLs, paths
- `CLAUDE.md` - all references
- `src/cloudlens/` → `src/hellocloud/` (directory + all imports)
- `tests/**/*.py` - imports
- `notebooks/**/*.md` - imports
- `.gitignore` - unversioned module path

## Next Steps After Compaction
1. Execute directory rename
2. Execute source rename
3. Execute import updates
4. Update README, pyproject.toml, CLAUDE.md
5. Stage changes
6. Create/update GitHub repo
7. Commit and push
