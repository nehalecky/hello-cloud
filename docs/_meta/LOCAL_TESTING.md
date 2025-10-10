# Local Documentation Testing

## Quick Reference

### Daily Development Workflow

```bash
# 1. Make changes to code/docstrings
vim src/cloud_sim/data_generation/workload_patterns.py

# 2. Regenerate API reference
just docs-api

# 3. Preview changes (auto-refreshes on file edits)
just docs-preview

# Opens http://localhost:4444/
# Edit any .qmd file → browser auto-refreshes!
```

### One-Time Setup

```bash
# Install just (if not installed)
brew install just

# Install Quarto (if not installed)
brew install --cask quarto

# Install Python docs dependencies
uv sync --group docs
```

### Available Commands

| Command | Description |
|---------|-------------|
| `just docs-api` | Generate API reference from docstrings |
| `just docs-preview` | Preview with auto-refresh (port 4444) |
| `just docs-build` | Build static site to `docs/_site/` |
| `just docs` | API + build (full rebuild) |
| `just docs-clean` | Remove build artifacts |

### Manual Commands

If you prefer not to use `just`:

```bash
# Generate API reference
uv run quartodoc build --config docs/_quarto.yml

# Preview (auto-refreshes)
quarto preview docs/

# Build static site
quarto render docs/

# Clean build
rm -rf docs/_site docs/.quarto
```

## Testing Checklist

Before committing docs changes:

- [ ] **API reference generated**: `just docs-api`
- [ ] **Preview looks good**: `just docs-preview` → check http://localhost:4444/
- [ ] **All links work**: Click through navigation
- [ ] **Code examples render**: Check syntax highlighting
- [ ] **No warnings**: Review build output
- [ ] **Build succeeds**: `just docs-build` completes without errors

## Common Issues

### Issue: "quartodoc: command not found"
**Solution**: Install docs dependencies
```bash
uv sync --group docs
```

### Issue: "quarto: command not found"
**Solution**: Install Quarto CLI
```bash
brew install --cask quarto
```

### Issue: API reference out of date
**Solution**: Regenerate after code changes
```bash
just docs-api
```

### Issue: Preview shows old content
**Solution**: Hard refresh browser (Cmd+Shift+R) or restart preview server

### Issue: Build fails with "port already in use"
**Solution**: Kill existing process
```bash
lsof -ti:4444 | xargs kill -9
```

## File Watching

The preview server automatically watches:
- ✅ All `.qmd` files in `docs/`
- ✅ Theme files in `docs/_theme/`
- ✅ Quarto config (`_quarto.yml`)
- ❌ Python source code (you must run `just docs-api` after changing docstrings)

## Build Artifacts

Generated files (gitignored):
- `docs/_site/` - Complete static website
- `docs/.quarto/` - Quarto build cache
- `docs/objects.json` - Search index metadata

## Performance Tips

1. **Use preview for development** - Much faster than full builds
2. **Incremental builds** - Quarto only rebuilds changed files
3. **Skip API regeneration** - If only editing content, skip `docs-api`
4. **Clear cache if weird** - `just docs-clean` then rebuild

## Integration with Testing

The docs can be built as part of your test suite:

```bash
# Run all checks (lint + test + docs)
just check

# This runs:
# - just lint
# - just test
# - just docs
```

## CI/CD Notes

For later GitHub Actions setup:
- Docs build on every push to `main`
- Auto-deploy to GitHub Pages
- Preview deployments on PRs (optional)

See `.github/workflows/docs.yml` for CI configuration (when ready).

---

**Last updated**: 2025-10-10
**For CI setup**: See `MIGRATION_SUMMARY.md` → Step 5
