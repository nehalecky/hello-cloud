# Deployment Guide

This site is automatically deployed to GitHub Pages via GitHub Actions.

## GitHub Pages Setup

**One-time configuration in repository settings:**

1. Go to repository **Settings** → **Pages**
2. Under **Source**, select: **GitHub Actions**
3. Done! No additional configuration needed.

## CI/CD Workflow

The `.github/workflows/docs.yml` workflow automatically:

1. **Executes notebooks** with outputs (`just nb-execute-all`)
2. **Generates API reference** via quartodoc
3. **Renders Quarto site** to static HTML
4. **Deploys to GitHub Pages**

**Triggers:**
- Push to `master` branch (docs/, notebooks/, src/hellocloud/ changes)
- Manual dispatch via Actions tab
- Pull requests (build only, no deploy)

**Site URL:** https://nehalecky.github.io/hello-cloud

## Local Preview

```bash
# Execute notebooks with outputs
just nb-execute-all

# Preview site (auto-refreshes)
quarto preview docs/

# Build site locally
quarto render docs/
```

## Workflow Details

**Build process:**
1. Checkout code
2. Setup Quarto 1.8.25
3. Setup Python 3.11 + uv
4. Install dependencies (`uv sync --all-extras`)
5. Execute all notebooks (MyST → .ipynb with outputs)
6. Generate API reference (quartodoc)
7. Render Quarto site (docs/ → docs/_site/)
8. Upload artifact

**Deploy process:**
- Deploys docs/_site/ to GitHub Pages
- Only runs on push to master (not PRs)
- Requires `pages: write` permission

## Troubleshooting

**Notebooks fail to execute:**
- Check notebook dependencies in pyproject.toml `[project.optional-dependencies]`
- Verify notebooks execute locally: `just nb-test-all`

**Quarto render fails:**
- Check Quarto version matches local (1.8.25)
- Verify _quarto.yml configuration

**GitHub Pages not updating:**
- Check Actions tab for workflow failures
- Ensure GitHub Pages source is set to "GitHub Actions"
- Verify `pages: write` permission in workflow

**Notebooks missing in published site:**
- Execute notebooks locally: `just nb-execute-all`
- Commit and push (CI will re-execute, but verifying locally helps debug)
