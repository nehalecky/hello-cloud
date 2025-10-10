# Documentation Directory Structure

## 📁 Clean Organization

```
docs/
├── _quarto.yml              ← Configuration (1 file)
├── README.md                ← Quick reference
│
├── _theme/                  ← Styling assets
│   ├── custom-light.scss
│   ├── custom-dark.scss
│   └── styles.css
│
├── _meta/                   ← Internal documentation
│   ├── README.md
│   ├── MIGRATION_SUMMARY.md
│   └── STRUCTURE.md (this file)
│
├── _archive/                ← Deprecated content
│   └── (old docs)
│
├── index.qmd                ← Landing page
│
├── tutorials/               ← Learning guides
│   ├── index.qmd
│   ├── data-exploration.qmd
│   ├── workload-signatures.qmd
│   └── gaussian-processes.qmd
│
├── how-to/                  ← Task guides
│   ├── index.qmd
│   ├── generate-synthetic-data.qmd
│   └── train-gp-models.qmd
│
├── concepts/                ← Explanatory docs
│   ├── index.qmd
│   ├── research/           (5 research papers)
│   └── design/             (6 architecture docs)
│
└── reference/               ← API docs (auto-generated)
    ├── index.qmd
    └── (12 API documentation files)
```

## 🎯 Before & After

### Before (Cluttered)
```
docs/
├── _quarto.yml
├── custom-light.scss       ← Clutter
├── custom-dark.scss        ← Clutter
├── styles.css              ← Clutter
├── README.md               ← Meta doc at root
├── MIGRATION_SUMMARY.md    ← Meta doc at root
├── index.qmd
├── tutorials/
├── how-to/
├── concepts/
└── reference/
```

### After (Clean)
```
docs/
├── _quarto.yml             ← Config only
├── README.md               ← User-facing
├── _theme/                 ← Organized assets
├── _meta/                  ← Internal docs
├── index.qmd
├── tutorials/              ← Content directories
├── how-to/
├── concepts/
└── reference/
```

## 🔍 Directory Purposes

| Directory | Purpose | Committed to Git? |
|-----------|---------|-------------------|
| `_theme/` | Styling (SCSS, CSS) | ✅ Yes |
| `_meta/` | Internal documentation | ✅ Yes |
| `_archive/` | Deprecated content | ✅ Yes |
| `_site/` | Generated website | ❌ No (gitignored) |
| `.quarto/` | Quarto cache | ❌ No (gitignored) |
| `tutorials/` | User-facing tutorials | ✅ Yes |
| `how-to/` | User-facing guides | ✅ Yes |
| `concepts/` | User-facing explanations | ✅ Yes |
| `reference/` | Auto-generated API docs | ✅ Yes |

## 📦 File Categories

### Configuration (1 file)
- `_quarto.yml` - Main Quarto configuration

### Styling (3 files in `_theme/`)
- `custom-light.scss` - Light theme
- `custom-dark.scss` - Dark theme
- `styles.css` - Additional CSS

### Meta Documentation (3 files in `_meta/`)
- `README.md` - Developer guide
- `MIGRATION_SUMMARY.md` - Migration record
- `STRUCTURE.md` - This file

### Content (28 files across content directories)
- **Tutorials**: 4 files (index + 3 guides)
- **How-To**: 3 files (index + 2 guides)
- **Concepts**: 12 files (index + research + design)
- **Reference**: 9 files (index + 8 API docs)
- **Root**: 2 files (README + landing page)

## 🧹 Maintenance

### What to Edit
- **Content**: `tutorials/`, `how-to/`, `concepts/`
- **Styling**: Files in `_theme/`
- **Config**: `_quarto.yml`

### What Not to Edit
- **Generated**: `reference/` (run `quartodoc build`)
- **Build output**: `_site/`, `.quarto/`

### When to Run Commands

**After adding API**: Generate reference docs
```bash
uv run quartodoc build --config docs/_quarto.yml
```

**After content changes**: Preview updates automatically
```bash
quarto preview docs/  # Already running? Changes auto-refresh!
```

## 🎨 Visual Hierarchy

```
Public-Facing Content
  ├── Landing (index.qmd)
  ├── Tutorials (learning)
  ├── How-To (tasks)
  ├── Concepts (understanding)
  └── Reference (information)

Internal Organization
  ├── _theme/ (styling)
  ├── _meta/ (documentation about docs)
  └── _archive/ (deprecated)

Generated (gitignored)
  ├── _site/ (built website)
  └── .quarto/ (build cache)
```

## ✨ Benefits

1. **Clean root** - Only config + content directories visible
2. **Organized assets** - Styling in one place (`_theme/`)
3. **Clear separation** - Internal vs. user-facing docs
4. **Easy navigation** - Fewer files at top level
5. **Git-friendly** - Generated files properly ignored

---

**Last updated**: 2025-10-10
**Maintained by**: Documentation migration project
