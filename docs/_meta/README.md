# Cloud Resource Simulator Documentation

This directory contains the **Quarto-based documentation** for Cloud Resource Simulator, following the [Diátaxis framework](https://diataxis.fr/).

## 🏗️ Structure

```
docs/
├── _quarto.yml              # Quarto configuration
├── index.qmd                # Landing page
├── tutorials/               # Learning-oriented guides
│   ├── data-exploration.qmd
│   ├── workload-signatures.qmd
│   └── gaussian-processes.qmd
├── how-to/                  # Task-oriented guides
│   ├── generate-synthetic-data.qmd
│   └── train-gp-models.qmd
├── concepts/                # Understanding-oriented docs
│   ├── research/           # Empirical research (35+ citations)
│   └── design/             # Architecture & design docs
├── reference/               # API documentation (auto-generated)
└── _site/                   # Generated website (gitignored)
```

## 🚀 Quick Start

### Prerequisites

1. **Install Quarto** (one-time setup):
   ```bash
   brew install --cask quarto
   ```

2. **Install Python dependencies**:
   ```bash
   uv sync --group docs
   ```

### Build & Preview

```bash
# Preview documentation locally
quarto preview docs/

# Or build static site
quarto render docs/

# Generate API reference
quartodoc build --config docs/_quarto.yml
```

### Hot Reload Development

The preview server **auto-refreshes** when you edit files:
1. Run `quarto preview docs/`
2. Edit any `.qmd` file
3. Save → browser refreshes automatically

## 📝 Creating Content

### Add a New Tutorial

```bash
# Create file
touch docs/tutorials/my-tutorial.qmd

# Add frontmatter
cat > docs/tutorials/my-tutorial.qmd << 'EOF'
---
title: "My Tutorial"
subtitle: "Learn something specific"
execute:
  eval: false
  echo: true
---

# My Tutorial

Content here...
EOF

# Add to navigation in _quarto.yml
```

### Add a New How-To Guide

```bash
# Create in how-to/ directory
touch docs/how-to/my-guide.qmd

# Follow task-oriented format:
# 1. Prerequisites
# 2. Quick start
# 3. Common tasks
# 4. Troubleshooting
```

### Add Research/Concept Doc

Research docs are already in `concepts/research/` and `concepts/design/`. Simply edit the `.qmd` files.

## 🎨 Styling & Theming

Customization files:
- `custom-light.scss` - Light theme (extends Flatly)
- `custom-dark.scss` - Dark theme (extends Darkly)
- `styles.css` - Additional CSS

## 🔧 Configuration

### _quarto.yml Structure

```yaml
project:
  type: website

website:
  title: "Cloud Resource Simulator"
  navbar: ...
  sidebar: ...

format:
  html:
    theme: [flatly, custom-light.scss]
    toc: true
    code-copy: true

execute:
  freeze: auto  # Re-execute only when source changes

quartodoc:  # API reference configuration
  package: cloud_sim
  sections: ...
```

## 📚 Documentation Philosophy (Diátaxis)

### Tutorials
**Learning-oriented** - Take the user by the hand through a series of steps

- Focus: Learning
- Analogy: Teaching a child to cook
- Examples: "Data Exploration Tutorial", "Gaussian Process Modeling"

### How-To Guides
**Task-oriented** - Show how to solve a specific problem

- Focus: Goals
- Analogy: Recipe in a cookbook
- Examples: "Generate Synthetic Data", "Train GP Models"

### Concepts (Explanation)
**Understanding-oriented** - Explain and clarify

- Focus: Knowledge
- Analogy: Article on culinary social history
- Examples: Research papers, design documents

### Reference
**Information-oriented** - Technical description

- Focus: Information
- Analogy: Encyclopedia article
- Examples: API documentation (auto-generated)

## 🚀 Deployment

### GitHub Pages

The site is automatically deployed via GitHub Actions on every push to `main`:

```yaml
# .github/workflows/docs.yml
- Build with Quarto
- Generate API docs with quartodoc
- Deploy to gh-pages branch
```

**Live site**: https://nehalecky.github.io/cloud-resource-simulator

### Manual Deployment

```bash
# Build site
quarto render docs/

# Deploy to GitHub Pages
quarto publish gh-pages docs/
```

## 🐛 Troubleshooting

### Issue: Quarto command not found

**Solution**: Install Quarto CLI
```bash
brew install --cask quarto
```

### Issue: quartodoc fails

**Solution**: Install Python dependencies
```bash
uv sync --group docs
quartodoc build --config docs/_quarto.yml
```

### Issue: Render fails with import errors

**Solution**: Install library in development mode
```bash
uv pip install -e .
```

### Issue: Images/plots don't render

**Solution**: Check `execute` settings in frontmatter
```yaml
---
execute:
  eval: true  # Enable code execution
  echo: true  # Show code
---
```

## 📖 Resources

- **Quarto Documentation**: https://quarto.org/docs/
- **Quartodoc**: https://machow.github.io/quartodoc/
- **Diátaxis Framework**: https://diataxis.fr/
- **Markdown Syntax**: https://quarto.org/docs/authoring/markdown-basics.html

## 🤝 Contributing

When adding documentation:

1. **Choose the right category** (Tutorial/How-To/Concept/Reference)
2. **Follow the format** of existing docs in that category
3. **Add to navigation** in `_quarto.yml`
4. **Preview locally** before committing
5. **Check links** work correctly

### Style Guide

- **Tutorials**: Conversational, step-by-step, complete examples
- **How-To**: Concise, task-focused, problem-solution format
- **Concepts**: Explanatory, deep-dive, cite sources
- **Reference**: Technical, complete, auto-generated

---

**Questions?** Open an issue on GitHub or see the [main README](../README.md).
