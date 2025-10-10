# Theme Assets

This directory contains styling and theme customization for the Quarto documentation site.

## Files

- **`custom-light.scss`** - Light theme customization (extends Flatly)
- **`custom-dark.scss`** - Dark theme customization (extends Darkly)
- **`styles.css`** - Additional CSS overrides and utilities

## Usage

These files are referenced in `_quarto.yml`:

```yaml
format:
  html:
    theme:
      light: [flatly, _theme/custom-light.scss]
      dark: [darkly, _theme/custom-dark.scss]
    css: _theme/styles.css
```

## Editing Themes

1. **SCSS files** - Modify variables and Bootstrap overrides
2. **CSS file** - Add custom styles and utilities
3. **Preview changes** - Run `quarto preview docs/` to see live updates

## Theme Structure

Based on [Bootswatch](https://bootswatch.com/):
- **Flatly** - Clean, modern light theme
- **Darkly** - Professional dark theme

Both extended with custom colors, typography, and component styling.
