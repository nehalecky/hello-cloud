#!/usr/bin/env python3
"""
Fix list formatting issues in markdown files.

Adds blank lines between text ending with ':' and list items.
"""

import re
from pathlib import Path


def fix_list_formatting(content):
    """Add blank lines before lists that immediately follow colons."""
    lines = content.split("\n")
    fixed_lines = []
    i = 0

    while i < len(lines):
        current = lines[i]
        fixed_lines.append(current)

        # Check if we need to insert a blank line
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            current_stripped = current.strip()
            next_stripped = next_line.strip()

            # Skip if we're in a code block
            if current_stripped.startswith("```") or current_stripped.startswith("    "):
                i += 1
                continue

            # Check if current line ends with colon and next starts a list
            if current_stripped.endswith(":") and (
                re.match(r"^\d+\.", next_stripped)
                or next_stripped.startswith("- ")
                or next_stripped.startswith("* ")
            ):
                # Don't add blank if there already is one
                if next_line.strip():  # Next line is not blank
                    fixed_lines.append("")  # Insert blank line

        i += 1

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a single markdown file."""
    content = filepath.read_text()
    fixed = fix_list_formatting(content)

    if fixed != content:
        filepath.write_text(fixed)
        return True
    return False


def main():
    """Fix list formatting in published documentation."""
    # Only process published docs, not archive or plans
    published_dirs = [
        Path("docs/research"),
        Path("docs/contributing"),
        Path("docs/reference"),
        Path("docs/getting-started"),
        Path("docs/tutorials"),
    ]

    # Also process top-level docs
    published_files = [Path("docs/index.md")]

    total_fixed = 0

    # Process directories
    for doc_dir in published_dirs:
        if not doc_dir.exists():
            continue

        for md_file in doc_dir.glob("*.md"):
            if process_file(md_file):
                print(f"✓ Fixed {md_file}")
                total_fixed += 1
            else:
                print(f"  Skipped {md_file.name} (no changes)")

    # Process individual files
    for md_file in published_files:
        if md_file.exists():
            if process_file(md_file):
                print(f"✓ Fixed {md_file}")
                total_fixed += 1

    print(f"\n{total_fixed} file(s) updated")


if __name__ == "__main__":
    main()
