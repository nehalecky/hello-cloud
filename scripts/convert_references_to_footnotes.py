#!/usr/bin/env python3
"""
Convert manual HTML anchor references to MkDocs footnote format.

Converts:
  [[1]](#ref-1) -> [^1]
  <a id="ref-1"></a>[1] Citation -> [^1]: Citation
"""

import re
import sys
from pathlib import Path


def convert_inline_references(content: str) -> str:
    """Convert inline reference links [[N]](#ref-N) to [^N]."""
    pattern = r"\[\[(\d+)\]\]\(#ref-\1\)"
    return re.sub(pattern, r"[^\1]", content)


def convert_reference_definitions(content: str) -> str:
    """Convert reference definitions to footnote format."""
    # Pattern: <a id="ref-N"></a>[N] Citation text
    pattern = r'<a id="ref-(\d+)"></a>\[(\d+)\]\s+'
    return re.sub(pattern, r"[^\1]: ", content)


def process_file(filepath: Path) -> tuple[bool, str]:
    """
    Process a single markdown file to convert references.

    Returns:
        (changed, new_content) tuple
    """
    content = filepath.read_text()
    original = content

    # Apply conversions
    content = convert_inline_references(content)
    content = convert_reference_definitions(content)

    changed = content != original
    return changed, content


def main():
    """Convert all research markdown files."""
    research_dir = Path(__file__).parent.parent / "docs" / "research"

    if not research_dir.exists():
        print(f"Error: {research_dir} does not exist")
        sys.exit(1)

    files_changed = 0
    for md_file in research_dir.glob("*.md"):
        changed, new_content = process_file(md_file)

        if changed:
            md_file.write_text(new_content)
            print(f"✓ Converted {md_file.name}")
            files_changed += 1
        else:
            print(f"  Skipped {md_file.name} (no changes)")

    print(f"\n{files_changed} file(s) updated")


if __name__ == "__main__":
    main()
