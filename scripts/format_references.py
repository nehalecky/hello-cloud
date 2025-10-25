#!/usr/bin/env python3
"""
Automate reference formatting in Markdown documentation.

Transforms:
1. Plain-text URLs → Markdown links
2. Adds HTML anchors to references
3. Converts in-text citations to anchor links
4. Validates citation/reference consistency
"""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Format references in Markdown docs")
    parser.add_argument("files", nargs="+", help="Markdown files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    for file_path in args.files:
        process_file(Path(file_path), dry_run=args.dry_run)


def split_document(content: str) -> tuple[str, str, str]:
    """
    Split document into pre-references, references, and post-references.

    Returns:
        (pre_content, references_section, post_content)
    """
    # Match "## References" or "## N. References" where N is digit
    pattern = r"^(#{1,6})\s+(\d+\.)?\s*References\s*$"

    lines = content.split("\n")
    ref_start_idx = None

    for i, line in enumerate(lines):
        if re.match(pattern, line, re.IGNORECASE):
            ref_start_idx = i
            break

    if ref_start_idx is None:
        return content, "", ""

    # Find next major section (same level heading) or end of document
    heading_level = len(re.match(r"^(#+)", lines[ref_start_idx]).group(1))
    ref_end_idx = len(lines)

    for i in range(ref_start_idx + 1, len(lines)):
        if re.match(f"^#{{{heading_level}}}\\s+", lines[i]):
            ref_end_idx = i
            break

    pre_content = "\n".join(lines[:ref_start_idx])
    references = "\n".join(lines[ref_start_idx:ref_end_idx])
    post_content = "\n".join(lines[ref_end_idx:]) if ref_end_idx < len(lines) else ""

    return pre_content, references, post_content


def format_reference_urls(references_text: str) -> tuple[str, int]:
    """
    Convert plain-text URLs in references to Markdown links.

    Pattern matches both single-line and multi-line references:

    With author:
    [N] Author. (Year). "Title." Publication. URL

    Without author:
    [N] "Title." (Year). Publication. URL

    Multi-line:
    [N] Author/Title. (Year). "Title."
        Publication details.
        https://example.com
        Optional annotation text.

    Converts to:
    <a id="ref-N"></a>[N] Author. (Year). ["Title."](URL) *Publication*.
    Optional annotation preserved.

    Returns:
        (formatted_text, count_of_changes)
    """
    # Regex pattern for reference entries (handles multi-line and both formats)
    # Matches from [N] through the URL, capturing everything in between
    # re.DOTALL allows . to match newlines
    # Two patterns: with author and without author
    pattern = re.compile(
        r"^\[(\d+)\]\s+"  # [N]
        r"(?:"  # Start non-capturing group for author variants
        r'(.+?)\.\s+\((\d{4})\)\.\s+"([^"]+)"\s*'  # Author. (Year). "Title"
        r"|"  # OR
        r'"([^"]+)"\s+\((\d{4})\)\.\s*'  # "Title" (Year). (note: space, not period after quote!)
        r")"  # End non-capturing group
        r"(.*?)"  # Everything between title and URL (may span lines)
        r"(https?://[^\s]+)"  # URL
        r"(.*?)(?=\n\[|\Z)",  # Optional text after URL until next ref or end
        re.MULTILINE | re.DOTALL,
    )

    changes = 0

    def replace_func(match):
        nonlocal changes
        changes += 1

        ref_num = match.group(1)

        # Check which pattern matched: with author (groups 2,3,4) or without (groups 5,6)
        if match.group(2) is not None:  # Pattern with author
            author = match.group(2)
            year = match.group(3)
            title = match.group(4)
            middle_text = match.group(7).strip()
        else:  # Pattern without author (title first)
            author = None
            title = match.group(5)
            year = match.group(6)
            middle_text = match.group(7).strip()

        url = match.group(8)
        after_url = match.group(9).strip()

        # Extract publication name from middle text
        # Remove leading/trailing whitespace, newlines, and markdown formatting
        publication = middle_text.replace("\n", " ").replace("  ", " ").strip(" .*")

        # Build formatted reference with anchor and Markdown link
        if author:
            result = (
                f'<a id="ref-{ref_num}"></a>'
                f"[{ref_num}] {author}. ({year}). "
                f'["{title}"]({url}) '
                f"*{publication}*."
            )
        else:
            result = (
                f'<a id="ref-{ref_num}"></a>'
                f'[{ref_num}] ["{title}"]({url}) ({year}). '
                f"*{publication}*."
            )

        # Preserve any annotation text after the URL
        if after_url:
            result += f"\n{after_url}"

        return result

    formatted = pattern.sub(replace_func, references_text)
    return formatted, changes


def link_citations(content: str, valid_refs: set) -> tuple[str, int]:
    """
    Convert in-text citations [N] to anchor links [[N]](#ref-N).

    Skips citations inside:
    - Code blocks (``` ```)
    - Inline code (` `)
    - Already formatted as [[N]](#ref-N)

    Args:
        content: Document content before References section
        valid_refs: Set of valid reference numbers from References section

    Returns:
        (formatted_content, count_of_changes)
    """
    changes = 0
    lines = content.split("\n")
    result_lines = []
    in_code_block = False

    for line in lines:
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            continue

        if in_code_block:
            result_lines.append(line)
            continue

        # Replace [N] with [[N]](#ref-N) outside of inline code
        def replace_citation(match):
            nonlocal changes
            ref_num = match.group(1)

            # Only link if reference exists
            if ref_num in valid_refs:
                changes += 1
                return f"[[{ref_num}]](#ref-{ref_num})"
            else:
                # Warn about missing reference but don't modify
                print(f"⚠️  Warning: Citation [{ref_num}] has no corresponding reference")
                return match.group(0)

        # Pattern: [N] not preceded by [ (avoids inline code and existing links)
        # Negative lookbehind: (?<![`\[])
        pattern = r"(?<![`\[])\[(\d+)\](?!`)"
        modified_line = re.sub(pattern, replace_citation, line)
        result_lines.append(modified_line)

    return "\n".join(result_lines), changes


def extract_reference_numbers(references_text: str) -> set:
    """Extract all reference numbers from References section."""
    pattern = r'<a id="ref-(\d+)"></a>\[(\d+)\]'
    matches = re.findall(pattern, references_text)
    return {num for _, num in matches}


def validate_references(pre_content: str, references: str) -> dict[str, list[str]]:
    """
    Validate citation/reference consistency.

    Returns dict with:
        - 'missing_refs': Citations with no corresponding reference
        - 'orphaned_refs': References never cited
    """
    # Extract cited reference numbers
    cited = set(re.findall(r"\[\[(\d+)\]\]\(#ref-\d+\)", pre_content))

    # Extract defined reference numbers
    defined = extract_reference_numbers(references)

    return {"missing_refs": sorted(cited - defined), "orphaned_refs": sorted(defined - cited)}


def process_file(path: Path, dry_run: bool = False):
    """Process a single Markdown file."""
    if not path.exists():
        print(f"❌ Error: {path} does not exist")
        return False

    # Read original content
    original_content = path.read_text()

    # Create backup
    backup_path = path.with_suffix(".md.bak")
    if not dry_run:
        backup_path.write_text(original_content)
        print(f"📝 Created backup: {backup_path}")

    # Split document
    pre_content, references, post_content = split_document(original_content)

    if not references:
        print(f"⚠️  No References section found in {path}")
        return False

    # Format references
    formatted_refs, ref_changes = format_reference_urls(references)

    # Extract valid reference numbers
    valid_refs = extract_reference_numbers(formatted_refs)

    # Link citations
    formatted_pre, cite_changes = link_citations(pre_content, valid_refs)

    # Validate
    issues = validate_references(formatted_pre, formatted_refs)

    # Report results
    print(f"✓ {path.name}:")
    print(f"  - {ref_changes} references formatted")
    print(f"  - {cite_changes} citations linked")

    if issues["missing_refs"]:
        print(f"  ⚠️  Missing references for citations: {issues['missing_refs']}")

    if issues["orphaned_refs"]:
        print(f"  ℹ️  Orphaned references (never cited): {issues['orphaned_refs']}")

    # Reconstruct document
    new_content = formatted_pre + "\n\n" + formatted_refs
    if post_content:
        new_content += "\n\n" + post_content

    # Write if not dry-run
    if not dry_run:
        path.write_text(new_content)
        print(f"✅ Updated {path}")
    else:
        print("🔍 Dry-run mode - no changes written")

    return True


if __name__ == "__main__":
    main()
