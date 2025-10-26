"""Convert doctest-style examples to markdown code blocks in Python docstrings."""

import re
import sys
from pathlib import Path


def convert_examples_in_file(file_path: Path) -> tuple[str, int]:
    """
    Convert doctest-style examples to markdown code blocks.

    Returns:
        Tuple of (converted content, number of conversions)
    """
    content = file_path.read_text()

    # Pattern to match Example sections with doctest format
    # Matches: Example:\n            >>> code\n            >>> code\n
    pattern = r"(Example:)\n((?:[ ]{12,}>>>.*\n?)+)"

    conversions = 0

    def convert_match(match):
        nonlocal conversions
        conversions += 1

        example_header = match.group(1)
        doctest_lines = match.group(2)

        # Extract indentation from first line
        indent_match = re.match(r"([ ]+)>>>", doctest_lines)
        if not indent_match:
            return match.group(0)  # No change if pattern doesn't match

        indent = indent_match.group(1)

        # Remove >>> and leading spaces, convert to regular Python
        code_lines = []
        for line in doctest_lines.split("\n"):
            if not line.strip():
                continue
            # Remove leading whitespace and >>>
            stripped = line.strip()
            if stripped.startswith(">>>"):
                stripped = stripped[3:].lstrip()
            code_lines.append(stripped)

        # Build markdown code block
        result = f"{example_header}\n{indent}``` python\n"
        for line in code_lines:
            result += f"{indent}{line}\n"
        result += f"{indent}```"

        return result

    converted_content = re.sub(pattern, convert_match, content)

    return converted_content, conversions


def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_doctest_to_markdown.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Converting examples in: {file_path}")
    converted_content, count = convert_examples_in_file(file_path)

    if count == 0:
        print("No doctest-style examples found.")
        sys.exit(0)

    # Write back
    file_path.write_text(converted_content)
    print(f"✓ Converted {count} Example sections to markdown code blocks")


if __name__ == "__main__":
    main()
