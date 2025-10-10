#!/usr/bin/env python3
"""
Batch conversion script: Polars → Ibis+DuckDB

This script performs systematic find-replace operations to convert
Polars DataFrame operations to Ibis expressions.
"""

import re
from pathlib import Path

# Conversion mappings
CONVERSIONS = [
    # Basic operations
    (r'\.collect\(\)', '.execute()'),
    (r'\.sort\(', '.order_by('),
    (r'pl\.len\(\)', '_.count()'),
    (r'pl\.col\([\'"](\w+)[\'"]\)', r'_.\1'),
    (r'pl\.lit\((.*?)\)', r'ibis.literal(\1)'),

    # Aggregations
    (r'\.agg\(\[([^\]]+)\]\)', r'.agg(\1)'),  # Remove list wrapping
    (r'pl\.col\([\'"](\w+)[\'"]\)\.sum\(\)\.alias\([\'"](\w+)[\'"]\)', r'\2=_.\1.sum()'),
    (r'pl\.col\([\'"](\w+)[\'"]\)\.mean\(\)\.alias\([\'"](\w+)[\'"]\)', r'\2=_.\1.mean()'),
    (r'pl\.col\([\'"](\w+)[\'"]\)\.n_unique\(\)', r'_.\1.nunique()'),

    # Schema operations
    (r'\.collect_schema\(\)', '.schema()'),
    (r'isinstance\(dtype, \(pl\.Date, pl\.Datetime\)\)', 'dtype.is_temporal()'),

    # Filtering
    (r'\.filter\(pl\.col\([\'"](\w+)[\'"]\)\s*([<>=!]+)\s*([^\)]+)\)', r'.filter(_.\1 \2 \3)'),

    # Selections
    (r'\.select\(\[([^\]]+)\]\)', r'.select(\1)'),
]

def convert_file(input_path: Path, output_path: Path = None):
    """Convert a Polars notebook to Ibis."""
    if output_path is None:
        output_path = input_path.with_suffix('.ibis.md')

    content = input_path.read_text()

    # Apply conversions
    for pattern, replacement in CONVERSIONS:
        content = re.sub(pattern, replacement, content)

    output_path.write_text(content)
    print(f"✅ Converted: {input_path} → {output_path}")
    print(f"   Applied {len(CONVERSIONS)} transformation rules")

if __name__ == '__main__':
    notebook = Path('notebooks/05_cloudzero_piedpiper_eda.md')
    convert_file(notebook)
