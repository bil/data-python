#!/usr/bin/python3
"""
Re-run notebooks, convert to markdown, strip.
"""

import subprocess
from pathlib import Path

# --- Configuration ---
SRC_DIR = Path("demo/ipynb")
DST_DIR = Path("demo/md")


def convert_notebooks():
    """Recursively find and convert notebooks."""
    if not SRC_DIR.exists():
        print(f"Error: Source directory {SRC_DIR} does not exist.")
        return

    # Ensure destination root exists
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Walk through notebooks
    for ipynb_path in SRC_DIR.rglob("*.ipynb"):
        # Skip hidden files
        if ipynb_path.name.startswith("."):
            continue

        # Determine relative path to mirror structure
        rel_path = ipynb_path.relative_to(SRC_DIR)

        # Determine destination subdirectory
        output_subdir = DST_DIR / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {rel_path} -> {DST_DIR / rel_path.with_suffix('.md')}")

        # Convert to Markdown
        cmd_convert = [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--execute",
            "--output-dir",
            str(output_subdir),
            str(ipynb_path),
        ]

        try:
            subprocess.run(cmd_convert, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {ipynb_path}:")
            print(e.stderr)
            continue

        # Strip outputs from source notebook
        cmd_strip = ["nbstripout", str(ipynb_path)]
        try:
            subprocess.run(cmd_strip, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: nbstripout failed for {ipynb_path}")
            raise e


if __name__ == "__main__":
    convert_notebooks()
