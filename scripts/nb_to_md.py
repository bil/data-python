#!/usr/bin/python3
"""
Re-run notebooks, convert to markdown using Quarto, strip outputs.
Processes both 'docs' and 'demo' directories in-place.
"""

import subprocess
from pathlib import Path

# --- Configuration ---
# List the directories you want to scan
TARGET_DIRS = [Path("docs"), Path("demo")]


def convert_notebooks():
    """Recursively find and convert notebooks in target directories."""

    for target_dir in TARGET_DIRS:
        if not target_dir.exists():
            print(f"Skipping {target_dir}: Directory does not exist.")
            continue

        print(f"\nScanning directory: {target_dir}/")

        # Walk through notebooks
        for ipynb_path in target_dir.rglob("*.ipynb"):
            # Skip hidden files or checkpoint files
            if ipynb_path.name.startswith(".") or "ipynb_checkpoints" in str(
                ipynb_path
            ):
                continue

            print(f"Converting: {ipynb_path}")

            # Quarto
            cmd_convert = [
                "quarto",
                "render",
                str(ipynb_path),
                "--to",
                "gfm",
                "--execute",
            ]

            try:
                subprocess.run(cmd_convert, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {ipynb_path}:")
                print(e.stderr)
                continue

            # --- STRIP OUTPUTS ---
            cmd_strip = ["nbstripout", str(ipynb_path)]
            try:
                subprocess.run(cmd_strip, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: nbstripout failed for {ipynb_path}")
                raise e


if __name__ == "__main__":
    convert_notebooks()
