#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Jekyll markdown tutorials.
Usage: python scripts/convert-notebooks.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
NOTEBOOKS = {
    "examples/user-guide/01-ordination.ipynb": {
        "title": "Ordination Analysis",
        "nav_order": 1
    },
    "examples/user-guide/02-statistics.ipynb": {
        "title": "Statistical Analysis",
        "nav_order": 2
    },
    "examples/user-guide/03-clustering.ipynb": {
        "title": "Clustering",
        "nav_order": 3
    },
    "examples/user-guide/04-ml.ipynb": {
        "title": "Machine Learning",
        "nav_order": 4
    }
}

OUTPUT_DIR = Path("docs/_tutorials")
REPO_ROOT = Path(__file__).parent.parent


def check_dependencies():
    """Check if jupyter nbconvert is installed."""
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--version"],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: jupyter nbconvert not found")
        print("Install with: pip install jupyter nbconvert")
        return False


def convert_notebook(notebook_path, config):
    """Convert a single notebook to markdown with Jekyll front matter."""
    notebook_path = REPO_ROOT / notebook_path
    filename = notebook_path.stem

    print(f"📝 Converting {filename}...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to markdown
    try:
        subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "markdown",
                "--output-dir", str(OUTPUT_DIR),
                str(notebook_path)
            ],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {filename}: {e}")
        return False

    # Add Jekyll front matter
    output_file = OUTPUT_DIR / f"{filename}.md"

    if not output_file.exists():
        print(f"❌ Warning: {output_file} not created")
        return False

    # Read converted content
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create front matter
    front_matter = f"""---
layout: default
title: "{config['title']}"
parent: Tutorials
nav_order: {config['nav_order']}
---

"""

    # Write with front matter
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(front_matter + content)

    print(f"✅ Created {output_file}")
    return True


def main():
    """Main conversion function."""
    os.chdir(REPO_ROOT)

    print("🔄 Converting notebooks to Jekyll markdown...\n")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Convert each notebook
    success_count = 0
    for notebook, config in NOTEBOOKS.items():
        if convert_notebook(notebook, config):
            success_count += 1

    print(f"\n🎉 Conversion complete! {success_count}/{len(NOTEBOOKS)} notebooks converted")
    print(f"\n📂 Tutorial files created in {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Review the converted files")
    print("  2. Test locally: cd docs && bundle exec jekyll serve")
    print("  3. Commit and push to deploy")


if __name__ == "__main__":
    main()
