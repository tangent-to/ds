#!/bin/bash
# Convert Jupyter notebooks to Jekyll markdown tutorials

set -e

echo "🔄 Converting notebooks to markdown..."

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "❌ Error: jupyter not found. Install with: pip install jupyter nbconvert"
    exit 1
fi

# Create output directory
mkdir -p docs/_tutorials

# Array of notebooks to convert
declare -A notebooks=(
    ["examples/user-guide/01-ordination.ipynb"]="Ordination Analysis|1"
    ["examples/user-guide/02-statistics.ipynb"]="Statistical Analysis|2"
    ["examples/user-guide/03-clustering.ipynb"]="Clustering|3"
    ["examples/user-guide/04-ml.ipynb"]="Machine Learning|4"
)

# Convert each notebook
for notebook in "${!notebooks[@]}"; do
    IFS='|' read -r title order <<< "${notebooks[$notebook]}"
    filename=$(basename "$notebook" .ipynb)

    echo "📝 Converting $filename..."

    # Convert to markdown
    jupyter nbconvert --to markdown \
        --output-dir=docs/_tutorials \
        "$notebook"

    # Add Jekyll front matter
    output_file="docs/_tutorials/${filename}.md"
    temp_file=$(mktemp)

    cat > "$temp_file" <<EOF
---
layout: default
title: "$title"
parent: Tutorials
nav_order: $order
---

EOF

    # Append converted content
    cat "$output_file" >> "$temp_file"
    mv "$temp_file" "$output_file"

    echo "✅ Created $output_file"
done

echo ""
echo "🎉 Conversion complete!"
echo ""
echo "📂 Tutorial files created in docs/_tutorials/"
echo ""
echo "Next steps:"
echo "  1. Review the converted files"
echo "  2. Test locally: cd docs && bundle exec jekyll serve"
echo "  3. Commit and push to deploy"
