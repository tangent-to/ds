# Scripts Directory

Automation scripts for documentation and build processes.

## Available Scripts

### 📓 convert-notebooks.py

Converts Jupyter notebooks to Jekyll markdown tutorials.

**Usage:**
```bash
python3 scripts/convert-notebooks.py

# Or via npm
npm run docs:convert
```

**Requirements:**
- Python 3.7+
- `pip install jupyter nbconvert`

**What it does:**
1. Converts `.ipynb` files from `examples/user-guide/` to markdown
2. Adds Jekyll front matter (title, parent, nav_order)
3. Outputs to `docs/_tutorials/`

---

### 📓 convert-notebooks.sh

Bash version of the notebook converter.

**Usage:**
```bash
bash scripts/convert-notebooks.sh
```

**Requirements:**
- Bash shell
- `pip install jupyter nbconvert`

---

### 📝 generate-docs.js

*Existing script for API documentation generation.*

**Usage:**
```bash
npm run docs:generate
```

---

## Quick Commands

```bash
# Convert notebooks to tutorials
npm run docs:convert

# Generate API docs
npm run docs:generate

# Test Jekyll site locally
cd docs && bundle exec jekyll serve
```

## Adding New Conversion Scripts

To add a new notebook for conversion:

1. Edit `convert-notebooks.py`
2. Add to the `NOTEBOOKS` dictionary:

```python
NOTEBOOKS = {
    # Existing notebooks...
    "path/to/new-notebook.ipynb": {
        "title": "New Tutorial Title",
        "nav_order": 5
    }
}
```

3. Run the script

## GitHub Actions

The conversion script runs automatically in CI/CD via `.github/workflows/jekyll.yml`

## Troubleshooting

**Error: jupyter not found**
```bash
pip install jupyter nbconvert
```

**Error: Permission denied**
```bash
chmod +x scripts/convert-notebooks.sh
chmod +x scripts/convert-notebooks.py
```

**Notebooks not converting**
- Check notebook paths are correct
- Ensure notebooks are valid JSON
- Check Python/Jupyter installation
