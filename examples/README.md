# Preprocessing Examples

This directory contains examples for using the declarative preprocessing API.

## Quick Start

```javascript
import * as aq from 'arquero';
import * as ds from '@tangent-to/ds';

const url = "https://raw.githubusercontent.com/tidyverse/ggplot2/e594b49fdd5e4d95bf1031edaf6c7ccfc0cdedb0/data-raw/diamonds.csv";
const diamondsTable = await aq.loadCSV(url);

const { data: diamonds } = ds.ml.preprocessing.preprocess({
  data: diamondsTable,
  parseNumeric: ['depth'],
  validCategories: {
    cut: ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    color: ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
    clarity: ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
  },
  labelEncode: [
    { column: 'cut', categories: ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'] }
  ],
  oneHotEncode: [
    { columns: ['color', 'clarity'], dropFirst: false, keepOriginal: false, prefix: true }
  ]
});

// Result: clean data ready for modeling!
```

## Examples

### Diamonds Dataset

- **`diamonds_clean.js`** - Recommended starting point using the new `preprocess()` API
- **`diamonds_simple.js`** - Basic preprocessing (uses `preprocessCategorical` alias)
- **`diamonds_complete.js`** - Full example with feature extraction
- **`diamonds_list_features.js`** - List features in different formats for copy-pasting
- **`diamonds_your_workflow.js`** - Before/after comparison showing the improvement
- **`diamonds_preprocessing.js`** - Train/test split workflow with `fitPreprocessor`

### Documentation

- **`README_preprocessing.md`** - Complete preprocessing guide
- **`../PREPROCESSING_SOLUTION.md`** - Summary of the solution to common issues

## Key Features

The `preprocess()` function provides a declarative API for:

1. **Parsing numeric columns** - Convert string columns to numbers
2. **Data validation** - Remove rows with invalid category values
3. **Label encoding** - Convert categories to integers (e.g., Fair=0, Good=1, etc.)
4. **One-hot encoding** - Create binary dummy columns for categorical variables

All in a single function call!

## Backward Compatibility

The old name `preprocessCategorical()` is still available as an alias to `preprocess()`, so existing code will continue to work.
