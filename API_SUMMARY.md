# tangent/ds API Patterns

This document summarizes the three API patterns available in the tangent/ds package.

## Overview

The package supports three complementary API patterns:

1. **Array API**: Pure numeric arrays (NumPy/scikit-learn style)
2. **Declarative Table API**: Column-based data with metadata (R/pandas style)  
3. **Recipe API**: Chainable preprocessing workflows with full inspection (NEW)

## 1. Array API

Works with raw numeric matrices (Array<Array<number>>).

### When to Use
- Porting code from NumPy/scikit-learn/MATLAB
- Working with pure numeric data without column names
- Performance is critical (minimal overhead)
- Simple workflows without complex preprocessing

### Example
```javascript
const X = [[1, 2], [3, 4], [5, 6]];
const y = [0, 1, 0];

const scaler = new ds.ml.preprocessing.StandardScaler();
scaler.fit(X);
const XScaled = scaler.transform(X);

const model = new ds.ml.KNNClassifier({ k: 3 });
model.fit(X, y);
const predictions = model.predict(XTest);
```

### Availability
- MVA (PCA, LDA, RDA): Full support
- Stats (GLM): Full support  
- ML (preprocessing, estimators): Full support

## 2. Declarative Table API

Works with structured data (array of objects) and column selectors.

### When to Use
- Working with structured data from CSV/JSON/databases
- Column names are important for interpretation
- Need metadata (encoders, transformers) to flow through
- Handling mixed data types (numeric and categorical)
- Coming from R, pandas, or SQL background

### Example
```javascript
const data = [
  { feature1: 1, feature2: 2, label: 'A' },
  { feature1: 3, feature2: 4, label: 'B' }
];

const split = ds.ml.validation.trainTestSplit(
  { data, X: ['feature1', 'feature2'], y: 'label' },
  { ratio: 0.7, shuffle: true }
);

const scaler = new ds.ml.preprocessing.StandardScaler()
  .fit({ data: split.train.data, columns: ['feature1', 'feature2'] });

const scaled = scaler.transform({
  data: split.train.data,
  columns: ['feature1', 'feature2'],
  encoders: split.train.metadata.encoders
});

const model = new ds.ml.KNNClassifier({ k: 3 })
  .fit({
    data: scaled.data,
    X: ['feature1', 'feature2'],
    y: 'label',
    encoders: scaled.metadata.encoders
  });
```

### Availability
- MVA (PCA, LDA, RDA): Full support
- Stats (GLM): Full support (including formula syntax)
- ML (preprocessing, estimators): Full support

### Key Features
- Column names preserved throughout
- Automatic handling of categorical encoding
- Missing value support (naOmit/omit_missing)
- Metadata flows through transformations

## 3. Recipe API

Chainable preprocessing workflows with full inspection capabilities.

### When to Use
- Building complex preprocessing pipelines
- Need to inspect intermediate transformations
- Applying same preprocessing to multiple datasets
- Preventing data leakage is critical
- Want readable, declarative workflow definitions
- Training models for production deployment

### Example
```javascript
const recipe = ds.ml.recipe({
  data: myData,
  X: ['feature1', 'feature2', 'category'],
  y: 'target'
})
  .parseNumeric(['feature1', 'feature2'])
  .oneHot(['category'], { dropFirst: false })
  .scale(['feature1', 'feature2'], { method: 'standard' })  // or 'minmax'
  .split({ ratio: 0.7, shuffle: true, seed: 42 });

// Execute recipe - fits all transformers on training data
const prepped = recipe.prep();

// Inspect everything
console.log(prepped.steps[0].output);  // After one-hot encoding
console.log(prepped.steps[1].output);  // After scaling
console.log(prepped.transformers.scale.means);  // Scaler parameters
console.log(prepped.transformers.oneHot);  // Encoder info

// Train model
const model = new ds.ml.KNNClassifier({ k: 3 })
  .fit({
    data: prepped.train.data,
    X: prepped.train.X,
    y: prepped.train.y,
    encoders: prepped.train.metadata.encoders
  });

// Apply to new data (uses fitted transformers)
const newPrepped = recipe.bake(newData);
const predictions = model.predict({
  data: newPrepped.data,
  X: newPrepped.X
});
```

### Availability
- ML preprocessing: Full support
- MVA methods: Not yet implemented
- Stats methods: Not yet implemented

### Key Features
- Chainable method calls define preprocessing steps
- prep() executes and returns inspectable results
- bake() applies fitted transformers to new data
- All intermediate outputs available for inspection
- Transformers stored and reusable
- Prevents data leakage (test data never used for fitting)

### Supported Methods

#### parseNumeric(columns)
Convert string columns to numeric values.

```javascript
.parseNumeric(['age', 'price'])
```

#### clean(validCategories)
Remove rows with invalid categorical values.

```javascript
.clean({ category: ['A', 'B', 'C'], status: ['active', 'pending'] })
```

#### oneHot(columns, options)
One-hot encode categorical columns.

```javascript
.oneHot(['category', 'region'], { dropFirst: true, prefix: true })
```

Options:
- `dropFirst`: Drop first category to avoid multicollinearity (default: true)
- `prefix`: Use column name as prefix for dummy variables (default: true)

#### scale(columns, options)
Scale numeric columns.

```javascript
.scale(['age', 'price', 'quantity'], { method: 'standard' })
```

Options:
- `method`: 'standard' (default) or 'minmax'
  - 'standard': Standardize to zero mean and unit variance
  - 'minmax': Scale to [0, 1] range

#### split(options)
Split data into train/test sets.

```javascript
.split({ ratio: 0.7, shuffle: true, seed: 42 })
```

Options:
- `ratio`: Training set ratio (default: 0.7)
- `shuffle`: Shuffle before splitting (default: true)
- `seed`: Random seed for reproducibility (default: null)

### prep() Result Structure

```javascript
{
  train: {
    data: Array<Object>,           // Training data
    X: Array<string>,              // Feature column names
    y: string,                     // Target column name
    metadata: { encoders: {...} }  // Label encoders for target
  },
  test: {
    data: Array<Object>,           // Test data
    X: Array<string>,              // Feature column names
    y: string,                     // Target column name
    metadata: { encoders: {...} }  // Label encoders for target
  },
  transformers: {
    scale: StandardScaler|MinMaxScaler,  // Fitted scaler
    oneHot: Map<string, Object>,         // Fitted encoders
    // ... other transformers
  },
  steps: [
    {
      name: string,                // Step name
      output: Array<Object>,       // Data after this step
      transformer: Object|null     // Fitted transformer (if applicable)
    },
    // ... one per step
  ],
  split: {
    train: Object,                 // Full train split info
    test: Object,                  // Full test split info
    ratio: number                  // Split ratio used
  }
}
```

## API Compatibility

All three patterns are compatible and can be mixed:

```javascript
// Start with Recipe
const prepped = recipe.prep();

// Use Table API for model
const model = new Classifier().fit({
  data: prepped.train.data,
  X: prepped.train.X,
  y: prepped.train.y
});

// Extract to Array API if needed
const Xarray = prepped.train.data.map(row => 
  prepped.train.X.map(col => row[col])
);
```

## Module Support Matrix

| Module | Array API | Table API | Recipe API |
|--------|-----------|-----------|------------|
| MVA (PCA, LDA, RDA) | Yes | Yes | Not yet |
| Stats (GLM) | Yes | Yes (+ Formula) | Not yet |
| ML Preprocessing | Yes | Yes | Yes |
| ML Estimators | Yes | Yes | Via prep() |

## Documentation

See `examples/user-guide/07-api.ipynb` for detailed examples of all three API patterns.

## Maintenance Notes

### Adding New Preprocessing Methods to Recipe

To add a new preprocessing method to Recipe:

1. Add a method to the Recipe class that pushes a step object:
```javascript
myMethod(columns, options = {}) {
  this.steps.push({
    name: 'myMethod',
    type: 'transform',  // or 'encode', 'scale', 'filter'
    columns,
    transformer: null,  // If method needs to be fitted
    fn: (data, transformer) => {
      if (!transformer) {
        // Fit: create and fit transformer
        const fitted = createTransformer(data, columns, options);
        return {
          data: fitted.transform(data),
          transformer: fitted
        };
      } else {
        // Transform: use fitted transformer
        return {
          data: transformer.transform(data),
          transformer
        };
      }
    }
  });
  return this;  // For chaining
}
```

2. The step function receives:
   - `data`: Current data (array of objects)
   - `transformer`: null when fitting (prep), fitted object when transforming (bake)

3. Return an object with:
   - `data`: Transformed data
   - `transformer`: Fitted transformer (for reuse in bake)

4. Update JSDoc with clear examples and parameter descriptions

### Design Principles

1. **Array API**: Minimal abstraction, maximum performance
2. **Table API**: Preserve metadata, handle mixed types
3. **Recipe API**: Inspectable, reusable, prevents leakage

Choose the pattern that matches your data structure and workflow needs.
