---
layout: default
title: Core Utilities
parent: API Reference
nav_order: 5
permalink: /api/core
---

# Core Utilities API
{: .no_toc }

Math, linear algebra, data manipulation, formula parsing, and optimization.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `ds.core` module provides foundational building blocks:

- **math** - Descriptive statistics and mathematical utilities
- **linalg** - Linear algebra operations (SVD, eigendecomposition, matrix operations)
- **table** - Data normalization, column extraction, and categorical encoding
- **formula** - R-style formula parsing (`'y ~ x1 + x2'`)
- **optimize** - Gradient-based optimizers (Adam, SGD, RMSProp)
- **persistence** - Model serialization and deserialization
- **spatial** - Spatial data structures (KD-tree)

---

## Math (`ds.core.math`)

### Descriptive Statistics

```javascript
ds.core.math.mean([1, 2, 3, 4, 5])        // 3
ds.core.math.median([1, 2, 3, 4, 5])      // 3
ds.core.math.variance([1, 2, 3, 4, 5])    // 2.5 (sample)
ds.core.math.stddev([1, 2, 3, 4, 5])      // 1.58...
ds.core.math.sum([1, 2, 3, 4, 5])         // 15
ds.core.math.min([1, 2, 3, 4, 5])         // 1
ds.core.math.max([1, 2, 3, 4, 5])         // 5
```

All functions accept an optional `options` parameter with `naOmit` (boolean) to skip `NaN` values.

Aliases: `ds.core.math.std` and `ds.core.math.sd` are aliases for `stddev`.

---

### quantile

Compute quantile(s) at probability `p`.

```javascript
ds.core.math.quantile(data, p, options)
```

**Parameters:**
- `data` (Array&lt;number&gt;): Numeric array
- `p` (number or Array&lt;number&gt;): Probability between 0 and 1
- `options.naOmit` (boolean): Skip NaN values

**Returns:** number or Array&lt;number&gt;

```javascript
// Single quantile
const q25 = ds.core.math.quantile(data, 0.25);

// Multiple quantiles
const [q25, q50, q75] = ds.core.math.quantile(data, [0.25, 0.5, 0.75]);
```

---

### summaryQuantiles

Compute a summary of quantiles at specified probabilities.

```javascript
ds.core.math.summaryQuantiles(arr, probs, options)
```

**Parameters:**
- `arr` (Array&lt;number&gt;): Data
- `probs` (Array&lt;number&gt;): Probabilities (default: `[0, 0.25, 0.5, 0.75, 1]`)

---

### range

Generate a sequence of evenly spaced numbers.

```javascript
ds.core.math.range(start, stop, step)
```

```javascript
ds.core.math.range(0, 10, 2)   // [0, 2, 4, 6, 8, 10]
ds.core.math.range(1, 5)       // [1, 2, 3, 4, 5]
ds.core.math.range(1, 2, 0.5)  // [1, 1.5, 2]
```

---

### Guards

Utility functions for validating numeric values:

```javascript
ds.core.math.approxEqual(a, b, tolerance)  // Floating-point comparison
ds.core.math.guardFinite(value, name)      // Throws if not finite
ds.core.math.guardPositive(value, name)    // Throws if <= 0
ds.core.math.guardProbability(value, name) // Throws if outside [0, 1]
```

### Constants

```javascript
ds.core.math.EPSILON  // 1e-10
ds.core.math.PI       // Math.PI
ds.core.math.E        // Math.E
```

---

## Linear Algebra (`ds.core.linalg`)

Matrix operations wrapping the `ml-matrix` library.

### Matrix Operations

```javascript
ds.core.linalg.transpose(A)       // Matrix transpose
ds.core.linalg.mmul(A, B)         // Matrix multiplication A * B
ds.core.linalg.inverse(A)         // Matrix inverse (square)
ds.core.linalg.pseudoInverse(A)   // Pseudo-inverse via SVD
ds.core.linalg.toMatrix(data)     // Convert array to Matrix object
```

All functions accept `Array<Array<number>>` or `Matrix` objects.

---

### Decompositions

#### svd

Singular Value Decomposition.

```javascript
const { U, s, V } = ds.core.linalg.svd(data)
// data ~ U * diag(s) * V'
```

#### eig

Eigenvalue decomposition of a square matrix.

```javascript
const { values, vectors } = ds.core.linalg.eig(data)
```

---

### covarianceMatrix

Compute the covariance matrix (rows = observations, columns = variables).

```javascript
const cov = ds.core.linalg.covarianceMatrix(data, center)
```

**Parameters:**
- `data` (Array&lt;Array&lt;number&gt;&gt;): Data matrix
- `center` (boolean): Subtract means first (default: `true`)

---

### solveLeastSquares

Solve the least squares problem: minimize ||Ax - b||^2.

```javascript
const x = ds.core.linalg.solveLeastSquares(A, b)
```

---

### Matrix Class

The `Matrix` class from `ml-matrix` is re-exported for direct use:

```javascript
const { Matrix } = ds.core.linalg;
const m = new Matrix([[1, 2], [3, 4]]);
```

---

## Table Utilities (`ds.core.table`)

Unified interface for working with table-like data (arrays of objects or Arquero tables).

### Data Normalization

```javascript
ds.core.table.normalize(data)              // Normalize to Array<Object>
ds.core.table.toMatrix(data, columns)      // Extract numeric matrix
ds.core.table.toVector(data, column)       // Extract single column array
ds.core.table.toColumns(data, columns)     // Extract column key-value pairs
ds.core.table.getColumns(data)             // Get all column names
ds.core.table.filter(data, predicate)      // Filter rows
ds.core.table.select(data, columns)        // Select columns
```

---

### prepareX

Prepare a feature matrix from table data with optional categorical encoding.

```javascript
const result = ds.core.table.prepareX({
  columns: ['col1', 'col2'],
  data: myData,
  omit_missing: true,
  encode: 'onehot',
  encoders: null
})
// Returns: { X, columns, n, rows, encoders }
```

### prepareXY

Prepare feature matrix `X` and response vector `y` together.

```javascript
const result = ds.core.table.prepareXY({
  X: ['feature1', 'feature2'],
  y: 'target',
  data: myData
})
// Returns: { X, y, columnsX, n, rows, encoders }
```

---

### applyColumns

Reattach transformed values back onto row objects.

```javascript
const updated = ds.core.table.applyColumns(rows, columns, matrix, { copy: true })
```

---

### oneHotEncodeTable

One-hot encode categorical columns in a table.

```javascript
const { data, dummyInfo } = ds.core.table.oneHotEncodeTable({
  data: myData,
  columns: ['species', 'island'],
  dropFirst: true,
  keepOriginal: false,
  prefix: true
})
```

---

### LabelEncoder

Encode categorical labels to integers.

```javascript
const encoder = new ds.core.table.LabelEncoder();
encoder.fit(['cat', 'dog', 'cat', 'bird']);

const encoded = encoder.transform(['cat', 'bird']);  // [0, 2]
const decoded = encoder.inverseTransform([0, 2]);    // ['cat', 'bird']

encoder.classes_     // ['bird', 'cat', 'dog']
encoder.classIndex   // Map { 'bird' => 0, 'cat' => 1, 'dog' => 2 }
```

**Methods:** `fit()`, `transform()`, `fitTransform()`, `inverseTransform()`, `toJSON()`, `LabelEncoder.fromJSON()`

---

### OneHotEncoder

One-hot encode categorical values.

```javascript
const encoder = new ds.core.table.OneHotEncoder({ handleUnknown: 'ignore' });

// Array API
encoder.fit(['red', 'green', 'blue']);
const vectors = encoder.transform(['red', 'green']);
// [[1, 0, 0], [0, 1, 0]]

// Declarative API
const encoded = encoder.fitTransform({
  data: penguins,
  columns: ['species', 'island'],
  dropFirst: true
});
```

**Methods:** `fit()`, `transform()`, `fitTransform()`, `getFeatureNames(prefix)`, `toJSON()`, `OneHotEncoder.fromJSON()`

---

## Formula (`ds.core`)

R-style formula parsing for model specifications.

### parseFormula

Parse a formula string into a structured object.

```javascript
const parsed = ds.core.parseFormula('y ~ x1 + x2 + (1 | group)', data)
```

**Returns:**
```javascript
{
  response: { variable: 'y', transform: null },
  fixed: [ /* term objects */ ],
  random: { /* random effects structure */ },
  original: 'y ~ x1 + x2 + (1 | group)'
}
```

**Supported syntax:**
- `y ~ x1 + x2` - Multiple predictors
- `y ~ x1 * x2` - Interactions (expands to `x1 + x2 + x1:x2`)
- `y ~ log(x1) + sqrt(x2)` - Transformations
- `y ~ I(x^2)` - Inline expressions
- `y ~ poly(x, 3)` - Polynomials
- `y ~ x1 + (1 | group)` - Random intercepts
- `y ~ x1 + (1 + time | subject)` - Random slopes

### applyFormula

Apply a formula to data, extracting design matrix and response.

```javascript
const result = ds.core.applyFormula('y ~ x1 * x2', data)
```

**Returns:** `{ X, y, groups, randomEffects, columnNames, parsed }`

---

## Optimization (`ds.core.optimize`)

Gradient-based optimizers for custom loss functions.

### Available Optimizers

| Class | Description |
|-------|-------------|
| `GradientDescent` | Vanilla gradient descent with optional line search |
| `MomentumOptimizer` | SGD with momentum |
| `RMSProp` | Adaptive learning rates per parameter |
| `AdamOptimizer` | Adaptive moment estimation |

### Usage

```javascript
const optimizer = ds.core.optimize.createOptimizer('adam', {
  learningRate: 0.01,
  maxIter: 1000
});

const result = optimizer.minimize(
  (params) => ({ loss: computeLoss(params), gradient: computeGrad(params) }),
  initialParams
);
// Returns: { x: optimizedParams, history: [...] }
```

### Factory

```javascript
ds.core.optimize.createOptimizer(name, options)
```

**Names:** `'gd'`, `'sgd'`, `'momentum'`, `'rmsprop'`, `'adam'`

---

## Persistence (`ds.core.persistence`)

Save and load fitted models as JSON.

### saveModel / loadModel

```javascript
const json = ds.core.persistence.saveModel(fittedModel);
// Store json string...

const restored = ds.core.persistence.loadModel(json);
```

Models are saved with metadata:
- `__tangentds__: true` - Marker flag
- `version` - Library version
- `timestamp` - ISO timestamp
- `modelType` - Auto-detected type (e.g., `'pca'`, `'kmeans'`, `'linear_model'`)

### makeSaveable

Create a wrapper with a `.save()` method:

```javascript
const saveable = ds.core.persistence.makeSaveable(model);
const json = saveable.save();
```

All estimator classes also support `.toJSON()` / `.fromJSON()` for persistence.

---

## Spatial (`ds.core.spatial`)

Spatial data structures for efficient nearest-neighbor queries.

### KDTree

```javascript
const tree = ds.core.spatial.buildKDTree(points);
// Or: const tree = new ds.core.spatial.KDTree(points);
```

---

## See Also

- [Statistics API](statistics) - Statistical models built on core utilities
- [Machine Learning API](machine-learning) - ML models using preprocessing and optimization
