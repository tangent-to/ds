---
layout: default
title: Machine Learning
parent: API Reference
nav_order: 2
---

# Machine Learning API
{: .no_toc }

Supervised learning models, validation, and preprocessing utilities.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `ds.ml` module provides:
- **Models**: KNN, MLP (neural networks)
- **Preprocessing**: Scaling, encoding, pipelines
- **Validation**: Train/test split, cross-validation
- **Tuning**: Grid search
- **Metrics**: Accuracy, R², RMSE
- **Recipe API**: Chainable preprocessing workflows

---

## Classification Models

### KNNClassifier

K-Nearest Neighbors classifier.

```javascript
new ds.ml.KNNClassifier(options)
```

#### Options

```javascript
{
  k: number,              // Number of neighbors (default: 5)
  weight: string,         // 'uniform' or 'distance' (default: 'uniform')
  metric: string          // 'euclidean' (default)
}
```

#### Methods

##### `.fit()`

**Array API:**
```javascript
model.fit(X, y)
```

**Table API:**
```javascript
model.fit({
  data: trainData,
  X: ['feature1', 'feature2'],
  y: 'label',
  encoders: metadata.encoders  // Optional
})
```

##### `.predict()`

**Array API:**
```javascript
const predictions = model.predict(XTest)
```

**Table API:**
```javascript
const predictions = model.predict({
  data: testData,
  X: ['feature1', 'feature2'],
  encoders: metadata.encoders  // Optional: decode to strings
})
```

##### `.predictProba()`

Get probability estimates.

```javascript
const probabilities = model.predictProba(XTest)
// [[0.8, 0.2], [0.3, 0.7], ...]
```

#### Example

```javascript
const knn = new ds.ml.KNNClassifier({ k: 5, weight: 'distance' });

knn.fit({
  data: trainData,
  X: ['sepal_length', 'sepal_width'],
  y: 'species'
});

const predictions = knn.predict({
  data: testData,
  X: ['sepal_length', 'sepal_width']
});

const probabilities = knn.predictProba(XTest);
```

---

### MLPClassifier

Multilayer Perceptron (neural network) classifier.

```javascript
new ds.ml.MLPClassifier(options)
```

#### Options

```javascript
{
  hiddenLayers: Array<number>,  // Neurons per hidden layer (default: [100])
  activation: string,            // 'relu', 'tanh', 'sigmoid' (default: 'relu')
  learningRate: number,          // Learning rate (default: 0.001)
  maxIter: number,              // Maximum iterations (default: 200)
  batchSize: number,            // Batch size (default: 'auto')
  solver: string,               // 'adam', 'sgd' (default: 'adam')
  alpha: number,                // L2 regularization (default: 0.0001)
  earlyStop: boolean,           // Early stopping (default: false)
  validationFraction: number    // Validation split for early stopping (default: 0.1)
}
```

#### Example

```javascript
const mlp = new ds.ml.MLPClassifier({
  hiddenLayers: [50, 30],      // 2 hidden layers
  activation: 'relu',
  learningRate: 0.01,
  maxIter: 300
});

mlp.fit({
  data: scaledTrainData,       // MLP requires scaled features!
  X: features,
  y: 'species'
});

const predictions = mlp.predict({ data: scaledTestData, X: features });
```

---

## Regression Models

### MLPRegressor

Multilayer Perceptron for regression.

```javascript
new ds.ml.MLPRegressor(options)
```

Same options as `MLPClassifier`.

#### Example

```javascript
const mlp = new ds.ml.MLPRegressor({
  hiddenLayers: [64, 32],
  activation: 'relu',
  learningRate: 0.001,
  maxIter: 500
});

mlp.fit({
  data: scaledTrain,
  X: ['carat', 'depth', 'table'],
  y: 'price'
});

const predictions = mlp.predict({ data: scaledTest, X: features });
```

---

## Preprocessing

### StandardScaler

Standardize features to zero mean and unit variance.

**Formula:** `z = (x - μ) / σ`

```javascript
new ds.ml.preprocessing.StandardScaler()
```

#### Methods

##### `.fit()`

**Array API:**
```javascript
scaler.fit(X)
```

**Table API:**
```javascript
scaler.fit({
  data: trainData,
  columns: ['feature1', 'feature2']
})
```

##### `.transform()`

**Array API:**
```javascript
const XScaled = scaler.transform(X)
```

**Table API:**
```javascript
const scaled = scaler.transform({
  data: trainData,
  columns: ['feature1', 'feature2'],
  encoders: metadata.encoders  // Pass through
})
// Returns: { data, metadata }
```

##### `.fitTransform()`

Convenience method: fit and transform in one step.

```javascript
const XScaled = scaler.fitTransform(X)
```

#### Example

```javascript
// Fit on training data only!
const scaler = new ds.ml.preprocessing.StandardScaler();
scaler.fit({ data: trainData, columns: numericFeatures });

// Transform both train and test with same parameters
const trainScaled = scaler.transform({ data: trainData, columns: numericFeatures });
const testScaled = scaler.transform({ data: testData, columns: numericFeatures });
```

**Important:** Always fit on training data, then transform both train and test.

---

### MinMaxScaler

Scale features to [0, 1] range.

**Formula:** `x_scaled = (x - min) / (max - min)`

```javascript
new ds.ml.preprocessing.MinMaxScaler()
```

Same API as `StandardScaler`.

```javascript
const scaler = new ds.ml.preprocessing.MinMaxScaler();
scaler.fit({ data: trainData, columns: features });
const scaled = scaler.transform({ data: testData, columns: features });
```

---

### OneHotEncoder

Convert categorical variables to binary columns.

```javascript
new ds.ml.preprocessing.OneHotEncoder()
```

#### Methods

##### `.fitTransform()`

```javascript
const encoded = encoder.fitTransform({
  data: myData,
  columns: ['color', 'cut']
})
```

**Returns:** Array of objects with binary columns
```javascript
[
  { color_red: 1, color_blue: 0, cut_fair: 0, cut_good: 1 },
  ...
]
```

#### Example

```javascript
const encoder = new ds.core.table.OneHotEncoder();

const encoded = encoder.fitTransform({
  data: penguins,
  columns: ['species', 'island'],
  dropFirst: true   // Optional: keep D-1 columns per feature
});

// Merge with original data
const withEncoded = penguins.map((row, i) => ({
  ...row,
  ...encoded[i]
}));
```
`dropFirst` defaults to `false`; set it to `true` to drop the first dummy column for each encoded feature when using the declarative API.

---

### LabelEncoder

Encode categorical labels to integers.

```javascript
new ds.core.table.LabelEncoder()
```

#### Methods

```javascript
encoder.fit(labels)                    // Fit on unique labels
const encoded = encoder.transform(labels)  // Transform to integers
const decoded = encoder.inverse(encoded)   // Transform back to labels
```

#### Properties

```javascript
encoder.classes_    // Array of unique classes
encoder.classIndex  // Map of class -> index
```

#### Example

```javascript
const encoder = new ds.core.table.LabelEncoder();
encoder.fit(['cat', 'dog', 'cat', 'bird']);

const encoded = encoder.transform(['cat', 'bird']);
// [0, 2]

const decoded = encoder.inverse([0, 2]);
// ['cat', 'bird']
```

---

## Recipe API

Chainable preprocessing workflows.

```javascript
const recipe = ds.ml.recipe(config)
  .parseNumeric(columns)
  .oneHot(columns, options)
  .scale(columns, options)
  .split(options)
```

### Creating a Recipe

```javascript
const recipe = ds.ml.recipe({
  data: myData,
  X: ['feature1', 'feature2', 'category'],
  y: 'target'
})
```

### Methods

#### `.parseNumeric(columns)`

Convert string columns to numbers.

```javascript
.parseNumeric(['age', 'price'])
```

#### `.clean(validCategories)`

Remove rows with invalid categories.

```javascript
.clean({
  category: ['A', 'B', 'C'],
  status: ['active', 'pending']
})
```

#### `.oneHot(columns, options)`

One-hot encode categorical columns.

```javascript
.oneHot(['category', 'region'], {
  dropFirst: true,   // Avoid multicollinearity
  prefix: true       // Use column name as prefix
})
```

#### `.scale(columns, options)`

Scale numeric columns.

```javascript
.scale(['age', 'price'], {
  method: 'standard'  // or 'minmax'
})
```

#### `.split(options)`

Split into train/test sets.

```javascript
.split({
  ratio: 0.7,
  shuffle: true,
  seed: 42
})
```

### Executing the Recipe

#### `.prep()`

Execute recipe and fit all transformers on training data.

```javascript
const prepped = recipe.prep()
```

**Returns:**
```javascript
{
  train: {
    data: Array<Object>,
    X: Array<string>,
    y: string,
    metadata: { encoders: {...} }
  },
  test: { /* same structure */ },
  transformers: {
    scale: StandardScaler,
    oneHot: Map<string, Object>
  },
  steps: [
    { name: 'oneHot', output: [...], transformer: {...} },
    { name: 'scale', output: [...], transformer: {...} }
  ]
}
```

#### `.bake(newData)`

Apply fitted transformers to new data.

```javascript
const newPrepped = recipe.bake(newData)
```

### Complete Example

```javascript
const recipe = ds.ml.recipe({
  data: diamondsData,
  X: ['carat', 'depth', 'table', 'cut', 'color'],
  y: 'price'
})
  .parseNumeric(['carat', 'depth', 'table', 'price'])
  .oneHot(['cut', 'color'], { dropFirst: false })
  .scale(['carat', 'depth', 'table'], { method: 'standard' })
  .split({ ratio: 0.8, shuffle: true, seed: 42 });

// Execute and fit transformers
const prepped = recipe.prep();

// Inspect intermediate steps
console.log(prepped.steps[0].output);  // After one-hot
console.log(prepped.steps[1].output);  // After scaling

// Train model
const model = new ds.ml.MLPRegressor({ hiddenLayers: [64, 32] });
model.fit({
  data: prepped.train.data,
  X: prepped.train.X,
  y: prepped.train.y
});

// Apply to new data (uses fitted transformers)
const newPrepped = recipe.bake(newDiamonds);
const predictions = model.predict({
  data: newPrepped.data,
  X: newPrepped.X
});
```

---

## Validation

### trainTestSplit

Split data into training and testing sets.

```javascript
ds.ml.validation.trainTestSplit(config, options)
```

**Array API:**
```javascript
const { XTrain, XTest, yTrain, yTest } = ds.ml.validation.trainTestSplit(
  X, y,
  { ratio: 0.7, shuffle: true, seed: 42 }
)
```

**Table API:**
```javascript
const split = ds.ml.validation.trainTestSplit(
  { data: myData, X: features, y: 'target' },
  { ratio: 0.7, shuffle: true, seed: 42 }
)

// Returns:
{
  train: {
    data: Array,
    X: Array<string>,
    y: string,
    metadata: { encoders: {...} }
  },
  test: { /* same structure */ }
}
```

---

### crossValidate

Perform k-fold cross-validation.

```javascript
ds.ml.validation.crossValidate(
  modelFn,
  scoreFn,
  data,
  options
)
```

**Parameters:**
- `modelFn`: `(XTrain, yTrain) => fittedModel`
- `scoreFn`: `(model, XTest, yTest) => score`
- `data`: Data configuration object
- `options`: `{ k: 5, shuffle: true, seed: 42 }`

**Returns:** Object with scores array

```javascript
{
  scores: [0.82, 0.85, 0.79, 0.83, 0.81],
  mean: () => 0.82,
  std: () => 0.02
}
```

#### Example

```javascript
const cv = ds.ml.validation.crossValidate(
  (Xtr, ytr) => new ds.ml.KNNClassifier({ k: 5 }).fit(Xtr, ytr),
  (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)),
  { data: myData, X: features, y: 'species' },
  { k: 5, shuffle: true }
);

console.log(`Mean accuracy: ${cv.scores.mean()}`);
console.log(`Std: ${cv.scores.std()}`);
```

---

## Hyperparameter Tuning

### GridSearchCV

Exhaustive search over parameter grid.

```javascript
ds.ml.tuning.GridSearchCV(
  modelFn,
  scoreFn,
  data,
  yData,
  paramGrid,
  cvOptions
)
```

**Parameters:**
- `modelFn`: `(Xtr, ytr, params) => fittedModel`
- `scoreFn`: `(model, Xte, yte) => score`
- `data`: Data configuration
- `yData`: null (when using table API)
- `paramGrid`: Object with parameter arrays
- `cvOptions`: Cross-validation options

**Returns:**
```javascript
{
  bestParams: Object,
  bestScore: number,
  cvResults: Array
}
```

#### Example

```javascript
const paramGrid = {
  k: [3, 5, 7, 11],
  weight: ['uniform', 'distance']
};

const grid = ds.ml.tuning.GridSearchCV(
  (Xtr, ytr, params) => new ds.ml.KNNClassifier(params).fit(Xtr, ytr),
  (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)),
  { data: trainData, X: features, y: 'species' },
  null,
  paramGrid,
  { k: 5, shuffle: true }
);

console.log('Best params:', grid.bestParams);
console.log('Best score:', grid.bestScore);

// Train final model with best params
const finalModel = new ds.ml.KNNClassifier(grid.bestParams);
finalModel.fit({ data: trainData, X: features, y: 'species' });
```

---

## Metrics

### Classification Metrics

#### accuracy

```javascript
const acc = ds.ml.metrics.accuracy(yTrue, yPred)
// Fraction of correct predictions
```

#### confusionMatrix

```javascript
const cm = ds.ml.metrics.confusionMatrix(yTrue, yPred)
// 2D array: rows=true, cols=pred
```

### Regression Metrics

#### r2Score

R² (coefficient of determination)

```javascript
const r2 = ds.ml.metrics.r2Score(yTrue, yPred)
// 1.0 = perfect, 0.0 = baseline, <0 = worse than baseline
```

#### rmse

Root Mean Squared Error

```javascript
const error = ds.ml.metrics.rmse(yTrue, yPred)
// Lower is better
```

#### mae

Mean Absolute Error

```javascript
const error = ds.ml.metrics.mae(yTrue, yPred)
```

---

## Common Workflows

### Classification Pipeline

```javascript
// 1. Split data
const split = ds.ml.validation.trainTestSplit(
  { data, X: features, y: 'label' },
  { ratio: 0.7, shuffle: true, seed: 42 }
);

// 2. Scale features
const scaler = new ds.ml.preprocessing.StandardScaler();
scaler.fit({ data: split.train.data, columns: features });

const trainScaled = scaler.transform({ data: split.train.data, columns: features });
const testScaled = scaler.transform({ data: split.test.data, columns: features });

// 3. Grid search
const grid = ds.ml.tuning.GridSearchCV(
  (Xtr, ytr, params) => new ds.ml.KNNClassifier(params).fit(Xtr, ytr),
  (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)),
  { data: trainScaled.data, X: features, y: 'label' },
  null,
  { k: [3, 5, 7], weight: ['uniform', 'distance'] },
  { k: 5 }
);

// 4. Final model
const model = new ds.ml.KNNClassifier(grid.bestParams);
model.fit({ data: trainScaled.data, X: features, y: 'label' });

// 5. Evaluate
const predictions = model.predict({ data: testScaled.data, X: features });
const accuracy = ds.ml.metrics.accuracy(
  split.test.data.map(d => d.label),
  predictions
);
```

### Regression Pipeline with Recipe

```javascript
// 1. Define recipe
const recipe = ds.ml.recipe({ data, X: allFeatures, y: 'price' })
  .parseNumeric(numericFeatures)
  .oneHot(categoricalFeatures)
  .scale(numericFeatures)
  .split({ ratio: 0.8, shuffle: true, seed: 42 });

// 2. Prep (fit transformers)
const prepped = recipe.prep();

// 3. Train model
const model = new ds.ml.MLPRegressor({
  hiddenLayers: [64, 32],
  learningRate: 0.001
});

model.fit({
  data: prepped.train.data,
  X: prepped.train.X,
  y: prepped.train.y
});

// 4. Evaluate
const predictions = model.predict({
  data: prepped.test.data,
  X: prepped.test.X
});

const yTrue = prepped.test.data.map(d => d[prepped.test.y]);
const r2 = ds.ml.metrics.r2Score(yTrue, predictions);
const rmse = ds.ml.metrics.rmse(yTrue, predictions);
```

---

## See Also

- [Tutorial: Machine Learning](../tutorials/04-ml)
- [Examples](../examples#machine-learning)
- [Recipe API Guide](https://github.com/tangent-to/ds/blob/main/API_SUMMARY.md)
