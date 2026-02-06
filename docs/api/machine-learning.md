---
layout: default
title: Machine Learning
parent: API Reference
nav_order: 2
permalink: /api/machine-learning
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
- **Models**: KNN, Decision Trees, Random Forests, GAMs, MLP (neural networks)
- **Clustering**: K-Means, DBSCAN, HCA
- **Preprocessing**: Scaling, encoding, pipelines
- **Validation**: Train/test split, cross-validation
- **Tuning**: Grid search
- **Metrics**: Accuracy, R-squared, RMSE, F1
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
```

---

### KNNRegressor

K-Nearest Neighbors regressor.

```javascript
new ds.ml.KNNRegressor(options)
```

Same API as `KNNClassifier`. Returns continuous predictions instead of classes.

---

### DecisionTreeClassifier

Decision tree for classification.

```javascript
new ds.ml.DecisionTreeClassifier(options)
```

#### Options

```javascript
{
  maxDepth: number,       // Maximum tree depth (default: Infinity)
  minSamplesSplit: number // Minimum samples to split (default: 2)
}
```

#### Methods

- `.fit(X, y)` or `.fit({ data, X, y })`
- `.predict(X)` or `.predict({ data, X })`
- `.predictProba(X)` - Probability estimates

---

### DecisionTreeRegressor

Decision tree for regression.

```javascript
new ds.ml.DecisionTreeRegressor(options)
```

Same options as `DecisionTreeClassifier`.

---

### RandomForestClassifier

Random forest ensemble for classification.

```javascript
new ds.ml.RandomForestClassifier(options)
```

#### Options

```javascript
{
  nEstimators: number,    // Number of trees (default: 100)
  maxDepth: number,       // Max depth per tree (default: Infinity)
  maxFeatures: string,    // Feature subset: 'sqrt', 'log2', or number
  seed: number            // Random seed
}
```

#### Methods

- `.fit(X, y)` or `.fit({ data, X, y })`
- `.predict(X)` or `.predict({ data, X })`
- `.predictProba(X)` - Probability estimates
- `.featureImportances()` - Feature importance scores

---

### RandomForestRegressor

Random forest ensemble for regression.

```javascript
new ds.ml.RandomForestRegressor(options)
```

Same API as `RandomForestClassifier`.

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
  activation: string,           // 'relu', 'tanh', 'sigmoid' (default: 'relu')
  learningRate: number,         // Learning rate (default: 0.001)
  maxIter: number,              // Maximum iterations (default: 200)
  batchSize: number,            // Batch size (default: 'auto')
  solver: string,               // 'adam', 'sgd' (default: 'adam')
  alpha: number,                // L2 regularization (default: 0.0001)
  earlyStop: boolean,           // Early stopping (default: false)
  validationFraction: number    // Validation split (default: 0.1)
}
```

#### Example

```javascript
const mlp = new ds.ml.MLPClassifier({
  hiddenLayers: [50, 30],
  activation: 'relu',
  learningRate: 0.01,
  maxIter: 300
});

mlp.fit({
  data: scaledTrainData,  // MLP requires scaled features
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

### PolynomialRegressor

Polynomial regression.

```javascript
new ds.ml.PolynomialRegressor(options)
```

#### Options

```javascript
{
  degree: number  // Polynomial degree (default: 2)
}
```

---

### GAMRegressor / GAMClassifier

Generalized Additive Models.

```javascript
new ds.ml.GAMRegressor(options)
new ds.ml.GAMClassifier(options)
```

---

## Clustering

### KMeans

Partition data into `k` clusters minimizing within-cluster sum of squares.

```javascript
new ds.ml.KMeans(options)
```

#### Options

```javascript
{
  k: number,       // Number of clusters (default: 3)
  maxIter: number,  // Maximum iterations (default: 300)
  tol: number,      // Convergence tolerance (default: 1e-4)
  seed: number      // Random seed
}
```

#### Methods

- `.fit(X)` or `.fit({ data, columns })`
- `.predict(X)` - Assign new points to nearest centroid
- `.silhouetteScore(X, labels)` - Compute silhouette score
- `.summary()` - Iterations, inertia, convergence, centroids
- `.toJSON()` / `KMeans.fromJSON()` - Persistence

#### Example

```javascript
const km = new ds.ml.KMeans({ k: 3, seed: 42 });
km.fit({
  data: iris,
  columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
});

console.log(km.labels);     // Cluster assignments
console.log(km.centroids);  // Cluster centers
```

---

### HCA

Hierarchical agglomerative clustering.

```javascript
new ds.ml.HCA(options)
```

#### Options

```javascript
{
  linkage: string,      // 'single', 'complete', 'average', 'ward' (default: 'average')
  omit_missing: boolean  // Drop rows with NaN (default: true)
}
```

#### Methods

- `.fit(X)` or `.fit({ data, columns })`
- `.cut(k)` - Return cluster labels for `k` clusters
- `.cutHeight(height)` - Cut dendrogram at a distance threshold
- `.summary()` - Linkage, observations, merge count, max distance
- `.toJSON()` / `HCA.fromJSON()`

#### Example

```javascript
const hca = new ds.ml.HCA({ linkage: 'ward' });
hca.fit({
  data: penguins,
  columns: ['bill_length_mm', 'flipper_length_mm', 'body_mass_g']
});

const labels = hca.cut(3);
```

---

## Preprocessing

### StandardScaler

Standardize features to zero mean and unit variance: `z = (x - mean) / std`.

```javascript
new ds.ml.preprocessing.StandardScaler()
```

#### Methods

- `.fit({ data, columns })` - Compute mean and std from training data
- `.transform({ data, columns })` - Apply standardization
- `.fitTransform(X)` - Fit and transform in one step

**Important:** Always fit on training data, then transform both train and test.

```javascript
const scaler = new ds.ml.preprocessing.StandardScaler();
scaler.fit({ data: trainData, columns: numericFeatures });

const trainScaled = scaler.transform({ data: trainData, columns: numericFeatures });
const testScaled = scaler.transform({ data: testData, columns: numericFeatures });
```

---

### MinMaxScaler

Scale features to [0, 1] range: `x_scaled = (x - min) / (max - min)`.

```javascript
new ds.ml.preprocessing.MinMaxScaler()
```

Same API as `StandardScaler`.

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

### Steps

| Method | Description |
|--------|-------------|
| `.parseNumeric(columns)` | Convert string columns to numbers |
| `.clean(validCategories)` | Remove rows with invalid categories |
| `.oneHot(columns, { dropFirst })` | One-hot encode categorical columns |
| `.scale(columns, { method })` | Scale numeric columns (`'standard'` or `'minmax'`) |
| `.split({ ratio, shuffle, seed })` | Split into train/test sets |

### Execution

#### `.prep()`

Execute recipe and fit all transformers on training data.

```javascript
const prepped = recipe.prep()
```

**Returns:**
```javascript
{
  train: { data, X, y, metadata },
  test: { data, X, y, metadata },
  transformers: { scale, oneHot },
  steps: [...]
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

const prepped = recipe.prep();

const model = new ds.ml.MLPRegressor({ hiddenLayers: [64, 32] });
model.fit({
  data: prepped.train.data,
  X: prepped.train.X,
  y: prepped.train.y
});

// Apply to new data (uses fitted transformers)
const newPrepped = recipe.bake(newDiamonds);
const predictions = model.predict({ data: newPrepped.data, X: newPrepped.X });
```

---

## Validation

### trainTestSplit

Split data into training and testing sets.

**Table API:**
```javascript
const split = ds.ml.validation.trainTestSplit(
  { data: myData, X: features, y: 'target' },
  { ratio: 0.7, shuffle: true, seed: 42 }
)

// Returns: { train: { data, X, y, metadata }, test: { ... } }
```

---

### crossValidate

Perform k-fold cross-validation.

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
```

---

## Metrics

### Classification

```javascript
ds.ml.metrics.accuracy(yTrue, yPred)       // Fraction correct
ds.ml.metrics.confusionMatrix(yTrue, yPred) // 2D array
ds.ml.metrics.f1Score(yTrue, yPred)         // F1 score
```

### Regression

```javascript
ds.ml.metrics.r2Score(yTrue, yPred)  // R-squared (1.0 = perfect)
ds.ml.metrics.rmse(yTrue, yPred)     // Root Mean Squared Error
ds.ml.metrics.mae(yTrue, yPred)      // Mean Absolute Error
ds.ml.metrics.mse(yTrue, yPred)      // Mean Squared Error
```

---

## Pipelines

### Pipeline

Chain preprocessing and model steps.

```javascript
new ds.ml.Pipeline(steps)
```

### GridSearchCV (pipeline-level)

```javascript
ds.ml.GridSearchCV(fitFn, scoreFn, X, y, paramGrid, { k: 5 })
```

---

## See Also

- [Statistics API](statistics) - GLM, hypothesis tests
- [Multivariate Analysis API](multivariate) - PCA, LDA, RDA
- [Visualization API](visualization) - ROC curves, confusion matrices
