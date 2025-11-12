---
layout: default
title: Multivariate Analysis
parent: API Reference
nav_order: 3
---

# Multivariate Analysis API
{: .no_toc }

Ordination, canonical correlation, and clustering utilities.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `ds.mva` namespace covers techniques that explain structure in multiple variables at once:

- **PCA** - project data onto orthogonal axes that maximize total variance.
- **LDA** - find discriminant axes that maximize between-class separation relative to within-class scatter.
- **RDA** - regress multivariate responses onto predictors and perform PCA in the fitted space.
- **CCA** - maximize correlation between two multivariate blocks (`X` <-> `Y`).
- **HCA** and **KMeans** - complementary clustering estimators from `ds.ml` commonly used alongside ordinations.

These estimators understand both the numeric **Array API** (`Array<Array<number>>`) and the declarative **Table API** (`{ data, X, y }`). Use `ds.plot.ordiplot(model).show(Plot)` to render biplots from any fitted ordination.

### Supported Estimators

| Estimator | Type | Goal | Works With |
|-----------|------|------|------------|
| `ds.mva.PCA` | Transformer | Maximize variance `w^T S w` subject to `||w|| = 1` | Array + Table |
| `ds.mva.LDA` | Classifier | Maximize Fisher ratio `|S_B| / |S_W|` for labeled groups | Array + Table |
| `ds.mva.RDA` | Transformer | Explain responses `Y` via predictors `X` before ordination | Array + Table |
| `ds.mva.CCA` | Transformer | Maximize correlation between `X` and `Y` scores | Array + Table |
| `ds.ml.HCA` | Estimator | Agglomerative clustering with linkage strategies | Array + Table |
| `ds.ml.KMeans` | Estimator | Partition data into `k` centroids minimizing SSE | Array + Table |

---

## Data Inputs

### Array API

Pass numeric matrices directly:

```javascript
const scores = pca.fit(X).transform(Xnew);
```

- `X` and `Y` are `Array<Array<number>>`.
- Optional `opts` allow `center`, `scale`, `linkage`, etc.

### Declarative Table API

Provide structured data with column selectors:

```javascript
pca.fit({
  data: penguins,
  columns: ['bill_length_mm', 'bill_depth_mm'],
  center: true,
  scale: true
});
```

- Automatically handles column lookup, missing-value filtering, and metadata.
- Use `response` + `predictors` for RDA, `X` + `y` for LDA, `X` + `Y` for CCA.

---

## Principal Component Analysis (PCA)

PCA finds orthogonal directions `w` that maximize variance `w^T S w` subject to `||w|| = 1`. Components are eigenvectors of the covariance (or correlation) matrix `S`.

### Constructor

```javascript
const pca = new ds.mva.PCA({
  center: true,
  scale: false,
  columns: null,
  omit_missing: true,
  scaling: 2
});
```

- `center` (boolean): subtract column means.
- `scale` (boolean): divide by standard deviation (correlation PCA).
- `columns` (string[]): Table API selectors; optional with Array API.
- `omit_missing` (boolean): drop rows containing `NaN`.
- `scaling` (1|2): Scaling applied to scores/loadings (Hill 1979).

### Methods

- `.fit(X[, opts])` or `.fit({ data, columns, ... })`
- `.transform(X)` - project new data onto PCs.
- `.cumulativeVariance()` - returns running sum of variance explained.
- `.getScores(type, scaled)` - `'sites'|'samples'` vs `'variables'|'loadings'`.
- `.summary()` - eigenvalues, variance explained, centering/scaling flags.
- `.toJSON()` / `PCA.fromJSON()` - model persistence.

### Examples

```javascript
// Declarative workflow
const pca = new ds.mva.PCA({ center: true, scale: true });
pca.fit({
  data: penguins,
  columns: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
});

const scores = pca.getScores('sites');
const loadings = pca.getScores('variables');
ds.plot.ordiplot(pca.model).show(Plot);
```

```javascript
// Numeric matrices
const X = [[5.1, 3.5], [4.9, 3.0], [4.7, 3.2]];
pca.fit(X);
const projected = pca.transform(X);
```

---

## Linear Discriminant Analysis (LDA)

LDA finds projection vectors `w` that maximize the Fisher criterion `w^T S_B w / w^T S_W w` (between-class scatter vs within-class scatter). Provides dimensionality reduction and supervised classification.

### Constructor

```javascript
const lda = new ds.mva.LDA({ scale: false, scaling: 2 });
```

- `scale`: Standardize features before solving eigenproblem.
- `scaling`: Score scaling (matches PCA options).

### Methods

- `.fit(X, y[, opts])` or `.fit({ X, y, data, omit_missing })`
- `.transform(X)` - discriminant scores for visualization.
- `.predict(X)` - class predictions.
- `.getScores(type, scaled)` - `'sites'` vs `'variables'`.
- `.summary()` - classes, eigenvalues, number of components.
- `.toJSON()` / `LDA.fromJSON()`.

### Example

```javascript
const lda = new ds.mva.LDA({ scale: true });
lda.fit({
  data: iris,
  X: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
  y: 'species'
});

const preds = lda.predict({
  data: iris,
  X: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
});
const axes = lda.getScores('variables');
```

---

## Redundancy Analysis (RDA)

RDA combines multivariate regression with PCA: regress responses `Y` onto predictors `X`, then perform PCA on fitted values. Constrained axes maximize variance explained by `X`.

### Constructor

```javascript
const rda = new ds.mva.RDA({
  scale: false,
  omit_missing: true,
  scaling: 2,
  constrained: true
});
```

- `scale`: Standardize responses and predictors.
- `omit_missing`: Drop rows with `NaN` before fitting.
- `scaling`: Score scaling for biplots.
- `constrained`: `true` to use constrained ordination; `false` for partial RDA.

### Methods

- `.fit(Y, X[, opts])` or `.fit({ data, response, predictors, ... })`
- `.transform(Y, X)` - project new rows using fitted axes.
- `.getScores(type, scaled)` - `'sites'|'responses'|'constraints'`.
- `.summary()` - constrained variance, eigenvalues, sample counts.
- `.toJSON()` / `RDA.fromJSON()`.

### Example

```javascript
const rda = new ds.mva.RDA({ scale: true });
rda.fit({
  data: drought,
  response: ['soil_moisture', 'soil_temp'],
  predictors: ['precip', 'evap', 'wind']
});

const siteScores = rda.getScores('sites');
const envVectors = rda.getScores('constraints');
```

---

## Canonical Correlation Analysis (CCA)

CCA finds weight vectors `w_x`, `w_y` that maximize the correlation between projected blocks `X w_x` and `Y w_y`. Multiple pairs of canonical variates are extracted sequentially.

### Constructor

```javascript
const cca = new ds.mva.CCA({
  center: true,
  scale: false
});
```

Pass any numeric hyperparameters needed by the functional API (for most uses, centering/scaling flags are sufficient).

### Methods

- `.fit(X, Y[, opts])` or `.fit({ data, X, Y })`
- `.transformX(X)` / `.transformY(Y)` - canonical variate scores for each block.
- `.transform(X, Y)` - project both simultaneously.
- `.summary()` - sample count, number of components, canonical correlations.
- `.toJSON()` / `CCA.fromJSON()`.

### Example

```javascript
const cca = new ds.mva.CCA();
cca.fit({
  data: study,
  X: ['blood_pressure', 'cholesterol'],
  Y: ['vo2max', 'time_to_exhaustion']
});

const serumScores = cca.transformX({
  data: study,
  X: ['blood_pressure', 'cholesterol']
});
const fitnessScores = cca.transformY({
  data: study,
  Y: ['vo2max', 'time_to_exhaustion']
});
```

---

## Hierarchical Clustering (HCA)

`ds.ml.HCA` wraps the agglomerative clustering routines (`single`, `complete`, `average`, `ward`). It builds a dendrogram by iteratively merging the closest clusters until a single cluster remains.

### Constructor

```javascript
const hca = new ds.ml.HCA({
  linkage: 'average',   // 'single' | 'complete' | 'average' | 'ward'
  omit_missing: true
});
```

### Methods

- `.fit(X[, opts])` or `.fit({ data, columns, linkage, omit_missing })`
- `.cut(k)` - return cluster labels for `k` clusters.
- `.cutHeight(height)` - cut dendrogram at a distance threshold.
- `.summary()` - linkage, number of observations, merge count, max distance.
- `.toJSON()` / `HCA.fromJSON()`.

### Example

```javascript
const hca = new ds.ml.HCA({ linkage: 'ward' });
hca.fit({
  data: penguins,
  columns: ['bill_length_mm', 'flipper_length_mm', 'body_mass_g']
});

const labels = hca.cut(3);
```

---

## K-Means Clustering

`ds.ml.KMeans` partitions observations into `k` centroids by minimizing the within-cluster sum of squares

```
J = Sigma_i Sigma_{x in C_i} ||x - mu_i||^2
```

### Constructor

```javascript
const kmeans = new ds.ml.KMeans({
  k: 3,
  maxIter: 300,
  tol: 1e-4,
  seed: 42
});
```

### Methods

- `.fit(X[, opts])` or `.fit({ data, columns, k, ... })`
- `.predict(X)` - assign new rows to nearest centroid.
- `.silhouetteScore(X, labels)` - quality metric (defaults to fitted labels).
- `.summary()` - iterations, inertia, convergence flag, centroids.
- `.toJSON()` / `KMeans.fromJSON()`.

### Example

```javascript
const km = new ds.ml.KMeans({ k: 4, seed: 10 });
km.fit({
  data: iris,
  columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
});

const clusterIds = km.predict({
  data: iris,
  columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
});

console.log(km.summary());
```

---

## Workflow Tips

- Use ordination scores plus clustering labels to annotate biplots.
- Combine `ds.mva.PCA` with `ds.plot.ordiplot(...).show(Plot)` for quick diagnostics.
- Persist fitted ordinations using `.toJSON()` and reload in browsers with `.fromJSON()`.
- `omit_missing` defaults to `true` so that ordinations operate on complete cases; set to `false` only when you have filled missing data manually.

