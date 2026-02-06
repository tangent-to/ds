---
layout: default
title: Multivariate Analysis
parent: API Reference
nav_order: 3
permalink: /api/multivariate
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
- **CCA** - maximize correlation between two multivariate blocks (`X` and `Y`).

These estimators understand both the numeric **Array API** (`Array<Array<number>>`) and the declarative **Table API** (`{ data, X, y }`). Use `ds.plot.ordiplot(model).show(Plot)` to render biplots from any fitted ordination.

### Supported Estimators

| Estimator | Type | Goal |
|-----------|------|------|
| `ds.mva.PCA` | Transformer | Maximize variance |
| `ds.mva.LDA` | Classifier | Maximize between-class separation |
| `ds.mva.RDA` | Transformer | Explain responses via predictors |
| `ds.mva.CCA` | Transformer | Maximize cross-block correlation |

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

PCA finds orthogonal directions that maximize variance. Components are eigenvectors of the covariance (or correlation) matrix.

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

**Options:**
- `center` (boolean): Subtract column means (default: `true`)
- `scale` (boolean): Divide by standard deviation for correlation PCA (default: `false`)
- `columns` (string[]): Table API column selectors
- `omit_missing` (boolean): Drop rows containing `NaN` (default: `true`)
- `scaling` (1 or 2): Score/loading scaling (default: `2`)

### Methods

| Method | Description |
|--------|-------------|
| `.fit(X)` or `.fit({ data, columns })` | Fit the PCA model |
| `.transform(X)` | Project new data onto PCs |
| `.cumulativeVariance()` | Running sum of variance explained |
| `.getScores(type, scaled)` | `'sites'` or `'variables'` scores |
| `.summary()` | Eigenvalues, variance explained |
| `.toJSON()` / `PCA.fromJSON()` | Model persistence |

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

LDA finds projection vectors that maximize the Fisher criterion: between-class scatter vs within-class scatter. Provides dimensionality reduction and supervised classification.

### Constructor

```javascript
const lda = new ds.mva.LDA({ scale: false, scaling: 2 });
```

**Options:**
- `scale` (boolean): Standardize features before solving
- `scaling` (1 or 2): Score scaling

### Methods

| Method | Description |
|--------|-------------|
| `.fit(X, y)` or `.fit({ X, y, data })` | Fit discriminant axes |
| `.transform(X)` | Discriminant scores for visualization |
| `.predict(X)` | Class predictions |
| `.getScores(type, scaled)` | `'sites'` vs `'variables'` |
| `.summary()` | Classes, eigenvalues, components |
| `.toJSON()` / `LDA.fromJSON()` | Persistence |

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

**Options:**
- `scale` (boolean): Standardize responses and predictors
- `omit_missing` (boolean): Drop rows with NaN
- `scaling` (1 or 2): Score scaling for biplots
- `constrained` (boolean): Use constrained ordination

### Methods

| Method | Description |
|--------|-------------|
| `.fit(Y, X)` or `.fit({ data, response, predictors })` | Fit the RDA model |
| `.transform(Y, X)` | Project new rows |
| `.getScores(type, scaled)` | `'sites'`, `'responses'`, or `'constraints'` |
| `.summary()` | Constrained variance, eigenvalues |
| `.toJSON()` / `RDA.fromJSON()` | Persistence |

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

CCA finds weight vectors that maximize the correlation between projected blocks `X` and `Y`.

### Constructor

```javascript
const cca = new ds.mva.CCA({
  center: true,
  scale: false
});
```

### Methods

| Method | Description |
|--------|-------------|
| `.fit(X, Y)` or `.fit({ data, X, Y })` | Fit canonical variates |
| `.transformX(X)` / `.transformY(Y)` | Canonical scores per block |
| `.transform(X, Y)` | Project both simultaneously |
| `.summary()` | Canonical correlations |
| `.toJSON()` / `CCA.fromJSON()` | Persistence |

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
```

---

## Composition Analysis

The `ds.mva.composition` namespace provides tools for compositional data (e.g., proportions that sum to 1).

---

## Functional APIs

In addition to the class-based estimators, lower-level functional namespaces are available:

```javascript
ds.mva.pca   // functional PCA
ds.mva.lda   // functional LDA
ds.mva.rda   // functional RDA
ds.mva.cca   // functional CCA
```

These return raw result objects and are useful for advanced workflows.

---

## Workflow Tips

- Combine ordination scores with clustering labels to annotate biplots.
- Use `ds.plot.ordiplot(model).show(Plot)` for quick visual diagnostics.
- Persist fitted ordinations with `.toJSON()` and reload with `.fromJSON()`.
- `omit_missing` defaults to `true` so ordinations operate on complete cases.

---

## See Also

- [Statistics API](statistics) - GLM, hypothesis tests
- [Machine Learning API](machine-learning) - Clustering (KMeans, HCA)
- [Visualization API](visualization) - Ordination plots, dendrograms
