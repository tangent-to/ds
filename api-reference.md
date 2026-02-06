---
layout: default
title: API Reference
nav_order: 4
has_children: true
permalink: /api
---

# API Reference
{: .no_toc }

Complete reference documentation for all tangent/ds modules and functions.
{: .fs-6 .fw-300 }

---

## Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| [Statistics](api/statistics) | `ds.stats` | GLM, hypothesis tests, distributions, model comparison |
| [Machine Learning](api/machine-learning) | `ds.ml` | KNN, trees, forests, MLP, preprocessing, validation |
| [Multivariate Analysis](api/multivariate) | `ds.mva` | PCA, LDA, RDA, CCA ordination methods |
| [Visualization](api/visualization) | `ds.plot` | Observable Plot configs for biplots, ROC, diagnostics |
| [Core Utilities](api/core) | `ds.core` | Math, linear algebra, tables, formulas, optimization |

---

## Usage Pattern

Most tangent/ds classes follow the **fit-predict** pattern:

```javascript
import * as ds from '@tangent/ds';

// 1. Create a model
const model = new ds.ml.KNNClassifier({ k: 5 });

// 2. Fit to data (Table API)
model.fit({ data: myData, X: ['feature1', 'feature2'], y: 'target' });

// 3. Make predictions
const predictions = model.predict({ data: newData, X: ['feature1', 'feature2'] });
```

### Input Styles

All estimators support multiple input styles:

**Table API** (recommended):
```javascript
model.fit({ data: myData, X: ['col1', 'col2'], y: 'target' })
```

**Array API**:
```javascript
model.fit(X, y)
```

**Formula API** (GLM only):
```javascript
model.fit({ formula: 'y ~ x1 + x2', data: myData })
```

### Serialization

All models support persistence via JSON:

```javascript
const json = model.toJSON();
const restored = ModelClass.fromJSON(json);
```
