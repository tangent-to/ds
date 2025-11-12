---
layout: default
title: API Reference
nav_order: 4
has_children: true
permalink: /api
---

# API Reference

Complete reference documentation for all tangent/ds modules and functions.

## Modules

### Multivariate analysis
Ordination (PCA, LDA, RDA) and clustering techniques.

### Statistics
Statistical analysis functions and models including GLM, t-tests, and ANOVA.

### Machine learning
Supervised learning algorithms, validation, and preprocessing utilities.

### Plotting
Visualization functions for Observable Plot integration.

---

## Usage pattern

Most tangent/ds classes follow this pattern:

```javascript
import * as ds from '@tangent/ds';
const model = new ds.ml.KNNClassifier({ k: 5 });
model.fit({ data: myData, X: features, y: 'target' });
const predictions = model.predict({ data: newData, X: features });
```
