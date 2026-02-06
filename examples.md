---
layout: default
title: Examples
nav_order: 5
permalink: /examples
---

# Examples

Quick code snippets to get you started. For in-depth tutorials, see the [Tutorials](tutorials) section.

## Statistical analysis

### T-test

```javascript
import * as ds from '@tangent/ds';

const group1 = [5.1, 4.9, 4.7, 4.6, 5.0];
const group2 = [7.0, 6.4, 6.9, 6.5, 6.3];

const result = ds.stats.ttest(group1, group2);
console.log(`p-value: ${result.pValue}`);
```

### Linear regression

```javascript
const glm = new ds.stats.GLM({ family: 'gaussian' });
glm.fit({
  X: ['height', 'weight'],
  y: 'blood_pressure',
  data: healthData
});

console.log(glm.summary());
```

---

## Machine learning

### Classification

```javascript
const knn = new ds.ml.KNNClassifier({ k: 5 });
knn.fit({
  data: trainData,
  X: ['feature1', 'feature2', 'feature3'],
  y: 'species'
});

const predictions = knn.predict({ data: testData, X: features });
```

### Cross-validation

```javascript
const scores = ds.ml.validation.crossValidate(
  (Xtr, ytr) => new ds.ml.KNNClassifier({ k: 5 }).fit(Xtr, ytr),
  (model, Xte, yte) => ds.ml.metrics.accuracy(yte, model.predict(Xte)),
  { data: myData, X: features, y: 'target' },
  { k: 5, shuffle: true }
);

console.log(`Mean accuracy: ${scores.mean()}`);
```

---

## Ordination

### PCA

```javascript
import * as Plot from '@observablehq/plot';

const pca = new ds.mva.PCA({ center: true, scale: true });
pca.fit({ data: myData, X: features });

// Visualize
const biplot = ds.plot.ordiplot(pca.model);
document.body.appendChild(biplot.show(Plot));
```

### LDA

```javascript
const lda = new ds.mva.LDA();
lda.fit({
  data: myData,
  X: features,
  y: 'group'
});

const biplot = ds.plot.ordiplot(lda.model, {
  colorBy: myData.map(d => d.group)
});
```

---

## Clustering

### K-Means

```javascript
const kmeans = new ds.ml.KMeans({ k: 3, random_state: 42 });
kmeans.fit({
  data: myData,
  columns: features,
  standardize: true
});

console.log(`Cluster labels: ${kmeans.labels}`);
```

### Hierarchical clustering

```javascript
const hca = new ds.ml.HCA({ linkage: 'ward' });
hca.fit({ data: myData, X: features });

// Plot dendrogram
const dendrogram = ds.plot.plotHCA(hca.model);
```

---

## More examples

For complete, runnable examples with explanations, check out:

- [Observable Notebooks](https://observablehq.com/@essi) — interactive tutorials
- [Tutorials](tutorials)
