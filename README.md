# @tangent.to/ds

A browser-friendly data science library in modern JavaScript (ESM).

## Installation

```bash
npm install @tangent.to/ds
```

## Quick Start

```javascript
import { core, stats, ml, mva, plot } from '@tangent.to/ds';

// Linear algebra
const transposed = core.linalg.transpose([[1, 2], [3, 4]]);

// Generalized Linear Models
const model = new stats.GLM({ family: 'gaussian' });
model.fit(X, y);
const predictions = model.predict(X_new);

// K-Means clustering
const kmeans = new ml.KMeans({ k: 3 });
kmeans.fit(data);

// PCA
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(data);
```

## Modules

- **core** - Linear algebra, tables, math, optimization, formulas
- **stats** - Distributions, GLM/GLMM, hypothesis tests, model comparison
- **ml** - Machine learning algorithms (clustering, classification, regression)
- **mva** - Multivariate analysis (PCA, LDA, RDA, CCA, HCA)
- **plot** - Observable Plot configuration generators

## Statistics (stats)

### Generalized Linear Models (GLM)

The `GLM` class unifies all regression models with a consistent interface:

```javascript
// Linear regression (Gaussian family)
const lm = new stats.GLM({ family: 'gaussian' });
lm.fit(X, y);

// Logistic regression (Binomial family)
const logit = new stats.GLM({ family: 'binomial', link: 'logit' });
logit.fit(X, y);

// Multiclass classification (one-vs-rest)
const multiclass = new stats.GLM({ family: 'binomial', multiclass: 'ovr' });
multiclass.fit({ X: ['feature1', 'feature2'], y: 'species', data });
const predictions = multiclass.predict({ X: ['feature1', 'feature2'], data: newData });

// Poisson regression
const poisson = new stats.GLM({ family: 'poisson' });
poisson.fit(X, y);

// Mixed-effects model (GLMM) with random intercepts
const lmm = new stats.GLM({
  family: 'gaussian',
  randomEffects: { intercept: groups }
});
lmm.fit(X, y);

// Access model results
console.log(model.summary());      // Detailed summary
console.log(model.coefficients);   // Coefficient estimates
const predictions = model.predict(X_new);
```

### Formula Syntax

```javascript
const model = new stats.GLM({ family: 'gaussian' });
model.fit({ formula: 'y ~ x1 + x2', data });
```

### Distributions

```javascript
// Normal distribution
stats.normal.pdf(x, { mean: 0, sd: 1 });
stats.normal.cdf(x, { mean: 0, sd: 1 });
stats.normal.quantile(p, { mean: 0, sd: 1 });

// Other distributions: uniform, gamma, beta
stats.gamma.pdf(x, { shape: 1, scale: 1 });
stats.beta.pdf(x, { alpha: 1, beta: 1 });
```

### Hypothesis Tests

```javascript
// One-sample t-test
const result = stats.hypothesis.oneSampleTTest(data, { mu0: 0 });

// Two-sample t-test
const result = stats.hypothesis.twoSampleTTest(group1, group2);

// Chi-square test
const result = stats.hypothesis.chiSquareTest(observed, expected);

// One-way ANOVA
const result = stats.hypothesis.oneWayAnova(groups);
```

### Model Comparison

```javascript
// Compare models with AIC/BIC
const comparison = stats.compareModels([model1, model2, model3]);

// Likelihood ratio test for nested models
const lrt = stats.likelihoodRatioTest(model1, model2);
```

## Machine Learning (ml)

### K-Means Clustering

```javascript
const model = new ml.KMeans({ k: 3, maxIter: 100 });
model.fit(data);
console.log(model.labels);      // Cluster assignments
console.log(model.centroids);   // Cluster centers
```

### K-Nearest Neighbors

```javascript
// Classification
const knn = new ml.KNNClassifier({ k: 5 });
knn.fit(X_train, y_train);
const predictions = knn.predict(X_test);

// Regression
const knn = new ml.KNNRegressor({ k: 5 });
```

### Decision Trees & Random Forests

```javascript
// Decision tree
const dt = new ml.DecisionTreeClassifier({ maxDepth: 5 });
dt.fit(X_train, y_train);

// Random forest
const rf = new ml.RandomForestClassifier({ nEstimators: 100 });
rf.fit(X_train, y_train);
```

### Polynomial Regression

```javascript
const poly = new ml.PolynomialRegressor({ degree: 3 });
poly.fit(X, y);
```

### Neural Networks

```javascript
const mlp = new ml.MLPRegressor({
  layerSizes: [10, 8, 1],
  activation: 'relu',
  epochs: 100,
  learningRate: 0.01
});
mlp.fit(X_train, y_train);
```

### Model Selection

```javascript
// Cross-validation
const scores = ml.validation.crossValidate(model, X, y, { cv: 5 });

// Grid search
import { GridSearchCV } from '@tangent.to/ds/ml';
const result = GridSearchCV(fitFn, scoreFn, X, y, paramGrid, { k: 5 });
```

## Multivariate Analysis (mva)

### PCA

```javascript
const pca = new mva.PCA({ center: true, scale: false });
pca.fit(X);
console.log(pca.model.explainedVarianceRatio);
const X_transformed = pca.transform(X);
```

### LDA

```javascript
const lda = new mva.LDA();
lda.fit(X, y);
const X_transformed = lda.transform(X);
```

### RDA (Redundancy Analysis)

```javascript
const rda = new mva.RDA();
rda.fit(response, explanatory);
```

### Hierarchical Clustering

```javascript
const hca = new ml.HCA({ linkage: 'ward' });
hca.fit(data);
console.log(hca.model.dendrogram);
```

## Visualization (plot)

Returns Observable Plot configurations:

```javascript
// ROC curve
const config = plot.plotROC(yTrue, yPred);

// Confusion matrix
const config = plot.plotConfusionMatrix(yTrue, yPred);

// PCA biplot (unified ordination plot)
const config = plot.ordiplot(pca.model, { type: 'pca', showLoadings: true, loadingFactor: 0 });

// GLM diagnostics
const config = plot.diagnosticDashboard(model);
```

## Metrics

```javascript
// Classification
ml.metrics.accuracy(yTrue, yPred);
ml.metrics.f1Score(yTrue, yPred);

// Regression
ml.metrics.mse(yTrue, yPred);
ml.metrics.r2(yTrue, yPred);
```

## Testing

```bash
npm test                # Watch mode
npm run test:run        # Run once
npm run test:coverage   # With coverage
```

## Documentation

See [docs/API.md](docs/API.md) for complete API reference.

### User Guides

- [Multiclass Classification](examples/user-guide/05-multiclass.md) - Guide to multiclass GLMs
- [Testing Against R](examples/user-guide/06-testing-against-R.md) - Comparing outputs with R

## Examples

See `examples/` directory for working examples:
- `examples/stats/multinomial_glm_example.js` - Multiclass classification demo

## License

GPL-3.0
