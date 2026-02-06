---
layout: default
title: Visualization
parent: API Reference
nav_order: 4
permalink: /api/visualization
---

# Visualization API
{: .no_toc }

Publication-ready plots powered by Observable Plot.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `ds.plot` module generates **Observable Plot configuration objects**. Each function returns a config object with a `.show(Plot)` method that renders the visualization.

```javascript
import * as Plot from '@observablehq/plot';

const config = ds.plot.ordiplot(pca.model);
document.body.appendChild(config.show(Plot));
```

**Categories:**
- **Ordination** - Biplots for PCA, LDA, RDA
- **Clustering** - Dendrograms, silhouette plots
- **Classification** - ROC, precision-recall, confusion matrix, calibration
- **Diagnostics** - Residual plots, Q-Q plots for GLMs
- **Interpretation** - Feature importance, partial dependence, learning curves

---

## Ordination Plots

### ordiplot

Unified ordination plot for PCA, LDA, and RDA results.

```javascript
ds.plot.ordiplot(result, options)
```

**Parameters:**
- `result` (Object): Ordination result from PCA, LDA, or RDA `.model`
- `options` (Object):
  - `type` (string): `'pca'`, `'lda'`, `'rda'` (auto-detected if omitted)
  - `colorBy` (Array): Group labels for coloring points
  - `labels` (Array&lt;string&gt;): Point labels
  - `showLoadings` (boolean): Show loading vectors (PCA/RDA)
  - `showCentroids` (boolean): Show class centroids (LDA)
  - `showConvexHulls` (boolean): Show convex hulls around groups
  - `axis1` (number): First axis (default: 1)
  - `axis2` (number): Second axis (default: 2)
  - `width` (number): Plot width (default: 640)
  - `height` (number): Plot height (default: 400)
  - `loadingFactor` (number): Multiplier for loading vectors (default: 1; set 0 for auto)
  - `predictorFactor` (number): Multiplier for predictor arrows in RDA

**Returns:** Plot configuration with `.show(Plot)` method

**Example:**
```javascript
import * as Plot from '@observablehq/plot';

const pca = new ds.mva.PCA({ center: true, scale: true });
pca.fit({ data: penguins, columns: numericCols });

const biplot = ds.plot.ordiplot(pca.model, {
  colorBy: penguins.map(d => d.Species),
  showLoadings: true
});

document.body.appendChild(biplot.show(Plot));
```

---

### plotScree

Scree plot showing variance explained by each component.

```javascript
ds.plot.plotScree(result, options)
```

**Parameters:**
- `result` (Object): PCA/ordination result with `varianceExplained`
- `options` (Object):
  - `width` (number): Plot width (default: 640)
  - `height` (number): Plot height (default: 300)
  - `cumulative` (boolean): Show cumulative variance (default: false)

---

## Clustering Plots

### plotHCA

Generate dendrogram data from hierarchical clustering results.

```javascript
ds.plot.plotHCA(result)
```

**Parameters:**
- `result` (Object): HCA result from `ds.ml.HCA`

**Returns:** Dendrogram tree structure

---

### dendrogramLayout

Convert dendrogram data to layout coordinates for rendering.

```javascript
ds.plot.dendrogramLayout(dendrogramData, options)
```

**Parameters:**
- `dendrogramData` (Object): Result from `plotHCA`
- `options` (Object):
  - `width` (number): default 640
  - `height` (number): default 400
  - `orientation` (string): `'vertical'` (default)

**Returns:** Layout with nodes and links coordinates

---

### plotSilhouette

Silhouette plot displaying per-sample cluster quality scores.

```javascript
ds.plot.plotSilhouette(options, plotOptions)
```

**Parameters:**
- `options` (Object):
  - `samples` (Array): Precomputed from `ds.ml.silhouette.silhouetteSamples()`
  - `data` (Array): Data matrix (alternative to `samples`)
  - `labels` (Array): Cluster labels (used with `data`)
  - `sorted` (boolean): Sort by silhouette score (default: true)
- `plotOptions` (Object):
  - `width` (number): default 720
  - `height` (number): default 420
  - `showAverageLines` (boolean): default true

**Returns:** Plot configuration with `.show(Plot)` method

---

### HDBSCAN Visualizations

```javascript
ds.plot.plotHDBSCAN(model, options)           // Condensed tree (default)
ds.plot.plotCondensedTree(model, options)      // Condensed cluster tree
ds.plot.plotHDBSCANDendrogram(model, options)  // HDBSCAN dendrogram
ds.plot.plotClusterMembership(model, data, options)  // Membership probabilities
ds.plot.plotClusterStability(model, options)    // Cluster stability
ds.plot.plotHDBSCANDashboard(model, data, options)  // All-in-one dashboard
```

---

## Classification Metrics

### plotROC

ROC curve with AUC.

```javascript
ds.plot.plotROC(yTrue, yProb, options)
```

**Parameters:**
- `yTrue` (Array&lt;number&gt;): True binary labels (0 or 1)
- `yProb` (Array&lt;number&gt;): Predicted probabilities for positive class
- `options` (Object):
  - `width` (number): default 500
  - `height` (number): default 500
  - `showDiagonal` (boolean): Show random baseline (default: true)

**Returns:** Plot configuration with AUC value

**Example:**
```javascript
const roc = ds.plot.plotROC(yTrue, model.predictProba(XTest).map(p => p[1]));
document.body.appendChild(roc.show(Plot));
```

---

### plotPrecisionRecall

Precision-recall curve with average precision.

```javascript
ds.plot.plotPrecisionRecall(yTrue, yProb, options)
```

**Parameters:**
- `yTrue` (Array&lt;number&gt;): True binary labels
- `yProb` (Array&lt;number&gt;): Predicted probabilities
- `options` (Object):
  - `width`, `height`: default 500
  - `showBaseline` (boolean): default true

---

### plotConfusionMatrix

Confusion matrix heatmap.

```javascript
ds.plot.plotConfusionMatrix(yTrue, yPred, options)
```

**Parameters:**
- `yTrue` (Array): True labels
- `yPred` (Array): Predicted labels
- `options` (Object):
  - `width`, `height`: default 500
  - `normalize` (boolean): Normalize values (default: false)
  - `labels` (Array): Custom class labels

**Example:**
```javascript
const cm = ds.plot.plotConfusionMatrix(yTrue, predictions, { normalize: true });
document.body.appendChild(cm.show(Plot));
```

---

### plotCalibration

Calibration curve showing how well predicted probabilities match actual frequencies.

```javascript
ds.plot.plotCalibration(yTrue, yProb, options)
```

**Parameters:**
- `yTrue` (Array&lt;number&gt;): True binary labels
- `yProb` (Array&lt;number&gt;): Predicted probabilities
- `options.nBins` (number): Number of bins (default: 10)

---

## GLM Diagnostics

### diagnosticDashboard

Generate all four diagnostic plots for a fitted GLM.

```javascript
const plots = ds.plot.diagnosticDashboard(model, options)
// Returns array of 4 plot specifications
```

**Includes:** Residual plot, Q-Q plot, scale-location plot, residuals-leverage plot.

---

### residualPlot

Residuals vs fitted values.

```javascript
ds.plot.residualPlot(model, options)
```

### scaleLocationPlot

Scale-location plot (sqrt of standardized residuals vs fitted).

```javascript
ds.plot.scaleLocationPlot(model, options)
```

### qqPlot

Q-Q plot for normality check of residuals.

```javascript
ds.plot.qqPlot(model, options)
```

### residualsLeveragePlot

Residuals vs leverage with Cook's distance.

```javascript
ds.plot.residualsLeveragePlot(model, options)
```

---

### effectPlot

Effect plot for a specific predictor variable.

```javascript
ds.plot.effectPlot(model, variable, data, options)
```

**Parameters:**
- `model` (Object): Fitted GLM
- `variable` (string): Variable name
- `data` (Object): Original data
- `options.grid` (number): Grid points (default: 50)
- `options.confidence` (number): Confidence level (default: 0.95)

---

### partialResidualPlot

Component + residual plot for a specific predictor.

```javascript
ds.plot.partialResidualPlot(model, variable, X, options)
```

---

## Model Interpretation

### plotFeatureImportance

Feature importance bar plot.

```javascript
ds.plot.plotFeatureImportance(importances, options)
```

**Parameters:**
- `importances` (Array&lt;Object&gt;): `[{ feature, importance }, ...]`
- `options.topN` (number): Top N features to display (default: 10)

---

### plotPartialDependence

Partial dependence plot.

```javascript
ds.plot.plotPartialDependence(pdResult, options)
```

**Parameters:**
- `pdResult` (Object): `{ values, predictions, feature }`

---

### plotCorrelationMatrix

Correlation matrix heatmap.

```javascript
ds.plot.plotCorrelationMatrix(corrResult, options)
```

**Parameters:**
- `corrResult` (Object): `{ matrix, features }`

---

### plotResiduals

Generic residual plot from precomputed data.

```javascript
ds.plot.plotResiduals(residualData, options)
```

**Parameters:**
- `residualData` (Object): `{ fitted, residuals, standardized }`
- `options.standardized` (boolean): Use standardized residuals (default: false)

---

### plotQQ

Generic Q-Q plot from precomputed data.

```javascript
ds.plot.plotQQ(residualData, options)
```

---

### plotLearningCurve

Learning curve showing train/test scores vs training size.

```javascript
ds.plot.plotLearningCurve(lcResult, options)
```

**Parameters:**
- `lcResult` (Object): `{ trainSizes, trainScores, testScores }`

---

## Renderers

### createD3DendrogramRenderer

Create a D3-based dendrogram renderer for environments without Observable Plot.

```javascript
ds.plot.createD3DendrogramRenderer(options)
```

---

## Common Pattern

All plot functions follow the same usage pattern:

```javascript
import * as Plot from '@observablehq/plot';

// 1. Generate config
const config = ds.plot.plotROC(yTrue, yProb);

// 2. Render with Observable Plot
const element = config.show(Plot);

// 3. Add to DOM
document.body.appendChild(element);
```

---

## See Also

- [Multivariate Analysis API](multivariate) - Ordination models for biplots
- [Statistics API](statistics) - GLMs for diagnostic plots
- [Machine Learning API](machine-learning) - Models for classification metrics
