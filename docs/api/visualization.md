---
layout: default
title: Visualization
parent: API Reference
nav_order: 4
has_children: true
permalink: /api/visualization
---
# plot

## Functions

### plotROC()

```ts
function plotROC(
   yTrue, 
   yProb, 
   options?): Object;
```

Defined in: [src/plot/classification.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/classification.js#L15)

Generate ROC curve plot configuration

#### Parameters

##### yTrue

`number`[]

True binary labels (0 or 1)

##### yProb

`number`[]

Predicted probabilities for positive class

##### options?

`Object` = `{}`

{width, height, showDiagonal}

#### Returns

`Object`

Plot configuration with ROC curve data and AUC

***

### plotPrecisionRecall()

```ts
function plotPrecisionRecall(
   yTrue, 
   yProb, 
   options?): Object;
```

Defined in: [src/plot/classification.js:125](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/classification.js#L125)

Generate precision-recall curve plot configuration

#### Parameters

##### yTrue

`number`[]

True binary labels (0 or 1)

##### yProb

`number`[]

Predicted probabilities for positive class

##### options?

`Object` = `{}`

{width, height, showBaseline}

#### Returns

`Object`

Plot configuration with precision-recall curve and average precision

***

### plotConfusionMatrix()

```ts
function plotConfusionMatrix(
   yTrue, 
   yPred, 
   options?): Object;
```

Defined in: [src/plot/classification.js:230](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/classification.js#L230)

Generate confusion matrix plot configuration

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

##### options?

`Object` = `{}`

{width, height, normalize, labels}

#### Returns

`Object`

Plot configuration with confusion matrix

***

### plotCalibration()

```ts
function plotCalibration(
   yTrue, 
   yProb, 
   options?): Object;
```

Defined in: [src/plot/classification.js:337](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/classification.js#L337)

Generate calibration curve plot configuration
Shows how well predicted probabilities match actual frequencies

#### Parameters

##### yTrue

`number`[]

True binary labels (0 or 1)

##### yProb

`number`[]

Predicted probabilities

##### options?

`Object` = `{}`

{width, height, nBins}

#### Returns

`Object`

Plot configuration with calibration curve

***

### residualPlot()

```ts
function residualPlot(model, options?): Object;
```

Defined in: [src/plot/diagnostics.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L14)

Generate residual vs fitted plot

#### Parameters

##### model

`Object`

Fitted GLM model

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### scaleLocationPlot()

```ts
function scaleLocationPlot(model, options?): Object;
```

Defined in: [src/plot/diagnostics.js:47](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L47)

Generate scale-location plot (sqrt of standardized residuals vs fitted)

#### Parameters

##### model

`Object`

Fitted GLM model

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### qqPlot()

```ts
function qqPlot(model, options?): Object;
```

Defined in: [src/plot/diagnostics.js:86](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L86)

Generate Q-Q plot for normality check

#### Parameters

##### model

`Object`

Fitted GLM model

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### residualsLeveragePlot()

```ts
function residualsLeveragePlot(model, options?): Object;
```

Defined in: [src/plot/diagnostics.js:143](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L143)

Generate residuals vs leverage plot (Cook's distance)

#### Parameters

##### model

`Object`

Fitted GLM model

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### diagnosticDashboard()

```ts
function diagnosticDashboard(model, options?): Object[];
```

Defined in: [src/plot/diagnostics.js:192](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L192)

Generate all diagnostic plots in a dashboard

#### Parameters

##### model

`Object`

Fitted GLM model

##### options?

`Object` = `{}`

Options for individual plots

#### Returns

`Object`[]

Array of Plot specifications

***

### effectPlot()

```ts
function effectPlot(
   model, 
   variable, 
   data, 
   options?): Object;
```

Defined in: [src/plot/diagnostics.js:209](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L209)

Generate effect plot for a specific predictor

#### Parameters

##### model

`Object`

Fitted GLM model

##### variable

`string`

Variable name

##### data

`Object`

Original data

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### partialResidualPlot()

```ts
function partialResidualPlot(
   model, 
   variable, 
   X, 
   options?): Object;
```

Defined in: [src/plot/diagnostics.js:276](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/diagnostics.js#L276)

Generate partial residual plot (component + residual plot)

#### Parameters

##### model

`Object`

Fitted GLM model

##### variable

`string`

Variable name

##### X

`any`[]

Original predictor matrix

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### ordiplot()

```ts
function ordiplot(result, options?): Object;
```

Defined in: [src/plot/ordiplot.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/ordiplot.js#L32)

Generate unified ordination plot configuration
Works with PCA, LDA, and RDA results

#### Parameters

##### result

`Object`

Ordination result (from PCA, LDA, or RDA)

##### options?

Configuration options

###### type

`string` = `null`

Type of ordination ('pca', 'lda', 'rda') - auto-detected if not specified

###### colorBy

`string` \| `Object` \| `any`[] \| `Iterable`\<`any`, `any`, `any`\> \| `null` = `null`

Group values for points: an array,
  any iterable (e.g. an Arquero column), a { data, column } descriptor, or the name of a
  column in the data the model was fit on (requires a declarative fit({ data, ... }))

###### labels

`string` \| `Object` \| `any`[] \| `Iterable`\<`any`, `any`, `any`\> \| `null` = `null`

Labels for points (same accepted forms as colorBy)

###### showLoadings

`boolean` = `true`

Show loading vectors (PCA/RDA only)

###### showCentroids

`boolean` = `false`

Show class centroids (LDA only)

###### showConvexHulls

`boolean` = `false`

Show convex hulls around groups (optional)

###### axis1

`number` = `1`

First axis to plot (default: 1)

###### axis2

`number` = `2`

Second axis to plot (default: 2)

###### width

`number` = `640`

Plot width (default: 640)

###### height

`number` = `400`

Plot height (default: 400)

###### loadingScale

`number` = `3`

Scale factor for loading vectors (default: 3)

###### loadingFactor

`number` = `1`

Multiplier applied to loading vectors (default: 1, set 0 for auto)

###### predictorFactor

`number` \| `null` = `null`

Multiplier for predictor arrows (RDA only, default: inherits loadingFactor; set 0 for auto)

#### Returns

`Object`

Plot configuration

***

### plotHCA()

```ts
function plotHCA(result): Object;
```

Defined in: [src/plot/plotHCA.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHCA.js#L13)

Generate dendrogram data structure

#### Parameters

##### result

`Object`

HCA result from ml.hca.fit()

#### Returns

`Object`

Dendrogram tree structure

***

### dendrogramLayout()

```ts
function dendrogramLayout(dendrogramData, options?): Object;
```

Defined in: [src/plot/plotHCA.js:59](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHCA.js#L59)

Convert dendrogram to layout coordinates

#### Parameters

##### dendrogramData

`Object`

Result from plotHCA

##### options?

`Object` = `{}`

{width, height, orientation}

#### Returns

`Object`

Layout with coordinates

***

### plotCondensedTree()

```ts
function plotCondensedTree(model, options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:17](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L17)

Generate condensed cluster tree visualization configuration

#### Parameters

##### model

`Object`

HDBSCAN model or result from hdbscan.fit()

##### options?

Visualization options

###### width?

`number` = `800`

Plot width

###### height?

`number` = `600`

Plot height

###### showStability?

`boolean` = `true`

Show cluster stability scores

#### Returns

`Object`

Observable Plot-compatible configuration

***

### plotHDBSCANDendrogram()

```ts
function plotHDBSCANDendrogram(model, options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:86](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L86)

Generate dendrogram visualization from HDBSCAN hierarchy
Similar to HCA dendrogram but for HDBSCAN

#### Parameters

##### model

`Object`

HDBSCAN model or result from hdbscan.fit()

##### options?

`Object` = `{}`

Visualization options

#### Returns

`Object`

Dendrogram configuration

***

### plotClusterMembership()

```ts
function plotClusterMembership(
   model, 
   data?, 
   options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:146](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L146)

Visualize cluster membership probabilities

#### Parameters

##### model

`Object`

HDBSCAN model or result from hdbscan.fit()

##### data?

`number`[][] = `null`

Original data for scatter plot (optional)

##### options?

Visualization options

###### width?

`number` = `720`

Plot width

###### height?

`number` = `480`

Plot height

###### showNoise?

`boolean` = `true`

Show noise points

###### columns?

`string`[] = `...`

Column names for 2D projection

#### Returns

`Object`

Observable Plot-compatible configuration

***

### plotClusterStability()

```ts
function plotClusterStability(model, options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:242](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L242)

Visualize cluster stability and persistence

#### Parameters

##### model

`Object`

HDBSCAN model or result from hdbscan.fit()

##### options?

Visualization options

###### width?

`number` = `600`

Plot width

###### height?

`number` = `400`

Plot height

#### Returns

`Object`

Observable Plot-compatible configuration

***

### plotHDBSCANDashboard()

```ts
function plotHDBSCANDashboard(
   model, 
   data, 
   options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:301](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L301)

Create a comprehensive HDBSCAN visualization dashboard

#### Parameters

##### model

`Object`

HDBSCAN model

##### data

`number`[][]

Original data

##### options?

`Object` = `{}`

Visualization options

#### Returns

`Object`

Dashboard configuration with multiple plots

***

### plotHDBSCAN()

```ts
function plotHDBSCAN(model, options?): Object;
```

Defined in: [src/plot/plotHDBSCAN.js:313](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotHDBSCAN.js#L313)

#### Parameters

##### model

`any`

##### options?

#### Returns

`Object`

***

### plotScree()

```ts
function plotScree(result, options?): Object;
```

Defined in: [src/plot/plotScree.js:11](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotScree.js#L11)

Generate scree plot configuration for PCA/ordination results
Shows variance explained by each component

#### Parameters

##### result

`Object`

PCA/LDA/RDA result with varianceExplained

##### options?

`Object` = `{}`

{width, height, cumulative}

#### Returns

`Object`

Plot configuration

***

### plotSilhouette()

```ts
function plotSilhouette(options?, __namedParameters?): Object;
```

Defined in: [src/plot/plotSilhouette.js:73](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/plotSilhouette.js#L73)

Generate silhouette plot configuration displaying per-sample scores.
Accepts either precomputed samples or raw data/labels for convenience.

#### Parameters

##### options?

###### samples?

`any`[]

Output from ml.silhouette.silhouetteSamples()

###### data?

`any`[]

Data matrix used to compute silhouette scores

###### labels?

`any`[]

Cluster labels for each observation

###### sorted?

`boolean`

Whether to sort samples by silhouette desc

###### minSilhouette?

`number`

Minimum silhouette value displayed

###### maxSilhouette?

`number`

Maximum silhouette value displayed

###### clusterOptions?

`Object`

Options for cluster summary inset

##### \_\_namedParameters?

###### width?

`number` = `720`

###### height?

`number` = `420`

###### minSilhouette?

`number` = `-1`

###### maxSilhouette?

`number` = `1`

###### clusterInsetWidth?

`number` = `160`

###### clusterInsetHeight?

`number` = `160`

###### showAverageLines?

`boolean` = `true`

#### Returns

`Object`

Observable Plot-compatible configuration with `.show()`

***

### createD3DendrogramRenderer()

```ts
function createD3DendrogramRenderer(d3, options?): Function;
```

Defined in: [src/plot/renderers/d3Dendrogram.js:52](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/renderers/d3Dendrogram.js#L52)

Build a dendrogram renderer that relies on user-supplied D3 modules for scaling.
The returned function is compatible with the `.show(renderer)` helper emitted by plotHCA.

Usage:
  import { plotHCA } from '@tangent.to/ds/plot';
  import { createD3DendrogramRenderer } from '@tangent.to/ds/plot/renderers/d3Dendrogram.js';
  const spec = plotHCA(model);
  const svg = spec.show(createD3DendrogramRenderer(d3));

#### Parameters

##### d3

`Object`

D3 namespace (only `scaleLinear` is used if available)

##### options?

`Object` = `{}`

Renderer options

#### Returns

`Function`

Renderer function accepted by config.show()

***

### resolveGroupValues()

```ts
function resolveGroupValues(
   spec, 
   result?, 
   name?): any[] | null;
```

Defined in: [src/plot/utils.js:26](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L26)

Normalize a colorBy (or labels) specification into a plain array of
per-observation values.

Accepted forms:
- an array (used as-is)
- any iterable, e.g. an Arquero column or a typed array (converted)
- a { data, column } descriptor (column extracted from table-like data)
- a column-name string, resolved against the source rows the model kept
  from a declarative fit (e.g. pca.fit({ data, columns }) stores the
  naOmit-filtered rows so values stay aligned with the scores)

#### Parameters

##### spec

`any`

colorBy specification

##### result?

`Object` \| `null`

Fitted model (for string column lookup)

##### name?

`string` = `'colorBy'`

Option name used in error messages

#### Returns

`any`[] \| `null`

Array of values, or null when spec is null

***

### plotFeatureImportance()

```ts
function plotFeatureImportance(importances, options?): Object;
```

Defined in: [src/plot/utils.js:68](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L68)

Generate feature importance bar plot configuration

#### Parameters

##### importances

`Object`[]

Feature importance array

##### options?

`Object` = `{}`

{width, height, topN}

#### Returns

`Object`

Plot configuration

***

### plotPartialDependence()

```ts
function plotPartialDependence(pdResult, options?): Object;
```

Defined in: [src/plot/utils.js:111](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L111)

Generate partial dependence plot configuration

#### Parameters

##### pdResult

`Object`

Result from partialDependence()

##### options?

`Object` = `{}`

{width, height, featureName}

#### Returns

`Object`

Plot configuration

***

### plotCorrelationMatrix()

```ts
function plotCorrelationMatrix(corrResult, options?): Object;
```

Defined in: [src/plot/utils.js:161](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L161)

Generate correlation matrix heatmap configuration

#### Parameters

##### corrResult

`Object`

Result from correlationMatrix()

##### options?

`Object` = `{}`

{width, height}

#### Returns

`Object`

Plot configuration

***

### plotResiduals()

```ts
function plotResiduals(residualData, options?): Object;
```

Defined in: [src/plot/utils.js:221](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L221)

Generate residual plot configuration

#### Parameters

##### residualData

`Object`

Result from residualPlotData()

##### options?

`Object` = `{}`

{width, height, standardized}

#### Returns

`Object`

Plot configuration

***

### plotQQ()

```ts
function plotQQ(residualData, options?): Object;
```

Defined in: [src/plot/utils.js:271](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L271)

Generate Q-Q plot configuration for normality check

#### Parameters

##### residualData

`Object`

Result from residualPlotData()

##### options?

`Object` = `{}`

{width, height}

#### Returns

`Object`

Plot configuration

***

### plotLearningCurve()

```ts
function plotLearningCurve(lcResult, options?): Object;
```

Defined in: [src/plot/utils.js:355](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/plot/utils.js#L355)

Generate learning curve plot configuration

#### Parameters

##### lcResult

`Object`

Result from learningCurve()

##### options?

`Object` = `{}`

{width, height}

#### Returns

`Object`

Plot configuration
