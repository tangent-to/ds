---
layout: default
title: dbscan
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/dbscan
---
# dbscan

## Functions

### fit()

```ts
function fit(X, options?): Object;
```

Defined in: [src/ml/dbscan.js:108](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/dbscan.js#L108)

Fit DBSCAN clustering model

#### Parameters

##### X

`any`

Data matrix (n samples × d features)

##### options?

`Object` = `{}`

{eps: neighborhood radius, minSamples: min points for core}

#### Returns

`Object`

{labels, nClusters, nNoise, coreSampleIndices}

***

### predict()

```ts
function predict(
   model, 
   X, 
   X_train, 
   eps): number[];
```

Defined in: [src/ml/dbscan.js:205](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/dbscan.js#L205)

Predict cluster labels for new data points
Note: DBSCAN doesn't naturally support prediction on new points.
This implementation assigns new points to the cluster of their nearest core point
if within eps distance, otherwise marks as noise.

#### Parameters

##### model

`Object`

Fitted model from fit()

##### X

`number`[][]

New data points

##### X\_train

`number`[][]

Original training data

##### eps

`number`

Maximum distance for neighborhood

#### Returns

`number`[]

Cluster labels
