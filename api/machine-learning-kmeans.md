---
layout: default
title: kmeans
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/kmeans
---
# kmeans

## Functions

### fit()

```ts
function fit(X, options?): Object;
```

Defined in: [src/ml/kmeans.js:166](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/kmeans.js#L166)

Fit k-means clustering model

#### Parameters

##### X

`any`

Data matrix (n samples × d features)

##### options?

`Object` = `{}`

{k: number of clusters, maxIter: max iterations, tol: tolerance}

#### Returns

`Object`

{labels, centroids, inertia, iterations, converged}

***

### predict()

```ts
function predict(model, X): number[];
```

Defined in: [src/ml/kmeans.js:268](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/kmeans.js#L268)

Predict cluster labels for new data

#### Parameters

##### model

`Object`

Fitted model from fit()

##### X

`number`[][]

New data points

#### Returns

`number`[]

Cluster labels

***

### silhouetteScore()

```ts
function silhouetteScore(X, labels): number;
```

Defined in: [src/ml/kmeans.js:280](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/kmeans.js#L280)

Compute silhouette score for clustering quality

#### Parameters

##### X

`number`[][]

Data points

##### labels

`number`[]

Cluster assignments

#### Returns

`number`

Silhouette score (range: -1 to 1, higher is better)
