---
layout: default
title: hca
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/hca
---
# hca

## Functions

### fit()

```ts
function fit(X, options?): Object;
```

Defined in: [src/ml/hca.js:132](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/hca.js#L132)

Fit hierarchical clustering

#### Parameters

##### X

`number`[][]

Data matrix

##### options?

`Object` = `{}`

{linkage: 'single'|'complete'|'average'|'ward'}

#### Returns

`Object`

{dendrogram, distances}

***

### cut()

```ts
function cut(model, k): number[];
```

Defined in: [src/ml/hca.js:233](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/hca.js#L233)

Cut a hierarchical clustering dendrogram into k flat clusters

#### Parameters

##### model

`Object`

Fitted HCA model with { dendrogram, linkage, n }

##### k

`number`

Desired number of clusters (1 ≤ k ≤ n)

#### Returns

`number`[]

Cluster label for each of the n samples

***

### cutHeight()

```ts
function cutHeight(model, height): number[];
```

Defined in: [src/ml/hca.js:287](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/hca.js#L287)

Cut a hierarchical clustering dendrogram at a given merge height

#### Parameters

##### model

`Object`

Fitted HCA model with { dendrogram, linkage, n }

##### height

`number`

Non-negative distance threshold; merges with distance above it are not applied

#### Returns

`number`[]

Cluster label for each of the n samples
