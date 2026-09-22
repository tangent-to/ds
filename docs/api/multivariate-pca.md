---
layout: default
title: pca
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/pca
---
# pca

## Functions

### fit()

```ts
function fit(X, options?): Object;
```

Defined in: [src/mva/pca.js:62](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/mva/pca.js#L62)

Fit PCA model

#### Parameters

##### X

`number`[][] \| [`Matrix`](/api/core/linalg#matrix)

Data matrix (n x p)

##### options?

`Object` = `{}`

{scale: boolean, center: boolean}

#### Returns

`Object`

PCA model

***

### transform()

```ts
function transform(model, X): Object[];
```

Defined in: [src/mva/pca.js:255](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/mva/pca.js#L255)

Transform new data using fitted PCA model

#### Parameters

##### model

`Object`

Fitted PCA model

##### X

`number`[][]

New data

#### Returns

`Object`[]

Transformed scores

***

### cumulativeVariance()

```ts
function cumulativeVariance(model): number[];
```

Defined in: [src/mva/pca.js:316](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/mva/pca.js#L316)

Get cumulative variance explained

#### Parameters

##### model

`Object`

Fitted PCA model

#### Returns

`number`[]

Cumulative variance explained
