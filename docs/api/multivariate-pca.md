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

Defined in: [src/mva/pca.js:62](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/pca.js#L62)

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

Defined in: [src/mva/pca.js:247](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/pca.js#L247)

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

Defined in: [src/mva/pca.js:308](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/pca.js#L308)

Get cumulative variance explained

#### Parameters

##### model

`Object`

Fitted PCA model

#### Returns

`number`[]

Cumulative variance explained
