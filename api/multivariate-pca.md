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

Defined in: [src/mva/pca.js:62](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/pca.js#L62)

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

Defined in: [src/mva/pca.js:241](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/pca.js#L241)

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

Defined in: [src/mva/pca.js:302](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/pca.js#L302)

Get cumulative variance explained

#### Parameters

##### model

`Object`

Fitted PCA model

#### Returns

`number`[]

Cumulative variance explained
