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

Defined in: [src/ml/hca.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/hca.js#L132)

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
function cut(model, k): any[];
```

Defined in: [src/ml/hca.js:227](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/hca.js#L227)

#### Parameters

##### model

`any`

##### k

`any`

#### Returns

`any`[]

***

### cutHeight()

```ts
function cutHeight(model, height): any[];
```

Defined in: [src/ml/hca.js:275](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/hca.js#L275)

#### Parameters

##### model

`any`

##### height

`any`

#### Returns

`any`[]
