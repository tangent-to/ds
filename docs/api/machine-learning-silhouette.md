---
layout: default
title: silhouette
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/silhouette
---
# silhouette

## Functions

### silhouetteSamples()

```ts
function silhouetteSamples(X, labels): object[];
```

Defined in: [src/ml/silhouette.js:60](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/silhouette.js#L60)

Compute the silhouette value for each sample

#### Parameters

##### X

  \| `number`[][]
  \| \{
  `data`: `number`[][];
  `columns`: `string`[];
\}

Data matrix (n × p) or an options object with data/columns

##### labels

`number`[] \| `string`[]

Cluster label for each sample (length n)

#### Returns

`object`[]

Per-sample silhouette records

***

### silhouetteByCluster()

```ts
function silhouetteByCluster(X, labels): object[];
```

Defined in: [src/ml/silhouette.js:130](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/silhouette.js#L130)

Group per-sample silhouette values by cluster and compute cluster averages

#### Parameters

##### X

  \| `number`[][]
  \| \{
  `data`: `number`[][];
  `columns`: `string`[];
\}

Data matrix (n × p) or an options object with data/columns

##### labels

`number`[] \| `string`[]

Cluster label for each sample (length n)

#### Returns

`object`[]

Clusters sorted by descending average silhouette
