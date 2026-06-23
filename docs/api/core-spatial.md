---
layout: default
title: spatial
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/spatial
---
# spatial

## Classes

### KDTree

Defined in: [src/core/spatial/kdtree.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L24)

KD-Tree for efficient spatial queries

#### Constructors

##### Constructor

```ts
new KDTree(points, metric?): KDTree;
```

Defined in: [src/core/spatial/kdtree.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L25)

###### Parameters

###### points

`any`

###### metric?

`string` = `'euclidean'`

###### Returns

[`KDTree`](#kdtree)

#### Properties

##### metric

```ts
metric: string;
```

Defined in: [src/core/spatial/kdtree.js:26](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L26)

##### dimensions

```ts
dimensions: any;
```

Defined in: [src/core/spatial/kdtree.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L27)

##### root

```ts
root: any;
```

Defined in: [src/core/spatial/kdtree.js:28](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L28)

#### Methods

##### knn()

```ts
knn(point, k): object[];
```

Defined in: [src/core/spatial/kdtree.js:64](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L64)

Find k nearest neighbors

###### Parameters

###### point

`number`[]

Query point

###### k

`number`

Number of neighbors

###### Returns

`object`[]

##### radiusSearch()

```ts
radiusSearch(point, radius): object[];
```

Defined in: [src/core/spatial/kdtree.js:100](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L100)

Find all neighbors within radius

###### Parameters

###### point

`number`[]

Query point

###### radius

`number`

Search radius

###### Returns

`object`[]

##### \_euclidean()

```ts
_euclidean(a, b): number;
```

Defined in: [src/core/spatial/kdtree.js:151](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L151)

###### Parameters

###### a

`any`

###### b

`any`

###### Returns

`number`

##### \_manhattan()

```ts
_manhattan(a, b): number;
```

Defined in: [src/core/spatial/kdtree.js:159](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L159)

###### Parameters

###### a

`any`

###### b

`any`

###### Returns

`number`

##### \_chebyshev()

```ts
_chebyshev(a, b): number;
```

Defined in: [src/core/spatial/kdtree.js:167](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L167)

###### Parameters

###### a

`any`

###### b

`any`

###### Returns

`number`

## Functions

### buildKDTree()

```ts
function buildKDTree(points, metric?): KDTree;
```

Defined in: [src/core/spatial/kdtree.js:212](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/spatial/kdtree.js#L212)

Build KD-tree from data

#### Parameters

##### points

`number`[][]

Data points

##### metric?

`string` \| `Function`

Distance metric

#### Returns

[`KDTree`](#kdtree)
