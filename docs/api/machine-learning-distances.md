---
layout: default
title: distances
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/distances
---
# distances

## Functions

### euclidean()

```ts
function euclidean(a, b): number;
```

Defined in: [src/ml/distances.js:12](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L12)

Euclidean distance (L2 norm)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

#### Returns

`number`

Euclidean distance

***

### manhattan()

```ts
function manhattan(a, b): number;
```

Defined in: [src/ml/distances.js:31](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L31)

Manhattan distance (L1 norm, taxicab distance)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

#### Returns

`number`

Manhattan distance

***

### minkowski()

```ts
function minkowski(
   a, 
   b, 
   p?): number;
```

Defined in: [src/ml/distances.js:50](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L50)

Minkowski distance (generalized Lp norm)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

##### p?

`number` = `2`

Order parameter (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev)

#### Returns

`number`

Minkowski distance

***

### chebyshev()

```ts
function chebyshev(a, b): number;
```

Defined in: [src/ml/distances.js:72](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L72)

Chebyshev distance (L∞ norm, maximum metric)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

#### Returns

`number`

Chebyshev distance

***

### cosine()

```ts
function cosine(a, b): number;
```

Defined in: [src/ml/distances.js:93](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L93)

Cosine distance (1 - cosine similarity)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

#### Returns

`number`

Cosine distance (0 = identical direction, 2 = opposite)

***

### hamming()

```ts
function hamming(a, b): number;
```

Defined in: [src/ml/distances.js:126](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L126)

Hamming distance (for categorical/binary data)
Counts the number of positions where elements differ

#### Parameters

##### a

`any`[]

First vector

##### b

`any`[]

Second vector

#### Returns

`number`

Hamming distance

***

### canberra()

```ts
function canberra(a, b): number;
```

Defined in: [src/ml/distances.js:146](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L146)

Canberra distance (weighted version of Manhattan distance)

#### Parameters

##### a

`number`[]

First vector

##### b

`number`[]

Second vector

#### Returns

`number`

Canberra distance

***

### gower()

```ts
function gower(
   a, 
   b, 
   options?): number;
```

Defined in: [src/ml/distances.js:172](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L172)

Gower distance for mixed-type data (numeric + categorical)
Handles missing values gracefully

#### Parameters

##### a

`any`[]

First vector (can contain numbers, strings, or null/undefined)

##### b

`any`[]

Second vector

##### options?

Configuration

###### types

`string`[] = `null`

Array indicating type of each feature: 'numeric' or 'categorical'

###### ranges

`number`[] = `null`

Array of ranges for numeric features (max - min)

#### Returns

`number`

Gower distance (0 = identical, 1 = maximally different)

***

### createGowerDistance()

```ts
function createGowerDistance(data, types?): Function;
```

Defined in: [src/ml/distances.js:231](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L231)

Create a Gower distance function with pre-computed ranges
Useful for KNN when you want to compute ranges once from training data

#### Parameters

##### data

`any`[][]

Training data to compute ranges from

##### types?

`string`[] = `null`

Feature types ('numeric' or 'categorical')

#### Returns

`Function`

Configured Gower distance function

***

### getDistanceFunction()

```ts
function getDistanceFunction(metric): Function;
```

Defined in: [src/ml/distances.js:283](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/distances.js#L283)

Get distance function by name

#### Parameters

##### metric

`string` \| `Function`

Metric name or custom function

#### Returns

`Function`

Distance function
