---
layout: default
title: utils
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/utils
---
# utils

## Functions

### setSeed()

```ts
function setSeed(value): void;
```

Defined in: [src/ml/utils.js:11](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/utils.js#L11)

Set random seed for reproducibility

#### Parameters

##### value

`number`

Seed value

#### Returns

`void`

***

### random()

```ts
function random(): number;
```

Defined in: [src/ml/utils.js:19](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/utils.js#L19)

Seeded random number generator (LCG)

#### Returns

`number`

Random number between 0 and 1

***

### randomInt()

```ts
function randomInt(min, max): number;
```

Defined in: [src/ml/utils.js:30](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/utils.js#L30)

Random integer in range [min, max)

#### Parameters

##### min

`number`

Minimum value (inclusive)

##### max

`number`

Maximum value (exclusive)

#### Returns

`number`

Random integer

***

### shuffle()

```ts
function shuffle(arr): any[];
```

Defined in: [src/ml/utils.js:39](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/utils.js#L39)

Shuffle array in place using seeded random

#### Parameters

##### arr

`any`[]

Array to shuffle

#### Returns

`any`[]

Shuffled array

***

### sample()

```ts
function sample(arr, k): any[];
```

Defined in: [src/ml/utils.js:54](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/utils.js#L54)

Sample without replacement

#### Parameters

##### arr

`any`[]

Array to sample from

##### k

`number`

Number of samples

#### Returns

`any`[]

Sampled elements
