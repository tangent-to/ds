---
layout: default
title: tuning
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/tuning
---
# tuning

## Variables

### distributions

```ts
const distributions: object;
```

Defined in: [src/ml/tuning.js:308](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/tuning.js#L308)

Create parameter distribution objects

#### Type Declaration

##### uniform

```ts
uniform: (low, high) => Object;
```

Uniform distribution

###### Parameters

###### low

`number`

###### high

`number`

###### Returns

`Object`

##### loguniform

```ts
loguniform: (low, high) => Object;
```

Log-uniform distribution (for learning rates, etc.)

###### Parameters

###### low

`number`

###### high

`number`

###### Returns

`Object`

##### randint

```ts
randint: (low, high) => Object;
```

Random integer

###### Parameters

###### low

`number`

###### high

`number`

###### Returns

`Object`

##### choice

```ts
choice: (options) => Object;
```

Choice from options

###### Parameters

###### options

`any`[]

###### Returns

`Object`

## Functions

### GridSearchCV()

```ts
function GridSearchCV(
   fitFn, 
   scoreFn, 
   X, 
   y, 
   paramGrid, 
   options?): Object;
```

Defined in: [src/ml/tuning.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/tuning.js#L20)

Grid Search Cross-Validation

#### Parameters

##### fitFn

`Function`

Function (X, y, params) => model

##### scoreFn

`Function`

Function (model, X, y) => score

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target values

##### paramGrid

`Object`

Parameter grid {param1: [values], param2: [values]}

##### options?

`Object` = `{}`

{k, shuffle, metric}

#### Returns

`Object`

{bestParams, bestScore, bestModel, results}

***

### RandomSearchCV()

```ts
function RandomSearchCV(
   fitFn, 
   scoreFn, 
   X, 
   y, 
   paramDistributions, 
   options?): Object;
```

Defined in: [src/ml/tuning.js:123](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/tuning.js#L123)

Random Search Cross-Validation

#### Parameters

##### fitFn

`Function`

Function (X, y, params) => model

##### scoreFn

`Function`

Function (model, X, y) => score

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target values

##### paramDistributions

`Object`

Parameter distributions

##### options?

`Object` = `{}`

{nIter, k, shuffle, seed}

#### Returns

`Object`

{bestParams, bestScore, bestModel, results}
