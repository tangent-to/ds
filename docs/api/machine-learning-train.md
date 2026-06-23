---
layout: default
title: train
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/train
---
# train

## Functions

### train()

```ts
function train(
   model, 
   X, 
   y, 
   options?): Object;
```

Defined in: [src/ml/train.js:17](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/train.js#L17)

Train a model with gradient-based optimization

#### Parameters

##### model

`Object`

Model with forward and backward methods

##### X

`number`[][]

Feature matrix

##### y

`number`[] \| `number`[][]

Target values

##### options?

`Object` = `{}`

Training options

#### Returns

`Object`

Training history

***

### trainFunction()

```ts
function trainFunction(
   lossFn, 
   params0, 
   options?): Object;
```

Defined in: [src/ml/train.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/train.js#L148)

Simple training loop for functions (not models)

#### Parameters

##### lossFn

`Function`

Loss function that returns {loss, gradient}

##### params0

`number`[]

Initial parameters

##### options?

`Object` = `{}`

Training options

#### Returns

`Object`

{params, history}

***

### earlyStopping()

```ts
function earlyStopping(patience?, minDelta?): Object;
```

Defined in: [src/ml/train.js:185](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/train.js#L185)

Early stopping callback

#### Parameters

##### patience?

`number` = `10`

Number of epochs to wait for improvement

##### minDelta?

`number` = `0`

Minimum change to qualify as improvement

#### Returns

`Object`

Callback object with state

***

### learningRateScheduler()

```ts
function learningRateScheduler(scheduleFn, optimizer): Object;
```

Defined in: [src/ml/train.js:222](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/train.js#L222)

Learning rate scheduler callback

#### Parameters

##### scheduleFn

`Function`

Function (epoch) => learningRate

##### optimizer

`Optimizer`

Optimizer to update

#### Returns

`Object`

Callback object

***

### modelCheckpoint()

```ts
function modelCheckpoint(metric?): Object;
```

Defined in: [src/ml/train.js:239](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/train.js#L239)

Model checkpoint callback

#### Parameters

##### metric?

`string` = `'valLoss'`

Metric to monitor ('loss' or 'valLoss')

#### Returns

`Object`

Callback object with state
