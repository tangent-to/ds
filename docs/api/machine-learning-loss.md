---
layout: default
title: loss
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/loss
---
# loss

## Functions

### mseLoss()

```ts
function mseLoss(yTrue, yPred): number;
```

Defined in: [src/ml/loss.js:13](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L13)

Mean Squared Error Loss

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

***

### maeLoss()

```ts
function maeLoss(yTrue, yPred): number;
```

Defined in: [src/ml/loss.js:33](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L33)

Mean Absolute Error Loss

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

***

### logLoss()

```ts
function logLoss(
   yTrue, 
   yPred, 
   epsilon?): number;
```

Defined in: [src/ml/loss.js:53](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L53)

Binary Cross-Entropy Loss (Log Loss)

#### Parameters

##### yTrue

`number`[]

True labels (0 or 1)

##### yPred

`number`[]

Predicted probabilities

##### epsilon?

`number` = `1e-15`

Small value to avoid log(0)

#### Returns

`number`

***

### crossEntropy()

```ts
function crossEntropy(
   yTrue, 
   yPred, 
   epsilon?): number;
```

Defined in: [src/ml/loss.js:77](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L77)

Categorical Cross-Entropy Loss

#### Parameters

##### yTrue

`number`[][]

One-hot encoded true labels

##### yPred

`number`[][]

Predicted probabilities

##### epsilon?

`number` = `1e-15`

Small value to avoid log(0)

#### Returns

`number`

***

### hingeLoss()

```ts
function hingeLoss(yTrue, yPred): number;
```

Defined in: [src/ml/loss.js:100](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L100)

Hinge Loss (for SVM)

#### Parameters

##### yTrue

`number`[]

True labels (-1 or 1)

##### yPred

`number`[]

Predicted scores

#### Returns

`number`

***

### huberLoss()

```ts
function huberLoss(
   yTrue, 
   yPred, 
   delta?): number;
```

Defined in: [src/ml/loss.js:120](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L120)

Huber Loss (robust to outliers)

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

##### delta?

`number` = `1.0`

Threshold for switching from quadratic to linear

#### Returns

`number`

***

### getLossFunction()

```ts
function getLossFunction(name): Function;
```

Defined in: [src/ml/loss.js:143](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/ml/loss.js#L143)

Get loss function by name

#### Parameters

##### name

`string`

Loss function name

#### Returns

`Function`

Loss function
