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
function mseLoss(yTrue, yPred): Object;
```

Defined in: [src/ml/loss.js:12](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L12)

Mean Squared Error Loss

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`Object`

{loss, gradient}

***

### maeLoss()

```ts
function maeLoss(yTrue, yPred): Object;
```

Defined in: [src/ml/loss.js:38](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L38)

Mean Absolute Error Loss

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`Object`

{loss, gradient}

***

### logLoss()

```ts
function logLoss(
   yTrue, 
   yPred, 
   epsilon?): Object;
```

Defined in: [src/ml/loss.js:65](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L65)

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

`Object`

{loss, gradient}

***

### crossEntropy()

```ts
function crossEntropy(
   yTrue, 
   yPred, 
   epsilon?): Object;
```

Defined in: [src/ml/loss.js:96](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L96)

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

`Object`

{loss, gradient}

***

### hingeLoss()

```ts
function hingeLoss(yTrue, yPred): Object;
```

Defined in: [src/ml/loss.js:130](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L130)

Hinge Loss (for SVM)

#### Parameters

##### yTrue

`number`[]

True labels (-1 or 1)

##### yPred

`number`[]

Predicted scores

#### Returns

`Object`

{loss, gradient}

***

### huberLoss()

```ts
function huberLoss(
   yTrue, 
   yPred, 
   delta?): Object;
```

Defined in: [src/ml/loss.js:160](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L160)

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

`Object`

{loss, gradient}

***

### getLossFunction()

```ts
function getLossFunction(name): Function;
```

Defined in: [src/ml/loss.js:195](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/loss.js#L195)

Get loss function by name

#### Parameters

##### name

`string`

Loss function name

#### Returns

`Function`

Loss function
