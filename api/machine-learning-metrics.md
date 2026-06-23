---
layout: default
title: metrics
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/metrics
---
# metrics

## Functions

### mse()

```ts
function mse(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:16](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L16)

Mean Squared Error

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

MSE

***

### rmse()

```ts
function rmse(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:34](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L34)

Root Mean Squared Error

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

RMSE

***

### mae()

```ts
function mae(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:44](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L44)

Mean Absolute Error

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

MAE

***

### r2()

```ts
function r2(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:62](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L62)

R² (coefficient of determination)

#### Parameters

##### yTrue

`number`[]

True values

##### yPred

`number`[]

Predicted values

#### Returns

`number`

R²

***

### accuracy()

```ts
function accuracy(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:88](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L88)

Accuracy score

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

#### Returns

`number`

Accuracy

***

### confusionMatrix()

```ts
function confusionMatrix(yTrue, yPred): Object;
```

Defined in: [src/ml/metrics.js:107](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L107)

Confusion matrix

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

#### Returns

`Object`

{matrix, labels}

***

### confusionMatrixText()

```ts
function confusionMatrixText(yTrue, yPred): string;
```

Defined in: [src/ml/metrics.js:129](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L129)

Format confusion matrix as text

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

#### Returns

`string`

Formatted confusion matrix

***

### precision()

```ts
function precision(
   yTrue, 
   yPred, 
   positiveLabel?): number;
```

Defined in: [src/ml/metrics.js:180](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L180)

Precision score (for binary classification)

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

##### positiveLabel?

`any` = `1`

Label to consider as positive

#### Returns

`number`

Precision

***

### recall()

```ts
function recall(
   yTrue, 
   yPred, 
   positiveLabel?): number;
```

Defined in: [src/ml/metrics.js:204](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L204)

Recall score (for binary classification)

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

##### positiveLabel?

`any` = `1`

Label to consider as positive

#### Returns

`number`

Recall

***

### f1()

```ts
function f1(
   yTrue, 
   yPred, 
   positiveLabel?): number;
```

Defined in: [src/ml/metrics.js:228](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L228)

F1 score

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

##### positiveLabel?

`any` = `1`

Label to consider as positive

#### Returns

`number`

F1 score

***

### logLoss()

```ts
function logLoss(
   yTrue, 
   yPred, 
   eps?): number;
```

Defined in: [src/ml/metrics.js:242](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L242)

Log loss (cross-entropy loss)

#### Parameters

##### yTrue

`any`[]

True labels (0 or 1)

##### yPred

`number`[]

Predicted probabilities

##### eps?

`number` = `1e-15`

Small constant to avoid log(0)

#### Returns

`number`

Log loss

***

### rocAuc()

```ts
function rocAuc(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:262](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L262)

ROC AUC score (simplified for binary classification)

#### Parameters

##### yTrue

`any`[]

True labels (0 or 1)

##### yPred

`number`[]

Predicted probabilities

#### Returns

`number`

AUC

***

### cohenKappa()

```ts
function cohenKappa(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:302](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L302)

Cohen's Kappa coefficient

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

#### Returns

`number`

Kappa

***

### adjustedRandIndex()

```ts
function adjustedRandIndex(yTrue, yPred): number;
```

Defined in: [src/ml/metrics.js:331](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/metrics.js#L331)

Adjusted Rand Index

#### Parameters

##### yTrue

`any`[]

True labels

##### yPred

`any`[]

Predicted labels

#### Returns

`number`

Adjusted Rand Index
