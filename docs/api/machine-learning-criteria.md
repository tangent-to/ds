---
layout: default
title: criteria
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/criteria
---
# criteria

## Functions

### gini()

```ts
function gini(labels): number;
```

Defined in: [src/ml/criteria.js:15](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L15)

Gini impurity for classification
Measures probability of misclassification
Lower is better (0 = pure node)

#### Parameters

##### labels

`any`[]

Array of labels

#### Returns

`number`

Gini impurity [0, 1]

***

### entropy()

```ts
function entropy(labels): number;
```

Defined in: [src/ml/criteria.js:41](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L41)

Entropy (information gain) for classification
Measures uncertainty/disorder in the data
Lower is better (0 = pure node)

#### Parameters

##### labels

`any`[]

Array of labels

#### Returns

`number`

Entropy [0, log2(n_classes)]

***

### variance()

```ts
function variance(values): number;
```

Defined in: [src/ml/criteria.js:69](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L69)

Variance for regression
Measures spread of continuous values
Lower is better (0 = all values equal)

#### Parameters

##### values

`number`[]

Array of numeric values

#### Returns

`number`

Variance

***

### mse()

```ts
function mse(values): number;
```

Defined in: [src/ml/criteria.js:83](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L83)

Mean Squared Error (MSE) for regression
Alternative to variance, measures prediction error

#### Parameters

##### values

`number`[]

Array of numeric values

#### Returns

`number`

MSE (same as variance for single node)

***

### mae()

```ts
function mae(values): number;
```

Defined in: [src/ml/criteria.js:93](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L93)

Mean Absolute Error (MAE) for regression
Robust alternative to MSE

#### Parameters

##### values

`number`[]

Array of numeric values

#### Returns

`number`

MAE

***

### classificationError()

```ts
function classificationError(labels): number;
```

Defined in: [src/ml/criteria.js:112](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L112)

Classification error (misclassification rate)
Simple impurity measure based on majority class

#### Parameters

##### labels

`any`[]

Array of labels

#### Returns

`number`

Classification error [0, 1]

***

### getCriterionFunction()

```ts
function getCriterionFunction(criterion, task?): Function;
```

Defined in: [src/ml/criteria.js:130](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L130)

Get impurity function by name

#### Parameters

##### criterion

`string` \| `Function`

Criterion name or custom function

##### task?

`string` = `'classification'`

'classification' or 'regression'

#### Returns

`Function`

Impurity function

***

### informationGain()

```ts
function informationGain(
   parentLabels, 
   leftLabels, 
   rightLabels, 
   impurityFn?): number;
```

Defined in: [src/ml/criteria.js:165](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/criteria.js#L165)

Compute information gain (reduction in impurity)

#### Parameters

##### parentLabels

`any`[]

Labels before split

##### leftLabels

`any`[]

Labels in left child

##### rightLabels

`any`[]

Labels in right child

##### impurityFn?

`Function` = `gini`

Impurity function (gini, entropy, etc.)

#### Returns

`number`

Information gain
