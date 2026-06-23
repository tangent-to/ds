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

Defined in: [src/ml/criteria.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L13)

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

Defined in: [src/ml/criteria.js:39](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L39)

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

Defined in: [src/ml/criteria.js:67](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L67)

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

Defined in: [src/ml/criteria.js:87](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L87)

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

Defined in: [src/ml/criteria.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L97)

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

Defined in: [src/ml/criteria.js:116](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L116)

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

Defined in: [src/ml/criteria.js:134](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L134)

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

Defined in: [src/ml/criteria.js:169](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/criteria.js#L169)

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
