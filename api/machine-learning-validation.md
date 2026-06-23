---
layout: default
title: validation
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/validation
---
# validation

## Functions

### trainTestSplit()

```ts
function trainTestSplit(
   X, 
   y?, 
   options?): 
  | {
  XTrain: any[];
  XTest: any[];
  trainIndices: number[];
  testIndices: number[];
  yTrain?: undefined;
  yTest?: undefined;
}
  | {
  XTrain: any[];
  XTest: any[];
  yTrain: any[];
  yTest: any[];
  trainIndices: number[];
  testIndices: number[];
};
```

Defined in: [src/ml/validation.js:238](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L238)

Split data into train and test sets
Supports both raw matrices and declarative table descriptors

#### Parameters

##### X

`any`

##### y?

`null` = `null`

##### options?

#### Returns

  \| \{
  `XTrain`: `any`[];
  `XTest`: `any`[];
  `trainIndices`: `number`[];
  `testIndices`: `number`[];
  `yTrain?`: `undefined`;
  `yTest?`: `undefined`;
\}
  \| \{
  `XTrain`: `any`[];
  `XTest`: `any`[];
  `yTrain`: `any`[];
  `yTest`: `any`[];
  `trainIndices`: `number`[];
  `testIndices`: `number`[];
\}

***

### kFold()

```ts
function kFold(
   X, 
   y, 
   k?, 
   shuffle?): Object[];
```

Defined in: [src/ml/validation.js:416](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L416)

K-Fold cross-validation generator

#### Parameters

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target values

##### k?

`number` = `5`

Number of folds

##### shuffle?

`boolean` = `false`

Whether to shuffle data

#### Returns

`Object`[]

Array of fold objects

***

### stratifiedKFold()

```ts
function stratifiedKFold(
   X, 
   y, 
   k?): Object[];
```

Defined in: [src/ml/validation.js:441](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L441)

Stratified K-Fold for classification with balanced class distribution

#### Parameters

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target labels

##### k?

`number` = `5`

Number of folds

#### Returns

`Object`[]

Array of fold objects

***

### groupKFold()

```ts
function groupKFold(
   X, 
   y, 
   groups, 
   k?): Object[];
```

Defined in: [src/ml/validation.js:468](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L468)

Group K-Fold keeping group membership intact

#### Parameters

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target values

##### groups

`any`[]

Group labels

##### k?

`number` = `5`

Number of folds

#### Returns

`Object`[]

Array of fold objects

***

### leaveOneOut()

```ts
function leaveOneOut(X, _y): Object[];
```

Defined in: [src/ml/validation.js:501](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L501)

Leave-One-Out cross-validation

#### Parameters

##### X

`any`[]

Feature matrix

##### \_y

`any`

#### Returns

`Object`[]

Array of fold objects

***

### shuffleSplit()

```ts
function shuffleSplit(
   X, 
   y, 
   options?): Object[];
```

Defined in: [src/ml/validation.js:522](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L522)

Shuffle Split - repeated random train-test splits

#### Parameters

##### X

`any`[]

Feature matrix

##### y

`any`[]

Target values

##### options?

`Object` = `{}`

{nSplits, testRatio, seed}

#### Returns

`Object`[]

Array of split objects

***

### crossValidate()

```ts
function crossValidate(
   fitFn, 
   scoreFn, 
   X, 
   y?, 
   folds?): object;
```

Defined in: [src/ml/validation.js:559](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/validation.js#L559)

Execute cross-validation with a model.

Array API:
  crossValidate(fitFn, scoreFn, X, y, folds?)

Declarative table API:
  crossValidate(fitFn, scoreFn, { data, X, y, encoders? }, options?)
Options can include { k, shuffle, folds } when using descriptors.

Returns:
  { scores, meanScore, stdScore, nFolds, metadata?, tableFolds? }
When invoked with a descriptor, metadata/tableFolds include the training encoders
and per-fold table views for further inspection.

#### Parameters

##### fitFn

`any`

##### scoreFn

`any`

##### X

`any`

##### y?

`null` = `null`

##### folds?

`null` = `null`

#### Returns

`object`

##### scores

```ts
scores: any[];
```

##### meanScore

```ts
meanScore: number;
```

##### stdScore

```ts
stdScore: number;
```

##### nFolds

```ts
nFolds: any = foldDefs.length;
```
