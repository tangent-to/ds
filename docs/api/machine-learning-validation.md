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
   options?): Object;
```

Defined in: [src/ml/validation.js:245](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L245)

Split data into train and test sets
Supports both raw matrices and declarative table descriptors

#### Parameters

##### X

`Object` \| `number`[][]

Design matrix (n × p) or a table descriptor ({ data, X, y, ... })

##### y?

`Object` \| `number`[] \| `string`[] \| `null`

Response vector, or options object when X is a table descriptor

##### options?

Split options

###### ratio?

`number`

Fraction of samples assigned to the train set (default 0.8)

###### shuffle?

`boolean`

Whether to shuffle indices before splitting (default true)

###### seed?

`number`

Optional random seed for reproducible shuffling

#### Returns

`Object`

Split result with XTrain/XTest, optional yTrain/yTest, and trainIndices/testIndices (or table views for descriptor input)

***

### kFold()

```ts
function kFold(
   X, 
   y, 
   k?, 
   shuffle?): Object[];
```

Defined in: [src/ml/validation.js:423](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L423)

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

Defined in: [src/ml/validation.js:448](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L448)

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

Defined in: [src/ml/validation.js:475](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L475)

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

Defined in: [src/ml/validation.js:508](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L508)

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

Defined in: [src/ml/validation.js:529](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L529)

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

Defined in: [src/ml/validation.js:573](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/validation.js#L573)

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

`Function`

Fits a model given (XTrain, yTrain) and returns the model

##### scoreFn

`Function`

Scores a model given (model, XTest, yTest) and returns a number

##### X

`Object` \| `number`[][]

Design matrix (n × p) or a table descriptor ({ data, X, y, encoders? })

##### y?

`Object` \| `number`[] \| `string`[] \| `null`

Response vector, or options object when X is a table descriptor

##### folds?

`Object`[] \| `null`

Optional fold definitions (each with trainIndices/testIndices); defaults to k-fold

#### Returns

`object`

Cross-validation results

##### scores

```ts
scores: number[];
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
nFolds: number;
```

##### metadata?

```ts
optional metadata?: Object;
```

##### tableFolds?

```ts
optional tableFolds?: Object[];
```
