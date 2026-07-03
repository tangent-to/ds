---
layout: default
title: interpret
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/interpret
---
# interpret

## Functions

### featureImportance()

```ts
function featureImportance(
   model, 
   X, 
   y, 
   scoreFn, 
   options?): Object[];
```

Defined in: [src/ml/interpret.js:18](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L18)

Compute feature importance via permutation

#### Parameters

##### model

`Object`

Fitted model with predict method

##### X

`number`[][]

Feature matrix

##### y

`number`[]

Target values

##### scoreFn

`Function`

Scoring function (yTrue, yPred) => score

##### options?

`Object` = `{}`

{nRepeats, seed}

#### Returns

`Object`[]

Feature importance scores

***

### coefficientImportance()

```ts
function coefficientImportance(model, featureNames?): Object[];
```

Defined in: [src/ml/interpret.js:71](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L71)

Compute coefficient-based feature importance (for linear models)

#### Parameters

##### model

`Object`

Fitted linear model with coefficients

##### featureNames?

`string`[] = `null`

Feature names (optional)

#### Returns

`Object`[]

Feature importance based on coefficients

***

### partialDependence()

```ts
function partialDependence(
   model, 
   X, 
   feature, 
   options?): Object;
```

Defined in: [src/ml/interpret.js:101](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L101)

Compute partial dependence for a feature

#### Parameters

##### model

`Object`

Fitted model with predict method

##### X

`number`[][]

Feature matrix

##### feature

`number`

Feature index

##### options?

`Object` = `{}`

{gridSize, percentiles}

#### Returns

`Object`

{values, predictions}

***

### residualPlotData()

```ts
function residualPlotData(
   model, 
   X, 
   y): Object;
```

Defined in: [src/ml/interpret.js:155](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L155)

Compute residual plot data

#### Parameters

##### model

`Object`

Fitted model with predict method

##### X

`number`[][]

Feature matrix

##### y

`number`[]

Target values

#### Returns

`Object`

{fitted, residuals, standardized}

***

### correlationMatrix()

```ts
function correlationMatrix(X, featureNames?): Object;
```

Defined in: [src/ml/interpret.js:185](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L185)

Compute correlation matrix

#### Parameters

##### X

`number`[][]

Feature matrix

##### featureNames?

`string`[] = `null`

Feature names (optional)

#### Returns

`Object`

{matrix, features}

***

### learningCurve()

```ts
function learningCurve(
   fitFn, 
   scoreFn, 
   X, 
   y, 
   options?): Object;
```

Defined in: [src/ml/interpret.js:244](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/interpret.js#L244)

Learning curve data (performance vs training size)

#### Parameters

##### fitFn

`Function`

Function to fit model: (X, y) => model

##### scoreFn

`Function`

Scoring function: (yTrue, yPred) => score

##### X

`number`[][]

Feature matrix

##### y

`any`[]

Target values

##### options?

`Object` = `{}`

{trainSizes, cv}

#### Returns

`Object`

{trainSizes, trainScores, testScores}
