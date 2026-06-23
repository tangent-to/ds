---
layout: default
title: polynomial
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/polynomial
---
# polynomial

## Functions

### polynomialFeatures()

```ts
function polynomialFeatures(X, degree): number[][];
```

Defined in: [src/ml/polynomial.js:30](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/polynomial.js#L30)

Create polynomial features from input

#### Parameters

##### X

`any`

Input features (n × 1 for univariate)

##### degree

`number`

Polynomial degree

#### Returns

`number`[][]

Polynomial features

***

### fit()

```ts
function fit(
   X, 
   y, 
   options?): Object;
```

Defined in: [src/ml/polynomial.js:75](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/polynomial.js#L75)

Fit polynomial regression model

#### Parameters

##### X

`number`[] \| `number`[][]

Input data (can be 1D for univariate)

##### y

`number`[]

Target values

##### options?

`Object` = `{}`

{degree: polynomial degree, intercept: include intercept}

#### Returns

`Object`

{coefficients, degree, fitted, residuals, rSquared}

***

### predict()

```ts
function predict(
   model, 
   X, 
   options?): number[];
```

Defined in: [src/ml/polynomial.js:120](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/polynomial.js#L120)

Predict using polynomial regression model

#### Parameters

##### model

`Object`

Fitted model from fit()

##### X

`number`[] \| `number`[][]

New data

##### options?

`Object` = `{}`

{intercept: boolean}

#### Returns

`number`[]

Predictions
