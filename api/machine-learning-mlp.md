---
layout: default
title: mlp
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/mlp
---
# mlp

## Functions

### createNetwork()

```ts
function createNetwork(layerSizes, activation?): Object[];
```

Defined in: [src/ml/mlp.js:157](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/mlp.js#L157)

Create MLP architecture

#### Parameters

##### layerSizes

`number`[]

Size of each layer [input, hidden1, ..., output]

##### activation?

`string` = `'relu'`

Activation function ('sigmoid', 'relu', 'tanh')

#### Returns

`Object`[]

Initialized layers

***

### fit()

```ts
function fit(
   X, 
   y, 
   options?): Object;
```

Defined in: [src/ml/mlp.js:192](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/mlp.js#L192)

Train MLP using mini-batch gradient descent

#### Parameters

##### X

`number`[][]

Training data

##### y

`number`[] \| `number`[][]

Target values

##### options?

`Object` = `{}`

Training options

#### Returns

`Object`

{layers, losses, epochs}

***

### predict()

```ts
function predict(model, X): number[][];
```

Defined in: [src/ml/mlp.js:335](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/mlp.js#L335)

Predict using trained MLP

#### Parameters

##### model

`Object`

Trained model from fit()

##### X

`number`[] \| `number`[][]

Input data

#### Returns

`number`[][]

Predictions

***

### evaluate()

```ts
function evaluate(
   model, 
   X, 
   y): Object;
```

Defined in: [src/ml/mlp.js:357](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/mlp.js#L357)

Evaluate model on test data

#### Parameters

##### model

`Object`

Trained model

##### X

`number`[][]

Test inputs

##### y

`number`[] \| `number`[][]

Test targets

#### Returns

`Object`

{mse, mae}
