---
layout: default
title: optimize
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/optimize
---
# optimize

## Classes

### GradientDescent

Defined in: [src/core/optimize.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L44)

Gradient Descent Optimizer (Batch and Stochastic)

#### Extends

- `Optimizer`

#### Constructors

##### Constructor

```ts
new GradientDescent(options?): GradientDescent;
```

Defined in: [src/core/optimize.js:45](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L45)

###### Parameters

###### options?

###### Returns

[`GradientDescent`](#gradientdescent)

###### Overrides

```ts
Optimizer.constructor
```

#### Properties

##### learningRate

```ts
learningRate: any;
```

Defined in: [src/core/optimize.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L13)

###### Inherited from

[`MomentumOptimizer`](#momentumoptimizer).[`learningRate`](#learningrate-1)

##### maxIter

```ts
maxIter: any;
```

Defined in: [src/core/optimize.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L14)

###### Inherited from

[`MomentumOptimizer`](#momentumoptimizer).[`maxIter`](#maxiter-1)

##### tol

```ts
tol: any;
```

Defined in: [src/core/optimize.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L15)

###### Inherited from

[`MomentumOptimizer`](#momentumoptimizer).[`tol`](#tol-1)

##### verbose

```ts
verbose: any;
```

Defined in: [src/core/optimize.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L16)

###### Inherited from

[`MomentumOptimizer`](#momentumoptimizer).[`verbose`](#verbose-1)

##### stochastic

```ts
stochastic: any;
```

Defined in: [src/core/optimize.js:47](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L47)

##### batchSize

```ts
batchSize: any;
```

Defined in: [src/core/optimize.js:48](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L48)

##### lineSearch

```ts
lineSearch: any;
```

Defined in: [src/core/optimize.js:49](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L49)

#### Methods

##### checkConvergence()

```ts
checkConvergence(gradient): boolean;
```

Defined in: [src/core/optimize.js:35](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L35)

Check convergence based on gradient norm

###### Parameters

###### gradient

`number`[]

###### Returns

`boolean`

###### Inherited from

```ts
Optimizer.checkConvergence
```

##### minimize()

```ts
minimize(
   lossFn, 
   x0, 
   options?): object;
```

Defined in: [src/core/optimize.js:52](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L52)

Minimize a loss function

###### Parameters

###### lossFn

`any`

Function that returns {loss, gradient}

###### x0

`any`

Initial parameters

###### options?

Additional options

###### Returns

`object`

{x, history}

###### x

```ts
x: any[];
```

###### history

```ts
history: object;
```

###### history.loss

```ts
loss: never[] = [];
```

###### history.gradNorm

```ts
gradNorm: never[] = [];
```

###### history.learningRate

```ts
learningRate: never[] = [];
```

###### Overrides

```ts
Optimizer.minimize
```

##### backtrackingLineSearch()

```ts
backtrackingLineSearch(
   lossFn, 
   x, 
   gradient, 
   currentLoss): number;
```

Defined in: [src/core/optimize.js:108](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L108)

Backtracking line search

###### Parameters

###### lossFn

`Function`

###### x

`number`[]

###### gradient

`number`[]

###### currentLoss

`number`

###### Returns

`number`

Step size

***

### MomentumOptimizer

Defined in: [src/core/optimize.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L139)

Momentum Optimizer

#### Extends

- `Optimizer`

#### Constructors

##### Constructor

```ts
new MomentumOptimizer(options?): MomentumOptimizer;
```

Defined in: [src/core/optimize.js:140](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L140)

###### Parameters

###### options?

###### Returns

[`MomentumOptimizer`](#momentumoptimizer)

###### Overrides

```ts
Optimizer.constructor
```

#### Properties

##### learningRate

```ts
learningRate: any;
```

Defined in: [src/core/optimize.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L13)

###### Inherited from

```ts
Optimizer.learningRate
```

##### maxIter

```ts
maxIter: any;
```

Defined in: [src/core/optimize.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L14)

###### Inherited from

```ts
Optimizer.maxIter
```

##### tol

```ts
tol: any;
```

Defined in: [src/core/optimize.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L15)

###### Inherited from

```ts
Optimizer.tol
```

##### verbose

```ts
verbose: any;
```

Defined in: [src/core/optimize.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L16)

###### Inherited from

```ts
Optimizer.verbose
```

##### momentum

```ts
momentum: any;
```

Defined in: [src/core/optimize.js:142](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L142)

#### Methods

##### checkConvergence()

```ts
checkConvergence(gradient): boolean;
```

Defined in: [src/core/optimize.js:35](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L35)

Check convergence based on gradient norm

###### Parameters

###### gradient

`number`[]

###### Returns

`boolean`

###### Inherited from

```ts
Optimizer.checkConvergence
```

##### minimize()

```ts
minimize(
   lossFn, 
   x0, 
   options?): object;
```

Defined in: [src/core/optimize.js:145](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L145)

Minimize a loss function

###### Parameters

###### lossFn

`any`

Function that returns {loss, gradient}

###### x0

`any`

Initial parameters

###### options?

Additional options

###### Returns

`object`

{x, history}

###### x

```ts
x: any[];
```

###### history

```ts
history: object;
```

###### history.loss

```ts
loss: never[] = [];
```

###### history.gradNorm

```ts
gradNorm: never[] = [];
```

###### Overrides

```ts
Optimizer.minimize
```

***

### RMSProp

Defined in: [src/core/optimize.js:189](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L189)

RMSProp Optimizer

#### Extends

- `Optimizer`

#### Constructors

##### Constructor

```ts
new RMSProp(options?): RMSProp;
```

Defined in: [src/core/optimize.js:190](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L190)

###### Parameters

###### options?

###### Returns

[`RMSProp`](#rmsprop)

###### Overrides

```ts
Optimizer.constructor
```

#### Properties

##### learningRate

```ts
learningRate: any;
```

Defined in: [src/core/optimize.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L13)

###### Inherited from

```ts
Optimizer.learningRate
```

##### maxIter

```ts
maxIter: any;
```

Defined in: [src/core/optimize.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L14)

###### Inherited from

```ts
Optimizer.maxIter
```

##### tol

```ts
tol: any;
```

Defined in: [src/core/optimize.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L15)

###### Inherited from

```ts
Optimizer.tol
```

##### verbose

```ts
verbose: any;
```

Defined in: [src/core/optimize.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L16)

###### Inherited from

```ts
Optimizer.verbose
```

##### decay

```ts
decay: any;
```

Defined in: [src/core/optimize.js:192](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L192)

##### epsilon

```ts
epsilon: any;
```

Defined in: [src/core/optimize.js:193](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L193)

#### Methods

##### checkConvergence()

```ts
checkConvergence(gradient): boolean;
```

Defined in: [src/core/optimize.js:35](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L35)

Check convergence based on gradient norm

###### Parameters

###### gradient

`number`[]

###### Returns

`boolean`

###### Inherited from

```ts
Optimizer.checkConvergence
```

##### minimize()

```ts
minimize(
   lossFn, 
   x0, 
   options?): object;
```

Defined in: [src/core/optimize.js:196](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L196)

Minimize a loss function

###### Parameters

###### lossFn

`any`

Function that returns {loss, gradient}

###### x0

`any`

Initial parameters

###### options?

Additional options

###### Returns

`object`

{x, history}

###### x

```ts
x: any[];
```

###### history

```ts
history: object;
```

###### history.loss

```ts
loss: never[] = [];
```

###### history.gradNorm

```ts
gradNorm: never[] = [];
```

###### Overrides

```ts
Optimizer.minimize
```

***

### AdamOptimizer

Defined in: [src/core/optimize.js:240](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L240)

Adam Optimizer (Adaptive Moment Estimation)

#### Extends

- `Optimizer`

#### Constructors

##### Constructor

```ts
new AdamOptimizer(options?): AdamOptimizer;
```

Defined in: [src/core/optimize.js:241](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L241)

###### Parameters

###### options?

###### Returns

[`AdamOptimizer`](#adamoptimizer)

###### Overrides

```ts
Optimizer.constructor
```

#### Properties

##### learningRate

```ts
learningRate: any;
```

Defined in: [src/core/optimize.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L13)

###### Inherited from

```ts
Optimizer.learningRate
```

##### maxIter

```ts
maxIter: any;
```

Defined in: [src/core/optimize.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L14)

###### Inherited from

```ts
Optimizer.maxIter
```

##### tol

```ts
tol: any;
```

Defined in: [src/core/optimize.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L15)

###### Inherited from

```ts
Optimizer.tol
```

##### verbose

```ts
verbose: any;
```

Defined in: [src/core/optimize.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L16)

###### Inherited from

```ts
Optimizer.verbose
```

##### beta1

```ts
beta1: any;
```

Defined in: [src/core/optimize.js:243](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L243)

##### beta2

```ts
beta2: any;
```

Defined in: [src/core/optimize.js:244](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L244)

##### epsilon

```ts
epsilon: any;
```

Defined in: [src/core/optimize.js:245](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L245)

#### Methods

##### checkConvergence()

```ts
checkConvergence(gradient): boolean;
```

Defined in: [src/core/optimize.js:35](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L35)

Check convergence based on gradient norm

###### Parameters

###### gradient

`number`[]

###### Returns

`boolean`

###### Inherited from

```ts
Optimizer.checkConvergence
```

##### minimize()

```ts
minimize(
   lossFn, 
   x0, 
   options?): object;
```

Defined in: [src/core/optimize.js:248](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L248)

Minimize a loss function

###### Parameters

###### lossFn

`any`

Function that returns {loss, gradient}

###### x0

`any`

Initial parameters

###### options?

Additional options

###### Returns

`object`

{x, history}

###### x

```ts
x: any[];
```

###### history

```ts
history: object;
```

###### history.loss

```ts
loss: never[] = [];
```

###### history.gradNorm

```ts
gradNorm: never[] = [];
```

###### Overrides

```ts
Optimizer.minimize
```

## Functions

### createOptimizer()

```ts
function createOptimizer(name, options?): Optimizer;
```

Defined in: [src/core/optimize.js:310](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/optimize.js#L310)

Convenience function to create optimizer by name

#### Parameters

##### name

`string`

Optimizer name

##### options?

`Object` = `{}`

Optimizer options

#### Returns

`Optimizer`

Optimizer instance
