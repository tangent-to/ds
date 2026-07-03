---
layout: default
title: ml/explain
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/ml-explain
---
# ml/explain

SHAP (SHapley Additive exPlanations) for @tangent.to/ds

Model explanations based on Shapley values from cooperative game theory.
Every explainer is *additive*: for an instance x,

  f(x) = baseValue + Σ_j φ_j(x)

where `baseValue` is the model's expected output over a background/reference
distribution and `φ_j` is the contribution of feature j.

Three explainers are provided:

 - `KernelExplainer`  model-agnostic KernelSHAP (Lundberg & Lee, 2017).
                      Works with any model exposing a numeric `predict`,
                      e.g. a `GaussianProcessRegressor`, `MLPRegressor`, GLM…
 - `TreeExplainer`    exact, fast path-dependent TreeSHAP (Lundberg et al.,
                      2018) for `DecisionTreeRegressor` / `RandomForestRegressor`
                      (and bare `DecisionTreeBase`). Uses node coverage.
 - `PermutationExplainer`  model-agnostic Shapley estimation by sampling
                      feature permutations (Štrumbelj & Kononenko, 2014).

Plus tidy-data helpers `summaryData` / `importanceData` that turn SHAP values
into row arrays ready for Observable Plot (beeswarm / bar).

## Classes

### KernelExplainer

Defined in: [src/ml/explain.js:164](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L164)

Model-agnostic SHAP via the KernelSHAP weighted-linear-regression estimator.

The "absent" features in a coalition are marginalised by substituting values
from a background dataset and averaging the model output (the interventional
expectation). Keep the background small (a sample or summary, ~20–100 rows)
for performance: cost ≈ nCoalitions × nBackground model evaluations.

#### Example

```ts
const ex = new KernelExplainer({ model: gp, background: Xref });
const { values, baseValue } = ex.shapValues(Xtest);
// values[i][j] is feature j's contribution for instance i
```

#### Constructors

##### Constructor

```ts
new KernelExplainer(opts?): KernelExplainer;
```

Defined in: [src/ml/explain.js:173](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L173)

###### Parameters

###### opts?

###### model?

`Object`

Fitted model with a numeric `predict(rows)`.

###### predict?

`Function`

Alternatively, a predict function.

###### background

`number`[][]

Reference rows used to marginalise
  absent features. Its mean prediction is the explanation `baseValue`.

###### featureNames?

`string`[]

Optional feature labels.

###### Returns

[`KernelExplainer`](#kernelexplainer)

#### Properties

##### \_predict

```ts
_predict: (rows) => number[];
```

Defined in: [src/ml/explain.js:177](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L177)

###### Parameters

###### rows

`number`[][]

###### Returns

`number`[]

##### background

```ts
background: any[];
```

Defined in: [src/ml/explain.js:178](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L178)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/explain.js:179](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L179)

##### featureNames

```ts
featureNames: string[];
```

Defined in: [src/ml/explain.js:180](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L180)

##### expectedValue

```ts
expectedValue: number;
```

Defined in: [src/ml/explain.js:183](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L183)

#### Methods

##### shapValues()

```ts
shapValues(X, opts?): object;
```

Defined in: [src/ml/explain.js:200](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L200)

Compute SHAP values.

###### Parameters

###### X

`number`[] \| `number`[][]

Instance(s) to explain.

###### opts?

###### nSamples?

`number` \| `"auto"`

Coalitions to evaluate when
  the feature count is large enough to require sampling. "auto" picks
  `2*M + 2048`. When `2^M` is below `maxExact`, all coalitions are used.

###### maxExact?

`number`

Exhaustively enumerate coalitions when
  `nFeatures <= maxExact`.

###### seed?

`number`

PRNG seed for the sampling path.

###### Returns

`object`

###### values

```ts
values: number[][];
```

###### baseValue

```ts
baseValue: number;
```

###### expectedValue

```ts
expectedValue: number;
```

###### featureNames

```ts
featureNames: string[];
```

***

### TreeExplainer

Defined in: [src/ml/explain.js:388](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L388)

Exact SHAP for tree models using the path-dependent algorithm of
Lundberg et al. (2018). Runs in O(T·L·D²) and uses each node's training
coverage (`nSamples`) as the conditional distribution.

Supports regression trees: `DecisionTreeRegressor`, `RandomForestRegressor`,
a bare `DecisionTreeBase`, or any object exposing a compatible root node.

#### Example

```ts
const ex = new TreeExplainer({ model: forest });
const { values, baseValue } = ex.shapValues(Xtest);
```

#### Constructors

##### Constructor

```ts
new TreeExplainer(opts?): TreeExplainer;
```

Defined in: [src/ml/explain.js:394](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L394)

###### Parameters

###### opts?

###### model

`Object`

Fitted tree or forest regressor.

###### featureNames?

`string`[]

###### Returns

[`TreeExplainer`](#treeexplainer)

#### Properties

##### roots

```ts
roots: any;
```

Defined in: [src/ml/explain.js:395](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L395)

##### nFeatures

```ts
nFeatures: number;
```

Defined in: [src/ml/explain.js:402](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L402)

##### featureNames

```ts
featureNames: string[];
```

Defined in: [src/ml/explain.js:403](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L403)

##### \_treeBase

```ts
_treeBase: any;
```

Defined in: [src/ml/explain.js:406](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L406)

##### expectedValue

```ts
expectedValue: number;
```

Defined in: [src/ml/explain.js:407](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L407)

#### Methods

##### shapValues()

```ts
shapValues(X): object;
```

Defined in: [src/ml/explain.js:415](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L415)

###### Parameters

###### X

`number`[] \| `number`[][]

###### Returns

`object`

###### values

```ts
values: number[][];
```

###### baseValue

```ts
baseValue: number;
```

###### expectedValue

```ts
expectedValue: number;
```

###### featureNames

```ts
featureNames: string[];
```

***

### PermutationExplainer

Defined in: [src/ml/explain.js:623](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L623)

Model-agnostic SHAP by sampling random feature orderings and accumulating
each feature's marginal contribution as it is "turned on" (its value swapped
from a background row to the instance's value). Uses antithetic pairs
(a permutation and its reverse) to reduce variance. Cheaper than KernelSHAP
for many features; converges to exact Shapley values as nPermutations grows.

#### Example

```ts
const ex = new PermutationExplainer({ model: gp, background: Xref });
const { values, baseValue } = ex.shapValues(Xtest, { nPermutations: 64 });
```

#### Constructors

##### Constructor

```ts
new PermutationExplainer(opts?): PermutationExplainer;
```

Defined in: [src/ml/explain.js:627](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L627)

###### Parameters

###### opts?

`Object` = `{}`

`{ model | predict, background, featureNames }`

###### Returns

[`PermutationExplainer`](#permutationexplainer)

#### Properties

##### \_predict

```ts
_predict: (rows) => number[];
```

Defined in: [src/ml/explain.js:631](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L631)

###### Parameters

###### rows

`number`[][]

###### Returns

`number`[]

##### background

```ts
background: any[];
```

Defined in: [src/ml/explain.js:632](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L632)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/explain.js:633](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L633)

##### featureNames

```ts
featureNames: any;
```

Defined in: [src/ml/explain.js:634](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L634)

##### expectedValue

```ts
expectedValue: number;
```

Defined in: [src/ml/explain.js:636](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L636)

#### Methods

##### shapValues()

```ts
shapValues(X, opts?): object;
```

Defined in: [src/ml/explain.js:647](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L647)

###### Parameters

###### X

`number`[] \| `number`[][]

###### opts?

###### nPermutations?

`number` = `64`

Antithetic permutation pairs.

###### seed?

`number` = `0`

###### Returns

`object`

###### values

```ts
values: number[][];
```

###### baseValue

```ts
baseValue: number;
```

###### expectedValue

```ts
expectedValue: number;
```

###### featureNames

```ts
featureNames: string[];
```

## Functions

### kernelShap()

```ts
function kernelShap(
   __namedParameters, 
   X, 
   opts): object;
```

Defined in: [src/ml/explain.js:723](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L723)

Convenience: KernelSHAP in one call. See [KernelExplainer](#kernelexplainer).

#### Parameters

##### \_\_namedParameters

###### model

`any`

###### predict

`any`

###### background

`any`

###### featureNames

`any`

##### X

`any`

##### opts

`any`

#### Returns

`object`

##### values

```ts
values: number[][];
```

##### baseValue

```ts
baseValue: number;
```

##### expectedValue

```ts
expectedValue: number;
```

##### featureNames

```ts
featureNames: string[];
```

***

### treeShap()

```ts
function treeShap(__namedParameters, X): object;
```

Defined in: [src/ml/explain.js:731](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L731)

Convenience: TreeSHAP in one call. See [TreeExplainer](#treeexplainer).

#### Parameters

##### \_\_namedParameters

###### model

`any`

###### featureNames

`any`

##### X

`any`

#### Returns

`object`

##### values

```ts
values: number[][];
```

##### baseValue

```ts
baseValue: number;
```

##### expectedValue

```ts
expectedValue: number;
```

##### featureNames

```ts
featureNames: string[];
```

***

### importanceData()

```ts
function importanceData(res): object[];
```

Defined in: [src/ml/explain.js:746](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L746)

Global feature importance = mean(|SHAP|) per feature, sorted descending.
Plug into `Plot.barX(importanceData(res), { x: "importance", y: "feature" })`.

#### Parameters

##### res

shapValues() output.

###### values

`number`[][]

###### featureNames?

`string`[]

#### Returns

`object`[]

***

### summaryData()

```ts
function summaryData(res, X): object[];
```

Defined in: [src/ml/explain.js:769](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/ml/explain.js#L769)

Tidy long-form rows for a beeswarm / summary plot: one row per
(instance, feature) carrying the SHAP value and the original feature value.
Plug into `Plot.dot(summaryData(res, X), { x: "shap", y: "feature", fill: "featureValue" })`.

#### Parameters

##### res

shapValues() output.

###### values

`number`[][]

###### featureNames?

`string`[]

##### X

`number`[][]

The explained instances (same order as res.values).

#### Returns

`object`[]
