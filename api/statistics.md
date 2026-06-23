---
layout: default
title: Statistics
parent: API Reference
nav_order: 1
has_children: true
permalink: /api/statistics
---
# stats

## Classes

### GLM

Defined in: [src/stats/estimators/GLM.js:19](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L19)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new GLM(params?): GLM;
```

Defined in: [src/stats/estimators/GLM.js:38](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L38)

###### Parameters

###### params?

Model parameters

###### family

`string`

GLM family (gaussian, binomial, poisson, gamma, inverse_gaussian, negative_binomial)

###### link

`string`

Link function (default: canonical link for family)

###### multiclass

`string`

Multiclass strategy: 'ovr' (one-vs-rest), 'multinomial' (softmax), or null (binary/regression)

###### randomEffects

`Object`

Random effects specification {intercept: [...], slopes: {...}}

###### intercept

`boolean`

Include intercept term (default: true)

###### maxIter

`number`

Maximum iterations (default: 100)

###### tol

`number`

Convergence tolerance (default: 1e-8 for GLM, 1e-6 for GLMM)

###### regularization

`Object`

Regularization {alpha, l1_ratio}

###### dispersion

`string` \| `number`

Dispersion estimation: 'estimate', 'fixed', or numeric

###### theta

`number`

Theta parameter for negative binomial (default: 1)

###### alpha

`number`

Significance level for confidence intervals (default: 0.05 for 95% CIs)

###### compress

`boolean`

Compress model to save memory (default: false)

###### keepFittedValues

`boolean`

Keep fitted values and residuals (default: true)

###### warnOnNoConvergence

`boolean`

Warn if model doesn't converge (default: true)

###### warnLargeDataset

`boolean`

Warn about large datasets in browser (default: true)

###### Returns

[`GLM`](#glm)

###### Overrides

```ts
Estimator.constructor
```

#### Properties

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

[`OneSampleTTest`](#onesamplettest).[`fitted`](#fitted-1)

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Estimator._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

[`OneSampleTTest`](#onesamplettest).[`_warnings`](#_warnings-1)

##### params

```ts
params: object;
```

Defined in: [src/stats/estimators/GLM.js:42](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L42)

###### family

```ts
family: string;
```

###### link

```ts
link: string;
```

###### multiclass

```ts
multiclass: string;
```

###### randomEffects

```ts
randomEffects: Object;
```

###### intercept

```ts
intercept: boolean;
```

###### maxIter

```ts
maxIter: number;
```

###### tol

```ts
tol: number;
```

###### regularization

```ts
regularization: Object;
```

###### dispersion

```ts
dispersion: string | number;
```

###### theta

```ts
theta: number;
```

###### alpha

```ts
alpha: number;
```

###### compress

```ts
compress: boolean;
```

###### keepFittedValues

```ts
keepFittedValues: boolean;
```

###### warnOnNoConvergence

```ts
warnOnNoConvergence: boolean;
```

###### warnLargeDataset

```ts
warnLargeDataset: boolean;
```

###### Inherited from

```ts
Estimator.params
```

##### \_model

```ts
_model: Object | null;
```

Defined in: [src/stats/estimators/GLM.js:71](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L71)

##### \_models

```ts
_models: 
  | {
}
  | null;
```

Defined in: [src/stats/estimators/GLM.js:72](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L72)

##### \_classes

```ts
_classes: any[] | null;
```

Defined in: [src/stats/estimators/GLM.js:73](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L73)

##### \_targetNames

```ts
_targetNames: any;
```

Defined in: [src/stats/estimators/GLM.js:74](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L74)

##### \_isMulticlass

```ts
_isMulticlass: boolean;
```

Defined in: [src/stats/estimators/GLM.js:75](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L75)

##### \_isMultiOutput

```ts
_isMultiOutput: boolean;
```

Defined in: [src/stats/estimators/GLM.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L76)

##### \_isMixed

```ts
_isMixed: boolean;
```

Defined in: [src/stats/estimators/GLM.js:77](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L77)

##### \_columnsX

```ts
_columnsX: any;
```

Defined in: [src/stats/estimators/GLM.js:78](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L78)

##### \_columnY

```ts
_columnY: any;
```

Defined in: [src/stats/estimators/GLM.js:79](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L79)

##### \_formula

```ts
_formula: any;
```

Defined in: [src/stats/estimators/GLM.js:208](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L208)

##### \_indicatorColumns

```ts
_indicatorColumns: string[] | undefined;
```

Defined in: [src/stats/estimators/GLM.js:382](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L382)

##### \_isMultinomial

```ts
_isMultinomial: boolean | undefined;
```

Defined in: [src/stats/estimators/GLM.js:448](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L448)

#### Accessors

##### coefficients

###### Get Signature

```ts
get coefficients(): any;
```

Defined in: [src/stats/estimators/GLM.js:85](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L85)

Get coefficients (for backward compatibility with lm interface)

###### Returns

`any`

##### intercept

###### Get Signature

```ts
get intercept(): boolean;
```

Defined in: [src/stats/estimators/GLM.js:93](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L93)

Get intercept flag (for backward compatibility)

###### Returns

`boolean`

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
Estimator.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
Estimator.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
Estimator.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
Estimator.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
Estimator.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
Estimator.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
Estimator.getWarningsByType
```

##### setParams()

```ts
setParams(params?): GLM;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GLM`](#glm)

###### Inherited from

```ts
Estimator.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
Estimator.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
Estimator.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
Estimator.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
Estimator._prepareArgsForFit
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
Estimator.transform
```

##### fit()

```ts
fit(...args): GLM;
```

Defined in: [src/stats/estimators/GLM.js:108](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L108)

Fit the GLM or GLMM

Supports multiple calling conventions:
- fit(X, y)
- fit(X, y, weights, offset)
- fit({ X, y, data })
- fit({ X, y, groups, data }) for mixed models
- fit('y ~ x1 + x2', data) - R-style formula
- fit({ formula: 'y ~ x1 + x2', data }) - formula in object

###### Parameters

###### args

...`any`[]

###### Returns

[`GLM`](#glm)

###### Overrides

```ts
Estimator.fit
```

##### \_parseRandomEffects()

```ts
_parseRandomEffects(opts, rows): object;
```

Defined in: [src/stats/estimators/GLM.js:686](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L686)

Parse random effects specification from table-style input

###### Parameters

###### opts

`any`

###### rows

`any`

###### Returns

`object`

###### intercept

```ts
intercept: any;
```

###### slopes

```ts
slopes: object;
```

##### \_extractColumn()

```ts
_extractColumn(
   columnName, 
   _data, 
   rows): any;
```

Defined in: [src/stats/estimators/GLM.js:713](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L713)

Extract a column from table data

###### Parameters

###### columnName

`any`

###### \_data

`any`

###### rows

`any`

###### Returns

`any`

##### predict()

```ts
predict(X, options?): any[];
```

Defined in: [src/stats/estimators/GLM.js:941](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L941)

Predict from the fitted model

###### Parameters

###### X

`Object` \| `any`[]

Predictors or table-style object

###### options?

Prediction options

###### type

`string`

Prediction type: 'link', 'response', 'class' (multiclass), 'proba' (multiclass)

###### interval

`boolean`

Compute confidence intervals (default: false)

###### level

`number`

Confidence level (default: 0.95)

###### allowNewGroups

`boolean`

For GLMM: allow new groups (default: true)

###### Returns

`any`[]

Predictions

###### Overrides

```ts
Estimator.predict
```

##### summary()

```ts
summary(options?): string;
```

Defined in: [src/stats/estimators/GLM.js:1015](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1015)

Get model summary (lme4-style for mixed models)

###### Parameters

###### options?

Summary options

###### alpha

`number`

Significance level for CIs (default: from constructor)

###### Returns

`string`

##### confint()

```ts
confint(alpha?): Object[];
```

Defined in: [src/stats/estimators/GLM.js:1036](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1036)

Compute confidence intervals for coefficients

###### Parameters

###### alpha?

`number` = `null`

Significance level (default: 0.05 for 95% CIs)

###### Returns

`Object`[]

Array of {lower, upper} for each coefficient

##### pvalues()

```ts
pvalues(): number[];
```

Defined in: [src/stats/estimators/GLM.js:1059](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1059)

Compute p-values for coefficients (Wald test)

Note: For mixed models, p-values are controversial and should be
interpreted with caution. Prefer confidence intervals.

###### Returns

`number`[]

P-values for each coefficient

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/stats/estimators/GLM.js:1138](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1138)

Jupyter notebook display support
Returns HTML representation for better notebook rendering

###### Returns

`string`

###### Overrides

```ts
Estimator._repr_html_
```

##### \_summaryGLM()

```ts
_summaryGLM(alpha?): string;
```

Defined in: [src/stats/estimators/GLM.js:1161](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1161)

Format GLM summary

###### Parameters

###### alpha?

`number` = `0.05`

###### Returns

`string`

##### \_summaryMulticlass()

```ts
_summaryMulticlass(alpha?): string;
```

Defined in: [src/stats/estimators/GLM.js:1225](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1225)

Format multiclass GLM summary

###### Parameters

###### alpha?

`number` = `0.05`

###### Returns

`string`

##### \_summaryMultiOutput()

```ts
_summaryMultiOutput(alpha?): string;
```

Defined in: [src/stats/estimators/GLM.js:1248](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1248)

Format multi-output GLM summary

###### Parameters

###### alpha?

`number` = `0.05`

###### Returns

`string`

##### \_summaryTrueMultinomial()

```ts
_summaryTrueMultinomial(alpha?): string;
```

Defined in: [src/stats/estimators/GLM.js:1289](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1289)

Format true multinomial logistic regression summary

###### Parameters

###### alpha?

`number` = `0.05`

###### Returns

`string`

##### \_summaryGLMHTML()

```ts
_summaryGLMHTML(): string;
```

Defined in: [src/stats/estimators/GLM.js:1373](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1373)

Format GLM summary as HTML for Jupyter

###### Returns

`string`

##### \_summaryGLMM()

```ts
_summaryGLMM(alpha?): string;
```

Defined in: [src/stats/estimators/GLM.js:1448](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1448)

Format GLMM summary (lme4-style, no p-values)

###### Parameters

###### alpha?

`number` = `0.05`

###### Returns

`string`

##### \_summaryGLMMHTML()

```ts
_summaryGLMMHTML(): string;
```

Defined in: [src/stats/estimators/GLM.js:1528](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1528)

Format GLMM summary as HTML for Jupyter

###### Returns

`string`

##### \_getCoefLabels()

```ts
_getCoefLabels(): any[];
```

Defined in: [src/stats/estimators/GLM.js:1632](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1632)

Get coefficient labels

###### Returns

`any`[]

##### score()

```ts
score(
   yTrue, 
   yPred, ...
   args): number;
```

Defined in: [src/stats/estimators/GLM.js:1660](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1660)

Score the model (R² for regression families, accuracy for binomial/multiclass)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### args

...`any`[]

###### Returns

`number`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/GLM.js:1709](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1709)

Serialize to JSON

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'GLM';
```

###### params

```ts
params: object;
```

###### params.family

```ts
family: string;
```

###### params.link

```ts
link: string;
```

###### params.multiclass

```ts
multiclass: string;
```

###### params.randomEffects

```ts
randomEffects: Object;
```

###### params.intercept

```ts
intercept: boolean;
```

###### params.maxIter

```ts
maxIter: number;
```

###### params.tol

```ts
tol: number;
```

###### params.regularization

```ts
regularization: Object;
```

###### params.dispersion

```ts
dispersion: string | number;
```

###### params.theta

```ts
theta: number;
```

###### params.alpha

```ts
alpha: number;
```

###### params.compress

```ts
compress: boolean;
```

###### params.keepFittedValues

```ts
keepFittedValues: boolean;
```

###### params.warnOnNoConvergence

```ts
warnOnNoConvergence: boolean;
```

###### params.warnLargeDataset

```ts
warnLargeDataset: boolean;
```

###### fitted

```ts
fitted: boolean;
```

###### model

```ts
model: Object | null;
```

###### models

```ts
models: 
  | {
[k: string]: any;
}
  | null;
```

###### classes

```ts
classes: any[] | null;
```

###### targetNames

```ts
targetNames: any;
```

###### isMulticlass

```ts
isMulticlass: boolean;
```

###### isMultiOutput

```ts
isMultiOutput: boolean;
```

###### isMultinomial

```ts
isMultinomial: boolean | undefined;
```

###### isMixed

```ts
isMixed: boolean;
```

###### columnsX

```ts
columnsX: any;
```

###### columnY

```ts
columnY: any;
```

###### Overrides

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj): GLM;
```

Defined in: [src/stats/estimators/GLM.js:1734](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/GLM.js#L1734)

Deserialize from JSON

###### Parameters

###### obj

`any`

###### Returns

[`GLM`](#glm)

###### Overrides

```ts
Estimator.fromJSON
```

***

### OneSampleTTest

Defined in: [src/stats/estimators/tests.js:69](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L69)

#### Extends

- `StatisticalTest`

#### Constructors

##### Constructor

```ts
new OneSampleTTest(params?): OneSampleTTest;
```

Defined in: [src/stats/estimators/tests.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L32)

###### Parameters

###### params?

###### Returns

[`OneSampleTTest`](#onesamplettest)

###### Inherited from

```ts
StatisticalTest.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L24)

###### constructor

```ts
constructor: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString()

```ts
toString(): string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString()

```ts
toLocaleString(): string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf()

```ts
valueOf(): Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty()

```ts
hasOwnProperty(v): boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf()

```ts
isPrototypeOf(v): boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable()

```ts
propertyIsEnumerable(v): boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
StatisticalTest.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
StatisticalTest._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
StatisticalTest._warnings
```

##### result

```ts
result: any;
```

Defined in: [src/stats/estimators/tests.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L34)

###### Inherited from

[`TwoSampleTTest`](#twosamplettest).[`result`](#result-1)

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
StatisticalTest.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
StatisticalTest.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
StatisticalTest.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
StatisticalTest.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
StatisticalTest._repr_html_
```

##### setParams()

```ts
setParams(params?): OneSampleTTest;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`OneSampleTTest`](#onesamplettest)

###### Inherited from

```ts
StatisticalTest.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
StatisticalTest.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
StatisticalTest.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
StatisticalTest.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
StatisticalTest._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.predict
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.transform
```

##### summary()

```ts
summary(): any;
```

Defined in: [src/stats/estimators/tests.js:37](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L37)

###### Returns

`any`

###### Inherited from

```ts
StatisticalTest.summary
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/tests.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L44)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string;
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### result

```ts
result: any;
```

###### Inherited from

```ts
StatisticalTest.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): StatisticalTest;
```

Defined in: [src/stats/estimators/tests.js:53](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L53)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

`StatisticalTest`

###### Inherited from

```ts
StatisticalTest.fromJSON
```

##### fit()

```ts
fit(sample, opts?): OneSampleTTest;
```

Defined in: [src/stats/estimators/tests.js:70](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L70)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### sample

`any`

###### opts?

###### Returns

[`OneSampleTTest`](#onesamplettest)

###### Overrides

```ts
StatisticalTest.fit
```

***

### TwoSampleTTest

Defined in: [src/stats/estimators/tests.js:126](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L126)

#### Extends

- `StatisticalTest`

#### Constructors

##### Constructor

```ts
new TwoSampleTTest(params?): TwoSampleTTest;
```

Defined in: [src/stats/estimators/tests.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L32)

###### Parameters

###### params?

###### Returns

[`TwoSampleTTest`](#twosamplettest)

###### Inherited from

```ts
StatisticalTest.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L24)

###### constructor

```ts
constructor: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString()

```ts
toString(): string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString()

```ts
toLocaleString(): string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf()

```ts
valueOf(): Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty()

```ts
hasOwnProperty(v): boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf()

```ts
isPrototypeOf(v): boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable()

```ts
propertyIsEnumerable(v): boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
StatisticalTest.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
StatisticalTest._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
StatisticalTest._warnings
```

##### result

```ts
result: any;
```

Defined in: [src/stats/estimators/tests.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L34)

###### Inherited from

```ts
StatisticalTest.result
```

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
StatisticalTest.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
StatisticalTest.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
StatisticalTest.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
StatisticalTest.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
StatisticalTest._repr_html_
```

##### setParams()

```ts
setParams(params?): TwoSampleTTest;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`TwoSampleTTest`](#twosamplettest)

###### Inherited from

```ts
StatisticalTest.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
StatisticalTest.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
StatisticalTest.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
StatisticalTest.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
StatisticalTest._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.predict
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.transform
```

##### summary()

```ts
summary(): any;
```

Defined in: [src/stats/estimators/tests.js:37](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L37)

###### Returns

`any`

###### Inherited from

```ts
StatisticalTest.summary
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/tests.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L44)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string;
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### result

```ts
result: any;
```

###### Inherited from

```ts
StatisticalTest.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): StatisticalTest;
```

Defined in: [src/stats/estimators/tests.js:53](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L53)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

`StatisticalTest`

###### Inherited from

```ts
StatisticalTest.fromJSON
```

##### fit()

```ts
fit(
   sample1, 
   sample2?, 
   opts?): TwoSampleTTest;
```

Defined in: [src/stats/estimators/tests.js:127](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L127)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### sample1

`any`

###### sample2?

`null` = `null`

###### opts?

###### Returns

[`TwoSampleTTest`](#twosamplettest)

###### Overrides

```ts
StatisticalTest.fit
```

***

### OneWayAnova

Defined in: [src/stats/estimators/tests.js:191](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L191)

#### Extends

- `StatisticalTest`

#### Constructors

##### Constructor

```ts
new OneWayAnova(params?): OneWayAnova;
```

Defined in: [src/stats/estimators/tests.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L32)

###### Parameters

###### params?

###### Returns

[`OneWayAnova`](#onewayanova)

###### Inherited from

```ts
StatisticalTest.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L24)

###### constructor

```ts
constructor: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString()

```ts
toString(): string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString()

```ts
toLocaleString(): string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf()

```ts
valueOf(): Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty()

```ts
hasOwnProperty(v): boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf()

```ts
isPrototypeOf(v): boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable()

```ts
propertyIsEnumerable(v): boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
StatisticalTest.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
StatisticalTest._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
StatisticalTest._warnings
```

##### result

```ts
result: any;
```

Defined in: [src/stats/estimators/tests.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L34)

###### Inherited from

```ts
StatisticalTest.result
```

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
StatisticalTest.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
StatisticalTest.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
StatisticalTest.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
StatisticalTest.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
StatisticalTest._repr_html_
```

##### setParams()

```ts
setParams(params?): OneWayAnova;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`OneWayAnova`](#onewayanova)

###### Inherited from

```ts
StatisticalTest.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
StatisticalTest.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
StatisticalTest.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
StatisticalTest.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
StatisticalTest._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.predict
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.transform
```

##### summary()

```ts
summary(): any;
```

Defined in: [src/stats/estimators/tests.js:37](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L37)

###### Returns

`any`

###### Inherited from

```ts
StatisticalTest.summary
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/tests.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L44)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string;
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### result

```ts
result: any;
```

###### Inherited from

```ts
StatisticalTest.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): StatisticalTest;
```

Defined in: [src/stats/estimators/tests.js:53](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L53)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

`StatisticalTest`

###### Inherited from

```ts
StatisticalTest.fromJSON
```

##### fit()

```ts
fit(groups, opts?): OneWayAnova;
```

Defined in: [src/stats/estimators/tests.js:192](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L192)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### groups

`any`

###### opts?

###### Returns

[`OneWayAnova`](#onewayanova)

###### Overrides

```ts
StatisticalTest.fit
```

***

### ChiSquareTest

Defined in: [src/stats/estimators/tests.js:223](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L223)

#### Extends

- `StatisticalTest`

#### Constructors

##### Constructor

```ts
new ChiSquareTest(params?): ChiSquareTest;
```

Defined in: [src/stats/estimators/tests.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L32)

###### Parameters

###### params?

###### Returns

[`ChiSquareTest`](#chisquaretest)

###### Inherited from

```ts
StatisticalTest.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L24)

###### constructor

```ts
constructor: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString()

```ts
toString(): string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString()

```ts
toLocaleString(): string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf()

```ts
valueOf(): Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty()

```ts
hasOwnProperty(v): boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf()

```ts
isPrototypeOf(v): boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable()

```ts
propertyIsEnumerable(v): boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
StatisticalTest.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
StatisticalTest._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
StatisticalTest._warnings
```

##### result

```ts
result: any;
```

Defined in: [src/stats/estimators/tests.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L34)

###### Inherited from

```ts
StatisticalTest.result
```

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
StatisticalTest.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
StatisticalTest.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
StatisticalTest.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
StatisticalTest.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
StatisticalTest._repr_html_
```

##### setParams()

```ts
setParams(params?): ChiSquareTest;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`ChiSquareTest`](#chisquaretest)

###### Inherited from

```ts
StatisticalTest.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
StatisticalTest.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
StatisticalTest.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
StatisticalTest.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
StatisticalTest._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.predict
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.transform
```

##### summary()

```ts
summary(): any;
```

Defined in: [src/stats/estimators/tests.js:37](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L37)

###### Returns

`any`

###### Inherited from

```ts
StatisticalTest.summary
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/tests.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L44)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string;
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### result

```ts
result: any;
```

###### Inherited from

```ts
StatisticalTest.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): StatisticalTest;
```

Defined in: [src/stats/estimators/tests.js:53](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L53)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

`StatisticalTest`

###### Inherited from

```ts
StatisticalTest.fromJSON
```

##### fit()

```ts
fit(
   observed, 
   expected?, 
   opts?): ChiSquareTest;
```

Defined in: [src/stats/estimators/tests.js:224](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L224)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### observed

`any`

###### expected?

`null` = `null`

###### opts?

###### Returns

[`ChiSquareTest`](#chisquaretest)

###### Overrides

```ts
StatisticalTest.fit
```

***

### TukeyHSD

Defined in: [src/stats/estimators/tests.js:273](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L273)

#### Extends

- `StatisticalTest`

#### Constructors

##### Constructor

```ts
new TukeyHSD(params?): TukeyHSD;
```

Defined in: [src/stats/estimators/tests.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L32)

###### Parameters

###### params?

###### Returns

[`TukeyHSD`](#tukeyhsd)

###### Inherited from

```ts
StatisticalTest.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L24)

###### constructor

```ts
constructor: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString()

```ts
toString(): string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString()

```ts
toLocaleString(): string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf()

```ts
valueOf(): Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty()

```ts
hasOwnProperty(v): boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf()

```ts
isPrototypeOf(v): boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable()

```ts
propertyIsEnumerable(v): boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
StatisticalTest.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
StatisticalTest._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
StatisticalTest._warnings
```

##### result

```ts
result: any;
```

Defined in: [src/stats/estimators/tests.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L34)

###### Inherited from

```ts
StatisticalTest.result
```

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
StatisticalTest.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
StatisticalTest.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
StatisticalTest.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
StatisticalTest.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L148)

Get warnings of a specific type

###### Parameters

###### type

`string`

Warning type

###### Returns

`Object`[]

Filtered warnings

###### Inherited from

```ts
StatisticalTest.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
StatisticalTest._repr_html_
```

##### setParams()

```ts
setParams(params?): TukeyHSD;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`TukeyHSD`](#tukeyhsd)

###### Inherited from

```ts
StatisticalTest.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
StatisticalTest.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
StatisticalTest.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L346)

Load model from JSON string

###### Parameters

###### jsonString

`string`

JSON representation

###### Returns

`Estimator`

Reconstructed estimator instance

###### Inherited from

```ts
StatisticalTest.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: Object[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: Object[];
  prepared: boolean;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X?: undefined;
  columns?: undefined;
  rows?: undefined;
  prepared?: undefined;
  raw: any[];
};
```

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L367)

Convenience helper: parse arguments passed to fit/predict/transform.

Supports declarative table-style inputs:
 - fit({ X, y, data, omit_missing })
 - fit({ data, columns, ... })

Returns an object { X, y, prepared, rows } where X/y are numeric arrays
if preparation was required, otherwise returns the original values.

Note: this helper only prepares numeric matrices/vectors using core table utilities;
it does not perform encoding of categorical predictors.

###### Parameters

###### args?

`any`[] = `[]`

###### Returns

  \| \{
  `X`: `any`[][];
  `y`: `any`[];
  `columnsX`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `Object`[];
  `prepared`: `boolean`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X?`: `undefined`;
  `columns?`: `undefined`;
  `rows?`: `undefined`;
  `prepared?`: `undefined`;
  `raw`: `any`[];
\}

###### Inherited from

```ts
StatisticalTest._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.predict
```

##### transform()

```ts
transform(): void;
```

Defined in: [src/core/estimators/estimator.js:431](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L431)

Transform should be implemented by transformers.

###### Returns

`void`

###### Inherited from

```ts
StatisticalTest.transform
```

##### summary()

```ts
summary(): any;
```

Defined in: [src/stats/estimators/tests.js:37](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L37)

###### Returns

`any`

###### Inherited from

```ts
StatisticalTest.summary
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/stats/estimators/tests.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L44)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string;
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### result

```ts
result: any;
```

###### Inherited from

```ts
StatisticalTest.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): StatisticalTest;
```

Defined in: [src/stats/estimators/tests.js:53](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L53)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

`StatisticalTest`

###### Inherited from

```ts
StatisticalTest.fromJSON
```

##### fit()

```ts
fit(groups, opts?): TukeyHSD;
```

Defined in: [src/stats/estimators/tests.js:274](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/estimators/tests.js#L274)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### groups

`any`

###### opts?

###### Returns

[`TukeyHSD`](#tukeyhsd)

###### Overrides

```ts
StatisticalTest.fit
```

## Variables

### normal

```ts
const normal: object;
```

Defined in: [src/stats/distribution.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L27)

Normal distribution

#### Type Declaration

##### pdf()

```ts
pdf(x, params?): number;
```

Probability density function

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{mean, sd}

###### Returns

`number`

Probability density

##### cdf()

```ts
cdf(x, params?): number;
```

Cumulative distribution function

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{mean, sd}

###### Returns

`number`

Cumulative probability

##### quantile()

```ts
quantile(p, params?): number;
```

Quantile function (inverse CDF)

###### Parameters

###### p

`number`

Probability

###### params?

`Object` = `{}`

{mean, sd}

###### Returns

`number`

Quantile

***

### uniform

```ts
const uniform: object;
```

Defined in: [src/stats/distribution.js:67](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L67)

#### Type Declaration

##### pdf()

```ts
pdf(x, params?): number;
```

Probability density function

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{min, max}

###### Returns

`number`

Probability density

##### cdf()

```ts
cdf(x, params?): number;
```

Cumulative distribution function

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{min, max}

###### Returns

`number`

Cumulative probability

##### quantile()

```ts
quantile(p, params?): number;
```

Quantile function

###### Parameters

###### p

`number`

Probability

###### params?

`Object` = `{}`

{min, max}

###### Returns

`number`

Quantile

***

### gamma

```ts
const gamma: object;
```

Defined in: [src/stats/distribution.js:120](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L120)

#### Type Declaration

##### pdf()

```ts
pdf(x, params?): number;
```

Probability density function

###### Parameters

###### x

`number`

Value (x > 0)

###### params?

`Object` = `{}`

{shape, scale}

###### Returns

`number`

Probability density

##### cdf()

```ts
cdf(x, params?): number;
```

Cumulative distribution function (approximation)

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{shape, scale}

###### Returns

`number`

Cumulative probability

##### quantile()

```ts
quantile(p, params?): number;
```

Quantile function (numerical approximation)

###### Parameters

###### p

`number`

Probability

###### params?

`Object` = `{}`

{shape, scale}

###### Returns

`number`

Quantile

***

### beta

```ts
const beta: object;
```

Defined in: [src/stats/distribution.js:244](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L244)

#### Type Declaration

##### pdf()

```ts
pdf(x, params?): number;
```

Probability density function

###### Parameters

###### x

`number`

Value (0 <= x <= 1)

###### params?

`Object` = `{}`

{alpha, beta}

###### Returns

`number`

Probability density

##### cdf()

```ts
cdf(x, params?): number;
```

Cumulative distribution function (approximation)

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{alpha, beta}

###### Returns

`number`

Cumulative probability

##### quantile()

```ts
quantile(p, params?): number;
```

Quantile function (numerical approximation)

###### Parameters

###### p

`number`

Probability

###### params?

`Object` = `{}`

{alpha, beta}

###### Returns

`number`

Quantile

***

### chisq

```ts
const chisq: object;
```

Defined in: [src/stats/distribution.js:336](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L336)

Chi-squared distribution (special case of gamma with shape=df/2, scale=2)

#### Type Declaration

##### pdf()

```ts
pdf(x, params?): number;
```

Probability density function

###### Parameters

###### x

`number`

Value (x >= 0)

###### params?

`Object` = `{}`

{df} degrees of freedom

###### Returns

`number`

Probability density

##### cdf()

```ts
cdf(x, params?): number;
```

Cumulative distribution function

###### Parameters

###### x

`number`

Value

###### params?

`Object` = `{}`

{df} degrees of freedom

###### Returns

`number`

Cumulative probability

##### quantile()

```ts
quantile(p, params?): number;
```

Quantile function (inverse CDF)

###### Parameters

###### p

`number`

Probability

###### params?

`Object` = `{}`

{df} degrees of freedom

###### Returns

`number`

Quantile

***

### oneSampleTTest

```ts
const oneSampleTTest: typeof OneSampleTTest = OneSampleTTest;
```

Defined in: [src/stats/index.js:48](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L48)

***

### twoSampleTTest

```ts
const twoSampleTTest: typeof TwoSampleTTest = TwoSampleTTest;
```

Defined in: [src/stats/index.js:49](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L49)

***

### chiSquareTest

```ts
const chiSquareTest: typeof ChiSquareTest = ChiSquareTest;
```

Defined in: [src/stats/index.js:50](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L50)

***

### oneWayAnova

```ts
const oneWayAnova: typeof OneWayAnova = OneWayAnova;
```

Defined in: [src/stats/index.js:51](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L51)

***

### tukeyHSD

```ts
const tukeyHSD: typeof TukeyHSD = TukeyHSD;
```

Defined in: [src/stats/index.js:52](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L52)

***

### pairedTTest

```ts
const pairedTTest: (sample1, sample2, options) => Object = pairedTTestFn;
```

Defined in: [src/stats/index.js:55](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L55)

Paired t-test for dependent samples

#### Parameters

##### sample1

`number`[]

First sample (before)

##### sample2

`number`[]

Second sample (after)

##### options?

`Object` = `{}`

{mu: hypothesized mean difference (default 0), alternative: 'two-sided'|'less'|'greater'}

#### Returns

`Object`

{statistic, pValue, df, meanDiff, se}

***

### mannWhitneyU

```ts
const mannWhitneyU: (sample1, sample2, options) => Object = mannWhitneyUFn;
```

Defined in: [src/stats/index.js:56](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L56)

Mann-Whitney U test (Wilcoxon rank-sum test)
Non-parametric alternative to two-sample t-test

#### Parameters

##### sample1

`number`[]

First sample

##### sample2

`number`[]

Second sample

##### options?

`Object` = `{}`

{alternative: 'two-sided'|'less'|'greater'}

#### Returns

`Object`

{statistic (U), pValue, alternative}

***

### kruskalWallis

```ts
const kruskalWallis: (groups) => Object = kruskalWallisFn;
```

Defined in: [src/stats/index.js:57](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L57)

Kruskal-Wallis H test
Non-parametric alternative to one-way ANOVA

#### Parameters

##### groups

`number`[][]

Array of group samples

#### Returns

`Object`

{statistic (H), pValue, df}

***

### hypothesis

```ts
const hypothesis: object;
```

Defined in: [src/stats/index.js:60](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/index.js#L60)

#### Type Declaration

##### oneSampleTTest

```ts
oneSampleTTest: (sample, options) => Object = oneSampleTTestFn;
```

One-sample t-test

###### Parameters

###### sample

`number`[]

Sample data

###### options?

`Object` = `{}`

{mu: hypothesized mean, alternative: 'two-sided'|'less'|'greater'}

###### Returns

`Object`

{statistic, pValue, df, mean, se}

##### twoSampleTTest

```ts
twoSampleTTest: (sample1, sample2, options) => Object = twoSampleTTestFn;
```

Two-sample t-test (assuming equal variances)

###### Parameters

###### sample1

`number`[]

First sample

###### sample2

`number`[]

Second sample

###### options?

`Object` = `{}`

{alternative: 'two-sided'|'less'|'greater'}

###### Returns

`Object`

{statistic, pValue, df, mean1, mean2, pooledSE}

##### pairedTTest

```ts
pairedTTest: (sample1, sample2, options) => Object = pairedTTestFn;
```

Paired t-test for dependent samples

###### Parameters

###### sample1

`number`[]

First sample (before)

###### sample2

`number`[]

Second sample (after)

###### options?

`Object` = `{}`

{mu: hypothesized mean difference (default 0), alternative: 'two-sided'|'less'|'greater'}

###### Returns

`Object`

{statistic, pValue, df, meanDiff, se}

##### chiSquareTest

```ts
chiSquareTest: (observed, expected) => Object = chiSquareTestFn;
```

Chi-square goodness of fit test

###### Parameters

###### observed

`number`[]

Observed frequencies

###### expected

`number`[]

Expected frequencies

###### Returns

`Object`

{statistic, pValue, df}

##### oneWayAnova

```ts
oneWayAnova: (groups) => Object = oneWayAnovaFn;
```

One-way ANOVA

###### Parameters

###### groups

`number`[][]

Array of group samples

###### Returns

`Object`

{statistic, pValue, dfBetween, dfWithin, MSbetween, MSwithin}

##### tukeyHSD

```ts
tukeyHSD: (groups, options) => Object = tukeyHSDFn;
```

Tukey's Honestly Significant Difference (HSD) test
Post-hoc test for pairwise comparisons after ANOVA

###### Parameters

###### groups

`number`[][]

Array of group samples

###### options?

`Object` = `{}`

{alpha: significance level (default 0.05), anovaResult: optional precomputed ANOVA result}

###### Returns

`Object`

{
  comparisons: Array of {groups: [i,j], diff, lowerCI, upperCI, pValue, significant},
  groupMeans: Array of group means,
  groupNames: Array of group indices,
  MSwithin: Mean square within groups,
  dfWithin: Degrees of freedom,
  alpha: Significance level
}

##### mannWhitneyU

```ts
mannWhitneyU: (sample1, sample2, options) => Object = mannWhitneyUFn;
```

Mann-Whitney U test (Wilcoxon rank-sum test)
Non-parametric alternative to two-sample t-test

###### Parameters

###### sample1

`number`[]

First sample

###### sample2

`number`[]

Second sample

###### options?

`Object` = `{}`

{alternative: 'two-sided'|'less'|'greater'}

###### Returns

`Object`

{statistic (U), pValue, alternative}

##### kruskalWallis

```ts
kruskalWallis: (groups) => Object = kruskalWallisFn;
```

Kruskal-Wallis H test
Non-parametric alternative to one-way ANOVA

###### Parameters

###### groups

`number`[][]

Array of group samples

###### Returns

`Object`

{statistic (H), pValue, df}

## Functions

### qchisq()

```ts
function qchisq(p, df): number;
```

Defined in: [src/stats/distribution.js:378](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/distribution.js#L378)

Shorthand for chi-squared quantile function (R-like API)

#### Parameters

##### p

`number`

Probability

##### df

`number`

Degrees of freedom

#### Returns

`number`

Quantile

***

### compareModels()

```ts
function compareModels(models, options?): Object;
```

Defined in: [src/stats/model\_comparison.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L14)

Compare multiple models with information criteria

#### Parameters

##### models

`Object`[]

Array of fitted GLM models with optional names

##### options?

`Object` = `{}`

Comparison options

#### Returns

`Object`

Comparison table and best model info

***

### likelihoodRatioTest()

```ts
function likelihoodRatioTest(
   model1, 
   model2, 
   _options?): Object;
```

Defined in: [src/stats/model\_comparison.js:115](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L115)

Perform likelihood ratio test for nested models

#### Parameters

##### model1

`Object`

Smaller (nested) model

##### model2

`Object`

Larger model

##### \_options?

#### Returns

`Object`

Test results

***

### pairwiseLRT()

```ts
function pairwiseLRT(models, _options?): Object;
```

Defined in: [src/stats/model\_comparison.js:177](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L177)

Compare multiple models and perform pairwise LRT

#### Parameters

##### models

`Object`[]

Array of fitted models (should be nested)

##### \_options?

#### Returns

`Object`

Matrix of pairwise comparisons

***

### modelSelectionPlot()

```ts
function modelSelectionPlot(models, options?): Object;
```

Defined in: [src/stats/model\_comparison.js:245](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L245)

Generate model selection plot specification

#### Parameters

##### models

`Object`[]

Array of fitted models

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### aicWeightPlot()

```ts
function aicWeightPlot(models, options?): Object;
```

Defined in: [src/stats/model\_comparison.js:330](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L330)

Generate AIC weight visualization

#### Parameters

##### models

`Object`[]

Array of fitted models

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### coefficientComparisonPlot()

```ts
function coefficientComparisonPlot(models, options?): Object;
```

Defined in: [src/stats/model\_comparison.js:367](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L367)

Generate coefficient comparison plot across models

#### Parameters

##### models

`Object`[]

Array of fitted models

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Observable Plot specification

***

### crossValidate()

```ts
function crossValidate(
   modelFactory, 
   X, 
   y, 
   options?): Object;
```

Defined in: [src/stats/model\_comparison.js:438](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L438)

K-fold cross-validation for model selection

#### Parameters

##### modelFactory

`Function`

Function that creates and returns a model

##### X

`any`[]

Predictor matrix

##### y

`any`[]

Response variable

##### options?

`Object` = `{}`

CV options

#### Returns

`Object`

Cross-validation results

***

### crossValidateModels()

```ts
function crossValidateModels(
   modelFactories, 
   X, 
   y, 
   options?): Object;
```

Defined in: [src/stats/model\_comparison.js:504](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/model_comparison.js#L504)

Compare models using cross-validation

#### Parameters

##### modelFactories

`Function`[]

Array of model factory functions

##### X

`any`[]

Predictor matrix

##### y

`any`[]

Response variable

##### options?

`Object` = `{}`

CV options

#### Returns

`Object`

CV comparison results

***

### cohensD()

```ts
function cohensD(
   sample1, 
   sample2, 
   options?): number;
```

Defined in: [src/stats/tests.js:660](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L660)

Cohen's d effect size for two samples
Standardized mean difference

#### Parameters

##### sample1

`number`[]

First sample

##### sample2

`number`[]

Second sample

##### options?

`Object` = `{}`

{pooled: use pooled SD (default true)}

#### Returns

`number`

Cohen's d

***

### etaSquared()

```ts
function etaSquared(anovaResult): number;
```

Defined in: [src/stats/tests.js:686](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L686)

Eta-squared effect size for ANOVA
Proportion of total variance explained by group differences

#### Parameters

##### anovaResult

`Object`

Result from oneWayAnova

#### Returns

`number`

Eta-squared

***

### omegaSquared()

```ts
function omegaSquared(anovaResult): number;
```

Defined in: [src/stats/tests.js:704](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L704)

Omega-squared effect size for ANOVA
Less biased estimate than eta-squared

#### Parameters

##### anovaResult

`Object`

Result from oneWayAnova

#### Returns

`number`

Omega-squared

***

### leveneTest()

```ts
function leveneTest(groups, options?): Object;
```

Defined in: [src/stats/tests.js:723](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L723)

Levene's test for equality of variances
Tests homogeneity of variance assumption (homoscedasticity)

#### Parameters

##### groups

`number`[][]

Array of group samples

##### options?

`Object` = `{}`

{center: 'mean'|'median'|'trimmed', trim: trim proportion for trimmed mean (default 0.1)}

#### Returns

`Object`

{statistic, pValue, df1, df2}

***

### pearsonCorrelation()

```ts
function pearsonCorrelation(x, y): Object;
```

Defined in: [src/stats/tests.js:776](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L776)

Pearson correlation coefficient with significance test

#### Parameters

##### x

`number`[]

First variable

##### y

`number`[]

Second variable

#### Returns

`Object`

{r, pValue, df, ci95}

***

### spearmanCorrelation()

```ts
function spearmanCorrelation(x, y): Object;
```

Defined in: [src/stats/tests.js:862](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L862)

Spearman rank correlation coefficient with significance test

#### Parameters

##### x

`number`[]

First variable

##### y

`number`[]

Second variable

#### Returns

`Object`

{rho, pValue, df}

***

### fisherExactTest()

```ts
function fisherExactTest(table, options?): Object;
```

Defined in: [src/stats/tests.js:922](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L922)

Fisher's exact test for 2x2 contingency tables

#### Parameters

##### table

`number`[][]

2x2 contingency table [[a,b],[c,d]]

##### options?

`Object` = `{}`

{alternative: 'two-sided'|'less'|'greater'}

#### Returns

`Object`

{pValue, oddsRatio, alternative}

***

### bonferroni()

```ts
function bonferroni(pValues, alpha?): Object;
```

Defined in: [src/stats/tests.js:1036](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L1036)

Bonferroni correction for multiple testing

#### Parameters

##### pValues

`number`[]

Array of p-values

##### alpha?

`number` = `0.05`

Family-wise error rate (default 0.05)

#### Returns

`Object`

{adjustedPValues, rejected, adjustedAlpha}

***

### holmBonferroni()

```ts
function holmBonferroni(pValues, alpha?): Object;
```

Defined in: [src/stats/tests.js:1056](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L1056)

Holm-Bonferroni correction for multiple testing
Sequentially rejective Bonferroni procedure (more powerful)

#### Parameters

##### pValues

`number`[]

Array of p-values

##### alpha?

`number` = `0.05`

Family-wise error rate (default 0.05)

#### Returns

`Object`

{adjustedPValues, rejected}

***

### fdr()

```ts
function fdr(pValues, alpha?): Object;
```

Defined in: [src/stats/tests.js:1097](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/stats/tests.js#L1097)

Benjamini-Hochberg FDR correction
Controls false discovery rate

#### Parameters

##### pValues

`number`[]

Array of p-values

##### alpha?

`number` = `0.05`

False discovery rate (default 0.05)

#### Returns

`Object`

{adjustedPValues, rejected, criticalValues}
