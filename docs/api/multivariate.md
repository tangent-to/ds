---
layout: default
title: Multivariate Analysis
parent: API Reference
nav_order: 3
has_children: true
permalink: /api/multivariate
---
# mva

## Namespaces

- [cca](/api/multivariate/cca)
- [composition](/api/multivariate/composition)
- [lda](/api/multivariate/lda)
- [pca](/api/multivariate/pca)
- [rda](/api/multivariate/rda)

## Classes

### CCA

Defined in: [src/mva/estimators/CCA.js:10](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L10)

#### Extends

- `Transformer`

#### Constructors

##### Constructor

```ts
new CCA(params?): CCA;
```

Defined in: [src/mva/estimators/CCA.js:11](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L11)

###### Parameters

###### params?

###### Returns

[`CCA`](#cca)

###### Overrides

```ts
Transformer.constructor
```

#### Properties

##### params

```ts
params: object;
```

Defined in: [src/core/estimators/estimator.js:24](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L24)

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
Transformer.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Transformer.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Transformer._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Transformer._warnings
```

##### model

```ts
model: Object | null;
```

Defined in: [src/mva/estimators/CCA.js:13](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L13)

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
Transformer.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
Transformer.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
Transformer.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
Transformer.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L148)

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
Transformer.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
Transformer._repr_html_
```

##### setParams()

```ts
setParams(params?): CCA;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`CCA`](#cca)

###### Inherited from

```ts
Transformer.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
Transformer.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
Transformer.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L346)

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
Transformer.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: any[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: any[];
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

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L367)

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
  `rows`: `any`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `any`[];
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
Transformer._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
Transformer.predict
```

##### fitTransform()

```ts
fitTransform(...args): void;
```

Defined in: [src/core/estimators/estimator.js:683](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L683)

Convenience: fit then transform
Returns transformed data.

###### Parameters

###### args

...`any`[]

###### Returns

`void`

###### Inherited from

```ts
Transformer.fitTransform
```

##### fit()

```ts
fit(
   X, 
   Y?, 
   opts?): CCA;
```

Defined in: [src/mva/estimators/CCA.js:26](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L26)

Fit the CCA model on paired data sets X and Y.

Accepts a positional numeric call (`fit(X, Y[, opts])`) or a declarative
`{ data, X, Y }` object (`fit({ data, X, Y, columnsX, columnsY })`).

###### Parameters

###### X

`Object` \| `number`[][]

First data matrix (n samples × p features), or a declarative `{ data, X, Y }` object

###### Y?

`number`[][] = `null`

Second data matrix (n samples × q features), for the positional call form

###### opts?

`Object` = `{}`

Fitting options (used for the positional call form)

###### Returns

[`CCA`](#cca)

The fitted estimator (for chaining)

###### Overrides

```ts
Transformer.fit
```

##### transformX()

```ts
transformX(X, opts?): Object[];
```

Defined in: [src/mva/estimators/CCA.js:53](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L53)

Project new X data onto the fitted X canonical variates.

###### Parameters

###### X

`Object` \| `number`[][]

X data matrix (n samples × p features), or a declarative `{ data, columns }` object

###### opts?

`Object` = `{}`

Transform options

###### Returns

`Object`[]

Canonical score objects, one per row (keyed `cca1`, `cca2`, ...)

##### transformY()

```ts
transformY(Y, opts?): Object[];
```

Defined in: [src/mva/estimators/CCA.js:66](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L66)

Project new Y data onto the fitted Y canonical variates.

###### Parameters

###### Y

`Object` \| `number`[][]

Y data matrix (n samples × q features), or a declarative `{ data, columns }` object

###### opts?

`Object` = `{}`

Transform options

###### Returns

`Object`[]

Canonical score objects, one per row (keyed `cca1`, `cca2`, ...)

##### transform()

```ts
transform(
   X, 
   Y, 
   opts?): Object;
```

Defined in: [src/mva/estimators/CCA.js:80](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L80)

Project new X and Y data onto their fitted canonical variates.

###### Parameters

###### X

`Object` \| `number`[][]

X data matrix (n samples × p features), or a declarative `{ data, columns }` object

###### Y

`Object` \| `number`[][]

Y data matrix (n samples × q features), or a declarative `{ data, columns }` object

###### opts?

`Object` = `{}`

Transform options

###### Returns

`Object`

Object with `xScores` and `yScores` arrays of canonical score objects

###### Overrides

```ts
Transformer.transform
```

##### summary()

```ts
summary(): object;
```

Defined in: [src/mva/estimators/CCA.js:87](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L87)

###### Returns

`object`

###### nSamples

```ts
nSamples: any;
```

###### nComponents

```ts
nComponents: any;
```

###### correlations

```ts
correlations: any;
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/mva/estimators/CCA.js:98](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L98)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'CCA';
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### model

```ts
model: Object | null;
```

###### Overrides

```ts
Transformer.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): CCA;
```

Defined in: [src/mva/estimators/CCA.js:107](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/CCA.js#L107)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`CCA`](#cca)

###### Overrides

```ts
Transformer.fromJSON
```

***

### LDA

Defined in: [src/mva/estimators/LDA.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L29)

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new LDA(params?): LDA;
```

Defined in: [src/mva/estimators/LDA.js:33](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L33)

###### Parameters

###### params?

`Object` = `{}`

optional hyperparameters (none required for basic LDA)

###### Returns

[`LDA`](#lda)

###### Overrides

```ts
Classifier.constructor
```

#### Properties

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L514)

###### Inherited from

```ts
Classifier.labelEncoder_
```

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L515)

###### Inherited from

```ts
Classifier.classes_
```

##### params

```ts
params: object;
```

Defined in: [src/mva/estimators/LDA.js:36](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L36)

###### scale

```ts
scale: boolean = false;
```

###### scaling

```ts
scaling: number = 2;
```

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
Classifier.params
```

##### model

```ts
model: Object | null;
```

Defined in: [src/mva/estimators/LDA.js:37](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L37)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/mva/estimators/LDA.js:38](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L38)

###### Inherited from

```ts
Classifier.fitted
```

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
Classifier.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
Classifier.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
Classifier.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
Classifier.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
Classifier.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
Classifier.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L148)

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
Classifier.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): LDA;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`LDA`](#lda)

###### Inherited from

```ts
Classifier.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
Classifier.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
Classifier.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L346)

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
Classifier.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: any[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: any[];
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

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L367)

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
  `rows`: `any`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `any`[];
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
Classifier._prepareArgsForFit
```

##### predictProba()

```ts
predictProba(_X): void;
```

Defined in: [src/core/estimators/estimator.js:531](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L531)

Predict probabilities - subclasses should override
Ensures model is fitted before prediction

###### Parameters

###### \_X

`any`

###### Returns

`void`

###### Inherited from

```ts
Classifier.predictProba
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L541)

Extract and store label encoder from prepared data

###### Parameters

###### prepared

`Object`

Result from prepareXY/prepareDataset

###### Returns

`boolean`

True if encoder was found and stored

###### Inherited from

```ts
Classifier._extractLabelEncoder
```

##### \_getClasses()

```ts
_getClasses(preparedY, onlyPresentClasses?): Object;
```

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L563)

Get unique classes from labels (encoded or raw)
If labelEncoder exists, preparedY is assumed to be numeric indices [0, 1, 2, ...]
Otherwise, creates classes from unique values in preparedY

###### Parameters

###### preparedY

`any`[]

Label array (numeric if encoded, or raw labels)

###### onlyPresentClasses?

`boolean` = `true`

If true, only return classes present in preparedY

###### Returns

`Object`

{ numericY, classes }

###### Inherited from

```ts
Classifier._getClasses
```

##### \_decodeLabels()

```ts
_decodeLabels(predictions): any[];
```

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L606)

Decode numeric predictions to original labels

###### Parameters

###### predictions

`any`[]

Numeric predictions or label strings

###### Returns

`any`[]

Decoded labels (or original if no encoder)

###### Inherited from

```ts
Classifier._decodeLabels
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L622)

Default accuracy scoring:
 - score(yTrue, yPred)
 - or score({ X, y, data }) which predicts internally

###### Parameters

###### yTrueOrOpts

`any`

###### yPred

`null`

###### \_opts?

###### args?

...`any`[] = `{}`

###### Returns

`number`

###### Inherited from

```ts
Classifier.score
```

##### \_accuracy()

```ts
_accuracy(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L644)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Classifier._accuracy
```

##### fit()

```ts
fit(
   X, 
   y?, 
   opts?): LDA;
```

Defined in: [src/mva/estimators/LDA.js:53](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L53)

Fit the LDA model.

Supports a positional numeric call (`fit(Xarray, yarray)`) or a declarative
`{ X, y, data }` object (`fit({ X: 'col'|'[cols]', y: 'label', data: tableLike, omit_missing, encoders })`).

###### Parameters

###### X

`Object` \| `number`[][]

Data matrix (n samples × p features), or a declarative `{ data, X, y }` object

###### y?

`number`[] \| `string`[]

Class labels, one per sample (positional call form)

###### opts?

Fitting options (used for the positional call form)

###### scale?

`boolean`

Whether to scale columns to unit variance

###### scaling?

`number`

Ordination scaling convention

###### Returns

[`LDA`](#lda)

The fitted estimator (for chaining)

###### Overrides

```ts
Classifier.fit
```

##### transform()

```ts
transform(X): Object[];
```

Defined in: [src/mva/estimators/LDA.js:100](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L100)

Transform input X to discriminant scores (delegates to functional transform).

Accepts a numeric array or a declarative object `{ X: cols, data: tableLike }`.

###### Parameters

###### X

`Object` \| `number`[][]

Data matrix (n samples × p features), or a declarative `{ data, X }` object

###### Returns

`Object`[]

Discriminant score objects, one per row, keyed by axis (e.g. `ld1`, `ld2`, ...)

###### Overrides

```ts
Classifier.transform
```

##### predict()

```ts
predict(X): number[] | string[];
```

Defined in: [src/mva/estimators/LDA.js:116](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L116)

Predict class labels for X.

Accepts a numeric array or a declarative object `{ X: cols, data: tableLike }`.
Returns decoded labels if a label encoder is present, otherwise numeric predictions.

###### Parameters

###### X

`Object` \| `number`[][]

Data matrix (n samples × p features), or a declarative `{ data, X }` object

###### Returns

`number`[] \| `string`[]

Predicted class label for each row

###### Overrides

```ts
Classifier.predict
```

##### summary()

```ts
summary(): object;
```

Defined in: [src/mva/estimators/LDA.js:130](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L130)

Return a small summary of the fitted model.

###### Returns

`object`

###### classes

```ts
classes: any;
```

###### nComponents

```ts
nComponents: any;
```

###### eigenvalues

```ts
eigenvalues: any;
```

###### scaling

```ts
scaling: number;
```

##### getScores()

```ts
getScores(type?, scaled?): any;
```

Defined in: [src/mva/estimators/LDA.js:149](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L149)

Retrieve site or variable scores (scaled or raw).

###### Parameters

###### type?

`"sites"` \| `"samples"` \| `"variables"` \| `"loadings"`

###### scaled?

`boolean` = `true`

###### Returns

`any`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/mva/estimators/LDA.js:175](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L175)

JSON serialization helper.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'LDA';
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### model

```ts
model: Object | null;
```

###### Overrides

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): LDA;
```

Defined in: [src/mva/estimators/LDA.js:187](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/LDA.js#L187)

Restore an instance from JSON produced by toJSON().

###### Parameters

###### obj?

###### Returns

[`LDA`](#lda)

###### Overrides

```ts
Classifier.fromJSON
```

***

### PCA

Defined in: [src/mva/estimators/PCA.js:30](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L30)

#### Extends

- `Transformer`

#### Constructors

##### Constructor

```ts
new PCA(params?): PCA;
```

Defined in: [src/mva/estimators/PCA.js:31](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L31)

###### Parameters

###### params?

###### Returns

[`PCA`](#pca)

###### Overrides

```ts
Transformer.constructor
```

#### Properties

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Transformer.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Transformer._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Transformer._warnings
```

##### params

```ts
params: object;
```

Defined in: [src/mva/estimators/PCA.js:34](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L34)

###### center

```ts
center: boolean = true;
```

###### scale

```ts
scale: boolean = false;
```

###### columns

```ts
columns: null = null;
```

###### omit\_missing

```ts
omit_missing: boolean = true;
```

###### scaling

```ts
scaling: number = 2;
```

###### Inherited from

```ts
Transformer.params
```

##### model

```ts
model: Object | null;
```

Defined in: [src/mva/estimators/PCA.js:35](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L35)

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
Transformer.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
Transformer.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
Transformer.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
Transformer.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L148)

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
Transformer.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
Transformer._repr_html_
```

##### setParams()

```ts
setParams(params?): PCA;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`PCA`](#pca)

###### Inherited from

```ts
Transformer.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
Transformer.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
Transformer.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L346)

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
Transformer.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: any[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: any[];
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

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L367)

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
  `rows`: `any`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `any`[];
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
Transformer._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
Transformer.predict
```

##### fitTransform()

```ts
fitTransform(...args): void;
```

Defined in: [src/core/estimators/estimator.js:683](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L683)

Convenience: fit then transform
Returns transformed data.

###### Parameters

###### args

...`any`[]

###### Returns

`void`

###### Inherited from

```ts
Transformer.fitTransform
```

##### fit()

```ts
fit(X, opts?): PCA;
```

Defined in: [src/mva/estimators/PCA.js:52](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L52)

Fit PCA on the provided data.

Accepts either a numeric matrix (`fit(X[, opts])`) or a declarative
`{ data, columns }` object (`fit({ data, columns, center, scale, omit_missing })`).

###### Parameters

###### X

`Object` \| `number`[][]

Data matrix (n samples × p features), or a declarative `{ data, columns }` object

###### opts?

Fitting options (used for the numeric-matrix call form)

###### center?

`boolean`

Whether to mean-center the columns

###### scale?

`boolean`

Whether to scale columns to unit variance

###### columns?

`string`[]

Column names to use for declarative inputs

###### omit_missing?

`boolean`

Whether to drop rows with missing values

###### scaling?

`number`

Ordination scaling convention

###### Returns

[`PCA`](#pca)

The fitted estimator (for chaining)

###### Overrides

```ts
Transformer.fit
```

##### transform()

```ts
transform(X): Object[];
```

Defined in: [src/mva/estimators/PCA.js:99](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L99)

Transform new data into principal-component scores using the fitted model.

Accepts numeric arrays or declarative table objects { data, columns }.

###### Parameters

###### X

`Object` \| `number`[][]

Data matrix (n samples × p features), or a declarative `{ data, columns }` object

###### Returns

`Object`[]

Score objects, one per row, keyed by component (e.g. `pc1`, `pc2`, ...)

###### Overrides

```ts
Transformer.transform
```

##### cumulativeVariance()

```ts
cumulativeVariance(): number[];
```

Defined in: [src/mva/estimators/PCA.js:125](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L125)

Helper to expose functional cumulative variance.

###### Returns

`number`[]

##### summary()

```ts
summary(): object;
```

Defined in: [src/mva/estimators/PCA.js:133](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L133)

Provide lightweight summary of the fitted model.

###### Returns

`object`

###### nComponents

```ts
nComponents: any;
```

###### eigenvalues

```ts
eigenvalues: any;
```

###### varianceExplained

```ts
varianceExplained: any;
```

###### cumulativeVariance

```ts
cumulativeVariance: number[];
```

###### centered

```ts
centered: boolean;
```

###### scaled

```ts
scaled: boolean;
```

###### scaling

```ts
scaling: number;
```

###### means

```ts
means: any;
```

###### sds

```ts
sds: any;
```

##### getScores()

```ts
getScores(type?, scaled?): any;
```

Defined in: [src/mva/estimators/PCA.js:156](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L156)

Retrieve site or variable scores with optional scaling.

###### Parameters

###### type?

`"sites"` \| `"samples"` \| `"variables"` \| `"loadings"`

###### scaled?

`boolean` = `true`

return scaled or raw coordinates

###### Returns

`any`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/mva/estimators/PCA.js:176](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L176)

Serialization helper for saving estimator state.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'PCA';
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### model

```ts
model: Object | null;
```

###### Overrides

```ts
Transformer.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): PCA;
```

Defined in: [src/mva/estimators/PCA.js:188](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/PCA.js#L188)

Restore PCA instance from JSON produced by toJSON().

###### Parameters

###### obj?

###### Returns

[`PCA`](#pca)

###### Overrides

```ts
Transformer.fromJSON
```

***

### RDA

Defined in: [src/mva/estimators/RDA.js:57](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L57)

#### Extends

- `Transformer`

#### Constructors

##### Constructor

```ts
new RDA(params?): RDA;
```

Defined in: [src/mva/estimators/RDA.js:58](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L58)

###### Parameters

###### params?

###### Returns

[`RDA`](#rda)

###### Overrides

```ts
Transformer.constructor
```

#### Properties

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Transformer.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Transformer._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Transformer._warnings
```

##### params

```ts
params: object;
```

Defined in: [src/mva/estimators/RDA.js:61](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L61)

###### scale

```ts
scale: boolean = false;
```

###### omit\_missing

```ts
omit_missing: boolean = true;
```

###### scaling

```ts
scaling: number = 2;
```

###### constrained

```ts
constrained: boolean = true;
```

###### Inherited from

```ts
Transformer.params
```

##### model

```ts
model: 
  | {
  omit_missing: boolean;
  constructor: Function;
  toString: string;
  toLocaleString: string;
  valueOf: Object;
  hasOwnProperty: boolean;
  isPrototypeOf: boolean;
  propertyIsEnumerable: boolean;
}
  | null;
```

Defined in: [src/mva/estimators/RDA.js:62](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L62)

###### Union Members

###### Type Literal

```ts
{
  omit_missing: boolean;
  constructor: Function;
  toString: string;
  toLocaleString: string;
  valueOf: Object;
  hasOwnProperty: boolean;
  isPrototypeOf: boolean;
  propertyIsEnumerable: boolean;
}
```

###### omit\_missing

```ts
omit_missing: boolean = omitMissing;
```

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

***

`null`

#### Methods

##### isFitted()

```ts
isFitted(): boolean;
```

Defined in: [src/core/estimators/estimator.js:36](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L36)

Check if model is fitted

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.isFitted
```

##### getState()

```ts
getState(): Object;
```

Defined in: [src/core/estimators/estimator.js:65](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L65)

Get comprehensive model state

###### Returns

`Object`

State information including fitted status, memory estimate, warnings

###### Inherited from

```ts
Transformer.getState
```

##### getMemoryUsage()

```ts
getMemoryUsage(): string;
```

Defined in: [src/core/estimators/estimator.js:97](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L97)

Get memory usage in human-readable format

###### Returns

`string`

Memory usage string (e.g., "2.3 MB" or "145 KB")

###### Inherited from

```ts
Transformer.getMemoryUsage
```

##### getWarnings()

```ts
getWarnings(): Object[];
```

Defined in: [src/core/estimators/estimator.js:124](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L124)

Get all warnings

###### Returns

`Object`[]

Array of warning objects

###### Inherited from

```ts
Transformer.getWarnings
```

##### hasWarnings()

```ts
hasWarnings(): boolean;
```

Defined in: [src/core/estimators/estimator.js:132](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L132)

Check if model has warnings

###### Returns

`boolean`

###### Inherited from

```ts
Transformer.hasWarnings
```

##### clearWarnings()

```ts
clearWarnings(): void;
```

Defined in: [src/core/estimators/estimator.js:139](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L139)

Clear all warnings

###### Returns

`void`

###### Inherited from

```ts
Transformer.clearWarnings
```

##### getWarningsByType()

```ts
getWarningsByType(type): Object[];
```

Defined in: [src/core/estimators/estimator.js:148](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L148)

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
Transformer.getWarningsByType
```

##### \_repr\_html\_()

```ts
_repr_html_(): string;
```

Defined in: [src/core/estimators/estimator.js:201](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L201)

Observable/Jupyter HTML representation

###### Returns

`string`

HTML representation

###### Inherited from

```ts
Transformer._repr_html_
```

##### setParams()

```ts
setParams(params?): RDA;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`RDA`](#rda)

###### Inherited from

```ts
Transformer.setParams
```

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/core/estimators/estimator.js:294](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L294)

Get a shallow copy of parameters.

###### Returns

`Object`

###### Inherited from

```ts
Transformer.getParams
```

##### save()

```ts
save(): string;
```

Defined in: [src/core/estimators/estimator.js:329](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L329)

Save model to JSON string

###### Returns

`string`

JSON representation of the model

###### Inherited from

```ts
Transformer.save
```

##### load()

```ts
static load(jsonString): Estimator;
```

Defined in: [src/core/estimators/estimator.js:346](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L346)

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
Transformer.load
```

##### \_prepareArgsForFit()

```ts
_prepareArgsForFit(args?): 
  | {
  X: any[][];
  y: any[];
  columnsX: any[];
  rows: any[];
  prepared: boolean;
  columns?: undefined;
  raw?: undefined;
}
  | {
  y?: undefined;
  columnsX?: undefined;
  X: any[][];
  columns: any[];
  rows: any[];
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

Defined in: [src/core/estimators/estimator.js:367](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L367)

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
  `rows`: `any`[];
  `prepared`: `boolean`;
  `columns?`: `undefined`;
  `raw?`: `undefined`;
\}
  \| \{
  `y?`: `undefined`;
  `columnsX?`: `undefined`;
  `X`: `any`[][];
  `columns`: `any`[];
  `rows`: `any`[];
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
Transformer._prepareArgsForFit
```

##### predict()

```ts
predict(): void;
```

Defined in: [src/core/estimators/estimator.js:424](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L424)

Predict should be implemented by supervised estimators.

###### Returns

`void`

###### Inherited from

```ts
Transformer.predict
```

##### fitTransform()

```ts
fitTransform(...args): void;
```

Defined in: [src/core/estimators/estimator.js:683](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/estimators/estimator.js#L683)

Convenience: fit then transform
Returns transformed data.

###### Parameters

###### args

...`any`[]

###### Returns

`void`

###### Inherited from

```ts
Transformer.fitTransform
```

##### fit()

```ts
fit(
   Y, 
   X?, 
   opts?): RDA;
```

Defined in: [src/mva/estimators/RDA.js:79](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L79)

Fit the RDA model, constraining the response table Y by the predictor table X.

Accepts a positional numeric call (`fit(Y, X[, opts])`) or a declarative
`{ data, response, predictors }` object.

###### Parameters

###### Y

`Object` \| `number`[][]

Response matrix (n samples × q responses), or a declarative `{ data, response, predictors }` object

###### X?

`number`[][] = `null`

Predictor/explanatory matrix (n samples × p predictors), for the positional call form

###### opts?

Fitting options

###### scale?

`boolean`

Whether to scale columns to unit variance

###### scaling?

`number`

Ordination scaling convention

###### constrained?

`boolean`

Whether to fit the constrained (RDA) rather than unconstrained ordination

###### omit_missing?

`boolean`

Whether to drop rows with missing values (declarative inputs)

###### Returns

[`RDA`](#rda)

The fitted estimator (for chaining)

###### Overrides

```ts
Transformer.fit
```

##### transform()

```ts
transform(
   Y, 
   X?, 
   opts?): Object[];
```

Defined in: [src/mva/estimators/RDA.js:152](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L152)

Transform new data into RDA canonical (site) scores using the fitted model.

Accepts positional numeric matrices (`transform(Y, X)`) or a declarative
`{ data, response, predictors }` object.

###### Parameters

###### Y

`Object` \| `number`[][]

Response matrix (n samples × q responses), or a declarative `{ data, response, predictors }` object

###### X?

`number`[][]

Predictor/explanatory matrix (n samples × p predictors), for the positional call form

###### opts?

Transform options

###### omit_missing?

`boolean`

Whether to drop rows with missing values (declarative inputs)

###### Returns

`Object`[]

Canonical score objects, one per row (keyed `rda1`, `rda2`, ...)

###### Overrides

```ts
Transformer.transform
```

##### summary()

```ts
summary(): object;
```

Defined in: [src/mva/estimators/RDA.js:178](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L178)

###### Returns

`object`

###### constrainedVariance

```ts
constrainedVariance: any;
```

###### eigenvalues

```ts
eigenvalues: any;
```

###### varianceExplained

```ts
varianceExplained: any;
```

###### samples

```ts
samples: any = n;
```

###### predictors

```ts
predictors: any = p;
```

###### responses

```ts
responses: any = q;
```

###### scaling

```ts
scaling: any;
```

###### constrained

```ts
constrained: any;
```

##### permutationTest()

```ts
permutationTest(options?): Object;
```

Defined in: [src/mva/estimators/RDA.js:211](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L211)

Global permutation test of the constrained ordination (vegan `anova.cca`).

###### Parameters

###### options?

###### permutations?

`number`

###### seed?

`number`

###### Returns

`Object`

pseudo-F, permutation p-value, df and inertia decomposition.

##### anova()

```ts
anova(options?): Object;
```

Defined in: [src/mva/estimators/RDA.js:219](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L219)

Alias for [RDA#permutationTest](#permutationtest) mirroring R's `anova(rda_model)`.

###### Parameters

###### options?

###### Returns

`Object`

##### getScores()

```ts
getScores(type?, scaled?): any;
```

Defined in: [src/mva/estimators/RDA.js:228](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L228)

Retrieve site (scores), response loadings, or predictor constraint scores.

###### Parameters

###### type?

  \| `"sites"`
  \| `"samples"`
  \| `"variables"`
  \| `"loadings"`
  \| `"predictors"`
  \| `"responses"`
  \| `"constraints"`

###### scaled?

`boolean` = `true`

###### Returns

`any`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/mva/estimators/RDA.js:263](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L263)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'RDA';
```

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### model

```ts
model: 
  | {
  omit_missing: boolean;
  constructor: Function;
  toString: string;
  toLocaleString: string;
  valueOf: Object;
  hasOwnProperty: boolean;
  isPrototypeOf: boolean;
  propertyIsEnumerable: boolean;
}
  | null;
```

###### Union Members

###### Type Literal

```ts
{
  omit_missing: boolean;
  constructor: Function;
  toString: string;
  toLocaleString: string;
  valueOf: Object;
  hasOwnProperty: boolean;
  isPrototypeOf: boolean;
  propertyIsEnumerable: boolean;
}
```

###### omit\_missing

```ts
omit_missing: boolean = omitMissing;
```

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

***

`null`

###### Overrides

```ts
Transformer.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): RDA;
```

Defined in: [src/mva/estimators/RDA.js:272](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/estimators/RDA.js#L272)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`RDA`](#rda)

###### Overrides

```ts
Transformer.fromJSON
```

## References

### CompositionalOutlierDetector

Re-exports [CompositionalOutlierDetector](/api/multivariate/composition#compositionaloutlierdetector)

***

### CompositionalImputer

Re-exports [CompositionalImputer](/api/multivariate/composition#compositionalimputer)
