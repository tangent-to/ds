---
layout: default
title: Machine Learning
parent: API Reference
nav_order: 2
has_children: true
permalink: /api/machine-learning
---
# ml

## Namespaces

- [criteria](/api/machine-learning/criteria)
- [dbscan](/api/machine-learning/dbscan)
- [distances](/api/machine-learning/distances)
- [ml/explain](/api/machine-learning/ml-explain)
- [hca](/api/machine-learning/hca)
- [interpret](/api/machine-learning/interpret)
- [kmeans](/api/machine-learning/kmeans)
- [loss](/api/machine-learning/loss)
- [metrics](/api/machine-learning/metrics)
- [mlp](/api/machine-learning/mlp)
- [polynomial](/api/machine-learning/polynomial)
- [preprocessing](/api/machine-learning/preprocessing)
- [silhouette](/api/machine-learning/silhouette)
- [train](/api/machine-learning/train)
- [tuning](/api/machine-learning/tuning)
- [utils](/api/machine-learning/utils)
- [validation](/api/machine-learning/validation)

## Classes

### ConsensusCluster

Defined in: [src/clustering/ConsensusCluster.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L29)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new ConsensusCluster(options?): ConsensusCluster;
```

Defined in: [src/clustering/ConsensusCluster.js:36](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L36)

###### Parameters

###### options?

###### estimators

`Object`[] = `[]`

Array of clustering estimators

###### threshold?

`number` = `0.5`

Minimum co-association for same cluster (0-1)

###### linkage?

`string` = `'single'`

Linkage for final clustering

###### Returns

[`ConsensusCluster`](#consensuscluster)

###### Overrides

```ts
Estimator.constructor
```

#### Properties

##### estimators

```ts
estimators: Object[];
```

Defined in: [src/clustering/ConsensusCluster.js:46](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L46)

##### threshold

```ts
threshold: number;
```

Defined in: [src/clustering/ConsensusCluster.js:47](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L47)

##### linkage

```ts
linkage: string;
```

Defined in: [src/clustering/ConsensusCluster.js:48](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L48)

##### coAssocMatrix

```ts
coAssocMatrix: any[][] | null;
```

Defined in: [src/clustering/ConsensusCluster.js:49](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L49)

##### labels

```ts
labels: any[] | null;
```

Defined in: [src/clustering/ConsensusCluster.js:50](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L50)

##### X\_train

```ts
X_train: Object | any[] | null;
```

Defined in: [src/clustering/ConsensusCluster.js:51](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L51)

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
Estimator.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Estimator.fitted
```

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

```ts
Estimator._warnings
```

#### Accessors

##### nClusters

###### Get Signature

```ts
get nClusters(): number;
```

Defined in: [src/clustering/ConsensusCluster.js:247](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L247)

Get number of clusters (excluding noise)

###### Returns

`number`

##### nNoise

###### Get Signature

```ts
get nNoise(): number;
```

Defined in: [src/clustering/ConsensusCluster.js:255](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L255)

Get number of noise points

###### Returns

`number`

##### agreementScore

###### Get Signature

```ts
get agreementScore(): number;
```

Defined in: [src/clustering/ConsensusCluster.js:265](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L265)

Get overall agreement score
Measures how consistent the input clusterings are

###### Returns

`number`

Score between 0 (no agreement) and 1 (perfect agreement)

#### Methods

##### fit()

```ts
fit(X): ConsensusCluster;
```

Defined in: [src/clustering/ConsensusCluster.js:59](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L59)

Fit consensus clustering

###### Parameters

###### X

`Object` \| `any`[]

Data to cluster (raw array or {data, columns})

###### Returns

[`ConsensusCluster`](#consensuscluster)

###### Overrides

```ts
Estimator.fit
```

##### getConsensusStrength()

```ts
getConsensusStrength(): number[];
```

Defined in: [src/clustering/ConsensusCluster.js:186](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L186)

Get consensus strength for each sample
Higher values = stronger agreement across models

###### Returns

`number`[]

Strength scores between 0 and 1

##### getClusterStrength()

```ts
getClusterStrength(): Object;
```

Defined in: [src/clustering/ConsensusCluster.js:221](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L221)

Get average consensus strength per cluster

###### Returns

`Object`

{cluster: avgStrength}

##### getEstimatorAgreement()

```ts
getEstimatorAgreement(): object[];
```

Defined in: [src/clustering/ConsensusCluster.js:286](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L286)

Get detailed comparison of input clusterings
Shows how each estimator contributed

###### Returns

`object`[]

##### summary()

```ts
summary(): 
  | {
  fitted: boolean;
  nEstimators: number;
  threshold: number;
  nClusters?: undefined;
  nNoise?: undefined;
  nSamples?: undefined;
  noiseRatio?: undefined;
  overallAgreement?: undefined;
  avgConsensusStrength?: undefined;
  clusterStrengths?: undefined;
}
  | {
  fitted: boolean;
  nEstimators: number;
  threshold: number;
  nClusters: number;
  nNoise: number;
  nSamples: number;
  noiseRatio: number;
  overallAgreement: number;
  avgConsensusStrength: number;
  clusterStrengths: Object;
};
```

Defined in: [src/clustering/ConsensusCluster.js:341](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L341)

Summary statistics

###### Returns

  \| \{
  `fitted`: `boolean`;
  `nEstimators`: `number`;
  `threshold`: `number`;
  `nClusters?`: `undefined`;
  `nNoise?`: `undefined`;
  `nSamples?`: `undefined`;
  `noiseRatio?`: `undefined`;
  `overallAgreement?`: `undefined`;
  `avgConsensusStrength?`: `undefined`;
  `clusterStrengths?`: `undefined`;
\}
  \| \{
  `fitted`: `boolean`;
  `nEstimators`: `number`;
  `threshold`: `number`;
  `nClusters`: `number`;
  `nNoise`: `number`;
  `nSamples`: `number`;
  `noiseRatio`: `number`;
  `overallAgreement`: `number`;
  `avgConsensusStrength`: `number`;
  `clusterStrengths`: `Object`;
\}

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/clustering/ConsensusCluster.js:370](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L370)

Serialization (simplified)

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'ConsensusCluster';
```

###### threshold

```ts
threshold: number;
```

###### linkage

```ts
linkage: string;
```

###### fitted

```ts
fitted: boolean;
```

###### labels

```ts
labels: any[] | null;
```

###### coAssocMatrix

```ts
coAssocMatrix: any[][] | null;
```

###### nEstimators

```ts
nEstimators: number;
```

###### Overrides

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj): ConsensusCluster;
```

Defined in: [src/clustering/ConsensusCluster.js:382](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/clustering/ConsensusCluster.js#L382)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj

`any`

###### Returns

[`ConsensusCluster`](#consensuscluster)

###### Overrides

```ts
Estimator.fromJSON
```

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): ConsensusCluster;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`ConsensusCluster`](#consensuscluster)

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
Estimator.predict
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

***

### DBSCAN

Defined in: [src/ml/estimators/DBSCAN.js:17](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L17)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new DBSCAN(params?): DBSCAN;
```

Defined in: [src/ml/estimators/DBSCAN.js:23](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L23)

###### Parameters

###### params?

{ eps, minSamples }

###### eps

`number` = `0.5`

Maximum distance between two points for neighborhood (default: 0.5)

###### minSamples

`number` = `5`

Minimum number of points to form a dense region (default: 5)

###### Returns

[`DBSCAN`](#dbscan)

###### Overrides

```ts
Estimator.constructor
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
Estimator.params
```

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

```ts
Estimator._warnings
```

##### eps

```ts
eps: number;
```

Defined in: [src/ml/estimators/DBSCAN.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L25)

##### minSamples

```ts
minSamples: number;
```

Defined in: [src/ml/estimators/DBSCAN.js:26](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L26)

##### model

```ts
model: Object | null;
```

Defined in: [src/ml/estimators/DBSCAN.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L29)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/ml/estimators/DBSCAN.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L30)

###### Inherited from

```ts
Estimator.fitted
```

##### X\_train

```ts
X_train: any;
```

Defined in: [src/ml/estimators/DBSCAN.js:31](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L31)

##### labels

```ts
labels: any;
```

Defined in: [src/ml/estimators/DBSCAN.js:85](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L85)

##### nClusters

```ts
nClusters: any;
```

Defined in: [src/ml/estimators/DBSCAN.js:86](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L86)

##### nNoise

```ts
nNoise: any;
```

Defined in: [src/ml/estimators/DBSCAN.js:87](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L87)

##### coreSampleIndices

```ts
coreSampleIndices: any;
```

Defined in: [src/ml/estimators/DBSCAN.js:88](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L88)

#### Accessors

##### coreSampleMask

###### Get Signature

```ts
get coreSampleMask(): any[];
```

Defined in: [src/ml/estimators/DBSCAN.js:128](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L128)

Get core sample mask (boolean array indicating which samples are core points)

###### Returns

`any`[]

##### components

###### Get Signature

```ts
get components(): any;
```

Defined in: [src/ml/estimators/DBSCAN.js:140](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L140)

Get components (core samples) - returns array of core sample data points

###### Returns

`any`

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): DBSCAN;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`DBSCAN`](#dbscan)

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
fit(X, opts?): DBSCAN;
```

Defined in: [src/ml/estimators/DBSCAN.js:43](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L43)

Fit the DBSCAN model.

Accepts:
 - numeric input: fit(Xarray, { eps, minSamples })
 - declarative input: fit({ data: tableLike, columns: ['c1','c2'], eps, ... })

Returns this.

###### Parameters

###### X

`any`

###### opts?

###### Returns

[`DBSCAN`](#dbscan)

###### Overrides

```ts
Estimator.fit
```

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/estimators/DBSCAN.js:106](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L106)

Predict cluster labels for new data.

Note: DBSCAN doesn't naturally support prediction on new points.
This assigns new points to the cluster of their nearest core point
if within eps distance, otherwise marks as noise (-1).

Accepts:
 - numeric array: predict([[x1,x2], [x1,x2], ...])
 - declarative: predict({ data: tableLike, columns: ['c1','c2'], omit_missing: true })

###### Parameters

###### X

`any`

###### Returns

`number`[]

###### Overrides

```ts
Estimator.predict
```

##### summary()

```ts
summary(): object;
```

Defined in: [src/ml/estimators/DBSCAN.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L148)

Convenience: return summary stats for fitted model

###### Returns

`object`

###### eps

```ts
eps: number;
```

###### minSamples

```ts
minSamples: number;
```

###### nClusters

```ts
nClusters: any;
```

###### nNoise

```ts
nNoise: any;
```

###### nSamples

```ts
nSamples: any;
```

###### nCore

```ts
nCore: any;
```

###### noiseRatio

```ts
noiseRatio: number;
```

###### coreRatio

```ts
coreRatio: number;
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/DBSCAN.js:171](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L171)

Serialization helper

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'DBSCAN';
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

###### X\_train

```ts
X_train: any;
```

###### Overrides

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): DBSCAN;
```

Defined in: [src/ml/estimators/DBSCAN.js:181](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DBSCAN.js#L181)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`DBSCAN`](#dbscan)

###### Overrides

```ts
Estimator.fromJSON
```

***

### DecisionTreeClassifier

Defined in: [src/ml/estimators/DecisionTree.js:526](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L526)

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new DecisionTreeClassifier(opts?): DecisionTreeClassifier;
```

Defined in: [src/ml/estimators/DecisionTree.js:527](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L527)

###### Parameters

###### opts?

###### Returns

[`DecisionTreeClassifier`](#decisiontreeclassifier)

###### Overrides

```ts
Classifier.constructor
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
Classifier.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Classifier.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L514)

###### Inherited from

[`GAMClassifier`](#gamclassifier).[`labelEncoder_`](#labelencoder_-1)

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L515)

###### Inherited from

[`GAMClassifier`](#gamclassifier).[`classes_`](#classes_-1)

##### tree

```ts
tree: DecisionTreeBase;
```

Defined in: [src/ml/estimators/DecisionTree.js:529](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L529)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): number[];
```

Defined in: [src/ml/estimators/DecisionTree.js:575](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L575)

Get feature importances

###### Returns

`number`[]

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
Classifier.isFitted
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
Classifier.getState
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
Classifier.getMemoryUsage
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
Classifier.getWarnings
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
Classifier.hasWarnings
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
Classifier.clearWarnings
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
Classifier.getWarningsByType
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
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): DecisionTreeClassifier;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`DecisionTreeClassifier`](#decisiontreeclassifier)

###### Inherited from

```ts
Classifier.setParams
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
Classifier.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Classifier.fromJSON
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
Classifier.save
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
Classifier.load
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
Classifier._prepareArgsForFit
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
Classifier.transform
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L541)

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

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L563)

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

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L606)

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

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L622)

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

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L644)

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
fit(X, y?): DecisionTreeClassifier;
```

Defined in: [src/ml/estimators/DecisionTree.js:532](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L532)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`DecisionTreeClassifier`](#decisiontreeclassifier)

###### Overrides

```ts
Classifier.fit
```

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/ml/estimators/DecisionTree.js:552](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L552)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`[]

###### Overrides

```ts
Classifier.predict
```

##### predictProba()

```ts
predictProba(X): any;
```

Defined in: [src/ml/estimators/DecisionTree.js:558](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L558)

Predict probabilities - subclasses should override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`

###### Overrides

```ts
Classifier.predictProba
```

##### apply()

```ts
apply(X): number[];
```

Defined in: [src/ml/estimators/DecisionTree.js:580](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L580)

Apply tree to X, return leaf indices

###### Parameters

###### X

`any`

###### Returns

`number`[]

##### decisionPath()

```ts
decisionPath(X): any[][];
```

Defined in: [src/ml/estimators/DecisionTree.js:585](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L585)

Return decision path

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### getDepth()

```ts
getDepth(): number;
```

Defined in: [src/ml/estimators/DecisionTree.js:590](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L590)

Get maximum depth of tree

###### Returns

`number`

##### getNLeaves()

```ts
getNLeaves(): number;
```

Defined in: [src/ml/estimators/DecisionTree.js:595](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L595)

Get number of leaves

###### Returns

`number`

##### exportTree()

```ts
exportTree(featureNames?, classNames?): string;
```

Defined in: [src/ml/estimators/DecisionTree.js:600](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L600)

Export tree to DOT format

###### Parameters

###### featureNames?

`null` = `null`

###### classNames?

`null` = `null`

###### Returns

`string`

##### exportText()

```ts
exportText(featureNames?): string;
```

Defined in: [src/ml/estimators/DecisionTree.js:605](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L605)

Export tree as ASCII text

###### Parameters

###### featureNames?

`null` = `null`

###### Returns

`string`

***

### DecisionTreeRegressor

Defined in: [src/ml/estimators/DecisionTree.js:610](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L610)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new DecisionTreeRegressor(opts?): DecisionTreeRegressor;
```

Defined in: [src/ml/estimators/DecisionTree.js:611](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L611)

###### Parameters

###### opts?

###### Returns

[`DecisionTreeRegressor`](#decisiontreeregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### tree

```ts
tree: DecisionTreeBase;
```

Defined in: [src/ml/estimators/DecisionTree.js:613](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L613)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): number[];
```

Defined in: [src/ml/estimators/DecisionTree.js:627](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L627)

Get feature importances

###### Returns

`number`[]

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): DecisionTreeRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`DecisionTreeRegressor`](#decisiontreeregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Regressor.fromJSON
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(X, y?): DecisionTreeRegressor;
```

Defined in: [src/ml/estimators/DecisionTree.js:616](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L616)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`DecisionTreeRegressor`](#decisiontreeregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X): any;
```

Defined in: [src/ml/estimators/DecisionTree.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L622)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`

###### Overrides

```ts
Regressor.predict
```

##### apply()

```ts
apply(X): number[];
```

Defined in: [src/ml/estimators/DecisionTree.js:632](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L632)

Apply tree to X, return leaf indices

###### Parameters

###### X

`any`

###### Returns

`number`[]

##### decisionPath()

```ts
decisionPath(X): any[][];
```

Defined in: [src/ml/estimators/DecisionTree.js:637](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L637)

Return decision path

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### getDepth()

```ts
getDepth(): number;
```

Defined in: [src/ml/estimators/DecisionTree.js:642](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L642)

Get maximum depth of tree

###### Returns

`number`

##### getNLeaves()

```ts
getNLeaves(): number;
```

Defined in: [src/ml/estimators/DecisionTree.js:647](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L647)

Get number of leaves

###### Returns

`number`

##### exportTree()

```ts
exportTree(featureNames?): string;
```

Defined in: [src/ml/estimators/DecisionTree.js:652](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L652)

Export tree to DOT format

###### Parameters

###### featureNames?

`null` = `null`

###### Returns

`string`

##### exportText()

```ts
exportText(featureNames?): string;
```

Defined in: [src/ml/estimators/DecisionTree.js:657](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/DecisionTree.js#L657)

Export tree as ASCII text

###### Parameters

###### featureNames?

`null` = `null`

###### Returns

`string`

***

### GAMRegressor

Defined in: [src/ml/estimators/GAM.js:210](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L210)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new GAMRegressor(opts?): GAMRegressor;
```

Defined in: [src/ml/estimators/GAM.js:211](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L211)

###### Parameters

###### opts?

###### Returns

[`GAMRegressor`](#gamregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### gam

```ts
gam: GAMBase;
```

Defined in: [src/ml/estimators/GAM.js:213](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L213)

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): GAMRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GAMRegressor`](#gamregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Regressor.fromJSON
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(X, y?): GAMRegressor;
```

Defined in: [src/ml/estimators/GAM.js:216](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L216)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`GAMRegressor`](#gamregressor)

###### Overrides

```ts
Regressor.fit
```

##### \_buildPenaltyMatrix()

```ts
_buildPenaltyMatrix(): Matrix;
```

Defined in: [src/ml/estimators/GAM.js:284](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L284)

###### Returns

[`Matrix`](/api/core/linalg#matrix)

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/estimators/GAM.js:311](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L311)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`number`[]

###### Overrides

```ts
Regressor.predict
```

##### predictWithInterval()

```ts
predictWithInterval(X, level?): Object[];
```

Defined in: [src/ml/estimators/GAM.js:326](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L326)

Get confidence intervals for predictions

###### Parameters

###### X

`any`[]

Input data

###### level?

`number` = `0.95`

Confidence level (default: 0.95)

###### Returns

`Object`[]

Array of { fitted, se, lower, upper } for each observation

##### summary()

```ts
summary(): Object;
```

Defined in: [src/ml/estimators/GAM.js:357](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L357)

###### Returns

`Object`

***

### GAMClassifier

Defined in: [src/ml/estimators/GAM.js:376](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L376)

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new GAMClassifier(opts?): GAMClassifier;
```

Defined in: [src/ml/estimators/GAM.js:377](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L377)

###### Parameters

###### opts?

###### Returns

[`GAMClassifier`](#gamclassifier)

###### Overrides

```ts
Classifier.constructor
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
Classifier.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Classifier.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L514)

###### Inherited from

```ts
Classifier.labelEncoder_
```

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L515)

###### Inherited from

```ts
Classifier.classes_
```

##### gam

```ts
gam: GAMBase;
```

Defined in: [src/ml/estimators/GAM.js:379](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L379)

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
Classifier.isFitted
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
Classifier.getState
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
Classifier.getMemoryUsage
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
Classifier.getWarnings
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
Classifier.hasWarnings
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
Classifier.clearWarnings
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
Classifier.getWarningsByType
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
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): GAMClassifier;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GAMClassifier`](#gamclassifier)

###### Inherited from

```ts
Classifier.setParams
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
Classifier.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Classifier.fromJSON
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
Classifier.save
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
Classifier.load
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
Classifier._prepareArgsForFit
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
Classifier.transform
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L541)

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

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L563)

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

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L606)

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

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L622)

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

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L644)

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
fit(X, y?): GAMClassifier;
```

Defined in: [src/ml/estimators/GAM.js:382](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L382)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`GAMClassifier`](#gamclassifier)

###### Overrides

```ts
Classifier.fit
```

##### \_buildPenaltyMatrix()

```ts
_buildPenaltyMatrix(): Matrix;
```

Defined in: [src/ml/estimators/GAM.js:434](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L434)

###### Returns

[`Matrix`](/api/core/linalg#matrix)

##### \_computeLinearPredictors()

```ts
_computeLinearPredictors(X): any[][];
```

Defined in: [src/ml/estimators/GAM.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L461)

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### predictProba()

```ts
predictProba(X): object[];
```

Defined in: [src/ml/estimators/GAM.js:485](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L485)

Predict probabilities - subclasses should override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`object`[]

###### Overrides

```ts
Classifier.predictProba
```

##### predict()

```ts
predict(X): (string | null)[];
```

Defined in: [src/ml/estimators/GAM.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L514)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

(`string` \| `null`)[]

###### Overrides

```ts
Classifier.predict
```

##### summary()

```ts
summary(): Object;
```

Defined in: [src/ml/estimators/GAM.js:533](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GAM.js#L533)

###### Returns

`Object`

***

### GaussianProcessRegressor

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:204](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L204)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new GaussianProcessRegressor(opts?): GaussianProcessRegressor;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:214](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L214)

###### Parameters

###### opts?

Options

###### kernel

`string` \| [`Kernel`](#kernel-1)

Kernel instance or type ('rbf', 'periodic', 'rational_quadratic')

###### lengthScale

`number`

Length scale for kernel (default: 1.0)

###### variance

`number`

Signal variance (default: 1.0)

###### alpha

`number`

Noise level / regularization (default: 1e-10)

###### noiseLevel

`number`

Alias for alpha

###### period

`number`

Period for periodic kernel

###### Returns

[`GaussianProcessRegressor`](#gaussianprocessregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### kernel

```ts
kernel: 
  | Kernel
  | RBF
  | Periodic
  | RationalQuadratic
  | Matern
  | ConstantKernel;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:219](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L219)

##### alpha

```ts
alpha: number;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:249](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L249)

##### optimize

```ts
optimize: boolean;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:255](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L255)

##### nRestarts

```ts
nRestarts: any;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:258](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L258)

##### \_seed

```ts
_seed: any;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:259](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L259)

##### \_XTrain

```ts
_XTrain: Matrix | null;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:262](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L262)

##### \_yTrain

```ts
_yTrain: any[] | null;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:263](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L263)

##### \_L

```ts
_L: any;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:264](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L264)

##### \_alphaVector

```ts
_alphaVector: any[] | null;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:265](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L265)

##### logMarginalLikelihood\_

```ts
logMarginalLikelihood_: number | null;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:266](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L266)

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): GaussianProcessRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GaussianProcessRegressor`](#gaussianprocessregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(X, y?): GaussianProcessRegressor;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:275](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L275)

Fit the GP to training data

###### Parameters

###### X

`Object` \| `any`[]

Training inputs (n x d) or { X, y, data }

###### y?

`any`[] = `null`

Training targets (n)

###### Returns

[`GaussianProcessRegressor`](#gaussianprocessregressor)

###### Overrides

```ts
Regressor.fit
```

##### logMarginalLikelihood()

```ts
logMarginalLikelihood(): number;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:333](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L333)

Log marginal likelihood of the training data under the current
hyperparameters: log p(y|X) = -½ yᵀK⁻¹y - ½ log|K| - n/2 log(2π).
Requires the model to have seen training data (via fit).

###### Returns

`number`

##### predict()

```ts
predict(X, opts?): Object | any[];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:410](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L410)

Predict at test points

###### Parameters

###### X

`any`[]

Test inputs (m x d)

###### opts?

Options

###### returnStd

`boolean`

Return standard deviations

###### Returns

`Object` \| `any`[]

Predictions, or { mean, std } if returnStd=true

###### Overrides

```ts
Regressor.predict
```

##### sample()

```ts
sample(
   X, 
   nSamples?, 
   seed?): any[][];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:451](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L451)

Sample from the posterior distribution

###### Parameters

###### X

`any`[]

Test inputs

###### nSamples?

`number` = `1`

Number of samples

###### seed?

`number` = `null`

Random seed for reproducibility

###### Returns

`any`[][]

Array of samples

##### samplePrior()

```ts
samplePrior(
   X, 
   nSamples?, 
   seed?): any[][];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:484](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L484)

Sample from the prior (unfitted GP)

###### Parameters

###### X

`any`[]

Input points

###### nSamples?

`number` = `1`

Number of samples

###### seed?

`number` = `null`

Random seed for reproducibility

###### Returns

`any`[][]

Array of samples

##### \_computePosteriorCovariance()

```ts
_computePosteriorCovariance(XTest, KStar): object;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:507](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L507)

###### Parameters

###### XTest

`any`

###### KStar

`any`

###### Returns

`object`

###### covarianceMatrix

```ts
covarianceMatrix: any;
```

###### diag

```ts
diag: any[];
```

##### \_solveCholesky()

```ts
_solveCholesky(L, y): any[];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:550](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L550)

###### Parameters

###### L

`any`

###### y

`any`

###### Returns

`any`[]

##### \_forwardSubstitution()

```ts
_forwardSubstitution(L, b): any[];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:555](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L555)

###### Parameters

###### L

`any`

###### b

`any`

###### Returns

`any`[]

##### \_backSubstitution()

```ts
_backSubstitution(L, b): any[];
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:570](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L570)

###### Parameters

###### L

`any`

###### b

`any`

###### Returns

`any`[]

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:585](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L585)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### type

```ts
type: string = 'GaussianProcessRegressor';
```

###### kernel

```ts
kernel: object;
```

###### kernel.type

```ts
type: string;
```

###### kernel.params

```ts
params: Object;
```

###### alpha

```ts
alpha: number;
```

###### fitted

```ts
fitted: boolean;
```

###### XTrain

```ts
XTrain: number[][] | null;
```

###### yTrain

```ts
yTrain: any[] | null;
```

###### L

```ts
L: any;
```

###### alphaVector

```ts
alphaVector: any[] | null;
```

###### Overrides

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(json): GaussianProcessRegressor;
```

Defined in: [src/ml/estimators/GaussianProcessRegressor.js:601](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GaussianProcessRegressor.js#L601)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### json

`any`

###### Returns

[`GaussianProcessRegressor`](#gaussianprocessregressor)

###### Overrides

```ts
Regressor.fromJSON
```

***

### GradientBoostingRegressor

Defined in: [src/ml/estimators/GradientBoosting.js:421](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L421)

Gradient boosting for regression (squared-error loss)

#### Example

```ts
const gbr = new GradientBoostingRegressor({ nEstimators: 200, learningRate: 0.05 });
gbr.fit(X, y);
const predictions = gbr.predict(Xnew);
```

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new GradientBoostingRegressor(opts?): GradientBoostingRegressor;
```

Defined in: [src/ml/estimators/GradientBoosting.js:422](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L422)

###### Parameters

###### opts?

###### Returns

[`GradientBoostingRegressor`](#gradientboostingregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### gb

```ts
gb: GradientBoostingBase;
```

Defined in: [src/ml/estimators/GradientBoosting.js:424](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L424)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): any[] | null;
```

Defined in: [src/ml/estimators/GradientBoosting.js:437](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L437)

###### Returns

`any`[] \| `null`

##### lossHistory

###### Get Signature

```ts
get lossHistory(): any[];
```

Defined in: [src/ml/estimators/GradientBoosting.js:442](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L442)

Per-stage training loss (MSE)

###### Returns

`any`[]

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): GradientBoostingRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GradientBoostingRegressor`](#gradientboostingregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Regressor.fromJSON
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(X, y?): GradientBoostingRegressor;
```

Defined in: [src/ml/estimators/GradientBoosting.js:427](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L427)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`GradientBoostingRegressor`](#gradientboostingregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X): any;
```

Defined in: [src/ml/estimators/GradientBoosting.js:433](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L433)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`

###### Overrides

```ts
Regressor.predict
```

***

### GradientBoostingClassifier

Defined in: [src/ml/estimators/GradientBoosting.js:455](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L455)

Gradient boosting for classification (logistic / multinomial deviance)

#### Example

```ts
const gbc = new GradientBoostingClassifier({ nEstimators: 100 });
gbc.fit({ X: ['bill_length', 'bill_depth'], y: 'species', data });
const labels = gbc.predict({ data: newData });
```

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new GradientBoostingClassifier(opts?): GradientBoostingClassifier;
```

Defined in: [src/ml/estimators/GradientBoosting.js:456](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L456)

###### Parameters

###### opts?

###### Returns

[`GradientBoostingClassifier`](#gradientboostingclassifier)

###### Overrides

```ts
Classifier.constructor
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
Classifier.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Classifier.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L514)

###### Inherited from

```ts
Classifier.labelEncoder_
```

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L515)

###### Inherited from

```ts
Classifier.classes_
```

##### gb

```ts
gb: GradientBoostingBase;
```

Defined in: [src/ml/estimators/GradientBoosting.js:458](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L458)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): any[] | null;
```

Defined in: [src/ml/estimators/GradientBoosting.js:505](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L505)

###### Returns

`any`[] \| `null`

##### lossHistory

###### Get Signature

```ts
get lossHistory(): any[];
```

Defined in: [src/ml/estimators/GradientBoosting.js:510](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L510)

Per-stage training loss (deviance)

###### Returns

`any`[]

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
Classifier.isFitted
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
Classifier.getState
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
Classifier.getMemoryUsage
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
Classifier.getWarnings
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
Classifier.hasWarnings
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
Classifier.clearWarnings
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
Classifier.getWarningsByType
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
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): GradientBoostingClassifier;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GradientBoostingClassifier`](#gradientboostingclassifier)

###### Inherited from

```ts
Classifier.setParams
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
Classifier.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Classifier.fromJSON
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
Classifier.save
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
Classifier.load
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
Classifier._prepareArgsForFit
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
Classifier.transform
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L541)

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

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L563)

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

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L606)

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

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L622)

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

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L644)

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
fit(X, y?): GradientBoostingClassifier;
```

Defined in: [src/ml/estimators/GradientBoosting.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L461)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`GradientBoostingClassifier`](#gradientboostingclassifier)

###### Overrides

```ts
Classifier.fit
```

##### predict()

```ts
predict(X): any;
```

Defined in: [src/ml/estimators/GradientBoosting.js:483](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L483)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`

###### Overrides

```ts
Classifier.predict
```

##### predictProba()

```ts
predictProba(X): Object[];
```

Defined in: [src/ml/estimators/GradientBoosting.js:493](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/GradientBoosting.js#L493)

Predict class probabilities

###### Parameters

###### X

`any`

###### Returns

`Object`[]

One object per row keyed by class label

###### Overrides

```ts
Classifier.predictProba
```

***

### HCA

Defined in: [src/ml/estimators/HCA.js:11](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L11)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new HCA(params?): HCA;
```

Defined in: [src/ml/estimators/HCA.js:12](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L12)

###### Parameters

###### params?

###### Returns

[`HCA`](#hca)

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

```ts
Estimator.fitted
```

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

```ts
Estimator._warnings
```

##### params

```ts
params: object;
```

Defined in: [src/ml/estimators/HCA.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L15)

###### linkage

```ts
linkage: string = 'average';
```

###### omit\_missing

```ts
omit_missing: boolean = true;
```

###### k

```ts
k: null = null;
```

###### Inherited from

```ts
Estimator.params
```

##### model

```ts
model: 
  | {
  omit_missing: any;
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

Defined in: [src/ml/estimators/HCA.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L16)

###### Union Members

###### Type Literal

```ts
{
  omit_missing: any;
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
omit_missing: any = omitMissing;
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

##### labels

```ts
labels: any[] | null;
```

Defined in: [src/ml/estimators/HCA.js:17](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L17)

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): HCA;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`HCA`](#hca)

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
Estimator.predict
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
fit(X, opts?): HCA;
```

Defined in: [src/ml/estimators/HCA.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L20)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### opts?

###### Returns

[`HCA`](#hca)

###### Overrides

```ts
Estimator.fit
```

##### cut()

```ts
cut(k): any[];
```

Defined in: [src/ml/estimators/HCA.js:67](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L67)

###### Parameters

###### k

`any`

###### Returns

`any`[]

##### cutHeight()

```ts
cutHeight(height): any[];
```

Defined in: [src/ml/estimators/HCA.js:72](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L72)

###### Parameters

###### height

`any`

###### Returns

`any`[]

##### summary()

```ts
summary(): object;
```

Defined in: [src/ml/estimators/HCA.js:77](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L77)

###### Returns

`object`

###### linkage

```ts
linkage: any;
```

###### n

```ts
n: any;
```

###### merges

```ts
merges: any = dendrogram.length;
```

###### maxDistance

```ts
maxDistance: any;
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/HCA.js:100](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L100)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'HCA';
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
  omit_missing: any;
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
  omit_missing: any;
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
omit_missing: any = omitMissing;
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

###### labels

```ts
labels: any[] | null;
```

###### Overrides

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): HCA;
```

Defined in: [src/ml/estimators/HCA.js:110](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/HCA.js#L110)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`HCA`](#hca)

###### Overrides

```ts
Estimator.fromJSON
```

***

### KMeans

Defined in: [src/ml/estimators/KMeans.js:17](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L17)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new KMeans(params?): KMeans;
```

Defined in: [src/ml/estimators/KMeans.js:21](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L21)

###### Parameters

###### params?

`Object` = `{}`

{ k, maxIter, tol, seed }

###### Returns

[`KMeans`](#kmeans)

###### Overrides

```ts
Estimator.constructor
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
Estimator.params
```

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

```ts
Estimator._warnings
```

##### k

```ts
k: any;
```

Defined in: [src/ml/estimators/KMeans.js:23](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L23)

##### maxIter

```ts
maxIter: any;
```

Defined in: [src/ml/estimators/KMeans.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L24)

##### tol

```ts
tol: any;
```

Defined in: [src/ml/estimators/KMeans.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L25)

##### seed

```ts
seed: any;
```

Defined in: [src/ml/estimators/KMeans.js:26](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L26)

##### model

```ts
model: Object | null;
```

Defined in: [src/ml/estimators/KMeans.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L29)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/ml/estimators/KMeans.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L30)

###### Inherited from

```ts
Estimator.fitted
```

##### labels

```ts
labels: any;
```

Defined in: [src/ml/estimators/KMeans.js:77](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L77)

##### centroids

```ts
centroids: any;
```

Defined in: [src/ml/estimators/KMeans.js:78](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L78)

##### inertia

```ts
inertia: any;
```

Defined in: [src/ml/estimators/KMeans.js:79](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L79)

##### iterations

```ts
iterations: any;
```

Defined in: [src/ml/estimators/KMeans.js:80](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L80)

##### converged

```ts
converged: any;
```

Defined in: [src/ml/estimators/KMeans.js:81](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L81)

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): KMeans;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`KMeans`](#kmeans)

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
fit(X, opts?): KMeans;
```

Defined in: [src/ml/estimators/KMeans.js:42](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L42)

Fit the KMeans model.

Accepts:
 - numeric input: fit(Xarray, { k, maxIter, tol, seed })
 - declarative input: fit({ data: tableLike, columns: ['c1','c2'], k, ... })

Returns this.

###### Parameters

###### X

`any`

###### opts?

###### Returns

[`KMeans`](#kmeans)

###### Overrides

```ts
Estimator.fit
```

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/estimators/KMeans.js:94](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L94)

Predict cluster labels for new data.

Accepts:
 - numeric array: predict([[x1,x2], [x1,x2], ...])
 - declarative: predict({ data: tableLike, columns: ['c1','c2'], omit_missing: true })

###### Parameters

###### X

`any`

###### Returns

`number`[]

###### Overrides

```ts
Estimator.predict
```

##### silhouetteScore()

```ts
silhouetteScore(X, labels?): number;
```

Defined in: [src/ml/estimators/KMeans.js:118](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L118)

Compute silhouette score for given X and labels (or use fitted labels if omitted).

Accepts:
 - numeric X array, and optional labels array
 - declarative object { data, columns } will be prepared

###### Parameters

###### X

`any`

###### labels?

`null` = `null`

###### Returns

`number`

##### summary()

```ts
summary(): object;
```

Defined in: [src/ml/estimators/KMeans.js:140](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L140)

Convenience: return summary stats for fitted model

###### Returns

`object`

###### k

```ts
k: any;
```

###### iterations

```ts
iterations: any;
```

###### inertia

```ts
inertia: any;
```

###### converged

```ts
converged: any;
```

###### centroids

```ts
centroids: any;
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/KMeans.js:154](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L154)

Serialization helper

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'KMeans';
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
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): KMeans;
```

Defined in: [src/ml/estimators/KMeans.js:163](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KMeans.js#L163)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`KMeans`](#kmeans)

###### Overrides

```ts
Estimator.fromJSON
```

***

### KNNClassifier

Defined in: [src/ml/estimators/KNN.js:283](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L283)

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new KNNClassifier(opts?): KNNClassifier;
```

Defined in: [src/ml/estimators/KNN.js:284](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L284)

###### Parameters

###### opts?

###### Returns

[`KNNClassifier`](#knnclassifier)

###### Overrides

```ts
Classifier.constructor
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
Classifier.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Classifier.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L514)

###### Inherited from

```ts
Classifier.labelEncoder_
```

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L515)

###### Inherited from

```ts
Classifier.classes_
```

##### knn

```ts
knn: KNNBase;
```

Defined in: [src/ml/estimators/KNN.js:286](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L286)

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
Classifier.isFitted
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
Classifier.getState
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
Classifier.getMemoryUsage
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
Classifier.getWarnings
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
Classifier.hasWarnings
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
Classifier.clearWarnings
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
Classifier.getWarningsByType
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
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): KNNClassifier;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`KNNClassifier`](#knnclassifier)

###### Inherited from

```ts
Classifier.setParams
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
Classifier.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Classifier.fromJSON
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
Classifier.save
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
Classifier.load
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
Classifier._prepareArgsForFit
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
Classifier.transform
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L541)

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

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L563)

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

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L606)

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

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L622)

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

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L644)

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
fit(X, y?): KNNClassifier;
```

Defined in: [src/ml/estimators/KNN.js:289](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L289)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`KNNClassifier`](#knnclassifier)

###### Overrides

```ts
Classifier.fit
```

##### predict()

```ts
predict(X, __namedParameters?): any[];
```

Defined in: [src/ml/estimators/KNN.js:296](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L296)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### \_\_namedParameters?

###### decode?

`boolean` = `...`

###### Returns

`any`[]

###### Overrides

```ts
Classifier.predict
```

##### predictProba()

```ts
predictProba(X, __namedParameters?): object[];
```

Defined in: [src/ml/estimators/KNN.js:331](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L331)

Predict probabilities - subclasses should override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### \_\_namedParameters?

###### decode?

`boolean` = `...`

###### Returns

`object`[]

###### Overrides

```ts
Classifier.predictProba
```

##### radiusNeighbors()

```ts
radiusNeighbors(X, radius): any[][];
```

Defined in: [src/ml/estimators/KNN.js:374](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L374)

Find neighbors within a given radius

###### Parameters

###### X

`any`[]

Query points

###### radius

`number`

Radius within which to find neighbors

###### Returns

`any`[][]

Indices of neighbors for each query point

##### kneighbors()

```ts
kneighbors(X, nNeighbors?): Object;
```

Defined in: [src/ml/estimators/KNN.js:393](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L393)

Find K nearest neighbors

###### Parameters

###### X

`any`[]

Query points

###### nNeighbors?

`number` = `null`

Number of neighbors (default: this.k)

###### Returns

`Object`

{distances, indices}

***

### KNNRegressor

Defined in: [src/ml/estimators/KNN.js:412](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L412)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new KNNRegressor(opts?): KNNRegressor;
```

Defined in: [src/ml/estimators/KNN.js:413](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L413)

###### Parameters

###### opts?

###### Returns

[`KNNRegressor`](#knnregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### knn

```ts
knn: KNNBase;
```

Defined in: [src/ml/estimators/KNN.js:415](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L415)

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): KNNRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`KNNRegressor`](#knnregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Regressor.fromJSON
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(X, y?): KNNRegressor;
```

Defined in: [src/ml/estimators/KNN.js:418](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L418)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### Returns

[`KNNRegressor`](#knnregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/estimators/KNN.js:423](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L423)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`number`[]

###### Overrides

```ts
Regressor.predict
```

##### radiusNeighbors()

```ts
radiusNeighbors(X, radius): any[][];
```

Defined in: [src/ml/estimators/KNN.js:455](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L455)

Find neighbors within a given radius

###### Parameters

###### X

`any`[]

Query points

###### radius

`number`

Radius within which to find neighbors

###### Returns

`any`[][]

Indices of neighbors for each query point

##### kneighbors()

```ts
kneighbors(X, nNeighbors?): Object;
```

Defined in: [src/ml/estimators/KNN.js:474](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/KNN.js#L474)

Find K nearest neighbors

###### Parameters

###### X

`any`[]

Query points

###### nNeighbors?

`number` = `null`

Number of neighbors (default: this.k)

###### Returns

`Object`

{distances, indices}

***

### MLPRegressor

Defined in: [src/ml/estimators/MLPRegressor.js:19](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L19)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new MLPRegressor(params?): MLPRegressor;
```

Defined in: [src/ml/estimators/MLPRegressor.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L20)

###### Parameters

###### params?

###### Returns

[`MLPRegressor`](#mlpregressor)

###### Overrides

```ts
Regressor.constructor
```

#### Properties

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### params

```ts
params: object;
```

Defined in: [src/ml/estimators/MLPRegressor.js:23](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L23)

###### layerSizes

```ts
layerSizes: null = null;
```

###### activation

```ts
activation: string = 'relu';
```

###### learningRate

```ts
learningRate: number = 0.01;
```

###### epochs

```ts
epochs: number = 100;
```

###### batchSize

```ts
batchSize: number = 32;
```

###### verbose

```ts
verbose: boolean = false;
```

###### omit\_missing

```ts
omit_missing: boolean = true;
```

###### Inherited from

```ts
Regressor.params
```

##### model

```ts
model: Object | null;
```

Defined in: [src/ml/estimators/MLPRegressor.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L24)

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): MLPRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`MLPRegressor`](#mlpregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(
   X, 
   y?, 
   opts?): MLPRegressor;
```

Defined in: [src/ml/estimators/MLPRegressor.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L27)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### opts?

###### Returns

[`MLPRegressor`](#mlpregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X): number[][];
```

Defined in: [src/ml/estimators/MLPRegressor.js:84](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L84)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`number`[][]

###### Overrides

```ts
Regressor.predict
```

##### evaluate()

```ts
evaluate(X, y): Object;
```

Defined in: [src/ml/estimators/MLPRegressor.js:109](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L109)

###### Parameters

###### X

`any`

###### y

`any`

###### Returns

`Object`

##### summary()

```ts
summary(): object;
```

Defined in: [src/ml/estimators/MLPRegressor.js:116](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L116)

###### Returns

`object`

###### epochs

```ts
epochs: any;
```

###### layerSizes

```ts
layerSizes: any;
```

###### finalLoss

```ts
finalLoss: any;
```

###### initialLoss

```ts
initialLoss: any;
```

###### losses

```ts
losses: any;
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/MLPRegressor.js:130](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L130)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'MLPRegressor';
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
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): MLPRegressor;
```

Defined in: [src/ml/estimators/MLPRegressor.js:139](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/MLPRegressor.js#L139)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`MLPRegressor`](#mlpregressor)

###### Overrides

```ts
Regressor.fromJSON
```

***

### PolynomialRegressor

Defined in: [src/ml/estimators/PolynomialRegressor.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L15)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new PolynomialRegressor(params?): PolynomialRegressor;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L16)

###### Parameters

###### params?

###### Returns

[`PolynomialRegressor`](#polynomialregressor)

###### Overrides

```ts
Regressor.constructor
```

#### Properties

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### params

```ts
params: object;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:19](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L19)

###### degree

```ts
degree: number = 2;
```

###### intercept

```ts
intercept: boolean = true;
```

###### omit\_missing

```ts
omit_missing: boolean = true;
```

###### Inherited from

```ts
Regressor.params
```

##### model

```ts
model: Object | null;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L20)

##### coef

```ts
coef: any;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:21](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L21)

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): PolynomialRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`PolynomialRegressor`](#polynomialregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(
   X, 
   y?, 
   opts?): PolynomialRegressor;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L24)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### opts?

###### Returns

[`PolynomialRegressor`](#polynomialregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X, __namedParameters?): number[];
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:69](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L69)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### \_\_namedParameters?

###### intercept?

`undefined` = `undefined`

###### Returns

`number`[]

###### Overrides

```ts
Regressor.predict
```

##### summary()

```ts
summary(): object;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:98](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L98)

###### Returns

###### constructor?

```ts
optional constructor?: Function;
```

The initial value of Object.prototype.constructor is the standard built-in Object constructor.

###### toString?

```ts
optional toString?: () => string;
```

Returns a string representation of an object.

###### Returns

`string`

###### toLocaleString?

```ts
optional toLocaleString?: () => string;
```

Returns a date converted to a string using the current locale.

###### Returns

`string`

###### valueOf?

```ts
optional valueOf?: () => Object;
```

Returns the primitive value of the specified object.

###### Returns

`Object`

###### hasOwnProperty?

```ts
optional hasOwnProperty?: (v) => boolean;
```

Determines whether an object has a property with the specified name.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

###### isPrototypeOf?

```ts
optional isPrototypeOf?: (v) => boolean;
```

Determines whether an object exists in another object's prototype chain.

###### Parameters

###### v

`Object`

Another object whose prototype chain is to be checked.

###### Returns

`boolean`

###### propertyIsEnumerable?

```ts
optional propertyIsEnumerable?: (v) => boolean;
```

Determines whether a specified property is enumerable.

###### Parameters

###### v

`PropertyKey`

A property name.

###### Returns

`boolean`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:103](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L103)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'PolynomialRegressor';
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

###### coefficients

```ts
coefficients: any;
```

###### Overrides

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): PolynomialRegressor;
```

Defined in: [src/ml/estimators/PolynomialRegressor.js:113](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/PolynomialRegressor.js#L113)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

###### Returns

[`PolynomialRegressor`](#polynomialregressor)

###### Overrides

```ts
Regressor.fromJSON
```

***

### RandomForestClassifier

Defined in: [src/ml/estimators/RandomForest.js:494](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L494)

#### Extends

- `Classifier`

#### Constructors

##### Constructor

```ts
new RandomForestClassifier(opts?): RandomForestClassifier;
```

Defined in: [src/ml/estimators/RandomForest.js:495](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L495)

###### Parameters

###### opts?

###### Returns

[`RandomForestClassifier`](#randomforestclassifier)

###### Overrides

```ts
Classifier.constructor
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
Classifier.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Classifier.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Classifier._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Classifier._warnings
```

##### labelEncoder\_

```ts
labelEncoder_: any;
```

Defined in: [src/core/estimators/estimator.js:514](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L514)

###### Inherited from

```ts
Classifier.labelEncoder_
```

##### classes\_

```ts
classes_: any;
```

Defined in: [src/core/estimators/estimator.js:515](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L515)

###### Inherited from

```ts
Classifier.classes_
```

##### forest

```ts
forest: RandomForestBase;
```

Defined in: [src/ml/estimators/RandomForest.js:497](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L497)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): any[] | number[] | null;
```

Defined in: [src/ml/estimators/RandomForest.js:559](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L559)

Get feature importances (MDI - Mean Decrease in Impurity)

###### Returns

`any`[] \| `number`[] \| `null`

##### oobScore

###### Get Signature

```ts
get oobScore(): number | null;
```

Defined in: [src/ml/estimators/RandomForest.js:566](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L566)

Get out-of-bag score (accuracy for classification)

###### Returns

`number` \| `null`

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
Classifier.isFitted
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
Classifier.getState
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
Classifier.getMemoryUsage
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
Classifier.getWarnings
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
Classifier.hasWarnings
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
Classifier.clearWarnings
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
Classifier.getWarningsByType
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
Classifier._repr_html_
```

##### setParams()

```ts
setParams(params?): RandomForestClassifier;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`RandomForestClassifier`](#randomforestclassifier)

###### Inherited from

```ts
Classifier.setParams
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
Classifier.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Classifier.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Classifier.fromJSON
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
Classifier.save
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
Classifier.load
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
Classifier._prepareArgsForFit
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
Classifier.transform
```

##### \_extractLabelEncoder()

```ts
_extractLabelEncoder(prepared): boolean;
```

Defined in: [src/core/estimators/estimator.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L541)

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

Defined in: [src/core/estimators/estimator.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L563)

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

Defined in: [src/core/estimators/estimator.js:606](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L606)

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

Defined in: [src/core/estimators/estimator.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L622)

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

Defined in: [src/core/estimators/estimator.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L644)

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
   sampleWeight?): RandomForestClassifier;
```

Defined in: [src/ml/estimators/RandomForest.js:500](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L500)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### sampleWeight?

`null` = `null`

###### Returns

[`RandomForestClassifier`](#randomforestclassifier)

###### Overrides

```ts
Classifier.fit
```

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/ml/estimators/RandomForest.js:526](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L526)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`[]

###### Overrides

```ts
Classifier.predict
```

##### predictProba()

```ts
predictProba(X): object[];
```

Defined in: [src/ml/estimators/RandomForest.js:532](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L532)

Predict probabilities - subclasses should override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`object`[]

###### Overrides

```ts
Classifier.predictProba
```

##### apply()

```ts
apply(X): any[][];
```

Defined in: [src/ml/estimators/RandomForest.js:573](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L573)

Apply trees in forest to X, return leaf indices

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### decisionPath()

```ts
decisionPath(X): any[][][];
```

Defined in: [src/ml/estimators/RandomForest.js:580](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L580)

Return decision path through the forest

###### Parameters

###### X

`any`

###### Returns

`any`[][][]

***

### RandomForestRegressor

Defined in: [src/ml/estimators/RandomForest.js:585](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L585)

#### Extends

- `Regressor`

#### Constructors

##### Constructor

```ts
new RandomForestRegressor(opts?): RandomForestRegressor;
```

Defined in: [src/ml/estimators/RandomForest.js:586](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L586)

###### Parameters

###### opts?

###### Returns

[`RandomForestRegressor`](#randomforestregressor)

###### Overrides

```ts
Regressor.constructor
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
Regressor.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Regressor.fitted
```

##### \_state

```ts
_state: object;
```

Defined in: [src/core/estimators/estimator.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L27)

###### Inherited from

```ts
Regressor._state
```

##### \_warnings

```ts
_warnings: any[];
```

Defined in: [src/core/estimators/estimator.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L29)

###### Inherited from

```ts
Regressor._warnings
```

##### forest

```ts
forest: RandomForestBase;
```

Defined in: [src/ml/estimators/RandomForest.js:588](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L588)

#### Accessors

##### featureImportances

###### Get Signature

```ts
get featureImportances(): any[] | number[] | null;
```

Defined in: [src/ml/estimators/RandomForest.js:604](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L604)

Get feature importances (MDI - Mean Decrease in Impurity)

###### Returns

`any`[] \| `number`[] \| `null`

##### oobScore

###### Get Signature

```ts
get oobScore(): number | null;
```

Defined in: [src/ml/estimators/RandomForest.js:611](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L611)

Get out-of-bag score (R^2 score for regression)

###### Returns

`number` \| `null`

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
Regressor.isFitted
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
Regressor.getState
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
Regressor.getMemoryUsage
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
Regressor.getWarnings
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
Regressor.hasWarnings
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
Regressor.clearWarnings
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
Regressor.getWarningsByType
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
Regressor._repr_html_
```

##### setParams()

```ts
setParams(params?): RandomForestRegressor;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`RandomForestRegressor`](#randomforestregressor)

###### Inherited from

```ts
Regressor.setParams
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
Regressor.getParams
```

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Regressor.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Regressor.fromJSON
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
Regressor.save
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
Regressor.load
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
Regressor._prepareArgsForFit
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
Regressor.transform
```

##### score()

```ts
score(
   yTrueOrOpts, 
   yPred, 
   _opts?, ...
   args?): number;
```

Defined in: [src/core/estimators/estimator.js:461](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L461)

Default R^2 scoring implementation:
  1 - SS_res / SS_tot

Accepts either:
 - arrays: score(yTrue, yPred)
 - table-style: score({ X, y, data }) where predict will be called internally

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
Regressor.score
```

##### \_r2()

```ts
_r2(yTrue, yPred): number;
```

Defined in: [src/core/estimators/estimator.js:489](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L489)

###### Parameters

###### yTrue

`any`

###### yPred

`any`

###### Returns

`number`

###### Inherited from

```ts
Regressor._r2
```

##### fit()

```ts
fit(
   X, 
   y?, 
   sampleWeight?): RandomForestRegressor;
```

Defined in: [src/ml/estimators/RandomForest.js:591](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L591)

Fit should be implemented by subclasses.
Return `this` for chaining.

###### Parameters

###### X

`any`

###### y?

`null` = `null`

###### sampleWeight?

`null` = `null`

###### Returns

[`RandomForestRegressor`](#randomforestregressor)

###### Overrides

```ts
Regressor.fit
```

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/ml/estimators/RandomForest.js:597](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L597)

Predict - subclasses must override
Ensures model is fitted before prediction

###### Parameters

###### X

`any`

###### Returns

`any`[]

###### Overrides

```ts
Regressor.predict
```

##### apply()

```ts
apply(X): any[][];
```

Defined in: [src/ml/estimators/RandomForest.js:618](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L618)

Apply trees in forest to X, return leaf indices

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### decisionPath()

```ts
decisionPath(X): any[][][];
```

Defined in: [src/ml/estimators/RandomForest.js:625](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/estimators/RandomForest.js#L625)

Return decision path through the forest

###### Parameters

###### X

`any`

###### Returns

`any`[][][]

***

### SimpleImputer

Defined in: [src/ml/impute.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L148)

Simple imputation strategies for missing values
Compatible with sklearn.impute.SimpleImputer

#### Example

```ts
const imputer = new SimpleImputer({ strategy: 'mean' });
imputer.fit(X_train);
const X_filled = imputer.transform(X_test);
```

#### Constructors

##### Constructor

```ts
new SimpleImputer(options?): SimpleImputer;
```

Defined in: [src/ml/impute.js:155](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L155)

###### Parameters

###### options?

###### strategy

`string` = `"mean"`

'mean', 'median', 'most_frequent', or 'constant' (default: 'mean')

###### fill_value

`string` \| `number` = `null`

Value to use for 'constant' strategy

###### copy

`boolean` = `true`

If true, create copy of X (default: true)

###### Returns

[`SimpleImputer`](#simpleimputer)

#### Properties

##### strategy

```ts
strategy: string;
```

Defined in: [src/ml/impute.js:169](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L169)

##### fill\_value

```ts
fill_value: string | number;
```

Defined in: [src/ml/impute.js:170](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L170)

##### copy

```ts
copy: boolean;
```

Defined in: [src/ml/impute.js:171](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L171)

##### statistics\_

```ts
statistics_: any[] | null;
```

Defined in: [src/ml/impute.js:172](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L172)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/impute.js:173](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L173)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/impute.js:174](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L174)

##### \_groupModels

```ts
_groupModels: Map<any, any> | null;
```

Defined in: [src/ml/impute.js:175](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L175)

##### \_originalData

```ts
_originalData: any;
```

Defined in: [src/ml/impute.js:176](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L176)

#### Methods

##### fit()

```ts
fit(X): SimpleImputer;
```

Defined in: [src/ml/impute.js:184](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L184)

Fit the imputer on training data

###### Parameters

###### X

`Object` \| `number`[][]

Training data, table object, or {data, columns, group} format

###### Returns

[`SimpleImputer`](#simpleimputer)

this

##### \_fitSingleModel()

```ts
_fitSingleModel(X): Object;
```

Defined in: [src/ml/impute.js:254](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L254)

Internal method to fit a single imputation model

###### Parameters

###### X

`number`[][]

2D array of numeric data

###### Returns

`Object`

Model statistics

##### transform()

```ts
transform(X): number[][] | Object[];
```

Defined in: [src/ml/impute.js:303](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L303)

Transform data by filling missing values

###### Parameters

###### X

`Object` \| `number`[][]

Data to transform, table object, or {data, columns, group} format

###### Returns

`number`[][] \| `Object`[]

Transformed data (array if input was table)

##### \_transformWithModel()

```ts
_transformWithModel(X, statistics): number[][];
```

Defined in: [src/ml/impute.js:386](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L386)

Internal method to transform data with specific statistics

###### Parameters

###### X

`number`[][]

Data to transform

###### statistics

`number`[]

Statistics to use for imputation

###### Returns

`number`[][]

Transformed data

##### fit\_transform()

```ts
fit_transform(X): number[][];
```

Defined in: [src/ml/impute.js:407](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L407)

Fit and transform in one step

###### Parameters

###### X

`Object` \| `number`[][]

Data to fit and transform

###### Returns

`number`[][]

Transformed data

***

### KNNImputer

Defined in: [src/ml/impute.js:426](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L426)

Imputation using k-Nearest Neighbors
Compatible with sklearn.impute.KNNImputer

Missing values are imputed using the mean value from the k nearest
neighbors found in the training set.

#### Example

```ts
const imputer = new KNNImputer({ n_neighbors: 5 });
imputer.fit(X_train);
const X_filled = imputer.transform(X_test);
```

#### Constructors

##### Constructor

```ts
new KNNImputer(options?): KNNImputer;
```

Defined in: [src/ml/impute.js:434](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L434)

###### Parameters

###### options?

###### n_neighbors

`number` = `5`

Number of neighbors to use (default: 5)

###### weights

`string` = `"uniform"`

'uniform' or 'distance' (default: 'uniform')

###### metric

`Function` = `null`

Distance function (default: euclidean)

###### copy

`boolean` = `true`

If true, create copy of X (default: true)

###### Returns

[`KNNImputer`](#knnimputer)

#### Properties

##### n\_neighbors

```ts
n_neighbors: number;
```

Defined in: [src/ml/impute.js:444](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L444)

##### weights

```ts
weights: string;
```

Defined in: [src/ml/impute.js:445](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L445)

##### metric

```ts
metric: Function;
```

Defined in: [src/ml/impute.js:446](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L446)

##### copy

```ts
copy: boolean;
```

Defined in: [src/ml/impute.js:447](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L447)

##### X\_

```ts
X_: any[] | null;
```

Defined in: [src/ml/impute.js:448](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L448)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/impute.js:449](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L449)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/impute.js:450](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L450)

##### \_columnTypes

```ts
_columnTypes: any;
```

Defined in: [src/ml/impute.js:493](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L493)

##### \_useGowerDistance

```ts
_useGowerDistance: boolean | undefined;
```

Defined in: [src/ml/impute.js:498](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L498)

#### Methods

##### \_euclideanDistance()

```ts
_euclideanDistance(a, b): number;
```

Defined in: [src/ml/impute.js:456](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L456)

Euclidean distance between two vectors (ignoring missing values)

###### Parameters

###### a

`any`

###### b

`any`

###### Returns

`number`

##### fit()

```ts
fit(X): KNNImputer;
```

Defined in: [src/ml/impute.js:483](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L483)

Fit the imputer on training data

###### Parameters

###### X

`Object` \| `number`[][]

Training data, table object, or {data, columns} format

###### Returns

[`KNNImputer`](#knnimputer)

this

##### transform()

```ts
transform(X, exclude_indices?): number[][] | Object[];
```

Defined in: [src/ml/impute.js:527](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L527)

Transform data by filling missing values using KNN

###### Parameters

###### X

`Object` \| `number`[][]

Data to transform, table object, or {data, columns} format

###### exclude\_indices?

`number`[] = `[]`

Row indices to exclude from neighbors (for fit_transform)

###### Returns

`number`[][] \| `Object`[]

Transformed data (array if input was table)

##### \_findNeighbors()

```ts
_findNeighbors(
   row, 
   k, 
   excludeIdx?): object[];
```

Defined in: [src/ml/impute.js:669](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L669)

Find k nearest neighbors for a given row

###### Parameters

###### row

`any`

###### k

`any`

###### excludeIdx?

`number` = `-1`

Row index to exclude (-1 for none)

###### Returns

`object`[]

##### fit\_transform()

```ts
fit_transform(X): number[][] | Object[];
```

Defined in: [src/ml/impute.js:702](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L702)

Fit and transform in one step

###### Parameters

###### X

`Object` \| `number`[][]

Data to fit and transform

###### Returns

`number`[][] \| `Object`[]

Transformed data

***

### IterativeImputer

Defined in: [src/ml/impute.js:742](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L742)

Multivariate imputation using chained equations (MICE algorithm)
Compatible with sklearn.impute.IterativeImputer

Models each feature with missing values as a function of other features,
and uses that estimate for imputation. It does so in an iterated round-robin
fashion: at each step, a feature column is designated as output y and the other
feature columns are treated as inputs X. A regressor is fit on (X, y) for known
values and used to predict missing values of y.

#### Example

```ts
const imputer = new IterativeImputer({ max_iter: 10 });
imputer.fit(X_train);
const X_filled = imputer.transform(X_test);
```

#### Constructors

##### Constructor

```ts
new IterativeImputer(options?): IterativeImputer;
```

Defined in: [src/ml/impute.js:753](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L753)

###### Parameters

###### options?

###### initial_strategy

`string` = `"mean"`

Initial imputation strategy (default: 'mean')

###### max_iter

`number` = `10`

Maximum number of imputation rounds (default: 10)

###### tol

`number` = `1e-3`

Tolerance for convergence (default: 1e-3)

###### min_value

`number` = `-Infinity`

Minimum possible imputed value (default: -Infinity)

###### max_value

`number` = `Infinity`

Maximum possible imputed value (default: Infinity)

###### verbose

`boolean` = `false`

Print progress (default: false)

###### copy

`boolean` = `true`

If true, create copy of X (default: true)

###### Returns

[`IterativeImputer`](#iterativeimputer)

#### Properties

##### initial\_strategy

```ts
initial_strategy: string;
```

Defined in: [src/ml/impute.js:762](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L762)

##### max\_iter

```ts
max_iter: number;
```

Defined in: [src/ml/impute.js:763](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L763)

##### tol

```ts
tol: number;
```

Defined in: [src/ml/impute.js:764](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L764)

##### min\_value

```ts
min_value: number;
```

Defined in: [src/ml/impute.js:765](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L765)

##### max\_value

```ts
max_value: number;
```

Defined in: [src/ml/impute.js:766](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L766)

##### verbose

```ts
verbose: boolean;
```

Defined in: [src/ml/impute.js:767](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L767)

##### copy

```ts
copy: boolean;
```

Defined in: [src/ml/impute.js:768](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L768)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/impute.js:769](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L769)

##### initial\_imputer\_

```ts
initial_imputer_: SimpleImputer | null;
```

Defined in: [src/ml/impute.js:770](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L770)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/impute.js:771](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L771)

##### n\_iter\_

```ts
n_iter_: number | null;
```

Defined in: [src/ml/impute.js:772](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L772)

#### Methods

##### \_fitLinearRegression()

```ts
_fitLinearRegression(X, y): Object;
```

Defined in: [src/ml/impute.js:781](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L781)

Fit a simple linear regression using pseudoinverse

###### Parameters

###### X

`number`[][]

Features

###### y

`number`[]

Target

###### Returns

`Object`

Model with coefficients and predict function

##### \_imputeFeature()

```ts
_imputeFeature(
   X, 
   featureIdx, 
   missing_mask): number[];
```

Defined in: [src/ml/impute.js:819](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L819)

Impute a single feature using other features

###### Parameters

###### X

`number`[][]

Data matrix (current working copy, all values filled)

###### featureIdx

`number`

Index of feature to impute

###### missing\_mask

`boolean`[][]

Original missingness mask

###### Returns

`number`[]

Imputed values for this feature

##### fit()

```ts
fit(X): IterativeImputer;
```

Defined in: [src/ml/impute.js:899](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L899)

Fit the imputer on training data

###### Parameters

###### X

`Object` \| `number`[][]

Training data, table object, or {data, columns} format

###### Returns

[`IterativeImputer`](#iterativeimputer)

this

##### transform()

```ts
transform(X): number[][] | Object[];
```

Defined in: [src/ml/impute.js:933](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L933)

Transform data by filling missing values using MICE

###### Parameters

###### X

`Object` \| `number`[][]

Data to transform, table object, or {data, columns} format

###### Returns

`number`[][] \| `Object`[]

Transformed data (array if input was table)

##### fit\_transform()

```ts
fit_transform(X): number[][];
```

Defined in: [src/ml/impute.js:1053](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L1053)

Fit and transform in one step

###### Parameters

###### X

`Object` \| `number`[][]

Data to fit and transform

###### Returns

`number`[][]

Transformed data

***

### Kernel

Defined in: [src/ml/kernels/base.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L13)

Abstract base class for GP kernels

#### Extended by

- [`RBF`](#rbf)
- [`Periodic`](#periodic)
- [`RationalQuadratic`](#rationalquadratic)
- [`ConstantKernel`](#constantkernel)
- [`SumKernel`](#sumkernel)
- [`Matern`](#matern)
- [`DotProduct`](#dotproduct)

#### Constructors

##### Constructor

```ts
new Kernel(): Kernel;
```

###### Returns

[`Kernel`](#kernel-1)

#### Methods

##### compute()

```ts
compute(_x1, _x2): number;
```

Defined in: [src/ml/kernels/base.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L20)

Compute covariance between two points

###### Parameters

###### \_x1

`any`

###### \_x2

`any`

###### Returns

`number`

Covariance value

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

##### getParams()

```ts
getParams(): Object;
```

Defined in: [src/ml/kernels/base.js:60](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L60)

Get kernel hyperparameters

###### Returns

`Object`

Hyperparameters

##### setParams()

```ts
setParams(_params): void;
```

Defined in: [src/ml/kernels/base.js:68](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L68)

Set kernel hyperparameters

###### Parameters

###### \_params

`any`

###### Returns

`void`

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

***

### ConstantKernel

Defined in: [src/ml/kernels/constant.js:11](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L11)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new ConstantKernel(valueOrOpts?): ConstantKernel;
```

Defined in: [src/ml/kernels/constant.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L15)

###### Parameters

###### valueOrOpts?

`number` \| `Object`

Constant value or options object

###### Returns

[`ConstantKernel`](#constantkernel)

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### value

```ts
value: any;
```

Defined in: [src/ml/kernels/constant.js:19](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L19)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(): any;
```

Defined in: [src/ml/kernels/constant.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L25)

Compute covariance between two points

###### Returns

`any`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/constant.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L29)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### value

```ts
value: any;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/constant.js:33](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/constant.js#L33)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### value

`any`

###### amplitude

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### DotProduct

Defined in: [src/ml/kernels/dot-product.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L15)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new DotProduct(sigma0OrOpts?): DotProduct;
```

Defined in: [src/ml/kernels/dot-product.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L24)

###### Parameters

###### sigma0OrOpts?

`number` \| `Object`

Inhomogeneity term, or `{ sigma0 }`.

###### Returns

[`DotProduct`](#dotproduct)

###### Example

```ts
new DotProduct(1.0)
new DotProduct({ sigma0: 1.0 })
```

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### sigma0

```ts
sigma0: any;
```

Defined in: [src/ml/kernels/dot-product.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L27)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/dot-product.js:33](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L33)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/dot-product.js:39](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L39)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### sigma0

```ts
sigma0: any;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params?): void;
```

Defined in: [src/ml/kernels/dot-product.js:43](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/dot-product.js#L43)

Set kernel hyperparameters

###### Parameters

###### params?

New parameters

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### Matern

Defined in: [src/ml/kernels/matern.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L13)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new Matern(
   lengthScaleOrOpts?, 
   nu?, 
   variance?): Matern;
```

Defined in: [src/ml/kernels/matern.js:14](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L14)

###### Parameters

###### lengthScaleOrOpts?

`number` = `1.0`

###### nu?

`number` = `1.5`

###### variance?

`number` = `1.0`

###### Returns

[`Matern`](#matern)

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### lengthScale

```ts
lengthScale: number;
```

Defined in: [src/ml/kernels/matern.js:23](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L23)

##### nu

```ts
nu: number;
```

Defined in: [src/ml/kernels/matern.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L24)

##### variance

```ts
variance: number;
```

Defined in: [src/ml/kernels/matern.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L25)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/matern.js:39](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L39)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/matern.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L76)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### lengthScale

```ts
lengthScale: number;
```

###### nu

```ts
nu: number;
```

###### variance

```ts
variance: number;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/matern.js:84](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/matern.js#L84)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### lengthScale

`any`

###### nu

`any`

###### variance

`any`

###### amplitude

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### Periodic

Defined in: [src/ml/kernels/periodic.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L15)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new Periodic(
   lengthScale?, 
   period?, 
   variance?): Periodic;
```

Defined in: [src/ml/kernels/periodic.js:21](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L21)

###### Parameters

###### lengthScale?

`number` = `1.0`

Length scale (default: 1.0)

###### period?

`number` = `1.0`

Period length (default: 1.0)

###### variance?

`number` = `1.0`

Signal variance (default: 1.0)

###### Returns

[`Periodic`](#periodic)

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### lengthScale

```ts
lengthScale: number;
```

Defined in: [src/ml/kernels/periodic.js:23](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L23)

##### period

```ts
period: number;
```

Defined in: [src/ml/kernels/periodic.js:24](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L24)

##### variance

```ts
variance: number;
```

Defined in: [src/ml/kernels/periodic.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L25)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/periodic.js:28](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L28)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/periodic.js:41](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L41)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### lengthScale

```ts
lengthScale: number;
```

###### period

```ts
period: number;
```

###### variance

```ts
variance: number;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/periodic.js:49](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/periodic.js#L49)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### lengthScale

`any`

###### period

`any`

###### variance

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### RationalQuadratic

Defined in: [src/ml/kernels/rational-quadratic.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L15)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new RationalQuadratic(
   lengthScaleOrOpts?, 
   alpha?, 
   variance?): RationalQuadratic;
```

Defined in: [src/ml/kernels/rational-quadratic.js:21](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L21)

###### Parameters

###### lengthScaleOrOpts?

`number` \| `Object`

Length scale or options object

###### alpha?

`number` = `1.0`

Scale mixture parameter (default: 1.0)

###### variance?

`number` = `1.0`

Signal variance (default: 1.0)

###### Returns

[`RationalQuadratic`](#rationalquadratic)

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### lengthScale

```ts
lengthScale: any;
```

Defined in: [src/ml/kernels/rational-quadratic.js:31](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L31)

##### alpha

```ts
alpha: any;
```

Defined in: [src/ml/kernels/rational-quadratic.js:32](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L32)

##### variance

```ts
variance: any;
```

Defined in: [src/ml/kernels/rational-quadratic.js:33](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L33)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/rational-quadratic.js:41](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L41)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/rational-quadratic.js:51](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L51)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### lengthScale

```ts
lengthScale: any;
```

###### alpha

```ts
alpha: any;
```

###### variance

```ts
variance: any;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/rational-quadratic.js:59](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rational-quadratic.js#L59)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### lengthScale

`any`

###### alpha

`any`

###### variance

`any`

###### amplitude

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### RBF

Defined in: [src/ml/kernels/rbf.js:15](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L15)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new RBF(lengthScaleOrOpts?, variance?): RBF;
```

Defined in: [src/ml/kernels/rbf.js:28](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L28)

###### Parameters

###### lengthScaleOrOpts?

`number` \| `Object`

Length scale or options object

###### variance?

`number` = `1.0`

Signal variance (default: 1.0)

###### Returns

[`RBF`](#rbf)

###### Examples

```ts
// Positional arguments (scikit-learn style)
new RBF(1.0, 1.0)
```

```ts
// Object arguments
new RBF({ lengthScale: 1.0, amplitude: 1.0 })
```

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### lengthScale

```ts
lengthScale: any;
```

Defined in: [src/ml/kernels/rbf.js:33](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L33)

##### variance

```ts
variance: any;
```

Defined in: [src/ml/kernels/rbf.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L34)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/rbf.js:42](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L42)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/rbf.js:55](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L55)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### lengthScale

```ts
lengthScale: any;
```

###### variance

```ts
variance: any;
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/rbf.js:62](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/rbf.js#L62)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### lengthScale

`any`

###### variance

`any`

###### amplitude

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### SumKernel

Defined in: [src/ml/kernels/sum.js:11](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L11)

Abstract base class for GP kernels

#### Extends

- [`Kernel`](#kernel-1)

#### Constructors

##### Constructor

```ts
new SumKernel(opts?): SumKernel;
```

Defined in: [src/ml/kernels/sum.js:16](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L16)

###### Parameters

###### opts?

###### kernels

[`Kernel`](#kernel-1)[]

Array of kernel instances to sum

###### Returns

[`SumKernel`](#sumkernel)

###### Overrides

[`Kernel`](#kernel-1).[`constructor`](#constructor-20)

#### Properties

##### kernels

```ts
kernels: Kernel[];
```

Defined in: [src/ml/kernels/sum.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L27)

#### Methods

##### call()

```ts
call(X1, X2?): Matrix;
```

Defined in: [src/ml/kernels/base.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L30)

Compute covariance matrix between sets of points

###### Parameters

###### X1

`any`

First set of points (n1 x d)

###### X2?

`any` = `null`

Second set of points (n2 x d). If omitted, computes K(X1, X1)

###### Returns

`Matrix`

Covariance matrix (n1 x n2)

###### Inherited from

[`Kernel`](#kernel-1).[`call`](#call)

##### clone()

```ts
clone(): Kernel;
```

Defined in: [src/ml/kernels/base.js:76](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/base.js#L76)

Clone the kernel with the same parameters

###### Returns

[`Kernel`](#kernel-1)

New kernel instance

###### Inherited from

[`Kernel`](#kernel-1).[`clone`](#clone)

##### compute()

```ts
compute(x1, x2): number;
```

Defined in: [src/ml/kernels/sum.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L30)

Compute covariance between two points

###### Parameters

###### x1

`any`

First point

###### x2

`any`

Second point

###### Returns

`number`

Covariance value

###### Overrides

[`Kernel`](#kernel-1).[`compute`](#compute)

##### getParams()

```ts
getParams(): object;
```

Defined in: [src/ml/kernels/sum.js:34](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L34)

Get kernel hyperparameters

###### Returns

`object`

Hyperparameters

###### kernels

```ts
kernels: object[];
```

###### Overrides

[`Kernel`](#kernel-1).[`getParams`](#getparams-17)

##### setParams()

```ts
setParams(params): void;
```

Defined in: [src/ml/kernels/sum.js:43](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/kernels/sum.js#L43)

Set kernel hyperparameters

###### Parameters

###### params

New parameters

###### kernels

`any`

###### Returns

`void`

###### Overrides

[`Kernel`](#kernel-1).[`setParams`](#setparams-17)

***

### IsolationForest

Defined in: [src/ml/outliers.js:192](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L192)

Isolation Forest for outlier detection
Compatible with sklearn.ensemble.IsolationForest

Detects outliers using ensemble of isolation trees.
Outliers are isolated closer to the root of the tree.

#### Example

```ts
const iso = new IsolationForest({ contamination: 0.1, n_estimators: 100 });
iso.fit(X_train);
const predictions = iso.predict(X_test);  // -1 for outliers, 1 for inliers
const scores = iso.score_samples(X_test); // Anomaly scores
```

#### Constructors

##### Constructor

```ts
new IsolationForest(options?): IsolationForest;
```

Defined in: [src/ml/outliers.js:202](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L202)

###### Parameters

###### options?

###### n_estimators

`number` = `100`

Number of trees (default: 100)

###### max_samples

`number` = `"auto"`

Samples to draw for each tree (default: 'auto' = min(256, n))

###### contamination

`number` = `0.1`

Expected proportion of outliers (default: 0.1)

###### max_features

`number` = `1.0`

Features to draw for each tree (default: 1.0 = all)

###### random_state

`number` = `null`

Random seed (default: null)

###### label_column

`string` = `"outlier"`

Name of output column for predictions (default: 'outlier')

###### Returns

[`IsolationForest`](#isolationforest)

#### Properties

##### n\_estimators

```ts
n_estimators: number;
```

Defined in: [src/ml/outliers.js:210](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L210)

##### max\_samples

```ts
max_samples: number;
```

Defined in: [src/ml/outliers.js:211](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L211)

##### contamination

```ts
contamination: number;
```

Defined in: [src/ml/outliers.js:212](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L212)

##### max\_features

```ts
max_features: number;
```

Defined in: [src/ml/outliers.js:213](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L213)

##### random\_state

```ts
random_state: number;
```

Defined in: [src/ml/outliers.js:214](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L214)

##### label\_column

```ts
label_column: string;
```

Defined in: [src/ml/outliers.js:215](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L215)

##### trees\_

```ts
trees_: any[] | null;
```

Defined in: [src/ml/outliers.js:217](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L217)

##### max\_samples\_

```ts
max_samples_: number | null;
```

Defined in: [src/ml/outliers.js:218](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L218)

##### offset\_

```ts
offset_: number | null;
```

Defined in: [src/ml/outliers.js:219](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L219)

##### threshold\_

```ts
threshold_: number | null;
```

Defined in: [src/ml/outliers.js:220](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L220)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/outliers.js:221](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L221)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/outliers.js:222](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L222)

##### \_originalData

```ts
_originalData: any;
```

Defined in: [src/ml/outliers.js:223](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L223)

##### \_groupModels

```ts
_groupModels: Map<any, any> | null;
```

Defined in: [src/ml/outliers.js:224](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L224)

#### Methods

##### fit()

```ts
fit(X): IsolationForest;
```

Defined in: [src/ml/outliers.js:232](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L232)

Fit the model

###### Parameters

###### X

`Object` \| `number`[][]

Training data (2D array or {data, columns, group})

###### Returns

[`IsolationForest`](#isolationforest)

this

##### \_fitSingleModel()

```ts
_fitSingleModel(X): Object;
```

Defined in: [src/ml/outliers.js:304](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L304)

Internal method to fit a single isolation forest model

###### Parameters

###### X

`number`[][]

2D array of numeric data

###### Returns

`Object`

Model parameters

##### score\_samples()

```ts
score_samples(X): number[];
```

Defined in: [src/ml/outliers.js:376](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L376)

Compute anomaly scores for samples
Lower (more negative) scores indicate outliers
Scores range approximately from -1 to 0

###### Parameters

###### X

`Object` \| `number`[][]

Data (2D array, {data, columns}, or array of objects)

###### Returns

`number`[]

Anomaly scores (negative values)

##### predict()

```ts
predict(X): number[] | Object[];
```

Defined in: [src/ml/outliers.js:449](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L449)

Predict if samples are outliers

###### Parameters

###### X

`Object` \| `number`[][]

Data (2D array or {data, columns, group})

###### Returns

`number`[] \| `Object`[]

Predictions: -1 for outliers, 1 for inliers (or table with outlier column)

##### \_predictWithModel()

```ts
_predictWithModel(X, model): number[];
```

Defined in: [src/ml/outliers.js:541](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L541)

Internal method to predict with a specific model

###### Parameters

###### X

`number`[][]

Data

###### model

`Object`

Model parameters

###### Returns

`number`[]

Predictions

##### transform()

```ts
transform(X): number[] | Object[];
```

Defined in: [src/ml/outliers.js:563](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L563)

Transform data by adding outlier labels
Alias for predict() - primary API for table-based workflows

###### Parameters

###### X

`Object` \| `number`[][]

Data (2D array or {data, columns, group})

###### Returns

`number`[] \| `Object`[]

Labels or table with outlier column

##### fit\_transform()

```ts
fit_transform(X): number[] | Object[];
```

Defined in: [src/ml/outliers.js:573](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L573)

Fit and transform in one step
Primary API for outlier detection with tables

###### Parameters

###### X

`Object` \| `number`[][]

Data (2D array or {data, columns, group})

###### Returns

`number`[] \| `Object`[]

Labels or table with outlier column

##### fit\_predict()

```ts
fit_predict(X): number[] | Object[];
```

Defined in: [src/ml/outliers.js:582](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L582)

Fit and predict in one step (sklearn compatibility)

###### Parameters

###### X

`Object` \| `number`[][]

Data (2D array or {data, columns, group})

###### Returns

`number`[] \| `Object`[]

Predictions: -1 for outliers, 1 for inliers (or table with outlier column)

***

### LocalOutlierFactor

Defined in: [src/ml/outliers.js:613](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L613)

Local Outlier Factor for outlier detection
Compatible with sklearn.neighbors.LocalOutlierFactor

Detects outliers using local density deviation.
LOF > 1 indicates outlier (lower local density than neighbors).

#### Example

```ts
const lof = new LocalOutlierFactor({ n_neighbors: 20, contamination: 0.1 });
lof.fit(X_train);
const predictions = lof.predict(X_test);  // -1 for outliers, 1 for inliers
```

#### Constructors

##### Constructor

```ts
new LocalOutlierFactor(options?): LocalOutlierFactor;
```

Defined in: [src/ml/outliers.js:622](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L622)

###### Parameters

###### options?

###### n_neighbors

`number` = `20`

Number of neighbors (default: 20)

###### algorithm

`string` = `"auto"`

'auto' (only option for now)

###### metric

`Function` = `null`

Distance function (default: euclidean)

###### contamination

`number` = `0.1`

Expected proportion of outliers (default: 0.1)

###### novelty

`string` = `false`

If true, can predict on new data (default: false)

###### Returns

[`LocalOutlierFactor`](#localoutlierfactor)

#### Properties

##### n\_neighbors

```ts
n_neighbors: number;
```

Defined in: [src/ml/outliers.js:633](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L633)

##### algorithm

```ts
algorithm: string;
```

Defined in: [src/ml/outliers.js:634](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L634)

##### metric

```ts
metric: Function;
```

Defined in: [src/ml/outliers.js:635](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L635)

##### contamination

```ts
contamination: number;
```

Defined in: [src/ml/outliers.js:636](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L636)

##### novelty

```ts
novelty: string;
```

Defined in: [src/ml/outliers.js:637](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L637)

##### X\_

```ts
X_: any[][] | null;
```

Defined in: [src/ml/outliers.js:639](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L639)

##### negative\_outlier\_factor\_

```ts
negative_outlier_factor_: any[] | null;
```

Defined in: [src/ml/outliers.js:640](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L640)

##### offset\_

```ts
offset_: number | null;
```

Defined in: [src/ml/outliers.js:641](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L641)

##### threshold\_

```ts
threshold_: number | null;
```

Defined in: [src/ml/outliers.js:642](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L642)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/outliers.js:643](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L643)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/outliers.js:644](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L644)

#### Accessors

##### negative\_outlier\_factor

###### Get Signature

```ts
get negative_outlier_factor(): number[];
```

Defined in: [src/ml/outliers.js:841](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L841)

Get negative outlier factor for each sample

###### Returns

`number`[]

Negative outlier factors

#### Methods

##### fit()

```ts
fit(X): LocalOutlierFactor;
```

Defined in: [src/ml/outliers.js:652](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L652)

Fit the model

###### Parameters

###### X

`Object` \| `number`[][]

Training data (2D array, {data, columns}, or array of objects)

###### Returns

[`LocalOutlierFactor`](#localoutlierfactor)

this

##### \_pairwiseDistances()

```ts
_pairwiseDistances(X): any[][];
```

Defined in: [src/ml/outliers.js:788](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L788)

Compute pairwise distances

###### Parameters

###### X

`any`

###### Returns

`any`[][]

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/outliers.js:808](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L808)

Predict if samples are outliers

###### Parameters

###### X

`Object` \| `number`[][]

Data (must be training data if novelty=false)

###### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

##### fit\_predict()

```ts
fit_predict(X): number[];
```

Defined in: [src/ml/outliers.js:833](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L833)

Fit and predict in one step

###### Parameters

###### X

`Object` \| `number`[][]

Data

###### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

***

### MahalanobisDistance

Defined in: [src/ml/outliers.js:861](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L861)

Mahalanobis distance-based outlier detection
Compatible with sklearn.covariance.EllipticEnvelope approach

Detects outliers based on statistical distance from the mean,
accounting for covariance structure. Uses pseudoinverse to handle
singular/near-singular covariance matrices.

#### Example

```ts
const md = new MahalanobisDistance({ contamination: 0.1 });
md.fit(X_train);
const predictions = md.predict(X_test);
```

#### Constructors

##### Constructor

```ts
new MahalanobisDistance(options?): MahalanobisDistance;
```

Defined in: [src/ml/outliers.js:867](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L867)

###### Parameters

###### options?

###### contamination

`number` = `0.1`

Expected proportion of outliers (default: 0.1)

###### use_chi2

`boolean` = `true`

Use chi-squared distribution for threshold (default: true)

###### Returns

[`MahalanobisDistance`](#mahalanobisdistance)

#### Properties

##### contamination

```ts
contamination: number;
```

Defined in: [src/ml/outliers.js:872](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L872)

##### use\_chi2

```ts
use_chi2: boolean;
```

Defined in: [src/ml/outliers.js:873](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L873)

##### mean\_

```ts
mean_: any[] | null;
```

Defined in: [src/ml/outliers.js:874](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L874)

##### precision\_

```ts
precision_: Matrix | null;
```

Defined in: [src/ml/outliers.js:875](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L875)

##### threshold\_

```ts
threshold_: number | null;
```

Defined in: [src/ml/outliers.js:876](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L876)

##### nFeatures\_

```ts
nFeatures_: number | null;
```

Defined in: [src/ml/outliers.js:877](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L877)

##### \_tableColumns

```ts
_tableColumns: any;
```

Defined in: [src/ml/outliers.js:878](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L878)

#### Accessors

##### mahalanobis\_distances

###### Get Signature

```ts
get mahalanobis_distances(): number[];
```

Defined in: [src/ml/outliers.js:1090](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1090)

Get Mahalanobis distances for fitted data

###### Returns

`number`[]

Mahalanobis distances

#### Methods

##### fit()

```ts
fit(X): MahalanobisDistance;
```

Defined in: [src/ml/outliers.js:886](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L886)

Fit the detector on training data

###### Parameters

###### X

`Object` \| `number`[][]

Training data (2D array, {data, columns}, or array of objects)

###### Returns

[`MahalanobisDistance`](#mahalanobisdistance)

this

##### \_mahalanobis\_distances()

```ts
_mahalanobis_distances(X): number[];
```

Defined in: [src/ml/outliers.js:982](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L982)

Compute Mahalanobis distances for samples

###### Parameters

###### X

`number`[][]

Data

###### Returns

`number`[]

Mahalanobis distances

##### score\_samples()

```ts
score_samples(X): number[];
```

Defined in: [src/ml/outliers.js:1012](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1012)

Compute Mahalanobis distances for samples

###### Parameters

###### X

`Object` \| `number`[][]

Data to score (2D array, {data, columns}, or array of objects)

###### Returns

`number`[]

Negative Mahalanobis distances (outliers have lower scores)

##### predict()

```ts
predict(X): number[];
```

Defined in: [src/ml/outliers.js:1067](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1067)

Predict if samples are outliers

###### Parameters

###### X

`Object` \| `number`[][]

Data

###### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

##### fit\_predict()

```ts
fit_predict(X): number[];
```

Defined in: [src/ml/outliers.js:1082](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1082)

Fit and predict in one step

###### Parameters

###### X

`Object` \| `number`[][]

Data

###### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

***

### Pipeline

Defined in: [src/ml/pipeline.js:20](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L20)

Pipeline class for chaining transformers and estimators

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new Pipeline(steps): Pipeline;
```

Defined in: [src/ml/pipeline.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L25)

Create a pipeline

###### Parameters

###### steps

`Object`[]

Array of transformers/estimators

###### Returns

[`Pipeline`](#pipeline)

###### Overrides

```ts
Estimator.constructor
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
Estimator.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Estimator.fitted
```

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

```ts
Estimator._warnings
```

##### steps

```ts
steps: Object[];
```

Defined in: [src/ml/pipeline.js:31](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L31)

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): Pipeline;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`Pipeline`](#pipeline)

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

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Estimator.fromJSON
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

##### fit()

```ts
fit(X, y?): Pipeline;
```

Defined in: [src/ml/pipeline.js:40](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L40)

Fit all steps

###### Parameters

###### X

`any`[]

Feature matrix

###### y?

`any`[] = `null`

Target values (optional)

###### Returns

[`Pipeline`](#pipeline)

this

###### Overrides

```ts
Estimator.fit
```

##### transform()

```ts
transform(X): any[];
```

Defined in: [src/ml/pipeline.js:73](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L73)

Transform data through all steps

###### Parameters

###### X

`any`[]

Feature matrix

###### Returns

`any`[]

Transformed data

###### Overrides

```ts
Estimator.transform
```

##### fitTransform()

```ts
fitTransform(X, y?): any[];
```

Defined in: [src/ml/pipeline.js:93](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L93)

Fit and transform in one step

###### Parameters

###### X

`any`[]

Feature matrix

###### y?

`any`[] = `null`

Target values (optional)

###### Returns

`any`[]

Transformed data

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/ml/pipeline.js:103](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L103)

Predict using final estimator

###### Parameters

###### X

`any`[]

Feature matrix

###### Returns

`any`[]

Predictions

###### Overrides

```ts
Estimator.predict
```

##### getFinalEstimator()

```ts
getFinalEstimator(): Object;
```

Defined in: [src/ml/pipeline.js:129](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L129)

Get final estimator

###### Returns

`Object`

Final step

***

### GridSearchCV

Defined in: [src/ml/pipeline.js:137](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L137)

Simple GridSearchCV for hyperparameter tuning

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new GridSearchCV(
   estimatorFn, 
   paramGrid, 
   scoreFn, 
   cv?): GridSearchCV;
```

Defined in: [src/ml/pipeline.js:145](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L145)

Create grid search

###### Parameters

###### estimatorFn

`Function`

Function that creates estimator: (params) => estimator

###### paramGrid

`Object`

Grid of parameters: {param1: [val1, val2], ...}

###### scoreFn

`Function`

Scoring function: (yTrue, yPred) => score

###### cv?

`number` = `5`

Number of cross-validation folds

###### Returns

[`GridSearchCV`](#gridsearchcv)

###### Overrides

```ts
Estimator.constructor
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
Estimator.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Estimator.fitted
```

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

```ts
Estimator._warnings
```

##### estimatorFn

```ts
estimatorFn: Function;
```

Defined in: [src/ml/pipeline.js:147](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L147)

##### paramGrid

```ts
paramGrid: Object;
```

Defined in: [src/ml/pipeline.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L148)

##### scoreFn

```ts
scoreFn: Function;
```

Defined in: [src/ml/pipeline.js:149](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L149)

##### cv

```ts
cv: number;
```

Defined in: [src/ml/pipeline.js:150](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L150)

##### bestParams

```ts
bestParams: any;
```

Defined in: [src/ml/pipeline.js:151](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L151)

##### bestScore

```ts
bestScore: number;
```

Defined in: [src/ml/pipeline.js:152](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L152)

##### bestEstimator

```ts
bestEstimator: any;
```

Defined in: [src/ml/pipeline.js:153](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L153)

##### cvResults

```ts
cvResults: any[];
```

Defined in: [src/ml/pipeline.js:154](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L154)

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): GridSearchCV;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`GridSearchCV`](#gridsearchcv)

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

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/estimators/estimator.js:302](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L302)

Serialize minimal model metadata.
Subclasses may override to include learned parameters.

###### Returns

`object`

###### params

```ts
params: Object;
```

###### fitted

```ts
fitted: boolean;
```

###### state

```ts
state: object;
```

###### warnings

```ts
warnings: any[];
```

###### Inherited from

```ts
Estimator.toJSON
```

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Estimator.fromJSON
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
fit(X, y): GridSearchCV;
```

Defined in: [src/ml/pipeline.js:189](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L189)

Perform grid search with cross-validation

###### Parameters

###### X

`any`[]

Feature matrix

###### y

`any`[]

Target values

###### Returns

[`GridSearchCV`](#gridsearchcv)

this

###### Overrides

```ts
Estimator.fit
```

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/ml/pipeline.js:264](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L264)

Predict using best estimator

###### Parameters

###### X

`any`[]

Feature matrix

###### Returns

`any`[]

Predictions

###### Overrides

```ts
Estimator.predict
```

##### getResults()

```ts
getResults(): Object[];
```

Defined in: [src/ml/pipeline.js:273](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/pipeline.js#L273)

Get results sorted by score

###### Returns

`Object`[]

Results

***

### Recipe

Defined in: [src/ml/recipe.js:143](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L143)

Recipe class for building inspectable preprocessing workflows

A Recipe defines a sequence of data preprocessing steps that can be:
1. Defined declaratively through chainable methods
2. Executed with prep() to fit transformers on training data
3. Applied to new data with bake() using fitted transformers
4. Inspected at every step to understand transformations

Supported preprocessing operations:

Data Cleaning:
- parseNumeric(): Convert string columns to numbers
- clean(): Remove rows with invalid categorical values

Missing Value Imputation:
- imputeMean(): Impute with mean values
- imputeMedian(): Impute with median values
- imputeMode(): Impute with mode (most frequent)
- imputeKNN(): Impute using K-Nearest Neighbors
- imputeIterative(): Impute using iterative MICE algorithm

Outlier Handling:
- removeOutliers(): Remove outliers using isolation forest, LOF, or Mahalanobis distance
- clipOutliers(): Clip outliers using IQR method

Encoding:
- oneHot(): One-hot encode categorical columns

Scaling:
- scale(): Scale numeric columns (standard or minmax)

Feature Engineering:
- createInteractions(): Create pairwise interaction features
- createPolynomial(): Create polynomial features
- binContinuous(): Bin continuous variables into discrete categories

Dimensionality Reduction:
- pca(): Principal Component Analysis
- lda(): Linear Discriminant Analysis (supervised)
- rda(): Redundancy Analysis (constrained ordination)

Sampling:
- upsample(): Oversample minority class for imbalanced data
- downsample(): Undersample majority class for imbalanced data

Feature Selection:
- selectByVariance(): Remove low-variance features
- selectByCorrelation(): Remove highly correlated features

Data Splitting:
- split(): Split into train/test sets

#### Example

```ts
// Complete workflow
const recipe = new Recipe({ data: myData, X: features, y: 'target' })
  .parseNumeric(['age', 'price'])
  .clean({ category: ['A', 'B', 'C'] })
  .oneHot(['category'])
  .scale(['age', 'price'], { method: 'standard' })
  .split({ ratio: 0.7, seed: 42 });

const result = recipe.prep();
// result.train.data - training data
// result.test.data - test data
// result.transformers - fitted transformers (scale, oneHot, etc.)
// result.steps - intermediate outputs for inspection
```

#### Constructors

##### Constructor

```ts
new Recipe(__namedParameters): Recipe;
```

Defined in: [src/ml/recipe.js:144](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L144)

###### Parameters

###### \_\_namedParameters

###### data

`any`

###### X

`any`

###### y

`any`

###### Returns

[`Recipe`](#recipe-3)

#### Properties

##### initialData

```ts
initialData: any;
```

Defined in: [src/ml/recipe.js:145](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L145)

##### X

```ts
X: any[];
```

Defined in: [src/ml/recipe.js:146](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L146)

##### y

```ts
y: any;
```

Defined in: [src/ml/recipe.js:147](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L147)

##### steps

```ts
steps: any[];
```

Defined in: [src/ml/recipe.js:148](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L148)

##### \_prepared

```ts
_prepared: boolean;
```

Defined in: [src/ml/recipe.js:149](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L149)

##### \_transformers

```ts
_transformers: object;
```

Defined in: [src/ml/recipe.js:150](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L150)

##### splitConfig

```ts
splitConfig: 
  | {
  ratio: number;
  shuffle: boolean;
  seed: number;
}
  | undefined;
```

Defined in: [src/ml/recipe.js:1496](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1496)

##### \_stepOutputs

```ts
_stepOutputs: 
  | (
  | {
  name: any;
  output: any;
  transformer: any;
}
  | {
  transformer?: undefined;
  name: any;
  output: any;
})[]
  | undefined;
```

Defined in: [src/ml/recipe.js:1578](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1578)

##### \_splitResult

```ts
_splitResult: 
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
}
  | undefined;
```

Defined in: [src/ml/recipe.js:1579](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1579)

#### Methods

##### parseNumeric()

```ts
parseNumeric(columns): Recipe;
```

Defined in: [src/ml/recipe.js:165](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L165)

Parse string columns as numeric

Converts string representations of numbers to actual numeric values.
Useful when CSV parsers incorrectly infer column types.

###### Parameters

###### columns

`string`[]

Column names to parse

###### Returns

[`Recipe`](#recipe-3)

this (for chaining)

###### Example

```ts
recipe.parseNumeric(['age', 'price', 'quantity']);
```

##### clean()

```ts
clean(validCategories): Recipe;
```

Defined in: [src/ml/recipe.js:195](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L195)

Clean categorical columns

###### Parameters

###### validCategories

`Object`

Map of column -> valid values

###### Returns

[`Recipe`](#recipe-3)

this

##### oneHot()

```ts
oneHot(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:223](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L223)

One-hot encode categorical columns

###### Parameters

###### columns

`string`[]

Columns to encode

###### options?

Encoding options

###### dropFirst

`boolean` = `true`

Drop first category

###### prefix

`boolean` = `true`

Use column name prefix

###### Returns

[`Recipe`](#recipe-3)

this

##### scale()

```ts
scale(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:293](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L293)

Scale numeric columns

###### Parameters

###### columns

`string`[]

Columns to scale

###### options?

Scaling options

###### method

`string` = `'standard'`

'standard' or 'minmax'

###### Returns

[`Recipe`](#recipe-3)

this

##### pca()

```ts
pca(options?): Recipe;
```

Defined in: [src/ml/recipe.js:342](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L342)

Apply Principal Component Analysis for dimensionality reduction

Reduces the dimensionality of numeric features by projecting them onto
principal components. This is a feature extraction/transformation step.

###### Parameters

###### options?

PCA options

###### columns

`string`[]

Columns to include in PCA

###### nComponents?

`number` = `null`

Number of components to keep (default: all)

###### scale?

`boolean` = `true`

Scale features before PCA

###### center?

`boolean` = `true`

Center features before PCA

###### Returns

[`Recipe`](#recipe-3)

this (for chaining)

###### Example

```ts
recipe.pca({
  columns: ['feature1', 'feature2', 'feature3'],
  nComponents: 2,
  scale: true
});
```

##### lda()

```ts
lda(options?): Recipe;
```

Defined in: [src/ml/recipe.js:439](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L439)

Apply Linear Discriminant Analysis for supervised dimensionality reduction

Reduces dimensionality while maximizing class separation. Requires a target
variable. This is both feature extraction and supervised learning.

###### Parameters

###### options?

LDA options

###### columns

`string`[]

Columns to include in LDA

###### nComponents?

`number` = `null`

Number of discriminants to keep

###### scale?

`boolean` = `false`

Scale features before LDA

###### Returns

[`Recipe`](#recipe-3)

this (for chaining)

###### Example

```ts
recipe.lda({
  columns: ['feature1', 'feature2', 'feature3'],
  nComponents: 2
});
```

##### rda()

```ts
rda(options?): Recipe;
```

Defined in: [src/ml/recipe.js:539](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L539)

Apply Redundancy Analysis for constrained ordination

RDA combines regression and PCA to find patterns in response variables
that are explained by predictor variables. Useful for ecological data.

###### Parameters

###### options?

RDA options

###### response

`string`[]

Response variable columns

###### predictors

`string`[]

Predictor variable columns

###### nComponents?

`number` = `null`

Number of RDA axes to keep

###### scale?

`boolean` = `false`

Scale variables before RDA

###### Returns

[`Recipe`](#recipe-3)

this (for chaining)

###### Example

```ts
recipe.rda({
  response: ['species1', 'species2', 'species3'],
  predictors: ['temperature', 'rainfall'],
  nComponents: 2
});
```

##### imputeMean()

```ts
imputeMean(columns): Recipe;
```

Defined in: [src/ml/recipe.js:618](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L618)

Impute missing values with mean

###### Parameters

###### columns

`string`[]

Columns to impute

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.imputeMean(['age', 'income']);
```

##### imputeMedian()

```ts
imputeMedian(columns): Recipe;
```

Defined in: [src/ml/recipe.js:653](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L653)

Impute missing values with median

###### Parameters

###### columns

`string`[]

Columns to impute

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.imputeMedian(['age', 'price']);
```

##### imputeMode()

```ts
imputeMode(columns): Recipe;
```

Defined in: [src/ml/recipe.js:688](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L688)

Impute missing values with mode (most frequent value)

###### Parameters

###### columns

`string`[]

Columns to impute

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.imputeMode(['category', 'status']);
```

##### imputeKNN()

```ts
imputeKNN(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:725](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L725)

Impute missing values using KNN

###### Parameters

###### columns

`string`[]

Columns to impute

###### options?

KNN imputer options

###### k

`number` = `5`

Number of neighbors (default 5)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.imputeKNN(['age', 'income'], { k: 3 });
```

##### imputeIterative()

```ts
imputeIterative(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:764](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L764)

Impute missing values using iterative imputation (MICE)

###### Parameters

###### columns

`string`[]

Columns to impute

###### options?

Iterative imputer options

###### maxIter

`number` = `10`

Maximum iterations (default 10)

###### tol

`number` = `0.001`

Convergence tolerance (default 0.001)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.imputeIterative(['age', 'income'], { maxIter: 20 });
```

##### removeOutliers()

```ts
removeOutliers(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:804](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L804)

Remove outliers from the dataset

###### Parameters

###### columns

`string`[]

Columns to check for outliers

###### options?

Outlier detection options

###### method

`string` = `'isolation_forest'`

Detection method: 'isolation_forest', 'lof', or 'mahalanobis'

###### contamination

`number` = `0.1`

Expected proportion of outliers (default 0.1)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.removeOutliers(['price', 'quantity'], { method: 'isolation_forest', contamination: 0.05 });
```

##### clipOutliers()

```ts
clipOutliers(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:844](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L844)

Clip outliers using IQR method

###### Parameters

###### columns

`string`[]

Columns to clip

###### options?

Clipping options

###### multiplier

`number` = `1.5`

IQR multiplier (default 1.5)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.clipOutliers(['price', 'age'], { multiplier: 1.5 });
```

##### createInteractions()

```ts
createInteractions(columns): Recipe;
```

Defined in: [src/ml/recipe.js:933](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L933)

Create pairwise interaction features

###### Parameters

###### columns

`string`[]

Columns to create interactions from

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.createInteractions(['feature1', 'feature2', 'feature3']);
// Creates: feature1_x_feature2, feature1_x_feature3, feature2_x_feature3
```

##### createPolynomial()

```ts
createPolynomial(columns, options?): Recipe;
```

Defined in: [src/ml/recipe.js:993](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L993)

Create polynomial features

###### Parameters

###### columns

`string`[]

Columns to create polynomials from

###### options?

Polynomial options

###### degree

`number` = `2`

Polynomial degree (default 2)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.createPolynomial(['age', 'income'], { degree: 2 });
// Creates: age^2, income^2
```

##### binContinuous()

```ts
binContinuous(column, options?): Recipe;
```

Defined in: [src/ml/recipe.js:1047](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1047)

Bin continuous variables into discrete categories

###### Parameters

###### column

`string`

Column to bin

###### options?

Binning options

###### bins

`number` = `5`

Number of bins (default 5)

###### labels

`string`[] = `null`

Custom bin labels

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.binContinuous('age', { bins: 5, labels: ['child', 'teen', 'adult', 'middle', 'senior'] });
```

##### upsample()

```ts
upsample(options?): Recipe;
```

Defined in: [src/ml/recipe.js:1160](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1160)

Upsample minority class for imbalanced classification

###### Parameters

###### options?

Upsampling options

###### targetRatio

`number` = `1.0`

Target ratio of minority to majority (default 1.0 for balanced)

###### seed

`number` = `null`

Random seed

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.upsample({ targetRatio: 1.0, seed: 42 });
```

##### downsample()

```ts
downsample(options?): Recipe;
```

Defined in: [src/ml/recipe.js:1223](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1223)

Downsample majority class for imbalanced classification

###### Parameters

###### options?

Downsampling options

###### strategy

`string` = `'balance'`

'balance' (equal classes) or 'ratio' (custom ratio)

###### targetRatio

`number` = `1.0`

Target ratio of majority to minority (for 'ratio' strategy)

###### seed

`number` = `null`

Random seed

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.downsample({ strategy: 'balance', seed: 42 });
```

##### selectByVariance()

```ts
selectByVariance(options?): Recipe;
```

Defined in: [src/ml/recipe.js:1299](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1299)

Remove low-variance features

###### Parameters

###### options?

Feature selection options

###### threshold

`number` = `0.0`

Variance threshold (default 0.0)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.selectByVariance({ threshold: 0.01 });
```

##### selectByCorrelation()

```ts
selectByCorrelation(options?): Recipe;
```

Defined in: [src/ml/recipe.js:1384](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1384)

Remove highly correlated features

###### Parameters

###### options?

Feature selection options

###### threshold

`number` = `0.95`

Correlation threshold (default 0.95)

###### Returns

[`Recipe`](#recipe-3)

this

###### Example

```ts
recipe.selectByCorrelation({ threshold: 0.9 });
```

##### split()

```ts
split(options?): Recipe;
```

Defined in: [src/ml/recipe.js:1495](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1495)

Split data into train/test sets

###### Parameters

###### options?

Split options

###### ratio

`number` = `0.7`

Training ratio (default 0.7)

###### shuffle

`boolean` = `true`

Shuffle before split

###### seed

`number` = `null`

Random seed

###### Returns

[`Recipe`](#recipe-3)

this

##### prep()

```ts
prep(): Object;
```

Defined in: [src/ml/recipe.js:1505](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1505)

Execute the recipe on the initial data
Returns train/test data and all fitted transformers

###### Returns

`Object`

Prepared data with train, test, transformers

##### bake()

```ts
bake(data): Object;
```

Defined in: [src/ml/recipe.js:1613](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1613)

Apply fitted transformers to new data

###### Parameters

###### data

`Object`[]

New data to transform

###### Returns

`Object`

Transformed data

##### summary()

```ts
summary(): string;
```

Defined in: [src/ml/recipe.js:1661](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L1661)

Get a summary of the recipe

###### Returns

`string`

Recipe summary

***

### BranchPipeline

Defined in: [src/pipeline/BranchPipeline.js:18](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L18)

#### Extends

- `Estimator`

#### Constructors

##### Constructor

```ts
new BranchPipeline(options?): BranchPipeline;
```

Defined in: [src/pipeline/BranchPipeline.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L25)

###### Parameters

###### options?

###### branches

\{
\} = `{}`

Named branches (estimators or pipelines)

###### combiner

`string` \| `Function` = `'vote'`

How to combine results: 'vote', 'average', 'max', or custom function

###### weights

`number`[] = `null`

Optional weights for each branch (for weighted voting)

###### Returns

[`BranchPipeline`](#branchpipeline)

###### Overrides

```ts
Estimator.constructor
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
Estimator.params
```

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/core/estimators/estimator.js:25](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L25)

###### Inherited from

```ts
Estimator.fitted
```

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

```ts
Estimator._warnings
```

##### branches

```ts
branches: object;
```

Defined in: [src/pipeline/BranchPipeline.js:27](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L27)

##### combiner

```ts
combiner: string | Function;
```

Defined in: [src/pipeline/BranchPipeline.js:28](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L28)

##### weights

```ts
weights: number[];
```

Defined in: [src/pipeline/BranchPipeline.js:29](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L29)

##### branchNames

```ts
branchNames: string[];
```

Defined in: [src/pipeline/BranchPipeline.js:30](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L30)

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
Estimator._repr_html_
```

##### setParams()

```ts
setParams(params?): BranchPipeline;
```

Defined in: [src/core/estimators/estimator.js:285](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L285)

Set parameters (mutates instance).

###### Parameters

###### params?

`Object` = `{}`

###### Returns

[`BranchPipeline`](#branchpipeline)

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

##### fromJSON()

```ts
static fromJSON(obj?): Estimator;
```

Defined in: [src/core/estimators/estimator.js:317](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/estimators/estimator.js#L317)

Basic deserialization. Subclasses should override if they need
to restore learned arrays / matrices.

###### Parameters

###### obj?

`Object` = `{}`

###### Returns

`Estimator`

###### Inherited from

```ts
Estimator.fromJSON
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
fit(X, y?): BranchPipeline;
```

Defined in: [src/pipeline/BranchPipeline.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L44)

Fit all branches

###### Parameters

###### X

`any`[]

Training data

###### y?

`any`[] = `null`

Target labels (optional)

###### Returns

[`BranchPipeline`](#branchpipeline)

###### Overrides

```ts
Estimator.fit
```

##### predict()

```ts
predict(X): any[];
```

Defined in: [src/pipeline/BranchPipeline.js:62](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L62)

Predict using all branches and combine results

###### Parameters

###### X

`any`[]

Data to predict

###### Returns

`any`[]

Combined predictions

###### Overrides

```ts
Estimator.predict
```

##### predictAll()

```ts
predictAll(X): object;
```

Defined in: [src/pipeline/BranchPipeline.js:88](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L88)

Get predictions from all branches without combining

###### Parameters

###### X

`any`[]

Data to predict

###### Returns

`object`

Predictions keyed by branch name

##### agreementScore()

```ts
agreementScore(X): number;
```

Defined in: [src/pipeline/BranchPipeline.js:255](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L255)

Get agreement score (how often branches agree)

###### Parameters

###### X

`any`[]

Data

###### Returns

`number`

Agreement score between 0 and 1

##### confidence()

```ts
confidence(X): number[];
```

Defined in: [src/pipeline/BranchPipeline.js:275](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L275)

Get per-sample confidence (based on branch agreement)

###### Parameters

###### X

`any`[]

Data

###### Returns

`number`[]

Confidence scores between 0 and 1

##### summary()

```ts
summary(): Object;
```

Defined in: [src/pipeline/BranchPipeline.js:300](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L300)

Summary statistics

###### Returns

`Object`

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/pipeline/BranchPipeline.js:313](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/pipeline/BranchPipeline.js#L313)

Serialization

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'BranchPipeline';
```

###### combiner

```ts
combiner: string;
```

###### weights

```ts
weights: number[];
```

###### fitted

```ts
fitted: boolean;
```

###### branches

```ts
branches: string[];
```

###### Overrides

```ts
Estimator.toJSON
```

## Functions

### simpleImpute()

```ts
function simpleImpute(X, options?): number[][];
```

Defined in: [src/ml/impute.js:1066](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L1066)

Simple imputation (functional interface)

#### Parameters

##### X

`number`[][]

Data with missing values

##### options?

`Object` = `{}`

Imputer options

#### Returns

`number`[][]

Imputed data

***

### knnImpute()

```ts
function knnImpute(X, options?): number[][];
```

Defined in: [src/ml/impute.js:1077](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L1077)

KNN imputation (functional interface)

#### Parameters

##### X

`number`[][]

Data with missing values

##### options?

`Object` = `{}`

Imputer options

#### Returns

`number`[][]

Imputed data

***

### iterativeImpute()

```ts
function iterativeImpute(X, options?): number[][];
```

Defined in: [src/ml/impute.js:1088](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/impute.js#L1088)

Iterative imputation (functional interface)

#### Parameters

##### X

`number`[][]

Data with missing values

##### options?

`Object` = `{}`

Imputer options

#### Returns

`number`[][]

Imputed data

***

### isolationForest()

```ts
function isolationForest(X, options?): number[];
```

Defined in: [src/ml/outliers.js:1106](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1106)

Isolation Forest (functional interface)

#### Parameters

##### X

`number`[][]

Data

##### options?

`Object` = `{}`

IsolationForest options

#### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

***

### localOutlierFactor()

```ts
function localOutlierFactor(X, options?): number[];
```

Defined in: [src/ml/outliers.js:1117](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1117)

Local Outlier Factor (functional interface)

#### Parameters

##### X

`number`[][]

Data

##### options?

`Object` = `{}`

LOF options

#### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

***

### mahalanobisDistance()

```ts
function mahalanobisDistance(X, options?): number[];
```

Defined in: [src/ml/outliers.js:1128](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/outliers.js#L1128)

Mahalanobis Distance (functional interface)

#### Parameters

##### X

`number`[][]

Data

##### options?

`Object` = `{}`

MahalanobisDistance options

#### Returns

`number`[]

Predictions: -1 for outliers, 1 for inliers

***

### recipe()

```ts
function recipe(options): Recipe;
```

Defined in: [src/ml/recipe.js:71](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/ml/recipe.js#L71)

Create a preprocessing recipe

Factory function to create a new Recipe instance. The recipe defines a sequence
of preprocessing operations that will be applied to data.

#### Parameters

##### options

Initial data descriptor

###### data

`Object`[]

Input data as array of objects

###### X

`string` \| `string`[]

Feature column names

###### y

`string`

Target column name

#### Returns

[`Recipe`](#recipe-3)

A recipe object for chaining preprocessing steps

#### Example

```ts
const recipe = ds.ml.recipe({
  data: penguins,
  X: ['bill_length', 'bill_depth', 'flipper_length'],
  y: 'species'
});
```
