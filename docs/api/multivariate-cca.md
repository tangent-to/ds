---
layout: default
title: cca
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/cca
---
# cca

## Functions

### fit()

```ts
function fit(
   X, 
   Y?, 
   options?): Object;
```

Defined in: [src/mva/cca.js:28](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/cca.js#L28)

Fit CCA model.

Accepts either numeric matrices (fit(XMatrix, YMatrix, options)) or a declarative
object: fit({ X: ['col1', ...], Y: ['colA', ...], data, omit_missing, center, scale }).

#### Parameters

##### X

`Object` \| `number`[][]

Design matrix (n × p) for the first dataset, or a declarative config object

##### Y?

`number`[][] = `null`

Design matrix (n × q) for the second dataset (ignored when X is declarative)

##### options?

Fitting options

###### center?

`boolean`

Center columns to zero mean (default true)

###### scale?

`boolean`

Scale columns to unit variance (default false)

###### columnsX?

`string`[]

Column names for X

###### columnsY?

`string`[]

Column names for Y

#### Returns

`Object`

Fitted CCA model

***

### transformX()

```ts
function transformX(
   model, 
   X, 
   options?): Object[];
```

Defined in: [src/mva/cca.js:211](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/cca.js#L211)

Project new X data onto the fitted X canonical variates

#### Parameters

##### model

`Object`

Fitted CCA model

##### X

`number`[][]

New X data matrix (n × p)

##### options?

`Object` = `{}`

Transform options

#### Returns

`Object`[]

Canonical score objects, one per row

***

### transformY()

```ts
function transformY(
   model, 
   Y, 
   options?): Object[];
```

Defined in: [src/mva/cca.js:235](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/cca.js#L235)

Project new Y data onto the fitted Y canonical variates

#### Parameters

##### model

`Object`

Fitted CCA model

##### Y

`number`[][]

New Y data matrix (n × q)

##### options?

`Object` = `{}`

Transform options

#### Returns

`Object`[]

Canonical score objects, one per row

***

### transform()

```ts
function transform(
   model, 
   X, 
   Y, 
   options?): Object;
```

Defined in: [src/mva/cca.js:260](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/cca.js#L260)

Project new X and Y data onto their fitted canonical variates

#### Parameters

##### model

`Object`

Fitted CCA model

##### X

`number`[][]

New X data matrix (n × p)

##### Y

`number`[][]

New Y data matrix (n × q)

##### options?

`Object` = `{}`

Transform options

#### Returns

`Object`

Object with xScores and yScores arrays of score objects
