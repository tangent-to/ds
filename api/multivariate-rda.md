---
layout: default
title: rda
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/rda
---
# rda

## Functions

### fit()

```ts
function fit(
   Y, 
   X, 
   options?): Object;
```

Defined in: [src/mva/rda.js:29](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/rda.js#L29)

Fit RDA model.

#### Parameters

##### Y

`number`[][]

Response matrix (n x q)

##### X

`number`[][]

Explanatory matrix (n x p)

##### options?

###### scale?

`boolean`

Standardise response variables before regression.

###### constrained?

`boolean`

When true, perform PCA on fitted values (constrained ordination); when false, perform PCA on residuals (unconstrained ordination).

#### Returns

`Object`

RDA model

***

### transform()

```ts
function transform(
   model, 
   Y, 
   X): Object[];
```

Defined in: [src/mva/rda.js:302](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/rda.js#L302)

Transform new data using fitted RDA model

#### Parameters

##### model

`Object`

Fitted RDA model

##### Y

`number`[][]

New response data

##### X

`number`[][]

New explanatory data

#### Returns

`Object`[]

Canonical scores
