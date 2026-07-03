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

Defined in: [src/mva/rda.js:29](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/mva/rda.js#L29)

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

Defined in: [src/mva/rda.js:302](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/mva/rda.js#L302)

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
