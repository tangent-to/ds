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

Defined in: [src/mva/rda.js:29](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/rda.js#L29)

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

### permutationTest()

```ts
function permutationTest(model, options?): Object;
```

Defined in: [src/mva/rda.js:373](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/rda.js#L373)

Permutation test of the global RDA (equivalent to vegan's
`anova.cca(model)`): tests H0 that the constraints explain no more response
variance than expected by chance. Under H0 the response rows are exchangeable,
so we permute the rows of the (centred) response matrix, refit, and compare the
permuted pseudo-F to the observed one. Because a row permutation leaves each
response column's total sum of squares unchanged, the total inertia and the
degrees of freedom are invariant, so only the constrained inertia is recomputed.

The pseudo-F is `(constrained inertia / dfModel) / (residual inertia / dfResidual)`
with `dfModel` the rank of the constraints and `dfResidual = n - dfModel - 1`,
matching vegan. The p-value uses the standard `(1 + #{F* >= F}) / (nperm + 1)`
correction. F, the proportion constrained and the df are divisor-invariant, so
they match vegan regardless of its n-1 inertia convention; the reported inertia
values use the n-1 divisor to match vegan's "Variance" column directly.

#### Parameters

##### model

`Object`

A constrained RDA model from fit().

##### options?

###### permutations?

`number`

Number of row permutations.

###### seed?

`number`

Seed for reproducibility.

#### Returns

`Object`

{ pseudoF, pValue, permutations, dfModel, dfResidual,
  constrainedInertia, residualInertia, totalInertia, constrainedProportion, eigenvalues }

***

### transform()

```ts
function transform(
   model, 
   Y, 
   X): Object[];
```

Defined in: [src/mva/rda.js:455](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/rda.js#L455)

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
