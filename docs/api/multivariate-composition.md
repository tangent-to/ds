---
layout: default
title: composition
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/composition
---
# composition

## Classes

### CompositionalImputer

Defined in: [src/mva/composition.js:571](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L571)

Fit/transform wrapper around [imputeMissing](#imputemissing) for leakage-free
cross-validation. `fit()` learns the CLR mean of a training composition
(zeros and missing cells treated alike as left-censored); `transform()`
completes each row of new data toward that learned mean, holding the observed
parts fixed. Because every incomplete row is completed individually (toward a
shared target mean), below-detection samples do NOT collapse onto one constant
coordinate - the property that motivates lrEM imputation in the first place - 
while test rows never influence the imputation model.

#### Example

```ts
const imp = new CompositionalImputer().fit(trainComp);
const trainZ = imp.transform(trainComp); // completed training rows
const testZ  = imp.transform(testComp);  // completed with train-only stats
```

#### Constructors

##### Constructor

```ts
new CompositionalImputer(opts?): CompositionalImputer;
```

Defined in: [src/mva/composition.js:577](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L577)

###### Parameters

###### opts?

###### maxIter?

`number` = `100`

EM iterations for the training fit.

###### tol?

`number` = `1e-9`

Convergence tolerance for the training fit.

###### Returns

[`CompositionalImputer`](#compositionalimputer)

#### Properties

##### maxIter

```ts
maxIter: number;
```

Defined in: [src/mva/composition.js:578](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L578)

##### tol

```ts
tol: number;
```

Defined in: [src/mva/composition.js:579](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L579)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/mva/composition.js:580](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L580)

##### meanClr

```ts
meanClr: any[] | undefined;
```

Defined in: [src/mva/composition.js:604](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L604)

##### D

```ts
D: any;
```

Defined in: [src/mva/composition.js:605](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L605)

#### Methods

##### fit()

```ts
fit(mat): CompositionalImputer;
```

Defined in: [src/mva/composition.js:593](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L593)

Learn the CLR mean of the (imputed) training composition.

###### Parameters

###### mat

`number`[][]

Training composition with zeros/missing.

###### Returns

[`CompositionalImputer`](#compositionalimputer)

this

##### transform()

```ts
transform(mat): number[][];
```

Defined in: [src/mva/composition.js:615](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L615)

Complete each row of `mat` toward the learned CLR mean.

###### Parameters

###### mat

`number`[][]

Composition with zeros/missing.

###### Returns

`number`[][]

Strictly-positive completed composition.

##### fitTransform()

```ts
fitTransform(mat): number[][];
```

Defined in: [src/mva/composition.js:647](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L647)

Convenience: fit then transform the same matrix.

###### Parameters

###### mat

`number`[][]

Composition with zeros/missing values (n samples × D parts)

###### Returns

`number`[][]

Strictly-positive completed composition

***

### CompositionalOutlierDetector

Defined in: [src/mva/composition.js:672](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L672)

Detect compositional outliers via the Mahalanobis distance in log-ratio
space, tested as a chi-squared variable (Filzmoser & Hron; Parent & Dafir,
1992).

Each observation's CLR (or ILR) vector is compared to a centroid using the
(pseudo-inverse) covariance; the squared Mahalanobis distance follows a
chi-squared distribution with `D − 1` degrees of freedom under compositional
normality. The centroid and covariance may be estimated from a reference
subpopulation (e.g. a high-yielding group) via `reference`.

#### Param

**mat**

Strictly-positive composition.

#### Param

**opts**

#### Param

**opts.reference**

Boolean mask selecting the rows
  that define the centroid/covariance (default: all rows).

#### Param

**opts.alpha**

Significance level for the outlier flag.

#### Param

**opts.transform**

Log-ratio coordinates to use.

#### Constructors

##### Constructor

```ts
new CompositionalOutlierDetector(opts?): CompositionalOutlierDetector;
```

Defined in: [src/mva/composition.js:678](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L678)

###### Parameters

###### opts?

###### transform?

`"clr"` \| `"ilr"` = `"clr"`

Log-ratio coordinates to use.

###### alpha?

`number` = `0.05`

Significance level for the outlier flag.

###### Returns

[`CompositionalOutlierDetector`](#compositionaloutlierdetector)

#### Properties

##### transform

```ts
transform: "clr" | "ilr";
```

Defined in: [src/mva/composition.js:679](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L679)

##### alpha

```ts
alpha: number;
```

Defined in: [src/mva/composition.js:680](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L680)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/mva/composition.js:681](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L681)

##### nParts

```ts
nParts: any;
```

Defined in: [src/mva/composition.js:695](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L695)

##### dim

```ts
dim: any;
```

Defined in: [src/mva/composition.js:696](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L696)

##### df

```ts
df: any;
```

Defined in: [src/mva/composition.js:697](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L697)

##### center

```ts
center: any[] | undefined;
```

Defined in: [src/mva/composition.js:702](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L702)

##### covInverse

```ts
covInverse: any;
```

Defined in: [src/mva/composition.js:713](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L713)

#### Methods

##### fit()

```ts
fit(mat): CompositionalOutlierDetector;
```

Defined in: [src/mva/composition.js:690](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L690)

Estimate the centroid and (pseudo-inverse) covariance in log-ratio space
from a reference composition - e.g. a healthy / high-yielding subpopulation.

###### Parameters

###### mat

`number`[][]

Strictly-positive reference composition.

###### Returns

[`CompositionalOutlierDetector`](#compositionaloutlierdetector)

this

##### distance()

```ts
distance(mat): number[];
```

Defined in: [src/mva/composition.js:719](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L719)

Squared Mahalanobis distance in log-ratio space for each row of `mat`.

###### Parameters

###### mat

`any`

###### Returns

`number`[]

##### pValue()

```ts
pValue(mat): number[];
```

Defined in: [src/mva/composition.js:736](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L736)

Chi-squared p-value (1 − CDF) for each row's Mahalanobis distance.

###### Parameters

###### mat

`any`

###### Returns

`number`[]

##### test()

```ts
test(mat): object;
```

Defined in: [src/mva/composition.js:746](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L746)

Test rows for compositional outlyingness against the fitted reference.

###### Parameters

###### mat

`number`[][]

Composition(s) to test (e.g. all
  experimental samples, or external standards to project).

###### Returns

`object`

###### distances

```ts
distances: number[];
```

###### pValues

```ts
pValues: number[];
```

###### outliers

```ts
outliers: boolean[];
```

###### df

```ts
df: number;
```

## Variables

### centralize

```ts
const centralize: (mat) => any[] | any[][] = center;
```

Defined in: [src/mva/composition.js:219](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L219)

Alias for center

Centers compositions by geometric mean

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

#### Returns

`any`[] \| `any`[][]

Centered composition

## Functions

### closure()

```ts
function closure(mat): any[] | any[][];
```

Defined in: [src/mva/composition.js:89](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L89)

Normalizes rows to sum to 1 (closure operation)

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

#### Returns

`any`[] \| `any`[][]

Closed composition (rows sum to 1)

***

### multiplicativeReplacement()

```ts
function multiplicativeReplacement(mat, delta?): any[] | any[][];
```

Defined in: [src/mva/composition.js:137](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L137)

Multiplicative replacement of zeros (Martín-Fernández, Barceló-Vidal &
Pawlowsky-Glahn, 2003).

Each row is first closed to sum to 1, so `delta` is a proportion of the row
total. For a closed row `x` and thresholds `δ_j`:

- `r_j = δ_j` where `x_j = 0`;
- `r_j = x_j · (1 − Σ_{k : x_k = 0} δ_k)` elsewhere.

Equivalently, on a row of total `c`, zeros become `δ_j · c` and the non-zero
parts are multiplied by `1 − Σ_{k : x_k = 0} δ_k`. The total does not change.
Non-zero parts are rescaled by a common factor, so their ratios are kept:
`r_i / r_j = x_i / x_j`. This is the subcompositional coherence that the
method exists for. A row with no zeros is only closed. The result is closed
(rows sum to 1).

Reference: Martín-Fernández, J. A., Barceló-Vidal, C. & Pawlowsky-Glahn, V.
(2003). Dealing with zeros and missing values in compositional data sets
using nonparametric imputation. *Mathematical Geology*, 35(3), 253–278.
https://doi.org/10.1023/A:1023866030544

#### Parameters

##### mat

`any`[] \| `any`[][]

Compositional data: non-negative, every
  row with a positive total. Rows need not be closed.

##### delta?

`number` \| `number`[]

Threshold for the zeros, as a
  proportion of the row total. Pass one number for every part, or an array
  with one threshold per part (default: 1e-6).

#### Returns

`any`[] \| `any`[][]

Closed composition with no zeros, in the shape
  of the input (vector or matrix).

#### Throws

If `mat` has a negative value, if a row has a zero total,
  if `delta` is not positive or (as an array) does not have one entry per
  part, or if a row's thresholds for its zeros sum to 1 or more.

***

### power()

```ts
function power(mat, pow): any[] | any[][];
```

Defined in: [src/mva/composition.js:187](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L187)

Raises components to a power and renormalizes

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

##### pow

`any`

#### Returns

`any`[] \| `any`[][]

Powered and renormalized composition

***

### center()

```ts
function center(mat): any[] | any[][];
```

Defined in: [src/mva/composition.js:201](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L201)

Centers compositions by geometric mean

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

#### Returns

`any`[] \| `any`[][]

Centered composition

***

### clr()

```ts
function clr(
   mat, 
   handleZeros?, 
   delta?): any[] | any[][];
```

Defined in: [src/mva/composition.js:228](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L228)

Centered log-ratio transformation (CLR)

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data (positive values)

##### handleZeros?

`boolean` = `false`

If true, replace zeros before transform

##### delta?

`number` \| `number`[]

Threshold for zeros, passed to [multiplicativeReplacement](#multiplicativereplacement)

#### Returns

`any`[] \| `any`[][]

CLR-transformed data

***

### clrInv()

```ts
function clrInv(mat): any[] | any[][];
```

Defined in: [src/mva/composition.js:252](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L252)

Inverse centered log-ratio transformation

#### Parameters

##### mat

`any`[] \| `any`[][]

CLR-transformed data

#### Returns

`any`[] \| `any`[][]

Composition (rows sum to 1)

***

### alr()

```ts
function alr(
   mat, 
   denomIdx?, 
   handleZeros?, 
   delta?): any[] | any[][];
```

Defined in: [src/mva/composition.js:269](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L269)

Additive log-ratio transformation (ALR)
Uses the last component as the reference denominator

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

##### denomIdx?

`number` \| `null`

Index of denominator component (default: last)

##### handleZeros?

`boolean` = `false`

If true, replace zeros before transform

##### delta?

`number` \| `number`[]

Threshold for zeros, passed to [multiplicativeReplacement](#multiplicativereplacement)

#### Returns

`any`[] \| `any`[][]

ALR-transformed data (dimension reduced by 1)

***

### alrInv()

```ts
function alrInv(mat, denomIdx?): any[] | any[][];
```

Defined in: [src/mva/composition.js:306](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L306)

Inverse additive log-ratio transformation

#### Parameters

##### mat

`any`[] \| `any`[][]

ALR-transformed data

##### denomIdx?

`number` \| `null`

Index where denominator was (default: last)

#### Returns

`any`[] \| `any`[][]

Composition (rows sum to 1)

***

### sbpBasis()

```ts
function sbpBasis(partition): any[][];
```

Defined in: [src/mva/composition.js:335](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L335)

Constructs orthonormal basis from sequential binary partition

#### Parameters

##### partition

`any`[][]

Sequential binary partition matrix

#### Returns

`any`[][]

Orthonormal basis for ILR

***

### ilr()

```ts
function ilr(
   mat, 
   basis?, 
   handleZeros?, 
   delta?): any[] | any[][];
```

Defined in: [src/mva/composition.js:369](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L369)

Isometric log-ratio transformation (ILR)

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

##### basis?

`any`[][] \| `null`

Orthonormal basis (default: Gram-Schmidt basis)

##### handleZeros?

`boolean` = `false`

If true, replace zeros before transform

##### delta?

`number` \| `number`[]

Threshold for zeros, passed to [multiplicativeReplacement](#multiplicativereplacement)

#### Returns

`any`[] \| `any`[][]

ILR-transformed data

***

### ilrInv()

```ts
function ilrInv(mat, basis?): any[] | any[][];
```

Defined in: [src/mva/composition.js:407](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L407)

Inverse isometric log-ratio transformation

#### Parameters

##### mat

`any`[] \| `any`[][]

ILR-transformed data

##### basis?

`any`[][] \| `null`

Orthonormal basis used in forward transform

#### Returns

`any`[] \| `any`[][]

Composition (rows sum to 1)

***

### inner()

```ts
function inner(x, y): number | any[];
```

Defined in: [src/mva/composition.js:439](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L439)

Computes inner product in the Aitchison simplex

#### Parameters

##### x

`any`[] \| `any`[][]

First composition

##### y

`any`[] \| `any`[][]

Second composition

#### Returns

`number` \| `any`[]

Inner product(s)

***

### imputeMissing()

```ts
function imputeMissing(mat, opts?): number[][];
```

Defined in: [src/mva/composition.js:486](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L486)

Impute missing values in compositional data, respecting the simplex.

Missing cells (`null`, `undefined` or `NaN`) are filled by an EM-style
iteration in centred-log-ratio (CLR) space: each incomplete row is updated so
that its CLR coordinates on the missing parts match the compositional
(CLR) mean of the complete observations, while the observed parts are
preserved. This is the log-ratio analogue of mean imputation and keeps the
imputed values strictly positive and coherent with the observed
sub-composition (cf. Martín-Fernández et al.; Palarea-Albaladejo &
Martín-Fernández, 2008). Combine with [multiplicativeReplacement](#multiplicativereplacement) to
additionally handle essential zeros.

#### Parameters

##### mat

`number`[][]

Composition with missing entries.

##### opts?

###### maxIter?

`number` = `100`

Maximum EM iterations.

###### tol?

`number` = `1e-9`

Convergence tolerance on the CLR mean.

#### Returns

`number`[][]

Completed, strictly-positive composition.

***

### compositionalOutliers()

```ts
function compositionalOutliers(mat, opts?): object;
```

Defined in: [src/mva/composition.js:773](https://github.com/tangent-to/ds/blob/09615975dac258601b705b966868b430eb321dc8/src/mva/composition.js#L773)

Detect compositional outliers via the Mahalanobis distance in log-ratio
space, tested as a chi-squared variable (Filzmoser & Hron; Parent & Dafir,
1992). Convenience wrapper around [CompositionalOutlierDetector](#compositionaloutlierdetector) that
fits on `mat` (or a `reference` subset of it) and tests `mat`.

For testing *new* points against the fitted reference (e.g. external
standards), fit a detector once and call `.test(newComposition)` - no manual
projection needed.

#### Parameters

##### mat

`number`[][]

Strictly-positive composition.

##### opts?

###### reference?

`boolean`[] = `null`

Mask selecting the rows that define
  the centroid/covariance (default: all rows).

###### alpha?

`number` = `0.05`

###### transform?

`"clr"` \| `"ilr"` = `"clr"`

#### Returns

`object`

##### distances

```ts
distances: number[];
```

##### pValues

```ts
pValues: number[];
```

##### outliers

```ts
outliers: boolean[];
```

##### df

```ts
df: number;
```

##### center

```ts
center: number[];
```

##### covInverse

```ts
covInverse: number[][];
```

##### detector

```ts
detector: CompositionalOutlierDetector;
```
