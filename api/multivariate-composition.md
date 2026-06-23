---
layout: default
title: composition
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/composition
---
# composition

## Classes

### CompositionalOutlierDetector

Defined in: [src/mva/composition.js:524](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L524)

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

Defined in: [src/mva/composition.js:530](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L530)

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

Defined in: [src/mva/composition.js:531](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L531)

##### alpha

```ts
alpha: number;
```

Defined in: [src/mva/composition.js:532](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L532)

##### fitted

```ts
fitted: boolean;
```

Defined in: [src/mva/composition.js:533](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L533)

##### nParts

```ts
nParts: any;
```

Defined in: [src/mva/composition.js:547](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L547)

##### dim

```ts
dim: any;
```

Defined in: [src/mva/composition.js:548](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L548)

##### df

```ts
df: any;
```

Defined in: [src/mva/composition.js:549](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L549)

##### center

```ts
center: any[] | undefined;
```

Defined in: [src/mva/composition.js:554](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L554)

##### covInverse

```ts
covInverse: 
  | number[][]
  | Matrix
  | undefined;
```

Defined in: [src/mva/composition.js:565](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L565)

#### Methods

##### fit()

```ts
fit(mat): CompositionalOutlierDetector;
```

Defined in: [src/mva/composition.js:542](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L542)

Estimate the centroid and (pseudo-inverse) covariance in log-ratio space
from a reference composition — e.g. a healthy / high-yielding subpopulation.

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

Defined in: [src/mva/composition.js:571](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L571)

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

Defined in: [src/mva/composition.js:588](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L588)

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

Defined in: [src/mva/composition.js:598](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L598)

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

Defined in: [src/mva/composition.js:167](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L167)

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

Defined in: [src/mva/composition.js:89](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L89)

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

Defined in: [src/mva/composition.js:110](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L110)

Replaces zeros with small delta values before closure

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data

##### delta?

`number` = `1e-6`

Replacement value for zeros (default: 1e-6)

#### Returns

`any`[] \| `any`[][]

Composition with zeros replaced

***

### power()

```ts
function power(mat, pow): any[] | any[][];
```

Defined in: [src/mva/composition.js:135](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L135)

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

Defined in: [src/mva/composition.js:149](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L149)

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

Defined in: [src/mva/composition.js:176](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L176)

Centered log-ratio transformation (CLR)

#### Parameters

##### mat

`any`[] \| `any`[][]

Input compositional data (positive values)

##### handleZeros?

`boolean` = `false`

If true, replace zeros before transform

##### delta?

`number` = `1e-6`

Replacement value for zeros

#### Returns

`any`[] \| `any`[][]

CLR-transformed data

***

### clrInv()

```ts
function clrInv(mat): any[] | any[][];
```

Defined in: [src/mva/composition.js:200](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L200)

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

Defined in: [src/mva/composition.js:217](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L217)

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

`number` = `1e-6`

Replacement value for zeros

#### Returns

`any`[] \| `any`[][]

ALR-transformed data (dimension reduced by 1)

***

### alrInv()

```ts
function alrInv(mat, denomIdx?): any[] | any[][];
```

Defined in: [src/mva/composition.js:254](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L254)

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

Defined in: [src/mva/composition.js:283](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L283)

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

Defined in: [src/mva/composition.js:317](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L317)

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

`number` = `1e-6`

Replacement value for zeros

#### Returns

`any`[] \| `any`[][]

ILR-transformed data

***

### ilrInv()

```ts
function ilrInv(mat, basis?): any[] | any[][];
```

Defined in: [src/mva/composition.js:355](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L355)

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

Defined in: [src/mva/composition.js:387](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L387)

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

Defined in: [src/mva/composition.js:434](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L434)

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

Defined in: [src/mva/composition.js:625](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/mva/composition.js#L625)

Detect compositional outliers via the Mahalanobis distance in log-ratio
space, tested as a chi-squared variable (Filzmoser & Hron; Parent & Dafir,
1992). Convenience wrapper around [CompositionalOutlierDetector](#compositionaloutlierdetector) that
fits on `mat` (or a `reference` subset of it) and tests `mat`.

For testing *new* points against the fitted reference (e.g. external
standards), fit a detector once and call `.test(newComposition)` — no manual
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
