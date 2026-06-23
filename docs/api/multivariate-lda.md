---
layout: default
title: lda
parent: Multivariate Analysis
grand_parent: API Reference
permalink: /api/multivariate/lda
---
# lda

## Functions

### fit()

```ts
function fit(
   X, 
   y, 
   options?): object;
```

Defined in: [src/mva/lda.js:22](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/lda.js#L22)

#### Parameters

##### X

`any`

##### y

`any`

##### options?

#### Returns

`object`

##### scores

```ts
scores: Object[];
```

##### loadings

```ts
loadings: Object[];
```

##### eigenvalues

```ts
eigenvalues: any = sortedEigenvalues;
```

##### rawScores

```ts
rawScores: number[][] = rawSiteMatrix;
```

##### rawLoadings

```ts
rawLoadings: number[][] = rawLoadingMatrix;
```

##### siteFactors

```ts
siteFactors: any = scaled.siteFactors;
```

##### loadingFactors

```ts
loadingFactors: any = scaled.loadingFactors;
```

##### scaling

```ts
scaling: number = appliedScaling;
```

##### axisSigns

```ts
axisSigns: number[];
```

##### exponent

```ts
exponent: any = scaled.exponent;
```

##### discriminantAxes

```ts
discriminantAxes: number[][];
```

##### sampleClasses

```ts
sampleClasses: any;
```

##### classMeans

```ts
classMeans: any[][] = classMeansOriginal;
```

##### classes

```ts
classes: any[];
```

##### overallMean

```ts
overallMean: number[];
```

##### projector

```ts
projector: number[][];
```

##### invScales

```ts
invScales: any[];
```

##### eigenvectors

```ts
eigenvectors: number[][];
```

##### classMeanScores

```ts
classMeanScores: any[][];
```

##### classStdScores

```ts
classStdScores: any[][];
```

##### featureNames

```ts
featureNames: any = variableNames;
```

##### labelEncoder

```ts
labelEncoder: any;
```

***

### transform()

```ts
function transform(model, X): Object[];
```

Defined in: [src/mva/lda.js:365](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/lda.js#L365)

#### Parameters

##### model

`any`

##### X

`any`

#### Returns

`Object`[]

***

### predict()

```ts
function predict(model, X): any[];
```

Defined in: [src/mva/lda.js:402](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/lda.js#L402)

#### Parameters

##### model

`any`

##### X

`any`

#### Returns

`any`[]
