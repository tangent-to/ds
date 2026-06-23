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
   options?): object;
```

Defined in: [src/mva/cca.js:19](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/cca.js#L19)

Fit CCA model.

Accepts either numeric matrices (fit(XMatrix, YMatrix, options)) or a declarative
object: fit({ X: ['col1', ...], Y: ['colA', ...], data, omit_missing, center, scale }).

#### Parameters

##### X

`any`

##### Y?

`null` = `null`

##### options?

#### Returns

`object`

##### type

```ts
type: string = 'cca';
```

##### nSamples

```ts
nSamples: any = n;
```

##### nFeaturesX

```ts
nFeaturesX: any = p;
```

##### nFeaturesY

```ts
nFeaturesY: any = q;
```

##### nComponents

```ts
nComponents: number = components;
```

##### correlations

```ts
correlations: any;
```

##### xWeights

```ts
xWeights: object[];
```

##### yWeights

```ts
yWeights: object[];
```

##### xScores

```ts
xScores: object[];
```

##### yScores

```ts
yScores: object[];
```

##### xMeans

```ts
xMeans: any[] = processedX.means;
```

##### xSds

```ts
xSds: any[] = processedX.sds;
```

##### yMeans

```ts
yMeans: any[] = processedY.means;
```

##### ySds

```ts
ySds: any[] = processedY.sds;
```

##### center

```ts
center: boolean;
```

##### scale

```ts
scale: boolean;
```

##### columnsX

```ts
columnsX: string[] = columnNamesX;
```

##### columnsY

```ts
columnsY: string[] = columnNamesY;
```

***

### transformX()

```ts
function transformX(
   model, 
   X, 
   options?): object[];
```

Defined in: [src/mva/cca.js:195](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/cca.js#L195)

#### Parameters

##### model

`any`

##### X

`any`

##### options?

#### Returns

`object`[]

***

### transformY()

```ts
function transformY(
   model, 
   Y, 
   options?): object[];
```

Defined in: [src/mva/cca.js:212](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/cca.js#L212)

#### Parameters

##### model

`any`

##### Y

`any`

##### options?

#### Returns

`object`[]

***

### transform()

```ts
function transform(
   model, 
   X, 
   Y, 
   options?): object;
```

Defined in: [src/mva/cca.js:229](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/mva/cca.js#L229)

#### Parameters

##### model

`any`

##### X

`any`

##### Y

`any`

##### options?

#### Returns

`object`

##### xScores

```ts
xScores: object[];
```

##### yScores

```ts
yScores: object[];
```
