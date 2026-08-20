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
   options?): Object;
```

Defined in: [src/mva/lda.js:34](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/lda.js#L34)

Fit LDA model

#### Parameters

##### X

`Object` \| `number`[][]

Design matrix (n × p), or a declarative config object with X/y/data

##### y

(`string` \| `number`)[]

Class label vector

##### options?

Fitting options

###### scale?

`boolean`

Scale features to unit variance (default false)

###### scaling?

`number`

Ordination scaling, 1 or 2 (default 2)

###### omit_missing?

`boolean`

Omit rows with missing values (alias of naOmit)

###### naOmit?

`boolean`

Omit rows with missing values (default true)

###### encoders?

`Object`

Label encoders for declarative input

#### Returns

`Object`

Fitted LDA model

***

### transform()

```ts
function transform(model, X): Object[];
```

Defined in: [src/mva/lda.js:402](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/lda.js#L402)

Project new data onto the fitted discriminant axes

#### Parameters

##### model

`Object`

Fitted LDA model

##### X

`number`[][]

New data matrix (n × p)

#### Returns

`Object`[]

Discriminant score objects, one per row

***

### predict()

```ts
function predict(model, X): (string | number)[];
```

Defined in: [src/mva/lda.js:451](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/mva/lda.js#L451)

Predict class labels for new data using nearest class-mean in discriminant space

#### Parameters

##### model

`Object`

Fitted LDA model

##### X

`Object` \| `number`[][]

New data matrix (n × p), or a declarative config object with data/X

#### Returns

(`string` \| `number`)[]

Predicted class label for each row
