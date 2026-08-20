---
layout: default
title: preprocessing
parent: Machine Learning
grand_parent: API Reference
permalink: /api/machine-learning/preprocessing
---
# preprocessing

## Classes

### StandardScaler

Defined in: [src/ml/preprocessing.js:43](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L43)

Standardize features by removing mean and scaling to unit variance

#### Constructors

##### Constructor

```ts
new StandardScaler(): StandardScaler;
```

Defined in: [src/ml/preprocessing.js:44](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L44)

###### Returns

[`StandardScaler`](#standardscaler)

#### Properties

##### means

```ts
means: any[] | null;
```

Defined in: [src/ml/preprocessing.js:45](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L45)

##### stds

```ts
stds: any[] | null;
```

Defined in: [src/ml/preprocessing.js:46](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L46)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/preprocessing.js:47](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L47)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:48](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L48)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:49](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L49)

#### Methods

##### fit()

```ts
fit(X): StandardScaler;
```

Defined in: [src/ml/preprocessing.js:57](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L57)

Compute mean and standard deviation

###### Parameters

###### X

`Object` \| `number`[][]

Feature matrix or {data, columns}

###### Returns

[`StandardScaler`](#standardscaler)

this

##### transform()

```ts
transform(X): Object | number[][];
```

Defined in: [src/ml/preprocessing.js:88](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L88)

Standardize features

###### Parameters

###### X

`Object` \| `number`[][]

Feature matrix or {data, columns}

###### Returns

`Object` \| `number`[][]

Scaled features or {data, columns, X}

##### fitTransform()

```ts
fitTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:147](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L147)

Fit and transform in one step

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

`number`[][]

Scaled features

##### inverseTransform()

```ts
inverseTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:156](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L156)

Inverse transform (unscale)

###### Parameters

###### X

`number`[][]

Scaled features

###### Returns

`number`[][]

Original scale features

***

### MinMaxScaler

Defined in: [src/ml/preprocessing.js:175](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L175)

Scale features to a given range [min, max]

#### Constructors

##### Constructor

```ts
new MinMaxScaler(__namedParameters?): MinMaxScaler;
```

Defined in: [src/ml/preprocessing.js:176](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L176)

###### Parameters

###### \_\_namedParameters?

###### featureRange?

`number`[] = `...`

###### Returns

[`MinMaxScaler`](#minmaxscaler)

#### Properties

##### featureRange

```ts
featureRange: number[];
```

Defined in: [src/ml/preprocessing.js:177](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L177)

##### dataMin

```ts
dataMin: any[] | null;
```

Defined in: [src/ml/preprocessing.js:178](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L178)

##### dataMax

```ts
dataMax: any[] | null;
```

Defined in: [src/ml/preprocessing.js:179](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L179)

##### nFeatures

```ts
nFeatures: number | null;
```

Defined in: [src/ml/preprocessing.js:180](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L180)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:181](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L181)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:182](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L182)

#### Methods

##### fit()

```ts
fit(X): MinMaxScaler;
```

Defined in: [src/ml/preprocessing.js:190](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L190)

Compute min and max for scaling

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

[`MinMaxScaler`](#minmaxscaler)

this

##### transform()

```ts
transform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:222](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L222)

Scale features to range

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

`number`[][]

Scaled features

##### fitTransform()

```ts
fitTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:282](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L282)

Fit and transform in one step

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

`number`[][]

Scaled features

##### inverseTransform()

```ts
inverseTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:291](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L291)

Inverse transform

###### Parameters

###### X

`number`[][]

Scaled features

###### Returns

`number`[][]

Original scale features

***

### LabelEncoder

Defined in: [src/ml/preprocessing.js:319](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L319)

Encode target labels with value between 0 and n_classes-1

Note: distinct from the internal core/table.js LabelEncoder used by
prepareX/prepareXY (which keeps first-seen class order and is persisted
with fitted models). This one sorts classes and supports table
descriptors; the two are intentionally separate.

#### Constructors

##### Constructor

```ts
new LabelEncoder(): LabelEncoder;
```

Defined in: [src/ml/preprocessing.js:320](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L320)

###### Returns

[`LabelEncoder`](#labelencoder)

#### Properties

##### classes

```ts
classes: any[] | null;
```

Defined in: [src/ml/preprocessing.js:321](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L321)

##### classMap

```ts
classMap: Map<any, number> | null;
```

Defined in: [src/ml/preprocessing.js:322](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L322)

##### \_tableColumn

```ts
_tableColumn: any;
```

Defined in: [src/ml/preprocessing.js:323](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L323)

#### Methods

##### \_extractLabelVector()

```ts
_extractLabelVector(input, __namedParameters?): 
  | {
  values: any[];
  rows: null;
  column: null;
}
  | {
  values: any[];
  rows: Object[];
  column: any;
};
```

Defined in: [src/ml/preprocessing.js:326](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L326)

###### Parameters

###### input

`any`

###### \_\_namedParameters?

###### fallbackColumn?

`null` = `null`

###### forTransform?

`boolean` = `false`

###### Returns

  \| \{
  `values`: `any`[];
  `rows`: `null`;
  `column`: `null`;
\}
  \| \{
  `values`: `any`[];
  `rows`: `Object`[];
  `column`: `any`;
\}

##### fit()

```ts
fit(y): LabelEncoder;
```

Defined in: [src/ml/preprocessing.js:353](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L353)

Fit label encoder

###### Parameters

###### y

`any`[]

Target labels

###### Returns

[`LabelEncoder`](#labelencoder)

this

##### transform()

```ts
transform(y): number[];
```

Defined in: [src/ml/preprocessing.js:366](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L366)

Transform labels to indices

###### Parameters

###### y

`any`[]

Target labels

###### Returns

`number`[]

Encoded labels

##### fitTransform()

```ts
fitTransform(y): number[];
```

Defined in: [src/ml/preprocessing.js:404](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L404)

Fit and transform in one step

###### Parameters

###### y

`any`[]

Target labels

###### Returns

`number`[]

Encoded labels

##### inverseTransform()

```ts
inverseTransform(y): any[];
```

Defined in: [src/ml/preprocessing.js:413](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L413)

Transform indices back to original labels

###### Parameters

###### y

`number`[]

Encoded labels

###### Returns

`any`[]

Original labels

***

### OneHotEncoder

Defined in: [src/ml/preprocessing.js:432](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L432)

Encode categorical features as one-hot numeric array

#### Constructors

##### Constructor

```ts
new OneHotEncoder(): OneHotEncoder;
```

Defined in: [src/ml/preprocessing.js:433](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L433)

###### Returns

[`OneHotEncoder`](#onehotencoder)

#### Properties

##### categories

```ts
categories: any[] | null;
```

Defined in: [src/ml/preprocessing.js:434](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L434)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/preprocessing.js:435](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L435)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:436](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L436)

#### Methods

##### \_prepareInput()

```ts
_prepareInput(X, __namedParameters?): 
  | {
  matrix: any[];
  rows: null;
  columns: null;
}
  | {
  matrix: any[][];
  rows: Object[];
  columns: any[];
};
```

Defined in: [src/ml/preprocessing.js:439](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L439)

###### Parameters

###### X

`any`

###### \_\_namedParameters?

###### fallbackColumns?

`null` = `null`

###### requireColumnsMessage?

`string` = `'OneHotEncoder: columns are required when using table data'`

###### Returns

  \| \{
  `matrix`: `any`[];
  `rows`: `null`;
  `columns`: `null`;
\}
  \| \{
  `matrix`: `any`[][];
  `rows`: `Object`[];
  `columns`: `any`[];
\}

##### fit()

```ts
fit(X): OneHotEncoder;
```

Defined in: [src/ml/preprocessing.js:473](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L473)

Fit encoder by discovering categories

###### Parameters

###### X

`Object` \| `any`[][]

Categorical features matrix or {data, columns}

###### Returns

[`OneHotEncoder`](#onehotencoder)

this

##### transform()

```ts
transform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:497](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L497)

Transform to one-hot encoding

###### Parameters

###### X

`any`[][]

Categorical features

###### Returns

`number`[][]

One-hot encoded features

##### fitTransform()

```ts
fitTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:570](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L570)

Fit and transform in one step

###### Parameters

###### X

`any`[][]

Categorical features

###### Returns

`number`[][]

One-hot encoded features

##### getFeatureNames()

```ts
getFeatureNames(): string[];
```

Defined in: [src/ml/preprocessing.js:578](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L578)

Get feature names after one-hot encoding

###### Returns

`string`[]

Feature names

***

### PolynomialFeatures

Defined in: [src/ml/preprocessing.js:598](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L598)

Generate polynomial and interaction features

#### Constructors

##### Constructor

```ts
new PolynomialFeatures(__namedParameters?): PolynomialFeatures;
```

Defined in: [src/ml/preprocessing.js:599](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L599)

###### Parameters

###### \_\_namedParameters?

###### degree?

`number` = `2`

###### includeBias?

`boolean` = `false`

###### Returns

[`PolynomialFeatures`](#polynomialfeatures)

#### Properties

##### degree

```ts
degree: number;
```

Defined in: [src/ml/preprocessing.js:600](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L600)

##### includeBias

```ts
includeBias: boolean;
```

Defined in: [src/ml/preprocessing.js:601](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L601)

##### nInputFeatures

```ts
nInputFeatures: any;
```

Defined in: [src/ml/preprocessing.js:602](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L602)

##### nOutputFeatures

```ts
nOutputFeatures: number | null;
```

Defined in: [src/ml/preprocessing.js:603](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L603)

##### \_featurePatterns

```ts
_featurePatterns: any[];
```

Defined in: [src/ml/preprocessing.js:604](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L604)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:605](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L605)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:606](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L606)

#### Methods

##### \_normalizeInput()

```ts
_normalizeInput(X, __namedParameters?): 
  | {
  matrix: any[][];
  tableInput: {
     prepared: {
        X: any[][];
        columns: any[];
        n: number;
        rows: any[];
        validIndices: any[];
        sourceLength: number;
        encoders: {
        };
     };
     naOmit: boolean;
  };
}
  | {
  matrix: any[];
  tableInput: null;
};
```

Defined in: [src/ml/preprocessing.js:609](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L609)

###### Parameters

###### X

`any`

###### \_\_namedParameters?

###### fallbackColumns?

`null` = `null`

###### fallbackNaOmit?

`boolean` = `true`

###### requireColumnsMessage?

`string` = `'PolynomialFeatures: columns are required when using table data'`

###### Returns

  \| \{
  `matrix`: `any`[][];
  `tableInput`: \{
     `prepared`: \{
        `X`: `any`[][];
        `columns`: `any`[];
        `n`: `number`;
        `rows`: `any`[];
        `validIndices`: `any`[];
        `sourceLength`: `number`;
        `encoders`: \{
        \};
     \};
     `naOmit`: `boolean`;
  \};
\}
  \| \{
  `matrix`: `any`[];
  `tableInput`: `null`;
\}

##### \_buildFeaturePatterns()

```ts
_buildFeaturePatterns(): void;
```

Defined in: [src/ml/preprocessing.js:634](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L634)

###### Returns

`void`

##### \_appendDegreePatterns()

```ts
_appendDegreePatterns(degree): void;
```

Defined in: [src/ml/preprocessing.js:651](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L651)

###### Parameters

###### degree

`any`

###### Returns

`void`

##### \_evaluatePattern()

```ts
_evaluatePattern(pattern, row): any;
```

Defined in: [src/ml/preprocessing.js:670](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L670)

###### Parameters

###### pattern

`any`

###### row

`any`

###### Returns

`any`

##### \_buildFeatureNames()

```ts
_buildFeatureNames(columns?): string[];
```

Defined in: [src/ml/preprocessing.js:678](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L678)

###### Parameters

###### columns?

`null` = `null`

###### Returns

`string`[]

##### fit()

```ts
fit(X): PolynomialFeatures;
```

Defined in: [src/ml/preprocessing.js:705](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L705)

Fit by determining input/output dimensions

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

[`PolynomialFeatures`](#polynomialfeatures)

this

##### transform()

```ts
transform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:731](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L731)

Transform to polynomial features

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

`number`[][]

Polynomial features

##### fitTransform()

```ts
fitTransform(X): number[][];
```

Defined in: [src/ml/preprocessing.js:779](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L779)

Fit and transform in one step

###### Parameters

###### X

`number`[][]

Feature matrix

###### Returns

`number`[][]

Polynomial features

## Variables

### preprocessCategorical

```ts
const preprocessCategorical: (options) => Object = preprocess;
```

Defined in: [src/ml/preprocessing.js:1141](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L1141)

Declarative preprocessing pipeline for tabular data
Handles numeric parsing, data validation, and categorical encoding in one step

#### Parameters

##### options

###### data

`Object` \| `any`[]

Input data (array of objects or Arquero table)

###### parseNumeric?

`string`[] = `null`

Column names to convert from string to numeric

###### validCategories?

`Object` = `null`

Validation rules for categorical columns (removes invalid rows)

###### labelEncode?

`Object`[] = `[]`

Columns to label encode: [{ column, outputColumn?, categories? }]

###### oneHotEncode?

`Object`[] = `[]`

Columns to one-hot encode: [{ columns, dropFirst?, prefix? }]

###### verbose?

`boolean` = `true`

Print preprocessing info

#### Returns

`Object`

{ data, info: { parsed, cleaned, labelEncoders, oneHotInfo } }

## Functions

### parseNumeric()

```ts
function parseNumeric(options): Object[];
```

Defined in: [src/ml/preprocessing.js:795](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L795)

Convert string columns to numeric
Useful when CSV parsers incorrectly infer column types

#### Parameters

##### options

###### data

`Object` \| `any`[]

Input data

###### columns

`string`[] = `[]`

Column names to convert

#### Returns

`Object`[]

Data with converted columns

***

### cleanCategorical()

```ts
function cleanCategorical(options): Object;
```

Defined in: [src/ml/preprocessing.js:822](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L822)

Clean and validate categorical columns
Removes rows with invalid categories

#### Parameters

##### options

###### data

`Object` \| `any`[]

Input data

###### validCategories

`Object` = `{}`

Map of column names to arrays of valid values

#### Returns

`Object`

{ data: cleaned data, removed: count of removed rows }

***

### labelEncode()

```ts
function labelEncode(options): Object;
```

Defined in: [src/ml/preprocessing.js:852](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L852)

Label encode a categorical column
Maps categories to integers (0, 1, 2, ...)

#### Parameters

##### options

###### data

`Object` \| `any`[]

Input data

###### column

`string`

Column to encode

###### outputColumn?

`string`

Name for encoded column (default: column + '_idx')

###### keepOriginal?

`boolean` = `true`

Keep original column

###### categories?

`any`[] = `null`

Predefined category order (optional)

#### Returns

`Object`

{ data, encoder, outputColumn }

***

### preprocess()

```ts
function preprocess(options): Object;
```

Defined in: [src/ml/preprocessing.js:903](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L903)

Declarative preprocessing pipeline for tabular data
Handles numeric parsing, data validation, and categorical encoding in one step

#### Parameters

##### options

###### data

`Object` \| `any`[]

Input data (array of objects or Arquero table)

###### parseNumeric?

`string`[] = `null`

Column names to convert from string to numeric

###### validCategories?

`Object` = `null`

Validation rules for categorical columns (removes invalid rows)

###### labelEncode?

`Object`[] = `[]`

Columns to label encode: [{ column, outputColumn?, categories? }]

###### oneHotEncode?

`Object`[] = `[]`

Columns to one-hot encode: [{ columns, dropFirst?, prefix? }]

###### verbose?

`boolean` = `true`

Print preprocessing info

#### Returns

`Object`

{ data, info: { parsed, cleaned, labelEncoders, oneHotInfo } }

***

### fitPreprocessor()

```ts
function fitPreprocessor(options): Object;
```

Defined in: [src/ml/preprocessing.js:1009](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L1009)

Fit a preprocessing pipeline and store the transformers
Use this on training data, then apply the same transformers to test data

#### Parameters

##### options

`Object`

Same as preprocessCategorical

#### Returns

`Object`

{ data, pipeline: reusable pipeline object }

***

### transformWithPipeline()

```ts
function transformWithPipeline(options): Object;
```

Defined in: [src/ml/preprocessing.js:1042](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/ml/preprocessing.js#L1042)

Transform new data using a fitted preprocessing pipeline

#### Parameters

##### options

###### data

`Object` \| `any`[]

New data to transform

###### pipeline

`Object`

Pipeline from fitPreprocessor

###### verbose?

`boolean` = `false`

Print info

#### Returns

`Object`

{ data }

## References

### trainTestSplit

Re-exports [trainTestSplit](/api/machine-learning/validation#traintestsplit)
