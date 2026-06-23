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

Defined in: [src/ml/preprocessing.js:43](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L43)

Standardize features by removing mean and scaling to unit variance

#### Constructors

##### Constructor

```ts
new StandardScaler(): StandardScaler;
```

Defined in: [src/ml/preprocessing.js:44](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L44)

###### Returns

[`StandardScaler`](#standardscaler)

#### Properties

##### means

```ts
means: any[] | null;
```

Defined in: [src/ml/preprocessing.js:45](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L45)

##### stds

```ts
stds: any[] | null;
```

Defined in: [src/ml/preprocessing.js:46](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L46)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/preprocessing.js:47](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L47)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:48](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L48)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:49](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L49)

#### Methods

##### fit()

```ts
fit(X): StandardScaler;
```

Defined in: [src/ml/preprocessing.js:57](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L57)

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

Defined in: [src/ml/preprocessing.js:103](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L103)

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

Defined in: [src/ml/preprocessing.js:164](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L164)

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

Defined in: [src/ml/preprocessing.js:173](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L173)

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

Defined in: [src/ml/preprocessing.js:192](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L192)

Scale features to a given range [min, max]

#### Constructors

##### Constructor

```ts
new MinMaxScaler(__namedParameters?): MinMaxScaler;
```

Defined in: [src/ml/preprocessing.js:193](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L193)

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

Defined in: [src/ml/preprocessing.js:194](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L194)

##### dataMin

```ts
dataMin: any[] | null;
```

Defined in: [src/ml/preprocessing.js:195](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L195)

##### dataMax

```ts
dataMax: any[] | null;
```

Defined in: [src/ml/preprocessing.js:196](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L196)

##### nFeatures

```ts
nFeatures: number | null;
```

Defined in: [src/ml/preprocessing.js:197](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L197)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:198](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L198)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:199](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L199)

#### Methods

##### fit()

```ts
fit(X): MinMaxScaler;
```

Defined in: [src/ml/preprocessing.js:207](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L207)

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

Defined in: [src/ml/preprocessing.js:239](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L239)

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

Defined in: [src/ml/preprocessing.js:299](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L299)

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

Defined in: [src/ml/preprocessing.js:308](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L308)

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

Defined in: [src/ml/preprocessing.js:336](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L336)

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

Defined in: [src/ml/preprocessing.js:337](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L337)

###### Returns

[`LabelEncoder`](#labelencoder)

#### Properties

##### classes

```ts
classes: any[] | null;
```

Defined in: [src/ml/preprocessing.js:338](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L338)

##### classMap

```ts
classMap: Map<any, number> | null;
```

Defined in: [src/ml/preprocessing.js:339](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L339)

##### \_tableColumn

```ts
_tableColumn: any;
```

Defined in: [src/ml/preprocessing.js:340](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L340)

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

Defined in: [src/ml/preprocessing.js:343](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L343)

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

Defined in: [src/ml/preprocessing.js:370](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L370)

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

Defined in: [src/ml/preprocessing.js:383](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L383)

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

Defined in: [src/ml/preprocessing.js:421](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L421)

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

Defined in: [src/ml/preprocessing.js:430](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L430)

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

Defined in: [src/ml/preprocessing.js:449](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L449)

Encode categorical features as one-hot numeric array

#### Constructors

##### Constructor

```ts
new OneHotEncoder(): OneHotEncoder;
```

Defined in: [src/ml/preprocessing.js:450](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L450)

###### Returns

[`OneHotEncoder`](#onehotencoder)

#### Properties

##### categories

```ts
categories: any[] | null;
```

Defined in: [src/ml/preprocessing.js:451](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L451)

##### nFeatures

```ts
nFeatures: any;
```

Defined in: [src/ml/preprocessing.js:452](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L452)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:453](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L453)

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

Defined in: [src/ml/preprocessing.js:456](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L456)

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

Defined in: [src/ml/preprocessing.js:490](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L490)

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

Defined in: [src/ml/preprocessing.js:514](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L514)

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

Defined in: [src/ml/preprocessing.js:587](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L587)

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

Defined in: [src/ml/preprocessing.js:595](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L595)

Get feature names after one-hot encoding

###### Returns

`string`[]

Feature names

***

### PolynomialFeatures

Defined in: [src/ml/preprocessing.js:615](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L615)

Generate polynomial and interaction features

#### Constructors

##### Constructor

```ts
new PolynomialFeatures(__namedParameters?): PolynomialFeatures;
```

Defined in: [src/ml/preprocessing.js:616](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L616)

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

Defined in: [src/ml/preprocessing.js:617](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L617)

##### includeBias

```ts
includeBias: boolean;
```

Defined in: [src/ml/preprocessing.js:618](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L618)

##### nInputFeatures

```ts
nInputFeatures: any;
```

Defined in: [src/ml/preprocessing.js:619](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L619)

##### nOutputFeatures

```ts
nOutputFeatures: number | null;
```

Defined in: [src/ml/preprocessing.js:620](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L620)

##### \_featurePatterns

```ts
_featurePatterns: any[];
```

Defined in: [src/ml/preprocessing.js:621](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L621)

##### \_tableColumns

```ts
_tableColumns: any[] | null;
```

Defined in: [src/ml/preprocessing.js:622](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L622)

##### \_tableNaOmit

```ts
_tableNaOmit: boolean;
```

Defined in: [src/ml/preprocessing.js:623](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L623)

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

Defined in: [src/ml/preprocessing.js:626](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L626)

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

Defined in: [src/ml/preprocessing.js:651](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L651)

###### Returns

`void`

##### \_appendDegreePatterns()

```ts
_appendDegreePatterns(degree): void;
```

Defined in: [src/ml/preprocessing.js:668](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L668)

###### Parameters

###### degree

`any`

###### Returns

`void`

##### \_evaluatePattern()

```ts
_evaluatePattern(pattern, row): any;
```

Defined in: [src/ml/preprocessing.js:687](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L687)

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

Defined in: [src/ml/preprocessing.js:695](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L695)

###### Parameters

###### columns?

`null` = `null`

###### Returns

`string`[]

##### fit()

```ts
fit(X): PolynomialFeatures;
```

Defined in: [src/ml/preprocessing.js:722](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L722)

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

Defined in: [src/ml/preprocessing.js:748](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L748)

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

Defined in: [src/ml/preprocessing.js:796](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L796)

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

Defined in: [src/ml/preprocessing.js:1158](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L1158)

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

Defined in: [src/ml/preprocessing.js:812](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L812)

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

Defined in: [src/ml/preprocessing.js:839](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L839)

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

Defined in: [src/ml/preprocessing.js:869](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L869)

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

Defined in: [src/ml/preprocessing.js:920](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L920)

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

Defined in: [src/ml/preprocessing.js:1026](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L1026)

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

Defined in: [src/ml/preprocessing.js:1059](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/ml/preprocessing.js#L1059)

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
