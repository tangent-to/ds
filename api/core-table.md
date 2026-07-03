---
layout: default
title: table
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/table
---
# table

## Classes

### LabelEncoder

Defined in: [src/core/table.js:226](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L226)

Simple Label Encoder for categorical labels -> integers

Note: this is the internal encoder used by prepareX/prepareXY and
persisted with fitted models (classes keep first-seen order). The
user-facing ml.preprocessing.LabelEncoder is a separate class with
different semantics (sorted classes, table-descriptor support).

handleUnknown controls what transform() does with categories not seen
during fit: 'error' (default) throws, 'ignore' maps them to NaN, and
'extend' registers them as new classes (legacy behaviour; unsafe at
predict time because indices grow past the fitted class set).

#### Constructors

##### Constructor

```ts
new LabelEncoder(__namedParameters?): LabelEncoder;
```

Defined in: [src/core/table.js:227](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L227)

###### Parameters

###### \_\_namedParameters?

###### handleUnknown?

`string` = `'error'`

###### Returns

[`LabelEncoder`](#labelencoder)

#### Properties

##### classes\_

```ts
classes_: any[];
```

Defined in: [src/core/table.js:228](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L228)

##### classIndex

```ts
classIndex: Map<any, any>;
```

Defined in: [src/core/table.js:229](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L229)

##### handleUnknown

```ts
handleUnknown: string;
```

Defined in: [src/core/table.js:230](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L230)

#### Methods

##### fit()

```ts
fit(values?): LabelEncoder;
```

Defined in: [src/core/table.js:233](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L233)

###### Parameters

###### values?

`any`[] = `[]`

###### Returns

[`LabelEncoder`](#labelencoder)

##### transform()

```ts
transform(values?): any[];
```

Defined in: [src/core/table.js:246](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L246)

###### Parameters

###### values?

`any`[] = `[]`

###### Returns

`any`[]

##### fitTransform()

```ts
fitTransform(values?): any[];
```

Defined in: [src/core/table.js:266](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L266)

###### Parameters

###### values?

`any`[] = `[]`

###### Returns

`any`[]

##### inverseTransform()

```ts
inverseTransform(indices?): any[];
```

Defined in: [src/core/table.js:271](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L271)

###### Parameters

###### indices?

`any`[] = `[]`

###### Returns

`any`[]

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/table.js:275](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L275)

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'LabelEncoder';
```

###### classes

```ts
classes: any[];
```

###### handleUnknown

```ts
handleUnknown: string;
```

##### fromJSON()

```ts
static fromJSON(obj?): LabelEncoder;
```

Defined in: [src/core/table.js:283](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L283)

###### Parameters

###### obj?

###### Returns

[`LabelEncoder`](#labelencoder)

***

### OneHotEncoder

Defined in: [src/core/table.js:299](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L299)

Simple OneHotEncoder for a single categorical column
Note: this encoder returns an array of arrays (one-hot vectors)

#### Constructors

##### Constructor

```ts
new OneHotEncoder(__namedParameters?): OneHotEncoder;
```

Defined in: [src/core/table.js:300](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L300)

###### Parameters

###### \_\_namedParameters?

###### handleUnknown?

`string` = `'ignore'`

###### Returns

[`OneHotEncoder`](#onehotencoder)

#### Properties

##### categories\_

```ts
categories_: any[];
```

Defined in: [src/core/table.js:301](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L301)

##### catIndex

```ts
catIndex: Map<any, any>;
```

Defined in: [src/core/table.js:302](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L302)

##### handleUnknown

```ts
handleUnknown: string;
```

Defined in: [src/core/table.js:303](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L303)

##### \_columnConfigs

```ts
_columnConfigs: Map<any, any> | null;
```

Defined in: [src/core/table.js:304](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L304)

##### \_columns

```ts
_columns: any[] | undefined;
```

Defined in: [src/core/table.js:368](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L368)

##### \_encoders

```ts
_encoders: Map<any, any> | undefined;
```

Defined in: [src/core/table.js:369](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L369)

#### Methods

##### fit()

```ts
fit(values?): OneHotEncoder;
```

Defined in: [src/core/table.js:307](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L307)

###### Parameters

###### values?

`any`[] = `[]`

###### Returns

[`OneHotEncoder`](#onehotencoder)

##### transform()

```ts
transform(values?): any[][];
```

Defined in: [src/core/table.js:320](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L320)

###### Parameters

###### values?

`any`[] = `[]`

###### Returns

`any`[][]

##### fitTransform()

```ts
fitTransform(valuesOrOptions?): Object[];
```

Defined in: [src/core/table.js:341](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L341)

###### Parameters

###### valuesOrOptions?

`any`[] = `[]`

###### Returns

`Object`[]

##### \_fitDeclarative()

```ts
_fitDeclarative(options): OneHotEncoder;
```

Defined in: [src/core/table.js:359](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L359)

Declarative API for fit

###### Parameters

###### options

`Object`

{ data, columns }

###### Returns

[`OneHotEncoder`](#onehotencoder)

this

##### \_transformDeclarative()

```ts
_transformDeclarative(options): Object[];
```

Defined in: [src/core/table.js:391](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L391)

Declarative API for transform

###### Parameters

###### options

`Object`

{ data }

###### Returns

`Object`[]

Array of objects with one-hot encoded columns

##### \_fitTransformDeclarative()

```ts
_fitTransformDeclarative(options): Object[];
```

Defined in: [src/core/table.js:424](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L424)

Declarative API for fitTransform

###### Parameters

###### options

`Object`

{ data, columns }

###### Returns

`Object`[]

Array of objects with one-hot encoded columns

##### getFeatureNames()

```ts
getFeatureNames(prefix?): string[];
```

Defined in: [src/core/table.js:433](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L433)

Get all feature names for declarative API

###### Parameters

###### prefix?

`string` = `''`

###### Returns

`string`[]

All feature names across all columns

##### toJSON()

```ts
toJSON(): object;
```

Defined in: [src/core/table.js:450](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L450)

###### Returns

`object`

###### \_\_class\_\_

```ts
__class__: string = 'OneHotEncoder';
```

###### categories

```ts
categories: any[];
```

##### fromJSON()

```ts
static fromJSON(obj?): OneHotEncoder;
```

Defined in: [src/core/table.js:454](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L454)

###### Parameters

###### obj?

###### Returns

[`OneHotEncoder`](#onehotencoder)

## Functions

### normalizeNaOmit()

```ts
function normalizeNaOmit(options?): boolean;
```

Defined in: [src/core/table.js:14](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L14)

Normalize naOmit/omit_missing parameter names
Accepts both R-style (naOmit) and Python-style (omit_missing)

#### Parameters

##### options?

`Object` = `{}`

Options object that may contain naOmit or omit_missing

#### Returns

`boolean`

The normalized boolean value (defaults to true)

***

### normalize()

```ts
function normalize(data): Object[];
```

Defined in: [src/core/table.js:39](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L39)

Normalize input to array of objects

#### Parameters

##### data

`Object` \| `Object`[]

Input data

#### Returns

`Object`[]

Array of row objects

***

### toMatrix()

```ts
function toMatrix(data, columns): Matrix;
```

Defined in: [src/core/table.js:55](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L55)

Convert table data to matrix

#### Parameters

##### data

`Object` \| `Object`[]

Input data

##### columns

`string`[]

Column names to extract

#### Returns

[`Matrix`](/api/core/linalg#matrix)

Matrix with selected columns

***

### toVector()

```ts
function toVector(data, column): number[];
```

Defined in: [src/core/table.js:87](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L87)

Convert table column to vector

#### Parameters

##### data

`Object` \| `Object`[]

Input data

##### column

`string`

Column name

#### Returns

`number`[]

1D array

***

### toColumns()

```ts
function toColumns(data, columns): Object;
```

Defined in: [src/core/table.js:109](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L109)

Extract multiple columns as arrays

#### Parameters

##### data

`Object` \| `Object`[]

Input data

##### columns

`string`[]

Column names

#### Returns

`Object`

Object with column names as keys and arrays as values

***

### getColumns()

```ts
function getColumns(data): string[];
```

Defined in: [src/core/table.js:125](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L125)

Get column names from data

#### Parameters

##### data

`Object` \| `Object`[]

Input data

#### Returns

`string`[]

Column names

***

### filter()

```ts
function filter(data, predicate): Object[];
```

Defined in: [src/core/table.js:139](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L139)

Filter rows based on predicate

#### Parameters

##### data

`Object` \| `Object`[]

Input data

##### predicate

`Function`

Filter function

#### Returns

`Object`[]

Filtered rows

***

### select()

```ts
function select(data, columns): Object[];
```

Defined in: [src/core/table.js:149](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L149)

Select specific columns

#### Parameters

##### data

`Object` \| `Object`[]

Input data

##### columns

`string`[]

Columns to select

#### Returns

`Object`[]

Rows with selected columns

***

### applyColumns()

```ts
function applyColumns(
   rows, 
   columns, 
   matrix, 
   options?): Object[];
```

Defined in: [src/core/table.js:170](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L170)

Apply a matrix of values to specific columns on each row.
Useful for re-attaching transformed feature matrices to table rows.

#### Parameters

##### rows

`Object`[]

Source rows (will be copied unless copy=false)

##### columns

`string`[]

Column names corresponding to matrix columns

##### matrix

`number`[][]

Values to assign per row/column

##### options?

`Object` = `{}`

{ copy: true } to control cloning behaviour

#### Returns

`Object`[]

Rows with columns assigned

***

### prepareX()

```ts
function prepareX(__namedParameters?): object;
```

Defined in: [src/core/table.js:478](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L478)

Prepare feature matrix X from table-like data.
Supports optional categorical encoding:
  prepareX({ columns, data, naOmit = true, encode = null, encoders = null })

encode can be:
  - null/false: no encoding (default)
  - true: auto-label encode any non-numeric columns
  - { colName: 'label' | 'onehot' } mapping per-column
encoders allows you to reuse previously fitted encoders per column.

Returns:
  { X, columns, n, rows, encoders } where encoders is a mapping of column->encoder used

#### Parameters

##### \_\_namedParameters?

###### encode?

`null` = `null`

###### encoders?

`null` = `null`

#### Returns

`object`

##### X

```ts
X: any[][];
```

##### columns

```ts
columns: any[] = finalColumnNames;
```

##### n

```ts
n: number = X.length;
```

##### rows

```ts
rows: any[] = preFiltered;
```

##### validIndices

```ts
validIndices: any[];
```

##### sourceLength

```ts
sourceLength: number = rows.length;
```

##### encoders

```ts
encoders: object = resolvedEncoders;
```

***

### attachSourceRows()

```ts
function attachSourceRows(model, prepared): Object;
```

Defined in: [src/core/table.js:646](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L646)

Attach naOmit alignment metadata to a fitted model as non-enumerable
properties, so plot helpers can realign external per-row values (colorBy,
labels) to the rows that survived missing-value filtering.

#### Parameters

##### model

`Object`

Fitted model to annotate

##### prepared

`Object`

A prepareX() result ({ rows, validIndices, sourceLength })

#### Returns

`Object`

The same model

***

### oneHotEncodeTable()

```ts
function oneHotEncodeTable(options?): Object;
```

Defined in: [src/core/table.js:676](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L676)

Utility to one-hot encode columns in a table-like object.

#### Parameters

##### options?

###### data

`Object` \| `any`[]

Array of row objects or table-like input

###### columns

`string` \| `string`[]

Column or columns to encode

###### dropFirst?

`boolean` = `true`

Drop first dummy (gives D-1 columns)

###### keepOriginal?

`boolean` = `false`

Keep the original categorical column

###### prefix?

`boolean` = `true`

Prefix generated column names with original column name

###### handleUnknown?

`string` = `'ignore'`

Behaviour for unseen categories

#### Returns

`Object`

{ data, dummyInfo }

***

### prepareXY()

```ts
function prepareXY(__namedParameters?): object;
```

Defined in: [src/core/table.js:751](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/table.js#L751)

Prepare feature matrix X and response vector y from table-like data.
Supports categorical encoding for X and y via `encode` option:
  prepareXY({ X, y, data, naOmit = true, encode = null, encoders = null })

encode semantics same as prepareX. For y, only 'label' encoding is supported
(maps categories to integer class labels). Supply `encoders` to reuse previously
fitted encoders (e.g., from a training split) and keep label IDs consistent.

Returns:
  { X, y, columnsX, n, rows, encoders } where encoders may include encoders.y

#### Parameters

##### \_\_namedParameters?

###### encode?

`null` = `null`

###### encoders?

`null` = `null`

#### Returns

`object`

##### X

```ts
X: any[][] = xPrep.X;
```

##### y

```ts
y: any[] = yvec;
```

##### columnsX

```ts
columnsX: any[] = xPrep.columns;
```

##### n

```ts
n: number = xPrep.n;
```

##### rows

```ts
rows: any[] = preFiltered;
```

##### validIndices

```ts
validIndices: any[];
```

##### sourceLength

```ts
sourceLength: number = rows.length;
```

##### encoders

```ts
encoders: object = encodersOut;
```
