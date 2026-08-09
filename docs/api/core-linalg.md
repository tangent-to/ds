---
layout: default
title: linalg
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/linalg
---
# linalg

## Classes

### SingularValueDecomposition

Defined in: [src/core/linalg.js:159](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L159)

SVD with the decomposition-object interface of ml-matrix, for
least-squares solves and pseudoinverses reusing one factorization.

#### Constructors

##### Constructor

```ts
new SingularValueDecomposition(data): SingularValueDecomposition;
```

Defined in: [src/core/linalg.js:163](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L163)

###### Parameters

###### data

`number`[][] \| [`Matrix`](#matrix)

Input matrix (any shape)

###### Returns

[`SingularValueDecomposition`](#singularvaluedecomposition)

#### Properties

##### \_U

```ts
_U: any;
```

Defined in: [src/core/linalg.js:165](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L165)

##### \_s

```ts
_s: any;
```

Defined in: [src/core/linalg.js:166](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L166)

##### \_V

```ts
_V: any;
```

Defined in: [src/core/linalg.js:167](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L167)

##### \_m

```ts
_m: any;
```

Defined in: [src/core/linalg.js:168](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L168)

##### \_n

```ts
_n: any;
```

Defined in: [src/core/linalg.js:169](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L169)

#### Accessors

##### leftSingularVectors

###### Get Signature

```ts
get leftSingularVectors(): Matrix;
```

Defined in: [src/core/linalg.js:172](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L172)

###### Returns

[`Matrix`](#matrix)

##### rightSingularVectors

###### Get Signature

```ts
get rightSingularVectors(): Matrix;
```

Defined in: [src/core/linalg.js:176](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L176)

###### Returns

[`Matrix`](#matrix)

##### diagonal

###### Get Signature

```ts
get diagonal(): any;
```

Defined in: [src/core/linalg.js:180](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L180)

###### Returns

`any`

#### Methods

##### \_cutoff()

```ts
_cutoff(): number;
```

Defined in: [src/core/linalg.js:184](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L184)

###### Returns

`number`

##### solve()

```ts
solve(b): Matrix;
```

Defined in: [src/core/linalg.js:193](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L193)

Minimum-norm least-squares solution of A x = b

###### Parameters

###### b

`number`[][] \| [`Matrix`](#matrix)

Right-hand side (column(s))

###### Returns

[`Matrix`](#matrix)

Solution x

##### inverse()

```ts
inverse(): Matrix;
```

Defined in: [src/core/linalg.js:210](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L210)

Pseudoinverse from the computed factorization

###### Returns

[`Matrix`](#matrix)

Pseudoinverse

***

### Matrix

Defined in: src/core/matrix.js:22

#### Constructors

##### Constructor

```ts
new Matrix(rowsOrData, columns?): Matrix;
```

Defined in: src/core/matrix.js:28

###### Parameters

###### rowsOrData

`number` \| `number`[][] \| [`Matrix`](#matrix)

Row count,
  nested array, or Matrix to copy

###### columns?

`number`

Column count when rowsOrData is a number

###### Returns

[`Matrix`](#matrix)

#### Properties

##### data

```ts
data: any;
```

Defined in: src/core/matrix.js:30

#### Accessors

##### rows

###### Get Signature

```ts
get rows(): any;
```

Defined in: src/core/matrix.js:71

###### Returns

`any`

##### columns

###### Get Signature

```ts
get columns(): any;
```

Defined in: src/core/matrix.js:75

###### Returns

`any`

#### Methods

##### zeros()

```ts
static zeros(rows, columns): Matrix;
```

Defined in: src/core/matrix.js:40

###### Parameters

###### rows

`any`

###### columns

`any`

###### Returns

[`Matrix`](#matrix)

##### ones()

```ts
static ones(rows, columns): Matrix;
```

Defined in: src/core/matrix.js:44

###### Parameters

###### rows

`any`

###### columns

`any`

###### Returns

[`Matrix`](#matrix)

##### eye()

```ts
static eye(rows, columns?): Matrix;
```

Defined in: src/core/matrix.js:50

###### Parameters

###### rows

`any`

###### columns?

`any` = `rows`

###### Returns

[`Matrix`](#matrix)

##### diag()

```ts
static diag(values): Matrix;
```

Defined in: src/core/matrix.js:56

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### columnVector()

```ts
static columnVector(values): Matrix;
```

Defined in: src/core/matrix.js:63

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### rowVector()

```ts
static rowVector(values): Matrix;
```

Defined in: src/core/matrix.js:67

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### get()

```ts
get(i, j): any;
```

Defined in: src/core/matrix.js:79

###### Parameters

###### i

`any`

###### j

`any`

###### Returns

`any`

##### set()

```ts
set(
   i, 
   j, 
   value): Matrix;
```

Defined in: src/core/matrix.js:83

###### Parameters

###### i

`any`

###### j

`any`

###### value

`any`

###### Returns

[`Matrix`](#matrix)

##### getRow()

```ts
getRow(i): any;
```

Defined in: src/core/matrix.js:88

###### Parameters

###### i

`any`

###### Returns

`any`

##### getColumn()

```ts
getColumn(j): any;
```

Defined in: src/core/matrix.js:92

###### Parameters

###### j

`any`

###### Returns

`any`

##### setRow()

```ts
setRow(i, values): Matrix;
```

Defined in: src/core/matrix.js:96

###### Parameters

###### i

`any`

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### setColumn()

```ts
setColumn(j, values): Matrix;
```

Defined in: src/core/matrix.js:101

###### Parameters

###### j

`any`

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### to2DArray()

```ts
to2DArray(): any;
```

Defined in: src/core/matrix.js:106

###### Returns

`any`

##### to1DArray()

```ts
to1DArray(): any;
```

Defined in: src/core/matrix.js:110

###### Returns

`any`

##### clone()

```ts
clone(): Matrix;
```

Defined in: src/core/matrix.js:114

###### Returns

[`Matrix`](#matrix)

##### mmul()

```ts
mmul(other): Matrix;
```

Defined in: src/core/matrix.js:123

Matrix product; returns a new Matrix.

###### Parameters

###### other

`number`[][] \| [`Matrix`](#matrix)

Right operand

###### Returns

[`Matrix`](#matrix)

this * other

##### transpose()

```ts
transpose(): Matrix;
```

Defined in: src/core/matrix.js:129

###### Returns

[`Matrix`](#matrix)

##### subMatrix()

```ts
subMatrix(
   startRow, 
   endRow, 
   startColumn, 
   endColumn): Matrix;
```

Defined in: src/core/matrix.js:139

###### Parameters

###### startRow

`any`

###### endRow

`any`

###### startColumn

`any`

###### endColumn

`any`

###### Returns

[`Matrix`](#matrix)

##### \_elementWise()

```ts
_elementWise(other, op): Matrix;
```

Defined in: src/core/matrix.js:147

###### Parameters

###### other

`any`

###### op

`any`

###### Returns

[`Matrix`](#matrix)

##### add()

```ts
add(other): Matrix;
```

Defined in: src/core/matrix.js:162

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### sub()

```ts
sub(other): Matrix;
```

Defined in: src/core/matrix.js:166

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### mul()

```ts
mul(other): Matrix;
```

Defined in: src/core/matrix.js:170

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### div()

```ts
div(other): Matrix;
```

Defined in: src/core/matrix.js:174

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### mean()

```ts
mean(by?): number | number[];
```

Defined in: src/core/matrix.js:183

Mean of all entries, or per-row/per-column means.

###### Parameters

###### by?

`"column"` \| `"row"`

Aggregation axis

###### Returns

`number` \| `number`[]

Grand mean, or one mean per row/column

##### max()

```ts
max(): number;
```

Defined in: src/core/matrix.js:201

###### Returns

`number`

##### min()

```ts
min(): number;
```

Defined in: src/core/matrix.js:209

###### Returns

`number`

## Functions

### toMatrix()

```ts
function toMatrix(data): Matrix;
```

Defined in: [src/core/linalg.js:20](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L20)

Convert array-like structure to Matrix

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Input data

#### Returns

[`Matrix`](#matrix)

Matrix object

***

### solveLeastSquares()

```ts
function solveLeastSquares(A, b): Matrix;
```

Defined in: [src/core/linalg.js:33](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L33)

Solve least squares problem: minimize ||Ax - b||^2

#### Parameters

##### A

`number`[][] \| [`Matrix`](#matrix)

Design matrix

##### b

`number`[] \| `number`[][] \| [`Matrix`](#matrix)

Target vector/matrix

#### Returns

[`Matrix`](#matrix)

Solution x

***

### covarianceMatrix()

```ts
function covarianceMatrix(data, center?): Matrix;
```

Defined in: [src/core/linalg.js:58](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L58)

Compute covariance matrix

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Data matrix (rows = observations)

##### center?

`boolean` = `true`

If true, center the data

#### Returns

[`Matrix`](#matrix)

Covariance matrix

***

### svd()

```ts
function svd(data): Object;
```

Defined in: [src/core/linalg.js:81](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L81)

Singular Value Decomposition (thin: U is m×k, V is n×k, k = min(m, n))

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Input matrix

#### Returns

`Object`

{U, s, V} where data ≈ U * diag(s) * V'

***

### eig()

```ts
function eig(data): Object;
```

Defined in: [src/core/linalg.js:97](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L97)

Eigenvalue decomposition of a symmetric matrix.
Eigenvalues are returned in descending order; eigenvectors are the
columns of `vectors`. Throws for non-symmetric input.

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Symmetric square matrix

#### Returns

`Object`

{values, vectors}

***

### mmul()

```ts
function mmul(A, B): Matrix;
```

Defined in: [src/core/linalg.js:111](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L111)

Matrix multiplication

#### Parameters

##### A

`number`[][] \| [`Matrix`](#matrix)

First matrix

##### B

`number`[][] \| [`Matrix`](#matrix)

Second matrix

#### Returns

[`Matrix`](#matrix)

A * B

***

### transpose()

```ts
function transpose(data): Matrix;
```

Defined in: [src/core/linalg.js:120](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L120)

Matrix transpose

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Input matrix

#### Returns

[`Matrix`](#matrix)

Transposed matrix

***

### inverse()

```ts
function inverse(data): Matrix;
```

Defined in: [src/core/linalg.js:129](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L129)

Matrix inverse

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Square matrix

#### Returns

[`Matrix`](#matrix)

Inverse matrix

***

### solve()

```ts
function solve(A, b): Matrix;
```

Defined in: [src/core/linalg.js:139](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L139)

Solve the linear system Ax = b (square A); throws if singular

#### Parameters

##### A

`number`[][] \| [`Matrix`](#matrix)

Square matrix

##### b

`number`[][] \| [`Matrix`](#matrix)

Right-hand side (column(s))

#### Returns

[`Matrix`](#matrix)

Solution x

***

### pseudoInverse()

```ts
function pseudoInverse(data): Matrix;
```

Defined in: [src/core/linalg.js:151](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/linalg.js#L151)

Moore-Penrose pseudoinverse via SVD with a singular-value cutoff
scaled by the matrix size and largest singular value (numpy
convention), so near-zero singular values are zeroed instead of
inverted into garbage for nearly rank-deficient matrices.

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Input matrix

#### Returns

[`Matrix`](#matrix)

Pseudoinverse
