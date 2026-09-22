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

Defined in: [src/core/linalg.js:218](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L218)

SVD with the decomposition-object interface of ml-matrix, for
least-squares solves and pseudoinverses reusing one factorization.

#### Constructors

##### Constructor

```ts
new SingularValueDecomposition(data): SingularValueDecomposition;
```

Defined in: [src/core/linalg.js:222](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L222)

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

Defined in: [src/core/linalg.js:224](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L224)

##### \_s

```ts
_s: any;
```

Defined in: [src/core/linalg.js:225](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L225)

##### \_V

```ts
_V: any;
```

Defined in: [src/core/linalg.js:226](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L226)

##### \_m

```ts
_m: any;
```

Defined in: [src/core/linalg.js:227](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L227)

##### \_n

```ts
_n: any;
```

Defined in: [src/core/linalg.js:228](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L228)

#### Accessors

##### leftSingularVectors

###### Get Signature

```ts
get leftSingularVectors(): Matrix;
```

Defined in: [src/core/linalg.js:231](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L231)

###### Returns

[`Matrix`](#matrix)

##### rightSingularVectors

###### Get Signature

```ts
get rightSingularVectors(): Matrix;
```

Defined in: [src/core/linalg.js:235](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L235)

###### Returns

[`Matrix`](#matrix)

##### diagonal

###### Get Signature

```ts
get diagonal(): any;
```

Defined in: [src/core/linalg.js:239](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L239)

###### Returns

`any`

#### Methods

##### \_cutoff()

```ts
_cutoff(): number;
```

Defined in: [src/core/linalg.js:243](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L243)

###### Returns

`number`

##### solve()

```ts
solve(b): Matrix;
```

Defined in: [src/core/linalg.js:252](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L252)

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

Defined in: [src/core/linalg.js:269](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L269)

Pseudoinverse from the computed factorization

###### Returns

[`Matrix`](#matrix)

Pseudoinverse

***

### Matrix

Defined in: [src/core/matrix.js:22](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L22)

#### Constructors

##### Constructor

```ts
new Matrix(rowsOrData, columns?): Matrix;
```

Defined in: [src/core/matrix.js:28](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L28)

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

Defined in: [src/core/matrix.js:30](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L30)

#### Accessors

##### rows

###### Get Signature

```ts
get rows(): any;
```

Defined in: [src/core/matrix.js:71](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L71)

###### Returns

`any`

##### columns

###### Get Signature

```ts
get columns(): any;
```

Defined in: [src/core/matrix.js:75](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L75)

###### Returns

`any`

#### Methods

##### zeros()

```ts
static zeros(rows, columns): Matrix;
```

Defined in: [src/core/matrix.js:40](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L40)

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

Defined in: [src/core/matrix.js:44](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L44)

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

Defined in: [src/core/matrix.js:50](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L50)

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

Defined in: [src/core/matrix.js:56](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L56)

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### columnVector()

```ts
static columnVector(values): Matrix;
```

Defined in: [src/core/matrix.js:63](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L63)

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### rowVector()

```ts
static rowVector(values): Matrix;
```

Defined in: [src/core/matrix.js:67](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L67)

###### Parameters

###### values

`any`

###### Returns

[`Matrix`](#matrix)

##### get()

```ts
get(i, j): any;
```

Defined in: [src/core/matrix.js:79](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L79)

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

Defined in: [src/core/matrix.js:83](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L83)

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

Defined in: [src/core/matrix.js:88](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L88)

###### Parameters

###### i

`any`

###### Returns

`any`

##### getColumn()

```ts
getColumn(j): any;
```

Defined in: [src/core/matrix.js:92](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L92)

###### Parameters

###### j

`any`

###### Returns

`any`

##### setRow()

```ts
setRow(i, values): Matrix;
```

Defined in: [src/core/matrix.js:96](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L96)

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

Defined in: [src/core/matrix.js:101](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L101)

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

Defined in: [src/core/matrix.js:106](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L106)

###### Returns

`any`

##### to1DArray()

```ts
to1DArray(): any;
```

Defined in: [src/core/matrix.js:110](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L110)

###### Returns

`any`

##### clone()

```ts
clone(): Matrix;
```

Defined in: [src/core/matrix.js:114](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L114)

###### Returns

[`Matrix`](#matrix)

##### mmul()

```ts
mmul(other): Matrix;
```

Defined in: [src/core/matrix.js:123](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L123)

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

Defined in: [src/core/matrix.js:129](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L129)

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

Defined in: [src/core/matrix.js:139](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L139)

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

Defined in: [src/core/matrix.js:147](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L147)

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

Defined in: [src/core/matrix.js:162](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L162)

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### sub()

```ts
sub(other): Matrix;
```

Defined in: [src/core/matrix.js:166](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L166)

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### mul()

```ts
mul(other): Matrix;
```

Defined in: [src/core/matrix.js:170](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L170)

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### div()

```ts
div(other): Matrix;
```

Defined in: [src/core/matrix.js:174](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L174)

###### Parameters

###### other

`any`

###### Returns

[`Matrix`](#matrix)

##### mean()

```ts
mean(by?): number | number[];
```

Defined in: [src/core/matrix.js:183](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L183)

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

Defined in: [src/core/matrix.js:201](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L201)

###### Returns

`number`

##### min()

```ts
min(): number;
```

Defined in: [src/core/matrix.js:209](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/matrix.js#L209)

###### Returns

`number`

## Functions

### toMatrix()

```ts
function toMatrix(data): Matrix;
```

Defined in: [src/core/linalg.js:24](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L24)

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

Defined in: [src/core/linalg.js:37](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L37)

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

Defined in: [src/core/linalg.js:62](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L62)

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

Defined in: [src/core/linalg.js:85](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L85)

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

Defined in: [src/core/linalg.js:101](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L101)

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

### eigGeneralized()

```ts
function eigGeneralized(A, B): Object;
```

Defined in: [src/core/linalg.js:123](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L123)

Generalized symmetric eigendecomposition: solve A x = lambda B x for
symmetric A and symmetric positive (semi)definite B. Eigenvalues are
returned in descending order; eigenvectors are the columns of `vectors`.

When B is positive definite the vectors are B-orthonormal (x'Bx = 1), as
from scipy's eigh(A, B). When B is singular the problem is solved on
range(B) and the vectors have unit euclidean length instead; `definite`
reports which case applied.

#### Parameters

##### A

`number`[][] \| [`Matrix`](#matrix)

Symmetric matrix

##### B

`number`[][] \| [`Matrix`](#matrix)

Symmetric positive (semi)definite matrix

#### Returns

`Object`

{values, vectors, definite}

***

### symmetricInverseSqrt()

```ts
function symmetricInverseSqrt(data): Matrix;
```

Defined in: [src/core/linalg.js:137](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L137)

Inverse square root of a symmetric positive semidefinite matrix: the
symmetric W with W A W = I on A's range, and 0 on its null space

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Symmetric positive semidefinite matrix

#### Returns

[`Matrix`](#matrix)

Symmetric inverse square root

***

### mmul()

```ts
function mmul(A, B): Matrix;
```

Defined in: [src/core/linalg.js:147](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L147)

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

Defined in: [src/core/linalg.js:156](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L156)

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

Defined in: [src/core/linalg.js:165](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L165)

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

Defined in: [src/core/linalg.js:175](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L175)

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

### cholesky()

```ts
function cholesky(data): Matrix;
```

Defined in: [src/core/linalg.js:185](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L185)

Cholesky factorization of a symmetric positive definite matrix

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Symmetric positive definite matrix

#### Returns

[`Matrix`](#matrix)

Lower triangular L with data = L * L'

#### Throws

When the matrix is not symmetric or not positive definite

***

### choleskySolve()

```ts
function choleskySolve(L, b): number[] | number[][];
```

Defined in: [src/core/linalg.js:198](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L198)

Solve A x = b (or A X = B) from the Cholesky factor L of A, by forward then
back substitution. Passing every right-hand side at once is much cheaper
than one call per column when building an inverse.

#### Parameters

##### L

`number`[][] \| [`Matrix`](#matrix)

Lower triangular factor

##### b

`number`[] \| `number`[][]

Right-hand side vector, or a
  matrix whose columns are right-hand sides

#### Returns

`number`[] \| `number`[][]

Solution, matching b's shape

***

### pseudoInverse()

```ts
function pseudoInverse(data): Matrix;
```

Defined in: [src/core/linalg.js:210](https://github.com/tangent-to/ds/blob/eb37453b7dcefe351c0f650c2bc55037ca8a784b/src/core/linalg.js#L210)

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
