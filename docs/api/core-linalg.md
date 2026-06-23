---
layout: default
title: linalg
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/linalg
---
# linalg

## Classes

### Matrix

Defined in: node\_modules/ml-matrix/matrix.d.ts:991

#### Extends

- `AbstractMatrix`

#### Constructors

##### Constructor

```ts
new Matrix(nRows, nColumns): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:992

###### Parameters

###### nRows

`number`

###### nColumns

`number`

###### Returns

[`Matrix`](#matrix)

###### Overrides

```ts
AbstractMatrix.constructor
```

##### Constructor

```ts
new Matrix(data): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:993

###### Parameters

###### data

`ArrayLike`\<`ArrayLike`\<`number`\>\>

###### Returns

[`Matrix`](#matrix)

###### Overrides

```ts
AbstractMatrix.constructor
```

##### Constructor

```ts
new Matrix(otherMatrix): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:994

###### Parameters

###### otherMatrix

`AbstractMatrix`

###### Returns

[`Matrix`](#matrix)

###### Overrides

```ts
AbstractMatrix.constructor
```

#### Properties

##### size

```ts
readonly size: number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:129

Total number of elements in the matrix.

###### Inherited from

```ts
AbstractMatrix.size
```

##### rows

```ts
readonly rows: number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:134

Number of rows of the matrix.

###### Inherited from

```ts
AbstractMatrix.rows
```

##### columns

```ts
readonly columns: number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:139

Number of columns of the matrix.

###### Inherited from

```ts
AbstractMatrix.columns
```

#### Methods

##### from1DArray()

```ts
static from1DArray(
   newRows, 
   newColumns, 
   newData): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:148

Constructs a matrix with the chosen dimensions from a 1D array.

###### Parameters

###### newRows

`number`

Number of rows.

###### newColumns

`number`

Number of columns.

###### newData

`ArrayLike`\<`number`\>

A 1D array containing data for the matrix.

###### Returns

[`Matrix`](#matrix)

The new matrix.

###### Inherited from

```ts
AbstractMatrix.from1DArray
```

##### rowVector()

```ts
static rowVector(newData): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:159

Creates a row vector, a matrix with only one row.

###### Parameters

###### newData

`ArrayLike`\<`number`\>

A 1D array containing data for the vector.

###### Returns

[`Matrix`](#matrix)

The new matrix.

###### Inherited from

```ts
AbstractMatrix.rowVector
```

##### columnVector()

```ts
static columnVector(newData): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:166

Creates a column vector, a matrix with only one column.

###### Parameters

###### newData

`ArrayLike`\<`number`\>

A 1D array containing data for the vector.

###### Returns

[`Matrix`](#matrix)

The new matrix.

###### Inherited from

```ts
AbstractMatrix.columnVector
```

##### zeros()

```ts
static zeros<_M>(rows, columns): _M;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:176

Creates a matrix with the given dimensions. Values will be set to zero.
This is equivalent to calling the Matrix constructor.

###### Type Parameters

###### _M

`_M` *extends* `AbstractMatrix` = [`Matrix`](#matrix)

is private. Don't override it.

###### Parameters

###### rows

`number`

Number of rows.

###### columns

`number`

Number of columns.

###### Returns

`_M`

The new matrix.

###### Inherited from

```ts
AbstractMatrix.zeros
```

##### ones()

```ts
static ones<M>(rows, columns): M;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:187

Creates a matrix with the given dimensions. Values will be set to one.

###### Type Parameters

###### M

`M` *extends* `AbstractMatrix` = [`Matrix`](#matrix)

###### Parameters

###### rows

`number`

Number of rows.

###### columns

`number`

Number of columns.

###### Returns

`M`

The new matrix.

###### Inherited from

```ts
AbstractMatrix.ones
```

##### rand()

```ts
static rand(
   rows, 
   columns, 
   options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:199

Creates a matrix with the given dimensions. Values will be randomly set.

###### Parameters

###### rows

`number`

Number of rows.

###### columns

`number`

Number of columns.

###### options?

`IRandomOptions`

Options object.

###### Returns

[`Matrix`](#matrix)

The new matrix.

###### Inherited from

```ts
AbstractMatrix.rand
```

##### random()

```ts
static random(
   rows, 
   columns, 
   options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:200

###### Parameters

###### rows

`number`

###### columns

`number`

###### options?

`IRandomOptions`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.random
```

##### randInt()

```ts
static randInt(
   rows, 
   columns, 
   options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:213

Creates a matrix with the given dimensions. Values will be random integers.

###### Parameters

###### rows

`number`

Number of rows.

###### columns

`number`

Number of columns.

###### options?

`IRandomIntOptions`

###### Returns

[`Matrix`](#matrix)

- The new matrix.

###### Inherited from

```ts
AbstractMatrix.randInt
```

##### eye()

```ts
static eye(
   rows, 
   columns?, 
   value?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:226

Creates an identity matrix with the given dimension. Values of the diagonal will be 1 and others will be 0.

###### Parameters

###### rows

`number`

Number of rows.

###### columns?

`number`

Number of columns. Default: `rows`.

###### value?

`number`

Value to fill the diagonal with. Default: `1`.

###### Returns

[`Matrix`](#matrix)

- The new identity matrix.

###### Inherited from

```ts
AbstractMatrix.eye
```

##### identity()

```ts
static identity(
   rows, 
   columns?, 
   value?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:231

Alias for [AbstractMatrix.eye](#eye).

###### Parameters

###### rows

`number`

###### columns?

`number`

###### value?

`number`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.identity
```

##### diag()

```ts
static diag(
   data, 
   rows?, 
   columns?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:240

Creates a diagonal matrix based on the given array.

###### Parameters

###### data

`ArrayLike`\<`number`\>

Array containing the data for the diagonal.

###### rows?

`number`

Number of rows. Default: `data.length`.

###### columns?

`number`

Number of columns. Default: `rows`.

###### Returns

[`Matrix`](#matrix)

- The new diagonal matrix.

###### Inherited from

```ts
AbstractMatrix.diag
```

##### diagonal()

```ts
static diagonal(
   data, 
   rows?, 
   columns?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:245

Alias for [AbstractMatrix.diag](#diag).

###### Parameters

###### data

`ArrayLike`\<`number`\>

###### rows?

`number`

###### columns?

`number`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.diagonal
```

##### min()

```ts
static min(matrix1, matrix2): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:254

Returns a matrix whose elements are the minimum between `matrix1` and `matrix2`.

###### Parameters

###### matrix1

`MaybeMatrix`

###### matrix2

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.min
```

##### max()

```ts
static max(matrix1, matrix2): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:261

Returns a matrix whose elements are the maximum between `matrix1` and `matrix2`.

###### Parameters

###### matrix1

`MaybeMatrix`

###### matrix2

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.max
```

##### checkMatrix()

```ts
static checkMatrix(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:267

Check that the provided value is a Matrix and tries to instantiate one if not.

###### Parameters

###### value

`any`

The value to check.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.checkMatrix
```

##### isMatrix()

```ts
static isMatrix(value): value is AbstractMatrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:273

Returns whether `value` is a Matrix.

###### Parameters

###### value

`any`

The value to check.

###### Returns

`value is AbstractMatrix`

###### Inherited from

```ts
AbstractMatrix.isMatrix
```

##### apply()

```ts
apply(callback): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:295

Applies a callback for each element of the matrix. The function is called in the matrix (this) context.

###### Parameters

###### callback

(`row`, `column`) => `void`

Function that will be called for each element in the matrix.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.apply
```

##### to1DArray()

```ts
to1DArray(): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:300

Returns a new 1D array filled row by row with the matrix values.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.to1DArray
```

##### to2DArray()

```ts
to2DArray(): number[][];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:305

Returns a 2D array containing a copy of the matrix data.

###### Returns

`number`[][]

###### Inherited from

```ts
AbstractMatrix.to2DArray
```

##### toJSON()

```ts
toJSON(): number[][];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:307

###### Returns

`number`[][]

###### Inherited from

```ts
AbstractMatrix.toJSON
```

##### isRowVector()

```ts
isRowVector(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:312

Returns whether the matrix has one row.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isRowVector
```

##### isColumnVector()

```ts
isColumnVector(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:317

Returns whether the matrix has one column.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isColumnVector
```

##### isVector()

```ts
isVector(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:322

Returns whether the matrix has one row or one column.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isVector
```

##### isSquare()

```ts
isSquare(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:327

Returns whether the matrix has the same number of rows and columns.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isSquare
```

##### isDistance()

```ts
isDistance(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:332

Returns whether the matrix is symmetric and diagonal values are equals to 0

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isDistance
```

##### isEmpty()

```ts
isEmpty(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:337

Returns whether the number of rows or columns (or both) is zero.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isEmpty
```

##### isSymmetric()

```ts
isSymmetric(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:342

Returns whether the matrix is square and has the same values on both sides of the diagonal.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isSymmetric
```

##### isEchelonForm()

```ts
isEchelonForm(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:347

Returns whether the matrix is in row echelon form.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isEchelonForm
```

##### isReducedEchelonForm()

```ts
isReducedEchelonForm(): boolean;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:352

Returns whether the matrix is in reduced row echelon form.

###### Returns

`boolean`

###### Inherited from

```ts
AbstractMatrix.isReducedEchelonForm
```

##### echelonForm()

```ts
echelonForm(): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:358

Returns the row echelon form of the matrix computed using gaussian
elimination.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.echelonForm
```

##### reducedEchelonForm()

```ts
reducedEchelonForm(): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:364

Returns the reduced row echelon form of the matrix computed using
gaussian elimination.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.reducedEchelonForm
```

##### repeat()

```ts
repeat(options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:374

Creates a new matrix that is a repetition of the current matrix. New matrix has rows times the number of
rows of the original matrix, and columns times the number of columns of the original matrix.

###### Parameters

###### options?

`IRepeatOptions`

###### Returns

[`Matrix`](#matrix)

###### Example

```ts
var matrix = new Matrix([[1, 2]]);
matrix.repeat({ rows: 2 }); // [[1, 2], [1, 2]]
```

###### Inherited from

```ts
AbstractMatrix.repeat
```

##### fill()

```ts
fill(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:380

Fills the matrix with a given value. All elements will be set to this value.

###### Parameters

###### value

`number`

New value.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.fill
```

##### neg()

```ts
neg(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:385

Negates the matrix. All elements will be multiplied by `-1`.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.neg
```

##### negate()

```ts
negate(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:390

Alias for AbstractMatrix.neg.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.negate
```

##### getRow()

```ts
getRow(index): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:396

Returns a new array with the values from the given row index.

###### Parameters

###### index

`number`

Row index.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.getRow
```

##### getRowVector()

```ts
getRowVector(index): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:402

Returns a new row vector with the values from the given row index.

###### Parameters

###### index

`number`

Row index.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.getRowVector
```

##### setRow()

```ts
setRow(index, array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:409

Sets a row at the given index.

###### Parameters

###### index

`number`

Row index.

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector to set.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.setRow
```

##### swapRows()

```ts
swapRows(row1, row2): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:416

Swap two rows.

###### Parameters

###### row1

`number`

First row index.

###### row2

`number`

Second row index.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.swapRows
```

##### getColumn()

```ts
getColumn(index): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:422

Returns a new array with the values from the given column index.

###### Parameters

###### index

`number`

Column index.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.getColumn
```

##### getColumnVector()

```ts
getColumnVector(index): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:428

Returns a new column vector with the values from the given column index.

###### Parameters

###### index

`number`

Column index.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.getColumnVector
```

##### setColumn()

```ts
setColumn(index, array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:435

Sets a column at the given index.

###### Parameters

###### index

`number`

Column index.

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector to set.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.setColumn
```

##### swapColumns()

```ts
swapColumns(column1, column2): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:442

Swap two columns.

###### Parameters

###### column1

`number`

First column index.

###### column2

`number`

Second column index.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.swapColumns
```

##### addRowVector()

```ts
addRowVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:448

Adds the values of a vector to each row.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.addRowVector
```

##### subRowVector()

```ts
subRowVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:454

Subtracts the values of a vector from each row.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.subRowVector
```

##### mulRowVector()

```ts
mulRowVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:460

Multiplies the values of a vector with each row.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mulRowVector
```

##### divRowVector()

```ts
divRowVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:466

Divides the values of each row by those of a vector.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.divRowVector
```

##### addColumnVector()

```ts
addColumnVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:472

Adds the values of a vector to each column.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.addColumnVector
```

##### subColumnVector()

```ts
subColumnVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:478

Subtracts the values of a vector from each column.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.subColumnVector
```

##### mulColumnVector()

```ts
mulColumnVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:484

Multiplies the values of a vector with each column.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mulColumnVector
```

##### divColumnVector()

```ts
divColumnVector(vector): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:490

Divides the values of each column by those of a vector.

###### Parameters

###### vector

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Array or vector.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.divColumnVector
```

##### mulRow()

```ts
mulRow(index, value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:497

Multiplies the values of a row with a scalar.

###### Parameters

###### index

`number`

Row index.

###### value

`number`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mulRow
```

##### mulColumn()

```ts
mulColumn(index, value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:504

Multiplies the values of a column with a scalar.

###### Parameters

###### index

`number`

Column index.

###### value

`number`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mulColumn
```

##### max()

###### Call Signature

```ts
max(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:509

Returns the maximum value of the matrix.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.max
```

###### Call Signature

```ts
max(by): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:515

Returns the maximum value by the given dimension.

###### Parameters

###### by

`MatrixDimension`

max by 'row' or 'column'.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.max
```

##### maxIndex()

```ts
maxIndex(): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:520

Returns the index of the maximum value.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.maxIndex
```

##### min()

###### Call Signature

```ts
min(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:525

Returns the minimum value of the matrix.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.min
```

###### Call Signature

```ts
min(by): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:531

Returns the minimum value by the given dimension.

###### Parameters

###### by

`MatrixDimension`

min by 'row' or 'column'.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.min
```

##### minIndex()

```ts
minIndex(): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:536

Returns the index of the minimum value.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.minIndex
```

##### maxRow()

```ts
maxRow(row): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:542

Returns the maximum value of one row.

###### Parameters

###### row

`number`

Row index.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.maxRow
```

##### maxRowIndex()

```ts
maxRowIndex(row): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:548

Returns the index of the maximum value of one row.

###### Parameters

###### row

`number`

Row index.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.maxRowIndex
```

##### minRow()

```ts
minRow(row): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:554

Returns the minimum value of one row.

###### Parameters

###### row

`number`

Row index.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.minRow
```

##### minRowIndex()

```ts
minRowIndex(row): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:560

Returns the index of the maximum value of one row.

###### Parameters

###### row

`number`

Row index.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.minRowIndex
```

##### maxColumn()

```ts
maxColumn(column): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:566

Returns the maximum value of one column.

###### Parameters

###### column

`number`

Column index.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.maxColumn
```

##### maxColumnIndex()

```ts
maxColumnIndex(column): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:572

Returns the index of the maximum value of one column.

###### Parameters

###### column

`number`

Column index.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.maxColumnIndex
```

##### minColumn()

```ts
minColumn(column): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:578

Returns the minimum value of one column.

###### Parameters

###### column

`number`

Column index.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.minColumn
```

##### minColumnIndex()

```ts
minColumnIndex(column): [number, number];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:584

Returns the index of the minimum value of one column.

###### Parameters

###### column

`number`

Column index.

###### Returns

\[`number`, `number`\]

###### Inherited from

```ts
AbstractMatrix.minColumnIndex
```

##### diag()

```ts
diag(): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:589

Returns an array containing the diagonal values of the matrix.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.diag
```

##### diagonal()

```ts
diagonal(): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:594

Alias for [AbstractMatrix.diag](#diag).

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.diagonal
```

##### norm()

```ts
norm(type?): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:600

Returns the norm of a matrix.

###### Parameters

###### type?

`"max"` \| `"frobenius"`

Norm type. Default: `'frobenius'`.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.norm
```

##### cumulativeSum()

```ts
cumulativeSum(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:605

Computes the cumulative sum of the matrix elements (in place, row by row).

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.cumulativeSum
```

##### dot()

```ts
dot(vector): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:611

Computes the dot (scalar) product between the matrix and another.

###### Parameters

###### vector

`AbstractMatrix`

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.dot
```

##### mmul()

```ts
mmul(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:617

Returns the matrix product between `this` and `other`.

###### Parameters

###### other

`MaybeMatrix`

Other matrix.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.mmul
```

##### mpow()

```ts
mpow(scalar): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:623

Returns the square matrix raised to the given power

###### Parameters

###### scalar

`number`

the non-negative integer power to raise this matrix to

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.mpow
```

##### strassen2x2()

```ts
strassen2x2(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:625

###### Parameters

###### other

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.strassen2x2
```

##### strassen3x3()

```ts
strassen3x3(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:627

###### Parameters

###### other

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.strassen3x3
```

##### mmulStrassen()

```ts
mmulStrassen(y): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:629

###### Parameters

###### y

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.mmulStrassen
```

##### scaleRows()

```ts
scaleRows(options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:635

Returns a new row-by-row scaled matrix.

###### Parameters

###### options?

`IScaleOptions`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.scaleRows
```

##### scaleColumns()

```ts
scaleColumns(options?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:644

Returns a new column-by-column scaled matrix.

###### Parameters

###### options?

`IScaleOptions`

###### Returns

[`Matrix`](#matrix)

###### Example

```ts
var matrix = new Matrix([[1, 2], [-1, 0]]);
var scaledMatrix = matrix.scaleColumns(); // [[1, 1], [0, 0]]
```

###### Inherited from

```ts
AbstractMatrix.scaleColumns
```

##### flipRows()

```ts
flipRows(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:646

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.flipRows
```

##### flipColumns()

```ts
flipColumns(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:648

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.flipColumns
```

##### kroneckerProduct()

```ts
kroneckerProduct(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:655

Returns the Kronecker product (also known as tensor product) between `this` and `other`.

###### Parameters

###### other

`MaybeMatrix`

Other matrix.

###### Returns

[`Matrix`](#matrix)

###### Link

https://en.wikipedia.org/wiki/Kronecker_product

###### Inherited from

```ts
AbstractMatrix.kroneckerProduct
```

##### kroneckerSum()

```ts
kroneckerSum(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:662

Returns the Kronecker sum between `this` and `other`.

###### Parameters

###### other

`MaybeMatrix`

Other matrix.

###### Returns

[`Matrix`](#matrix)

###### Link

https://en.wikipedia.org/wiki/Kronecker_product#Kronecker_sum

###### Inherited from

```ts
AbstractMatrix.kroneckerSum
```

##### tensorProduct()

```ts
tensorProduct(other): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:667

Alias for [AbstractMatrix.kroneckerProduct](#kroneckerproduct).

###### Parameters

###### other

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.tensorProduct
```

##### transpose()

```ts
transpose(): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:672

Transposes the matrix and returns a new one containing the result.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.transpose
```

##### sortRows()

```ts
sortRows(compareFunction?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:678

Sorts the rows in-place.

###### Parameters

###### compareFunction?

(`a`, `b`) => `number`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sortRows
```

##### sortColumns()

```ts
sortColumns(compareFunction?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:684

Sorts the columns in-place.

###### Parameters

###### compareFunction?

(`a`, `b`) => `number`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sortColumns
```

##### subMatrix()

```ts
subMatrix(
   startRow, 
   endRow, 
   startColumn, 
   endColumn): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:693

Returns a subset of the matrix.

###### Parameters

###### startRow

`number`

First row index.

###### endRow

`number`

Last row index.

###### startColumn

`number`

First column index.

###### endColumn

`number`

Last column index.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.subMatrix
```

##### subMatrixRow()

```ts
subMatrixRow(
   indices, 
   startColumn?, 
   endColumn?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:706

Returns a subset of the matrix based on an array of row indices.

###### Parameters

###### indices

`ArrayLike`\<`number`\>

Array containing the row indices.

###### startColumn?

`number`

First column index. Default: `0`.

###### endColumn?

`number`

Last column index. Default: `this.columns - 1`.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.subMatrixRow
```

##### subMatrixColumn()

```ts
subMatrixColumn(
   indices, 
   startRow?, 
   endRow?): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:718

Returns a subset of the matrix based on an array of column indices.

###### Parameters

###### indices

`ArrayLike`\<`number`\>

Array containing the column indices.

###### startRow?

`number`

First row index. Default: `0`.

###### endRow?

`number`

Last row index. Default: `this.rows - 1`.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.subMatrixColumn
```

##### setSubMatrix()

```ts
setSubMatrix(
   matrix, 
   startRow, 
   startColumn): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:730

Set a part of the matrix to the given sub-matrix.

###### Parameters

###### matrix

`MaybeMatrix`

The source matrix from which to extract values.

###### startRow

`number`

The index of the first row to set.

###### startColumn

`number`

The index of the first column to set.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.setSubMatrix
```

##### selection()

```ts
selection(rowIndices, columnIndices): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:742

Return a new matrix based on a selection of rows and columns.
Order of the indices matters and the same index can be used more than once.

###### Parameters

###### rowIndices

`ArrayLike`\<`number`\>

The row indices to select.

###### columnIndices

`ArrayLike`\<`number`\>

The column indices to select.

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.selection
```

##### trace()

```ts
trace(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:750

Returns the trace of the matrix (sum of the diagonal elements).

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.trace
```

##### clone()

```ts
clone(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:755

Creates an exact and independent copy of the matrix.

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.clone
```

##### copy()

```ts
static copy<M>(from, to): M;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:757

###### Type Parameters

###### M

`M` *extends* `AbstractMatrix`

###### Parameters

###### from

`AbstractMatrix`

###### to

`M`

###### Returns

`M`

###### Inherited from

```ts
AbstractMatrix.copy
```

##### sum()

###### Call Signature

```ts
sum(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:762

Returns the sum of all elements of the matrix.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.sum
```

###### Call Signature

```ts
sum(by): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:768

Returns the sum by the given dimension.

###### Parameters

###### by

`MatrixDimension`

sum by 'row' or 'column'.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.sum
```

##### product()

###### Call Signature

```ts
product(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:773

Returns the product of all elements of the matrix.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.product
```

###### Call Signature

```ts
product(by): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:779

Returns the product by the given dimension.

###### Parameters

###### by

`MatrixDimension`

product by 'row' or 'column'.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.product
```

##### mean()

###### Call Signature

```ts
mean(): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:784

Returns the mean of all elements of the matrix.

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.mean
```

###### Call Signature

```ts
mean(by): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:790

Returns the mean by the given dimension.

###### Parameters

###### by

`MatrixDimension`

mean by 'row' or 'column'.

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.mean
```

##### variance()

###### Call Signature

```ts
variance(options?): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:796

Returns the variance of all elements of the matrix.

###### Parameters

###### options?

`IVarianceOptions`

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.variance
```

###### Call Signature

```ts
variance(by, options?): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:803

Returns the variance by the given dimension.

###### Parameters

###### by

`MatrixDimension`

variance by 'row' or 'column'.

###### options?

`IVarianceByOptions`

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.variance
```

##### standardDeviation()

###### Call Signature

```ts
standardDeviation(options?): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:809

Returns the standard deviation of all elements of the matrix.

###### Parameters

###### options?

`IVarianceOptions`

###### Returns

`number`

###### Inherited from

```ts
AbstractMatrix.standardDeviation
```

###### Call Signature

```ts
standardDeviation(by, options?): number[];
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:816

Returns the standard deviation by the given dimension.

###### Parameters

###### by

`MatrixDimension`

standard deviation by 'row' or 'column'.

###### options?

`IVarianceByOptions`

###### Returns

`number`[]

###### Inherited from

```ts
AbstractMatrix.standardDeviation
```

##### center()

###### Call Signature

```ts
center(options?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:826

Center the matrix in-place. By default, the mean value of the matrix is
subtracted from every value.

###### Parameters

###### options?

`ICenterOptions`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.center
```

###### Call Signature

```ts
center(by, options?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:834

Center the matrix in-place. By default, the mean values in the give
dimension are subtracted from the values.

###### Parameters

###### by

`MatrixDimension`

center by 'row' or 'column'.

###### options?

`ICenterByOptions`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.center
```

##### scale()

###### Call Signature

```ts
scale(options?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:841

Scale the matrix in-place. By default, values are divided by their
standard deviation.

###### Parameters

###### options?

`IScaleOptions`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.scale
```

###### Call Signature

```ts
scale(by, options?): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:849

Scale the matrix in-place. By default, values are divided by the
standard deviation in the given dimension.

###### Parameters

###### by

`MatrixDimension`

scale by 'row' or 'column'.

###### options?

`IScaleByOptions`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.scale
```

##### toString()

```ts
toString(options?): string;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:851

###### Parameters

###### options?

`IToStringOptions`

###### Returns

`string`

###### Inherited from

```ts
AbstractMatrix.toString
```

##### \[iterator\]()

```ts
iterator: Generator<[number, number, number], void, void>;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:859

iterator from left to right, from top to bottom
yield [row, column, value]

###### Returns

`Generator`\<\[`number`, `number`, `number`\], `void`, `void`\>

###### Inherited from

```ts
AbstractMatrix.[iterator]
```

##### entries()

```ts
entries(): Generator<[number, number, number], void, void>;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:869

iterator from left to right, from top to bottom
yield [row, column, value]

###### Returns

`Generator`\<\[`number`, `number`, `number`\], `void`, `void`\>

###### Inherited from

```ts
AbstractMatrix.entries
```

##### values()

```ts
values(): Generator<number, void, void>;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:879

iterator from left to right, from top to bottom
yield value

###### Returns

`Generator`\<`number`, `void`, `void`\>

###### Inherited from

```ts
AbstractMatrix.values
```

##### add()

```ts
add(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:885

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.add
```

##### sub()

```ts
sub(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:886

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sub
```

##### subtract()

```ts
subtract(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:887

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.subtract
```

##### mul()

```ts
mul(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:888

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mul
```

##### multiply()

```ts
multiply(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:889

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.multiply
```

##### div()

```ts
div(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:890

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.div
```

##### divide()

```ts
divide(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:891

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.divide
```

##### mod()

```ts
mod(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:892

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.mod
```

##### modulus()

```ts
modulus(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:893

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.modulus
```

##### and()

```ts
and(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:894

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.and
```

##### or()

```ts
or(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:895

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.or
```

##### xor()

```ts
xor(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:896

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.xor
```

##### leftShift()

```ts
leftShift(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:897

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.leftShift
```

##### signPropagatingRightShift()

```ts
signPropagatingRightShift(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:898

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.signPropagatingRightShift
```

##### rightShift()

```ts
rightShift(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:899

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.rightShift
```

##### zeroFillRightShift()

```ts
zeroFillRightShift(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:900

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.zeroFillRightShift
```

##### add()

```ts
static add(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:902

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.add
```

##### sub()

```ts
static sub(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:903

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.sub
```

##### subtract()

```ts
static subtract(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:904

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.subtract
```

##### mul()

```ts
static mul(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:905

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.mul
```

##### multiply()

```ts
static multiply(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:906

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.multiply
```

##### div()

```ts
static div(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:907

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.div
```

##### divide()

```ts
static divide(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:908

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.divide
```

##### mod()

```ts
static mod(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:909

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.mod
```

##### modulus()

```ts
static modulus(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:910

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.modulus
```

##### and()

```ts
static and(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:911

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.and
```

##### or()

```ts
static or(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:912

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.or
```

##### xor()

```ts
static xor(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:913

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.xor
```

##### leftShift()

```ts
static leftShift(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:914

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.leftShift
```

##### signPropagatingRightShift()

```ts
static signPropagatingRightShift(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:915

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.signPropagatingRightShift
```

##### rightShift()

```ts
static rightShift(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:919

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.rightShift
```

##### zeroFillRightShift()

```ts
static zeroFillRightShift(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:920

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.zeroFillRightShift
```

##### not()

```ts
not(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:924

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.not
```

##### abs()

```ts
abs(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:925

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.abs
```

##### acos()

```ts
acos(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:926

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.acos
```

##### acosh()

```ts
acosh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:927

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.acosh
```

##### asin()

```ts
asin(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:928

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.asin
```

##### asinh()

```ts
asinh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:929

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.asinh
```

##### atan()

```ts
atan(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:930

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.atan
```

##### atanh()

```ts
atanh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:931

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.atanh
```

##### cbrt()

```ts
cbrt(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:932

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.cbrt
```

##### ceil()

```ts
ceil(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:933

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.ceil
```

##### clz32()

```ts
clz32(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:934

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.clz32
```

##### cos()

```ts
cos(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:935

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.cos
```

##### cosh()

```ts
cosh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:936

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.cosh
```

##### exp()

```ts
exp(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:937

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.exp
```

##### expm1()

```ts
expm1(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:938

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.expm1
```

##### floor()

```ts
floor(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:939

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.floor
```

##### fround()

```ts
fround(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:940

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.fround
```

##### log()

```ts
log(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:941

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.log
```

##### log1p()

```ts
log1p(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:942

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.log1p
```

##### log10()

```ts
log10(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:943

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.log10
```

##### log2()

```ts
log2(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:944

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.log2
```

##### round()

```ts
round(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:945

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.round
```

##### sign()

```ts
sign(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:946

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sign
```

##### sin()

```ts
sin(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:947

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sin
```

##### sinh()

```ts
sinh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:948

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sinh
```

##### sqrt()

```ts
sqrt(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:949

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.sqrt
```

##### tan()

```ts
tan(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:950

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.tan
```

##### tanh()

```ts
tanh(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:951

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.tanh
```

##### trunc()

```ts
trunc(): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:952

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.trunc
```

##### not()

```ts
static not(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:954

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.not
```

##### abs()

```ts
static abs(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:955

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.abs
```

##### acos()

```ts
static acos(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:956

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.acos
```

##### acosh()

```ts
static acosh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:957

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.acosh
```

##### asin()

```ts
static asin(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:958

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.asin
```

##### asinh()

```ts
static asinh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:959

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.asinh
```

##### atan()

```ts
static atan(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:960

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.atan
```

##### atanh()

```ts
static atanh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:961

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.atanh
```

##### cbrt()

```ts
static cbrt(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:962

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.cbrt
```

##### ceil()

```ts
static ceil(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:963

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.ceil
```

##### clz32()

```ts
static clz32(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:964

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.clz32
```

##### cos()

```ts
static cos(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:965

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.cos
```

##### cosh()

```ts
static cosh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:966

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.cosh
```

##### exp()

```ts
static exp(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:967

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.exp
```

##### expm1()

```ts
static expm1(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:968

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.expm1
```

##### floor()

```ts
static floor(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:969

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.floor
```

##### fround()

```ts
static fround(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:970

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.fround
```

##### log()

```ts
static log(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:971

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.log
```

##### log1p()

```ts
static log1p(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:972

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.log1p
```

##### log10()

```ts
static log10(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:973

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.log10
```

##### log2()

```ts
static log2(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:974

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.log2
```

##### round()

```ts
static round(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:975

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.round
```

##### sign()

```ts
static sign(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:976

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.sign
```

##### sin()

```ts
static sin(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:977

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.sin
```

##### sinh()

```ts
static sinh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:978

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.sinh
```

##### sqrt()

```ts
static sqrt(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:979

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.sqrt
```

##### tan()

```ts
static tan(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:980

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.tan
```

##### tanh()

```ts
static tanh(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:981

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.tanh
```

##### trunc()

```ts
static trunc(value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:982

###### Parameters

###### value

`MaybeMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.trunc
```

##### pow()

```ts
pow(value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:986

###### Parameters

###### value

`ScalarOrMatrix`

###### Returns

`this`

###### Inherited from

```ts
AbstractMatrix.pow
```

##### pow()

```ts
static pow(matrix, value): Matrix;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:988

###### Parameters

###### matrix

`MaybeMatrix`

###### value

`ScalarOrMatrix`

###### Returns

[`Matrix`](#matrix)

###### Inherited from

```ts
AbstractMatrix.pow
```

##### set()

```ts
set(
   rowIndex, 
   columnIndex, 
   value): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:996

Sets a given element of the matrix.

###### Parameters

###### rowIndex

`number`

Index of the element's row.

###### columnIndex

`number`

Index of the element's column.

###### value

`number`

The new value for the element.

###### Returns

`this`

###### Overrides

```ts
AbstractMatrix.set
```

##### get()

```ts
get(rowIndex, columnIndex): number;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:997

Returns the value of the given element of the matrix.

###### Parameters

###### rowIndex

`number`

Index of the element's row.

###### columnIndex

`number`

Index of the element's column.

###### Returns

`number`

- The value of the element.

###### Overrides

```ts
AbstractMatrix.get
```

##### removeColumn()

```ts
removeColumn(index): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1003

Removes a column from the matrix (in place).

###### Parameters

###### index

`number`

Column index.

###### Returns

`this`

##### removeRow()

```ts
removeRow(index): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1009

Removes a row from the matrix (in place).

###### Parameters

###### index

`number`

Row index.

###### Returns

`this`

##### addColumn()

###### Call Signature

```ts
addColumn(index, array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1016

Adds a new column to the matrix (in place).

###### Parameters

###### index

`number`

Column index. Default: `this.columns`.

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Column to add.

###### Returns

`this`

###### Call Signature

```ts
addColumn(array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1017

Adds a new column to the matrix (in place).

###### Parameters

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Column to add.

###### Returns

`this`

##### addRow()

###### Call Signature

```ts
addRow(index, array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1024

Adds a new row to the matrix (in place).

###### Parameters

###### index

`number`

Row index. Default: `this.rows`.

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Row to add.

###### Returns

`this`

###### Call Signature

```ts
addRow(array): this;
```

Defined in: node\_modules/ml-matrix/matrix.d.ts:1025

Adds a new row to the matrix (in place).

###### Parameters

###### array

`AbstractMatrix` \| `ArrayLike`\<`number`\>

Row to add.

###### Returns

`this`

## Functions

### toMatrix()

```ts
function toMatrix(data): Matrix;
```

Defined in: [src/core/linalg.js:13](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L13)

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

Defined in: [src/core/linalg.js:26](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L26)

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

Defined in: [src/core/linalg.js:51](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L51)

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

Defined in: [src/core/linalg.js:74](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L74)

Singular Value Decomposition

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

Defined in: [src/core/linalg.js:89](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L89)

Eigenvalue decomposition

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Square matrix

#### Returns

`Object`

{values, vectors}

***

### mmul()

```ts
function mmul(A, B): Matrix;
```

Defined in: [src/core/linalg.js:104](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L104)

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

Defined in: [src/core/linalg.js:113](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L113)

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

Defined in: [src/core/linalg.js:122](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L122)

Matrix inverse

#### Parameters

##### data

`number`[][] \| [`Matrix`](#matrix)

Square matrix

#### Returns

[`Matrix`](#matrix)

Inverse matrix

***

### pseudoInverse()

```ts
function pseudoInverse(data): Matrix;
```

Defined in: [src/core/linalg.js:128](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/linalg.js#L128)

#### Parameters

##### data

`any`

#### Returns

[`Matrix`](#matrix)
