---
layout: default
title: math
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/math
---
# math

## Variables

### EPSILON

```ts
const EPSILON: 1e-10 = 1e-10;
```

Defined in: [src/core/math.js:6](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L6)

Core mathematical utilities and constants

***

### PI

```ts
const PI: number = Math.PI;
```

Defined in: [src/core/math.js:7](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L7)

***

### E

```ts
const E: number = Math.E;
```

Defined in: [src/core/math.js:8](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L8)

***

### std

```ts
const std: (arr, sample, options?) => number = stddev;
```

Defined in: [src/core/math.js:152](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L152)

Alias for stddev (standard deviation)

Standard deviation of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### sample?

`boolean` = `true`

If true, use sample variance (n-1)

##### options?

`Object` = `{}`

{ naOmit?: boolean }

#### Returns

`number`

Standard deviation

#### Param

**arr**

Array of numbers

#### Param

**sample**

If true, use sample variance (n-1)

#### Param

**options**

{ naOmit?: boolean }

#### Returns

Standard deviation

***

### sd

```ts
const sd: (arr, sample, options?) => number = stddev;
```

Defined in: [src/core/math.js:161](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L161)

Alias for stddev (standard deviation) - R-style naming

Standard deviation of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### sample?

`boolean` = `true`

If true, use sample variance (n-1)

##### options?

`Object` = `{}`

{ naOmit?: boolean }

#### Returns

`number`

Standard deviation

#### Param

**arr**

Array of numbers

#### Param

**sample**

If true, use sample variance (n-1)

#### Param

**options**

{ naOmit?: boolean }

#### Returns

Standard deviation

## Functions

### approxEqual()

```ts
function approxEqual(
   a, 
   b, 
   tolerance?): boolean;
```

Defined in: [src/core/math.js:17](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L17)

Approximate equality comparison for floating point numbers

#### Parameters

##### a

`number`

First number

##### b

`number`

Second number

##### tolerance?

`number` = `EPSILON`

Tolerance for comparison

#### Returns

`boolean`

True if approximately equal

***

### guardFinite()

```ts
function guardFinite(value, name?): number;
```

Defined in: [src/core/math.js:28](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L28)

Guard against non-finite values

#### Parameters

##### value

`number`

Value to check

##### name?

`string` = `'value'`

Name for error message

#### Returns

`number`

The value if valid

#### Throws

If value is not finite

***

### guardPositive()

```ts
function guardPositive(value, name?): number;
```

Defined in: [src/core/math.js:42](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L42)

Guard against negative values

#### Parameters

##### value

`number`

Value to check

##### name?

`string` = `'value'`

Name for error message

#### Returns

`number`

The value if valid

#### Throws

If value is negative

***

### guardProbability()

```ts
function guardProbability(value, name?): number;
```

Defined in: [src/core/math.js:56](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L56)

Guard against values outside [0, 1]

#### Parameters

##### value

`number`

Value to check

##### name?

`string` = `'value'`

Name for error message

#### Returns

`number`

The value if valid

#### Throws

If value is outside [0, 1]

***

### sum()

```ts
function sum(arr, options?): number;
```

Defined in: [src/core/math.js:68](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L68)

Sum of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### options?

#### Returns

`number`

Sum

***

### mean()

```ts
function mean(arr, options?): number;
```

Defined in: [src/core/math.js:85](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L85)

Mean of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### options?

`Object` = `{}`

{ naOmit?: boolean }

#### Returns

`number`

Mean

***

### variance()

```ts
function variance(
   arr, 
   sample?, 
   options?): number;
```

Defined in: [src/core/math.js:104](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L104)

Variance of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### sample?

`boolean` = `true`

If true, use sample variance (n-1)

##### options?

`Object` = `{}`

{ naOmit?: boolean }

#### Returns

`number`

Variance

***

### stddev()

```ts
function stddev(
   arr, 
   sample?, 
   options?): number;
```

Defined in: [src/core/math.js:141](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L141)

Standard deviation of array

#### Parameters

##### arr

`number`[]

Array of numbers

##### sample?

`boolean` = `true`

If true, use sample variance (n-1)

##### options?

`Object` = `{}`

{ naOmit?: boolean }

#### Returns

`number`

Standard deviation

***

### quantile()

```ts
function quantile(
   arr, 
   p, 
   options?): any;
```

Defined in: [src/core/math.js:196](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L196)

#### Parameters

##### arr

`any`

##### p

`any`

##### options?

#### Returns

`any`

***

### median()

```ts
function median(arr, options?): any;
```

Defined in: [src/core/math.js:227](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L227)

#### Parameters

##### arr

`any`

##### options?

#### Returns

`any`

***

### percentile()

```ts
function percentile(
   arr, 
   value, 
   options?): number;
```

Defined in: [src/core/math.js:231](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L231)

#### Parameters

##### arr

`any`

##### value

`any`

##### options?

#### Returns

`number`

***

### summaryQuantiles()

```ts
function summaryQuantiles(
   arr, 
   probs?, 
   options?): object;
```

Defined in: [src/core/math.js:242](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L242)

#### Parameters

##### arr

`any`

##### probs?

`number`[] = `...`

##### options?

#### Returns

`object`

***

### min()

```ts
function min(arr, options?): number;
```

Defined in: [src/core/math.js:256](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L256)

Compute minimum value of an array

#### Parameters

##### arr

`number`[]

Array of numbers

##### options?

`Object` = `{}`

Options { naOmit: boolean }

#### Returns

`number`

Minimum value or NaN if empty

***

### max()

```ts
function max(arr, options?): number;
```

Defined in: [src/core/math.js:268](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L268)

Compute maximum value of an array

#### Parameters

##### arr

`number`[]

Array of numbers

##### options?

`Object` = `{}`

Options { naOmit: boolean }

#### Returns

`number`

Maximum value or NaN if empty

***

### range()

```ts
function range(
   start, 
   stop, 
   step?): number[];
```

Defined in: [src/core/math.js:286](https://github.com/tangent-to/ds/blob/2e2217c296d90f7a4afa5d8795d35e61ab8d6294/src/core/math.js#L286)

Generate a sequence of numbers

#### Parameters

##### start

`number`

Start value (inclusive)

##### stop

`number`

Stop value (inclusive)

##### step?

`number` = `1`

Step size (default: 1)

#### Returns

`number`[]

Array of evenly spaced numbers

#### Example

```ts
range(0, 10, 2) // [0, 2, 4, 6, 8, 10]
range(1, 5)     // [1, 2, 3, 4, 5]
range(1, 2, 0.5) // [1, 1.5, 2]
```
