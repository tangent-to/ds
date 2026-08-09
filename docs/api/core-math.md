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

Defined in: [src/core/math.js:6](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L6)

Core mathematical utilities and constants

***

### PI

```ts
const PI: number = Math.PI;
```

Defined in: [src/core/math.js:7](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L7)

***

### E

```ts
const E: number = Math.E;
```

Defined in: [src/core/math.js:8](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L8)

***

### std

```ts
const std: (arr, sample, options?) => number = stddev;
```

Defined in: [src/core/math.js:152](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L152)

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

Defined in: [src/core/math.js:161](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L161)

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

Defined in: [src/core/math.js:17](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L17)

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

Defined in: [src/core/math.js:28](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L28)

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

Defined in: [src/core/math.js:42](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L42)

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

Defined in: [src/core/math.js:56](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L56)

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

Defined in: [src/core/math.js:68](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L68)

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

Defined in: [src/core/math.js:85](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L85)

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

Defined in: [src/core/math.js:104](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L104)

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

Defined in: [src/core/math.js:141](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L141)

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
   options?): number | number[];
```

Defined in: [src/core/math.js:205](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L205)

Compute the quantile(s) of an array

#### Parameters

##### arr

`number`[]

Array of numbers

##### p

`number` \| `number`[]

Probability in [0, 1], or an array of probabilities

##### options?

Options

###### naOmit?

`boolean`

Omit non-finite values instead of throwing

###### method?

`string`

Interpolation method ('linear' or nearest)

#### Returns

`number` \| `number`[]

Quantile value, or array of quantiles if p is an array

***

### median()

```ts
function median(arr, options?): number;
```

Defined in: [src/core/math.js:244](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L244)

Compute the median of an array

#### Parameters

##### arr

`number`[]

Array of numbers

##### options?

Options

###### naOmit?

`boolean`

Omit non-finite values instead of throwing

###### method?

`string`

Interpolation method ('linear' or nearest)

#### Returns

`number`

Median value, or NaN if empty

***

### percentile()

```ts
function percentile(
   arr, 
   value, 
   options?): number;
```

Defined in: [src/core/math.js:256](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L256)

Compute the proportion of values less than or equal to a given value

#### Parameters

##### arr

`number`[]

Array of numbers

##### value

`number`

Threshold value

##### options?

Options

###### naOmit?

`boolean`

Omit non-finite values instead of throwing

#### Returns

`number`

Proportion in [0, 1], or NaN if empty

***

### summaryQuantiles()

```ts
function summaryQuantiles(
   arr, 
   probs?, 
   options?): Object;
```

Defined in: [src/core/math.js:276](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L276)

Compute a set of quantiles keyed by probability

#### Parameters

##### arr

`number`[]

Array of numbers

##### probs?

`number`[] = `...`

Probabilities in [0, 1] to compute

##### options?

Options

###### naOmit?

`boolean`

Omit non-finite values instead of throwing

###### method?

`string`

Interpolation method ('linear' or nearest)

#### Returns

`Object`

Object mapping each probability to its quantile value

***

### min()

```ts
function min(arr, options?): number;
```

Defined in: [src/core/math.js:290](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L290)

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

Defined in: [src/core/math.js:302](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L302)

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

Defined in: [src/core/math.js:320](https://github.com/tangent-to/ds/blob/906004976edc5a867a581f4e234a37a94ce2f592/src/core/math.js#L320)

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
