---
layout: default
title: Core Utilities
parent: API Reference
nav_order: 5
has_children: true
permalink: /api/core
---
# core

## Namespaces

- [linalg](/api/core/linalg)
- [math](/api/core/math)
- [optimize](/api/core/optimize)
- [persistence](/api/core/persistence)
- [spatial](/api/core/spatial)
- [table](/api/core/table)

## Functions

### parseFormula()

```ts
function parseFormula(formula, _data?): Object;
```

Defined in: [src/core/formula.js:19](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/formula.js#L19)

Parse an R-style formula string

#### Parameters

##### formula

`string`

Formula string like 'y ~ x1 + x2'

##### \_data?

`null` = `null`

#### Returns

`Object`

Parsed formula specification

***

### applyFormula()

```ts
function applyFormula(
   formula, 
   data, 
   options?): Object;
```

Defined in: [src/core/formula.js:415](https://github.com/tangent-to/ds/blob/fd643fde6ff506706e9da35a1f25e91a0026b50f/src/core/formula.js#L415)

Apply formula to data to extract design matrix and response

#### Parameters

##### formula

`string`

Formula string

##### data

`Object`[]

Data array

##### options?

`Object` = `{}`

Additional options

#### Returns

`Object`

{X, y, groups, columnNames}
