---
layout: default
title: persistence
parent: Core Utilities
grand_parent: API Reference
permalink: /api/core/persistence
---
# persistence

## Functions

### saveModel()

```ts
function saveModel(model): string;
```

Defined in: [src/core/persistence.js:11](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L11)

Save model to JSON

#### Parameters

##### model

`Object`

Model object with toJSON method or estimator instance

#### Returns

`string`

JSON string representation of the model

***

### loadModel()

```ts
function loadModel(json): Object;
```

Defined in: [src/core/persistence.js:44](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L44)

Load model from JSON

#### Parameters

##### json

`string`

JSON string representation

#### Returns

`Object`

Reconstructed model object

***

### addSerializationSupport()

```ts
function addSerializationSupport(EstimatorClass, toJSONFn): void;
```

Defined in: [src/core/persistence.js:113](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L113)

Add toJSON method to an estimator class prototype
This allows models to define their own serialization logic

#### Parameters

##### EstimatorClass

`Function`

Estimator class

##### toJSONFn

`Function`

Custom toJSON function

#### Returns

`void`

***

### serializeValue()

```ts
function serializeValue(value): any;
```

Defined in: [src/core/persistence.js:123](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L123)

Serialize model to file-safe object
Handles special types like undefined, Infinity, NaN

#### Parameters

##### value

`any`

Value to serialize

#### Returns

`any`

Serializable value

***

### deserializeValue()

```ts
function deserializeValue(value): any;
```

Defined in: [src/core/persistence.js:154](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L154)

Deserialize value from file-safe object

#### Parameters

##### value

`any`

Value to deserialize

#### Returns

`any`

Deserialized value

***

### makeSaveable()

```ts
function makeSaveable(model): Object;
```

Defined in: [src/core/persistence.js:186](https://github.com/tangent-to/ds/blob/edabdef9ecba7d49f301b52f886c73af8ca457ed/src/core/persistence.js#L186)

Create a saveable model wrapper
Adds save() method to any model object

#### Parameters

##### model

`Object`

Model object

#### Returns

`Object`

Model with save() method
