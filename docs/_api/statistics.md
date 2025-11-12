---
layout: default
title: Statistics
parent: API Reference
nav_order: 1
---

# Statistics API
{: .no_toc }

Statistical analysis functions and models.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The `ds.stats` module provides statistical functions for:
- Descriptive statistics (mean, median, standard deviation)
- Hypothesis testing (t-tests, ANOVA)
- Generalized Linear Models (GLM)
- Mixed-effects models

---

## Basic Statistics

### mean

Calculate the arithmetic mean.

```javascript
ds.stats.mean(values)
```

**Parameters:**
- `values` (Array<number>): Array of numeric values

**Returns:** number

**Example:**
```javascript
const avg = ds.stats.mean([1, 2, 3, 4, 5]);
// 3
```

---

### median

Calculate the median value.

```javascript
ds.stats.median(values)
```

**Parameters:**
- `values` (Array<number>): Array of numeric values

**Returns:** number

**Example:**
```javascript
const mid = ds.stats.median([1, 2, 3, 4, 5]);
// 3
```

---

### stddev

Calculate the standard deviation.

```javascript
ds.stats.stddev(values, options)
```

**Parameters:**
- `values` (Array<number>): Array of numeric values
- `options` (Object, optional):
  - `ddof` (number): Delta degrees of freedom (default: 1)

**Returns:** number

**Example:**
```javascript
const sd = ds.stats.stddev([1, 2, 3, 4, 5]);
// Sample standard deviation
```

---

### quantile

Calculate quantiles/percentiles.

```javascript
ds.stats.quantile(values, q)
```

**Parameters:**
- `values` (Array<number>): Array of numeric values
- `q` (number | Array<number>): Quantile(s) between 0 and 1

**Returns:** number | Array<number>

**Example:**
```javascript
// Single quantile
const q25 = ds.stats.quantile(data, 0.25);

// Multiple quantiles
const [q25, q50, q75] = ds.stats.quantile(data, [0.25, 0.5, 0.75]);
```

---

## Hypothesis Testing

### ttest

Perform Student's t-test for two independent samples.

```javascript
ds.stats.ttest(sample1, sample2, options)
```

**Parameters:**
- `sample1` (Array<number>): First sample
- `sample2` (Array<number>): Second sample
- `options` (Object, optional):
  - `alternative` (string): 'two-sided', 'less', or 'greater' (default: 'two-sided')
  - `mu` (number): Hypothesized difference in means (default: 0)

**Returns:** Object
```javascript
{
  statistic: number,     // t-statistic
  pValue: number,        // p-value
  degreesOfFreedom: number,
  mean1: number,         // Mean of sample1
  mean2: number,         // Mean of sample2
  alternative: string
}
```

**Example:**
```javascript
const groupA = [5.1, 4.9, 4.7, 4.6, 5.0];
const groupB = [7.0, 6.4, 6.9, 6.5, 6.3];

const result = ds.stats.ttest(groupA, groupB);
console.log(`t = ${result.statistic}, p = ${result.pValue}`);
// If p < 0.05, reject null hypothesis (means are different)
```

---

### anova

Perform one-way ANOVA.

```javascript
ds.stats.anova(groups)
```

**Parameters:**
- `groups` (Array<Array<number>>): Array of groups to compare

**Returns:** Object
```javascript
{
  statistic: number,     // F-statistic
  pValue: number,        // p-value
  dfBetween: number,     // Degrees of freedom between groups
  dfWithin: number,      // Degrees of freedom within groups
  ssBetween: number,     // Sum of squares between groups
  ssWithin: number       // Sum of squares within groups
}
```

**Example:**
```javascript
const groupA = [5.1, 4.9, 4.7];
const groupB = [6.0, 5.8, 6.2];
const groupC = [7.0, 6.9, 7.1];

const result = ds.stats.anova([groupA, groupB, groupC]);
console.log(`F = ${result.statistic}, p = ${result.pValue}`);
```

---

## Generalized Linear Models

### GLM

Fit generalized linear models including linear regression, logistic regression, and more.

```javascript
new ds.stats.GLM(options)
```

#### Constructor Options

```javascript
{
  family: string,          // 'gaussian', 'binomial', 'poisson'
  link: string,           // 'identity', 'log', 'logit', 'probit', etc.
  randomEffects: Object,  // For mixed-effects models (optional)
  multiclass: string      // 'ovr' (one-vs-rest) for multiclass (optional)
}
```

**Common Configurations:**

| Model Type | Family | Link | Use For |
|------------|--------|------|---------|
| Linear Regression | `gaussian` | `identity` | Continuous outcomes |
| Logistic Regression | `binomial` | `logit` | Binary outcomes (0/1) |
| Poisson Regression | `poisson` | `log` | Count data |

#### Methods

##### `.fit()`

Fit the model to data.

**Array API:**
```javascript
model.fit(X, y)
```

**Table API:**
```javascript
model.fit({
  data: myData,
  X: ['feature1', 'feature2'],
  y: 'outcome'
})
```

**Formula API:**
```javascript
model.fit({
  formula: 'outcome ~ feature1 + feature2',
  data: myData
})
```

##### `.predict()`

Make predictions.

**Array API:**
```javascript
const predictions = model.predict(XNew)
```

**Table API:**
```javascript
const predictions = model.predict({
  data: newData,
  X: ['feature1', 'feature2']
})
```

##### `.summary()`

Get model summary statistics.

```javascript
const summary = model.summary()
console.log(summary)
```

Returns formatted string with:
- Coefficient estimates
- Standard errors
- z-values
- Confidence intervals
- Model fit statistics (AIC, BIC, R²)

#### Examples

**Linear Regression:**

```javascript
const lm = new ds.stats.GLM({ family: 'gaussian' });

lm.fit({
  X: ['height', 'weight'],
  y: 'blood_pressure',
  data: healthData
});

console.log(lm.summary());

const predictions = lm.predict({
  data: newPatients,
  X: ['height', 'weight']
});
```

**Logistic Regression:**

```javascript
const logit = new ds.stats.GLM({ family: 'binomial', link: 'logit' });

logit.fit({
  X: ['beak_length', 'flipper_length'],
  y: 'is_adelie',  // Binary: 0 or 1
  data: penguins
});

const probabilities = logit.predict({
  data: newPenguins,
  X: ['beak_length', 'flipper_length']
});
```

**Multiclass Logistic Regression:**

```javascript
const multiclass = new ds.stats.GLM({
  family: 'binomial',
  multiclass: 'ovr'  // One-vs-rest strategy
});

multiclass.fit({
  X: ['feature1', 'feature2'],
  y: 'species',  // 3+ categories
  data: myData
});

// Fits 3 binary models (one per class)
// Predicts using the model with highest probability
```

**Mixed-Effects Model:**

```javascript
const lme = new ds.stats.GLM({
  family: 'gaussian',
  randomEffects: {
    intercept: data.map(d => d.site)  // Random intercepts by site
  }
});

lme.fit({
  X: ['treatment', 'age'],
  y: 'response',
  data: nestedData
});

console.log(lme.summary());
// Shows fixed effects and random effects variance
```

---

## Formula Syntax

GLM supports R-style formula syntax:

```javascript
// Simple linear model
formula: 'y ~ x'

// Multiple predictors
formula: 'y ~ x1 + x2 + x3'

// Interactions
formula: 'y ~ x1 * x2'  // x1 + x2 + x1:x2

// Categorical variables (auto-encoded)
formula: 'price ~ carat + cut + color'
```

---

## Model Diagnostics

### Coefficients

```javascript
model.coefficients
// { '(Intercept)': 2.5, 'feature1': 0.8, 'feature2': -0.3 }
```

### Fitted Values

```javascript
model.fitted
// [2.1, 3.4, 2.8, ...]
```

### Residuals

```javascript
model.residuals
// [-0.1, 0.2, -0.05, ...]
```

### Model Metrics

```javascript
model.deviance        // Model deviance
model.nullDeviance    // Null deviance
model.aic             // Akaike Information Criterion
model.bic             // Bayesian Information Criterion
model.pseudoR2        // Pseudo R² (for GLMs)
```

---

## Common Patterns

### Test for Group Differences

```javascript
// t-test for 2 groups
const result = ds.stats.ttest(groupA, groupB);

// ANOVA for 3+ groups
const result = ds.stats.anova([groupA, groupB, groupC]);
```

### Predict Continuous Outcome

```javascript
const model = new ds.stats.GLM({ family: 'gaussian' });
model.fit({ X: features, y: 'price', data: myData });
const predictions = model.predict({ data: newData, X: features });
```

### Predict Binary Outcome

```javascript
const model = new ds.stats.GLM({ family: 'binomial' });
model.fit({ X: features, y: 'is_fraud', data: transactions });
const probabilities = model.predict({ data: newTransactions, X: features });
```

### Account for Hierarchical Data

```javascript
const model = new ds.stats.GLM({
  family: 'gaussian',
  randomEffects: { intercept: data.map(d => d.groupId) }
});
model.fit({ X: features, y: 'outcome', data: nestedData });
```

---

## See Also

- [Tutorial: Statistical Analysis](../tutorials/02-statistics)
- [Examples](../examples#statistical-analysis)
- [GLM on Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model)
