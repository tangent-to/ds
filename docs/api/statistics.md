---
layout: default
title: Statistics
parent: API Reference
nav_order: 1
permalink: /api/statistics
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
- Hypothesis testing (t-tests, ANOVA, chi-square)
- Generalized Linear Models (GLM)
- Mixed-effects models (GLMM)
- Probability distributions
- Model comparison and selection

---

## Distributions

### normal

Normal (Gaussian) distribution functions.

```javascript
ds.stats.normal.pdf(x, { mean, sd })
ds.stats.normal.cdf(x, { mean, sd })
ds.stats.normal.quantile(p, { mean, sd })
```

**Parameters:**
- `x` (number): Value to evaluate
- `mean` (number): Mean (default: 0)
- `sd` (number): Standard deviation (default: 1)

**Example:**
```javascript
ds.stats.normal.pdf(0, { mean: 0, sd: 1 });
ds.stats.normal.cdf(1.96, { mean: 0, sd: 1 }); // ~0.975
ds.stats.normal.quantile(0.975, { mean: 0, sd: 1 }); // ~1.96
```

---

### uniform

Uniform distribution functions.

```javascript
ds.stats.uniform.pdf(x, { min, max })
ds.stats.uniform.cdf(x, { min, max })
ds.stats.uniform.quantile(p, { min, max })
```

---

### gamma

Gamma distribution functions.

```javascript
ds.stats.gamma.pdf(x, { shape, scale })
ds.stats.gamma.cdf(x, { shape, scale })
ds.stats.gamma.quantile(p, { shape, scale })
```

---

### beta

Beta distribution functions.

```javascript
ds.stats.beta.pdf(x, { alpha, beta })
ds.stats.beta.cdf(x, { alpha, beta })
ds.stats.beta.quantile(p, { alpha, beta })
```

---

## Hypothesis Testing

### Functional API

Convenience functions that return result objects directly.

#### oneSampleTTest

```javascript
ds.stats.hypothesis.oneSampleTTest(data, { mu0 })
```

**Parameters:**
- `data` (Array&lt;number&gt;): Sample data
- `mu0` (number): Hypothesized mean (default: 0)

**Returns:** Object
```javascript
{
  statistic: number,     // t-statistic
  pValue: number,        // p-value
  df: number,            // degrees of freedom
  mean: number,          // sample mean
  alternative: string
}
```

---

#### twoSampleTTest

```javascript
ds.stats.hypothesis.twoSampleTTest(sample1, sample2, options)
```

**Parameters:**
- `sample1` (Array&lt;number&gt;): First sample
- `sample2` (Array&lt;number&gt;): Second sample
- `options` (Object, optional):
  - `alternative` (string): `'two-sided'`, `'less'`, or `'greater'` (default: `'two-sided'`)
  - `mu` (number): Hypothesized difference in means (default: 0)

**Returns:** Object
```javascript
{
  statistic: number,
  pValue: number,
  df: number,
  mean1: number,
  mean2: number,
  pooledSE: number,
  alternative: string
}
```

**Example:**
```javascript
const groupA = [5.1, 4.9, 4.7, 4.6, 5.0];
const groupB = [7.0, 6.4, 6.9, 6.5, 6.3];

const result = ds.stats.hypothesis.twoSampleTTest(groupA, groupB);
console.log(`t = ${result.statistic}, p = ${result.pValue}`);
```

---

#### chiSquareTest

```javascript
ds.stats.hypothesis.chiSquareTest(observed, expected)
```

**Parameters:**
- `observed` (Array&lt;number&gt;): Observed frequencies
- `expected` (Array&lt;number&gt;): Expected frequencies

**Returns:** Object with `statistic`, `pValue`, `df`

---

#### oneWayAnova

```javascript
ds.stats.hypothesis.oneWayAnova(groups)
```

**Parameters:**
- `groups` (Array&lt;Array&lt;number&gt;&gt;): Array of groups to compare

**Returns:** Object
```javascript
{
  statistic: number,     // F-statistic
  pValue: number,
  dfBetween: number,
  dfWithin: number,
  ssBetween: number,
  ssWithin: number
}
```

**Example:**
```javascript
const groupA = [5.1, 4.9, 4.7];
const groupB = [6.0, 5.8, 6.2];
const groupC = [7.0, 6.9, 7.1];

const result = ds.stats.hypothesis.oneWayAnova([groupA, groupB, groupC]);
console.log(`F = ${result.statistic}, p = ${result.pValue}`);
```

---

### Class-based API

Estimator-style classes that follow the fit/predict pattern.

#### OneSampleTTest

```javascript
new ds.stats.OneSampleTTest(options)
```

**Methods:**
- `.fit(data)` - Perform the test
- `.summary()` - Formatted summary

#### TwoSampleTTest

```javascript
new ds.stats.TwoSampleTTest(options)
```

#### ChiSquareTest

```javascript
new ds.stats.ChiSquareTest()
```

#### OneWayAnova

```javascript
new ds.stats.OneWayAnova()
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
  family: string,          // 'gaussian', 'binomial', 'poisson', 'gamma'
  link: string,            // 'identity', 'log', 'logit', 'probit', etc.
  randomEffects: Object,   // For mixed-effects models (optional)
  multiclass: string       // 'ovr' (one-vs-rest) for multiclass (optional)
}
```

**Common Configurations:**

| Model Type | Family | Link | Use For |
|------------|--------|------|---------|
| Linear Regression | `gaussian` | `identity` | Continuous outcomes |
| Logistic Regression | `binomial` | `logit` | Binary outcomes (0/1) |
| Poisson Regression | `poisson` | `log` | Count data |
| Gamma Regression | `gamma` | `inverse` | Positive continuous, skewed |

#### Methods

##### `.fit()`

Fit the model to data. Supports three input styles:

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
- Model fit statistics (AIC, BIC, R-squared)

#### Properties

```javascript
model.coefficients   // { '(Intercept)': 2.5, 'feature1': 0.8, ... }
model.fitted         // [2.1, 3.4, 2.8, ...]
model.residuals      // [-0.1, 0.2, -0.05, ...]
model.deviance       // Model deviance
model.nullDeviance   // Null deviance
model.aic            // Akaike Information Criterion
model.bic            // Bayesian Information Criterion
model.pseudoR2       // Pseudo R-squared (for GLMs)
```

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
  y: 'is_adelie',
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
  multiclass: 'ovr'
});

multiclass.fit({
  X: ['feature1', 'feature2'],
  y: 'species',
  data: myData
});
```

**Mixed-Effects Model (GLMM):**

```javascript
const lme = new ds.stats.GLM({
  family: 'gaussian',
  randomEffects: {
    intercept: data.map(d => d.site)
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

// Transformations
formula: 'y ~ log(x1) + sqrt(x2)'

// Polynomials
formula: 'y ~ poly(x, 3)'

// Mixed effects
formula: 'y ~ x1 + (1 | group)'
formula: 'y ~ x1 + (1 + time | subject)'
```

---

## Model Comparison

### compareModels

Compare multiple models with AIC/BIC.

```javascript
const comparison = ds.stats.compareModels([model1, model2, model3])
```

### likelihoodRatioTest

Likelihood ratio test for nested models.

```javascript
const lrt = ds.stats.likelihoodRatioTest(model1, model2)
```

### pairwiseLRT

Pairwise likelihood ratio tests across all model pairs.

```javascript
const results = ds.stats.pairwiseLRT([model1, model2, model3])
```

### crossValidate / crossValidateModels

Cross-validate one or more models.

```javascript
ds.stats.crossValidate(model, data, options)
ds.stats.crossValidateModels([model1, model2], data, options)
```

---

## Common Patterns

### Test for Group Differences

```javascript
// t-test for 2 groups
const result = ds.stats.hypothesis.twoSampleTTest(groupA, groupB);

// ANOVA for 3+ groups
const result = ds.stats.hypothesis.oneWayAnova([groupA, groupB, groupC]);
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

- [Machine Learning API](machine-learning) - Supervised learning models
- [Visualization API](visualization) - Diagnostic plots for GLMs
