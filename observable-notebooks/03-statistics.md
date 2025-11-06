# Statistical Modeling with tangent-ds

*From simple t-tests to advanced mixed models*

---

## What is Statistical Modeling?

Statistical models help us **understand relationships** between variables and **make predictions**. We ask questions like:
- Does treatment A work better than treatment B?
- How does exercise affect health?
- Can we predict sales from advertising spend?

---

## Setup

```js
import * as ds from "@tangent.to/ds"
import * as Plot from "@observablehq/plot"
import * as vega from "vega-datasets"
```

---

## 📊 Hypothesis Testing Basics

Before building models, let's start with simple comparisons.

### One-Sample t-Test

**Question**: Is the average different from a specific value?

```js
// Example: Are cars from the 1970s fuel-efficient (>20 mpg average)?
carsData = vega.data.cars.json().filter(d => d.Miles_per_Gallon != null)
mpgValues = carsData.map(d => d.Miles_per_Gallon)
```

```js
ttest = new ds.stats.oneSampleTTest({mu: 20})
ttest.fit(mpgValues)
```

```js
md`
**Results**:
- Sample mean: ${ttest.summary().mean.toFixed(2)} mpg
- t-statistic: ${ttest.summary().tStatistic.toFixed(3)}
- p-value: ${ttest.summary().pValue.toFixed(4)}
- 95% CI: [${ttest.summary().confidenceInterval.lower.toFixed(2)}, ${ttest.summary().confidenceInterval.upper.toFixed(2)}]

**Interpretation**: ${ttest.summary().pValue < 0.05 ? "✅ The mean is significantly different from 20 mpg" : "❌ No significant difference from 20 mpg"}
`
```

### Two-Sample t-Test

**Question**: Are two groups different?

```js
// Compare MPG between American and European cars
americanMPG = carsData.filter(d => d.Origin === "USA").map(d => d.Miles_per_Gallon)
europeanMPG = carsData.filter(d => d.Origin === "Europe").map(d => d.Miles_per_Gallon)
```

```js
twoSampleTest = new ds.stats.twoSampleTTest()
twoSampleTest.fit(americanMPG, europeanMPG)
```

```js
md`
**Comparison**: American vs European cars
- American mean: ${d3.mean(americanMPG).toFixed(2)} mpg
- European mean: ${d3.mean(europeanMPG).toFixed(2)} mpg
- Difference: ${(d3.mean(europeanMPG) - d3.mean(americanMPG)).toFixed(2)} mpg
- p-value: ${twoSampleTest.summary().pValue.toFixed(4)}

**Conclusion**: European cars have ${twoSampleTest.summary().pValue < 0.05 ? "significantly" : "not significantly"} different fuel efficiency
`
```

```js
// Visualize the distributions
Plot.plot({
  width: 700,
  height: 300,
  marginLeft: 60,

  marks: [
    Plot.rectY(
      [...americanMPG.map(v => ({mpg: v, origin: "USA"})),
       ...europeanMPG.map(v => ({mpg: v, origin: "Europe"}))],
      Plot.binX(
        {y: "count"},
        {x: "mpg", fill: "origin", fillOpacity: 0.5, thresholds: 20}
      )
    ),

    Plot.ruleX([d3.mean(americanMPG)], {stroke: "#e15759", strokeWidth: 2}),
    Plot.ruleX([d3.mean(europeanMPG)], {stroke: "#4e79a7", strokeWidth: 2})
  ],

  x: {label: "Miles per Gallon →"},
  y: {label: "↑ Count"},
  color: {legend: true},
  caption: "Thick lines show group means"
})
```

---

## 📈 Linear Models with Formulas

Now let's move to **regression** - modeling relationships between variables. tangent-ds supports **R-style formulas** for intuitive model specification!

### Simple Linear Regression

**Formula notation**: `y ~ x` means "y depends on x"

```js
// Model: How does weight affect fuel efficiency?
model_simple = new ds.stats.GLM({family: "gaussian"})
model_simple.fit("Miles_per_Gallon ~ Weight_in_lbs", carsData)
```

```js
model_simple.summary()
```

The model equation is:

$$\text{MPG} = \beta_0 + \beta_1 \times \text{Weight} + \varepsilon$$

where $\varepsilon \sim N(0, \sigma^2)$

```js
// Visualize the fit
{
  const predictions = carsData.map(d => ({
    weight: d.Weight_in_lbs,
    actual: d.Miles_per_Gallon,
    predicted: model_simple.predict([[d.Weight_in_lbs]])[0]
  }))

  return Plot.plot({
    width: 800,
    height: 500,
    marginLeft: 60,
    grid: true,

    marks: [
      Plot.dot(predictions, {
        x: "weight",
        y: "actual",
        fill: "steelblue",
        fillOpacity: 0.5,
        r: 4,
        tip: true
      }),

      Plot.line(predictions, {
        x: "weight",
        y: "predicted",
        stroke: "red",
        strokeWidth: 3
      })
    ],

    x: {label: "Weight (lbs) →"},
    y: {label: "↑ Miles per Gallon"},
    caption: "Blue = actual data | Red line = model prediction"
  })
}
```

**Interpretation**: Each additional 1000 lbs decreases MPG by approximately ${Math.abs(model_simple.model.coefficients[1] / 1000).toFixed(2)} miles per gallon.

### Multiple Regression

Include multiple predictors with `+`:

```js
model_multi = new ds.stats.GLM({family: "gaussian"})
model_multi.fit("Miles_per_Gallon ~ Weight_in_lbs + Horsepower + Year", carsData)
```

```js
model_multi.summary()
```

**Coefficient interpretation**:
- **Weight**: Holding other variables constant, heavier cars get lower MPG
- **Horsepower**: More power = lower efficiency
- **Year**: Cars got more efficient over time (technology improved!)

### Interactions

Test if the effect of weight *depends on* the number of cylinders using `*`:

```js
model_interaction = new ds.stats.GLM({family: "gaussian"})
model_interaction.fit("Miles_per_Gallon ~ Weight_in_lbs * Cylinders", carsData)
```

The `*` expands to: `Weight + Cylinders + Weight:Cylinders`

---

## 🔍 Model Diagnostics

Good models need **assumption checking**! tangent-ds provides diagnostic plots.

### Residual Plot

Check for **non-linear patterns** and **heteroscedasticity** (non-constant variance):

```js
ds.plot.residualPlot(model_multi, {
  width: 700,
  height: 400
})
```

**What to look for**:
- ✅ Random scatter around zero (good!)
- ❌ Curved pattern (try transformations)
- ❌ Fan shape (variance increases - try log transform)

### QQ Plot

Check if residuals are **normally distributed**:

```js
ds.plot.qqPlot(model_multi, {
  width: 600,
  height: 400
})
```

**What to look for**:
- ✅ Points follow diagonal line (good!)
- ❌ Curved pattern (non-normal residuals)
- ❌ Heavy tails (outliers present)

### Scale-Location Plot

Check **homoscedasticity** (constant variance):

```js
ds.plot.scaleLocationPlot(model_multi, {
  width: 700,
  height: 400
})
```

### Diagnostic Dashboard

Get all four diagnostic plots at once:

```js
{
  const plots = ds.plot.diagnosticDashboard(model_multi, {
    width: 380,
    height: 300
  })

  return html`<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    ${plots.map(spec => Plot.plot(spec)).map(p => html`<div>${p}</div>`)}
  </div>`
}
```

---

## 🎲 Generalized Linear Models (GLM)

Not all outcomes are continuous! GLMs extend linear models to:
- **Binary outcomes** (yes/no, success/failure)
- **Count data** (number of events)
- **Positive continuous** (reaction times, prices)

### Logistic Regression

**Use case**: Predict binary outcomes (0/1, true/false)

Let's predict if a car is fuel-efficient (>25 mpg):

```js
// Create binary outcome
carsWithEfficiency = carsData.map(d => ({
  ...d,
  efficient: d.Miles_per_Gallon > 25 ? 1 : 0
}))
```

```js
logit_model = new ds.stats.GLM({family: "binomial", link: "logit"})
logit_model.fit("efficient ~ Weight_in_lbs + Horsepower", carsWithEfficiency)
```

```js
logit_model.summary()
```

The model uses the **logistic function**:

$$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2)}}$$

```js
// Visualize predictions
{
  const pred_probs = logit_model.predict(
    carsWithEfficiency.map(d => [d.Weight_in_lbs, d.Horsepower])
  )

  const plotData = carsWithEfficiency.map((d, i) => ({
    weight: d.Weight_in_lbs,
    actual: d.efficient,
    predicted: pred_probs[i]
  }))

  return Plot.plot({
    width: 800,
    height: 400,
    marginLeft: 60,

    marks: [
      // Actual values (jittered)
      Plot.dot(plotData, {
        x: "weight",
        y: d => d.actual + (Math.random() - 0.5) * 0.05,
        fill: d => d.actual === 1 ? "green" : "red",
        fillOpacity: 0.3,
        r: 3
      }),

      // Predicted probabilities
      Plot.line(
        plotData.sort((a, b) => a.weight - b.weight),
        {
          x: "weight",
          y: "predicted",
          stroke: "blue",
          strokeWidth: 3
        }
      )
    ],

    x: {label: "Weight (lbs) →"},
    y: {label: "↑ P(Efficient)", domain: [0, 1]},
    caption: "Green = efficient | Red = not efficient | Blue = predicted probability"
  })
}
```

### Poisson Regression

**Use case**: Count data (number of events)

```js
// Simulate count data: number of sales per day based on ad spend
salesData = {
  const n = 100
  return Array.from({length: n}, (_, i) => {
    const adSpend = (i + 1) * 50
    const lambda = Math.exp(2 + 0.005 * adSpend)
    const sales = Math.floor(lambda + (Math.random() - 0.5) * 10)
    return {adSpend, sales: Math.max(0, sales)}
  })
}
```

```js
poisson_model = new ds.stats.GLM({family: "poisson"})
poisson_model.fit("sales ~ adSpend", salesData)
```

The model assumes:

$$\log(\lambda) = \beta_0 + \beta_1 X$$

where $Y \sim \text{Poisson}(\lambda)$

```js
// Visualize
{
  const predictions = salesData.map(d => ({
    adSpend: d.adSpend,
    actual: d.sales,
    predicted: poisson_model.predict([[d.adSpend]])[0]
  }))

  return Plot.plot({
    width: 700,
    height: 400,
    marks: [
      Plot.dot(predictions, {x: "adSpend", y: "actual", fill: "gray", fillOpacity: 0.5}),
      Plot.line(predictions, {x: "adSpend", y: "predicted", stroke: "purple", strokeWidth: 3})
    ],
    x: {label: "Ad Spend ($) →"},
    y: {label: "↑ Sales (count)"}
  })
}
```

---

## 🎯 Advanced: Mixed Effects Models

**Hierarchical data** has grouping (students in schools, measurements over time, etc.). **Mixed effects models** account for correlation within groups.

### Random Intercepts

Different groups have different baselines:

```js
// Simulate student test scores in different schools
studentData = {
  const schools = ["School A", "School B", "School C"]
  const schoolEffects = [0, 5, -3]  // Random intercepts
  const n = 60

  return Array.from({length: n}, (_, i) => {
    const schoolIdx = i % 3
    const studyHours = Math.random() * 10
    const score = 60 + schoolEffects[schoolIdx] + 3 * studyHours + (Math.random() - 0.5) * 10
    return {
      school: schools[schoolIdx],
      studyHours,
      score
    }
  })
}
```

```js
// Random intercept model
mixed_model = new ds.stats.GLM({family: "gaussian"})
mixed_model.fit("score ~ studyHours + (1 | school)", studentData)
```

```js
mixed_model.summary()
```

**Interpretation**:
- **Fixed effect (studyHours)**: Average effect across all schools
- **Random effect (school)**: Each school has its own intercept
- **Variance components**: How much variation is between vs within schools

### Model Comparison

Compare models with different complexities:

```js
// Model 1: No school effect
model1 = new ds.stats.GLM({family: "gaussian"})
model1.fit("score ~ studyHours", studentData)

// Model 2: Fixed school effect
model2 = new ds.stats.GLM({family: "gaussian"})
model2.fit("score ~ studyHours + school", studentData)

// Model 3: Random school effect
model3 = mixed_model
```

```js
comparison = ds.stats.compareModels([model1, model2, model3], {criterion: "aic"})
```

```js
Inputs.table(comparison.table)
```

**How to read**:
- **AIC/BIC**: Lower is better
- **Δ AIC**: Difference from best model
- **AIC Weight**: Relative evidence for each model

```js
// Visualize comparison
Plot.plot(ds.stats.modelSelectionPlot([model1, model2, model3]).criterion)
```

### Likelihood Ratio Test

Test if the complex model is significantly better:

```js
lrt = ds.stats.likelihoodRatioTest(model1, model2)
```

```js
md`
${lrt.summary}

**Decision**: ${lrt.significant ? "✅ Use the more complex model" : "❌ Simpler model is adequate"}
`
```

---

## 🧮 Model Selection & Cross-Validation

### K-Fold Cross-Validation

Test model performance on held-out data:

```js
// Define model factory
modelFactory = () => new ds.stats.GLM({family: "gaussian"})
```

```js
// Prepare data
cv_X = carsData.map(d => [d.Weight_in_lbs, d.Horsepower])
cv_y = carsData.map(d => d.Miles_per_Gallon)
```

```js
cv_results = ds.stats.crossValidate(modelFactory, cv_X, cv_y, {
  k: 5,
  metric: "mse"
})
```

```js
md`
**Cross-Validation Results**:
- Mean MSE: ${cv_results.mean.toFixed(2)}
- Std Dev: ${cv_results.std.toFixed(2)}
- Folds: ${cv_results.k}

**Interpretation**: On average, predictions are off by ${Math.sqrt(cv_results.mean).toFixed(2)} mpg (RMSE)
`
```

---

## 🎨 Effect Plots

Visualize **marginal effects** - how the predicted value changes with one variable:

```js
ds.plot.effectPlot(model_multi, {
  variable: "Weight_in_lbs",
  data: carsData,
  width: 700,
  height: 400
})
```

---

## 📊 GLM Family Reference

| Family | Use Case | Link Function | Example |
|--------|----------|---------------|---------|
| **Gaussian** | Continuous outcome | Identity: $\mu$ | Height, temperature, income |
| **Binomial** | Binary outcome | Logit: $\log(\frac{p}{1-p})$ | Disease (yes/no), pass/fail |
| **Poisson** | Count data | Log: $\log(\lambda)$ | Number of events, accidents |
| **Gamma** | Positive continuous | Log or inverse | Waiting times, insurance claims |
| **Inverse Gaussian** | Positive, skewed | Inverse squared | Reaction times |
| **Negative Binomial** | Overdispersed counts | Log | Count data with extra variance |

---

## 💡 Best Practices

1. **Visualize first**: Plot your data before modeling
2. **Check assumptions**: Use diagnostic plots
3. **Start simple**: Begin with simple models, add complexity as needed
4. **Compare models**: Use AIC, cross-validation
5. **Interpret carefully**: Correlation ≠ causation
6. **Report uncertainty**: Always include confidence intervals
7. **Validate**: Test on new data when possible

---

## 🎓 Exercises

**Exercise 1**: Build a model predicting `Horsepower` from `Cylinders` and `Weight`. Check diagnostics.

**Exercise 2**: Use logistic regression to predict if a car is American (`Origin === "USA"`) based on characteristics.

**Exercise 3**: Create a Poisson model for a count outcome of your choice. Interpret the coefficients on the response scale (not log scale).

**Exercise 4**: Compare 3 models with different predictors. Which has the best AIC? Does cross-validation agree?

---

## 📚 Learn More

- [Statistical Rethinking (textbook)](https://xcelab.net/rm/statistical-rethinking/)
- [Mixed Models in R (lme4)](https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf)
- [GLM Guide](https://en.wikipedia.org/wiki/Generalized_linear_model)
- [tangent-ds Documentation](https://github.com/tangent-to/tangent-ds)

---

*Created with [tangent-ds](https://github.com/tangent-to/tangent-ds) - Statistical modeling in the browser*
