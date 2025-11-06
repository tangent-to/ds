# Machine Learning with tangent-ds

*Building predictive models in the browser*

---

## What is Machine Learning?

Machine Learning (ML) is about **learning patterns from data** to make predictions. Unlike traditional statistics, ML emphasizes **prediction accuracy** over interpretation.

**Key concepts**:
- 🎯 **Supervised Learning**: Learn from labeled examples (this is a cat, this is a dog)
- 🔍 **Unsupervised Learning**: Find patterns without labels (clustering, dimensionality reduction)
- 📊 **Training**: Learning from data
- 🎲 **Testing**: Evaluating on new data
- 🎚️ **Overfitting**: Model learns noise, not signal

**Applications**:
- 🖼️ Image recognition
- 💬 Spam detection
- 💰 Credit risk assessment
- 🏥 Disease diagnosis
- 📈 Stock price prediction

---

## Setup

```js
import * as ds from "@tangent.to/ds"
import * as Plot from "@observablehq/plot"
import * as vega from "vega-datasets"
```

---

## 🎯 Classification: Predicting Categories

Classification predicts **discrete labels** (species, success/fail, image class).

### Binary Classification: Iris Virginica Detection

Let's build a classifier to identify *Iris virginica* from petal measurements.

```js
// Load iris data
irisData = vega.data.iris.json()
```

```js
// Create binary classification task
irisClassData = irisData.map(d => ({
  ...d,
  isVirginica: d.species === "virginica" ? 1 : 0
}))
```

```js
// Prepare features (X) and target (y)
X_iris = irisClassData.map(d => [d.petalLength, d.petalWidth])
y_iris = irisClassData.map(d => d.isVirginica)
```

```js
// Train/test split (80/20)
{
  const n = X_iris.length
  const trainSize = Math.floor(n * 0.8)

  // Shuffle indices
  const indices = d3.shuffle(d3.range(n))

  train_idx = indices.slice(0, trainSize)
  test_idx = indices.slice(trainSize)
}
```

```js
X_train = train_idx.map(i => X_iris[i])
y_train = train_idx.map(i => y_iris[i])
X_test = test_idx.map(i => X_iris[i])
y_test = test_idx.map(i => y_iris[i])
```

### Logistic Regression Classifier

```js
// Train logistic regression
classifier = new ds.stats.GLM({family: "binomial"})
classifier.fit(X_train, y_train)
```

```js
// Predict on test set
y_pred_proba = classifier.predict(X_test)
y_pred = y_pred_proba.map(p => p > 0.5 ? 1 : 0)
```

```js
// Calculate accuracy
accuracy = y_pred.filter((pred, i) => pred === y_test[i]).length / y_test.length
```

```js
md`**Test Accuracy**: ${(accuracy * 100).toFixed(1)}% 🎯`
```

### Decision Boundary Visualization

```js
{
  // Create grid for decision boundary
  const petalLengthRange = d3.extent(X_iris, d => d[0])
  const petalWidthRange = d3.extent(X_iris, d => d[1])

  const gridPoints = []
  for (let pl = petalLengthRange[0]; pl <= petalLengthRange[1]; pl += 0.1) {
    for (let pw = petalWidthRange[0]; pw <= petalWidthRange[1]; pw += 0.1) {
      const prob = classifier.predict([[pl, pw]])[0]
      gridPoints.push({pl, pw, prob})
    }
  }

  return Plot.plot({
    width: 700,
    height: 500,
    marginLeft: 60,

    marks: [
      // Decision boundary heatmap
      Plot.raster(gridPoints, {
        x: "pl",
        y: "pw",
        fill: "prob",
        interpolate: "nearest"
      }),

      // Actual data points
      Plot.dot(irisClassData, {
        x: "petalLength",
        y: "petalWidth",
        stroke: d => d.isVirginica === 1 ? "gold" : "white",
        fill: d => d.isVirginica === 1 ? "gold" : "purple",
        r: 5,
        strokeWidth: 2,
        tip: true
      })
    ],

    color: {
      scheme: "purples",
      legend: true,
      label: "P(Virginica)"
    },

    x: {label: "Petal Length (cm) →"},
    y: {label: "↑ Petal Width (cm)"},
    caption: "Decision boundary (purple = low probability, yellow = high probability)"
  })
}
```

### Confusion Matrix

A **confusion matrix** shows where the model makes mistakes:

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | True Negative (TN) | False Positive (FP) |
| **Actual: Yes** | False Negative (FN) | True Positive (TP) |

```js
confusion = {
  let TP = 0, TN = 0, FP = 0, FN = 0

  y_pred.forEach((pred, i) => {
    if (pred === 1 && y_test[i] === 1) TP++
    else if (pred === 0 && y_test[i] === 0) TN++
    else if (pred === 1 && y_test[i] === 0) FP++
    else FN++
  })

  return {TP, TN, FP, FN}
}
```

```js
ds.plot.plotConfusionMatrix(confusion, {
  labels: ["Not Virginica", "Virginica"],
  width: 400,
  height: 400
})
```

**Metrics**:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ = ${((confusion.TP + confusion.TN) / (confusion.TP + confusion.TN + confusion.FP + confusion.FN)).toFixed(3)}$
- **Precision**: $\frac{TP}{TP + FP}$ = ${(confusion.TP / (confusion.TP + confusion.FP)).toFixed(3)}$ (when we predict "yes", how often are we right?)
- **Recall**: $\frac{TP}{TP + FN}$ = ${(confusion.TP / (confusion.TP + confusion.FN)).toFixed(3)}$ (of all actual "yes", how many did we catch?)
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ = harmonic mean

### ROC Curve

The **Receiver Operating Characteristic (ROC)** curve shows the trade-off between true positive rate and false positive rate:

```js
// Compute ROC curve
roc_data = {
  const thresholds = d3.range(0, 1.01, 0.01)

  return thresholds.map(thresh => {
    let TP = 0, FP = 0, TN = 0, FN = 0

    y_pred_proba.forEach((prob, i) => {
      const pred = prob > thresh ? 1 : 0
      if (pred === 1 && y_test[i] === 1) TP++
      else if (pred === 0 && y_test[i] === 0) TN++
      else if (pred === 1 && y_test[i] === 0) FP++
      else FN++
    })

    return {
      threshold: thresh,
      TPR: TP / (TP + FN),  // Sensitivity/Recall
      FPR: FP / (FP + TN),  // 1 - Specificity
      precision: TP / (TP + FP) || 0
    }
  })
}
```

```js
ds.plot.plotROC(roc_data.map(d => ({fpr: d.FPR, tpr: d.TPR})), {
  width: 500,
  height: 500
})
```

**AUC (Area Under Curve)**: Perfect classifier = 1.0, Random = 0.5

```js
// Calculate AUC (trapezoidal rule)
auc = d3.sum(roc_data.slice(1).map((d, i) => {
  const prev = roc_data[i]
  return (d.FPR - prev.FPR) * (d.TPR + prev.TPR) / 2
}))
```

```js
md`**AUC = ${auc.toFixed(3)}** ${"★".repeat(Math.floor(auc * 5))}`
```

### Precision-Recall Curve

For **imbalanced datasets** (rare events), Precision-Recall curve is more informative:

```js
ds.plot.plotPrecisionRecall(
  roc_data.filter(d => !isNaN(d.precision)).map(d => ({
    recall: d.TPR,
    precision: d.precision
  })),
  {width: 500, height: 500}
)
```

---

## 📈 Regression: Predicting Continuous Values

Regression predicts **continuous numbers** (price, temperature, sales).

### Example: Predicting Car Prices

```js
// Load car data
carsData = vega.data.cars.json().filter(d =>
  d.Horsepower != null &&
  d.Weight_in_lbs != null &&
  d.Miles_per_Gallon != null
)
```

```js
// Features: car characteristics
X_cars = carsData.map(d => [
  d.Horsepower,
  d.Weight_in_lbs / 1000,
  d.Cylinders,
  d.Acceleration
])

// Target: MPG (fuel efficiency)
y_cars = carsData.map(d => d.Miles_per_Gallon)
```

```js
// Train/test split
{
  const n = X_cars.length
  const trainSize = Math.floor(n * 0.75)
  const indices = d3.shuffle(d3.range(n))

  train_cars_idx = indices.slice(0, trainSize)
  test_cars_idx = indices.slice(trainSize)
}
```

```js
X_cars_train = train_cars_idx.map(i => X_cars[i])
y_cars_train = train_cars_idx.map(i => y_cars[i])
X_cars_test = test_cars_idx.map(i => X_cars[i])
y_cars_test = test_cars_idx.map(i => y_cars[i])
```

### Linear Regression

```js
// Train model
reg_model = new ds.stats.GLM({family: "gaussian"})
reg_model.fit(X_cars_train, y_cars_train)
```

```js
// Predictions
y_cars_pred = reg_model.predict(X_cars_test)
```

### Regression Metrics

```js
// Calculate metrics
reg_metrics = {
  const n = y_cars_test.length

  // Mean Squared Error
  const mse = d3.sum(y_cars_test.map((actual, i) =>
    Math.pow(actual - y_cars_pred[i], 2)
  )) / n

  // Root Mean Squared Error
  const rmse = Math.sqrt(mse)

  // Mean Absolute Error
  const mae = d3.sum(y_cars_test.map((actual, i) =>
    Math.abs(actual - y_cars_pred[i])
  )) / n

  // R-squared
  const y_mean = d3.mean(y_cars_test)
  const ss_tot = d3.sum(y_cars_test.map(y => Math.pow(y - y_mean, 2)))
  const ss_res = d3.sum(y_cars_test.map((actual, i) =>
    Math.pow(actual - y_cars_pred[i], 2)
  ))
  const r2 = 1 - (ss_res / ss_tot)

  return {mse, rmse, mae, r2}
}
```

```js
Inputs.table([
  {Metric: "RMSE", Value: reg_metrics.rmse.toFixed(2), Unit: "mpg"},
  {Metric: "MAE", Value: reg_metrics.mae.toFixed(2), Unit: "mpg"},
  {Metric: "R²", Value: reg_metrics.r2.toFixed(3), Unit: "(proportion of variance explained)"}
])
```

**Interpretation**:
- **RMSE**: On average, predictions are off by ${reg_metrics.rmse.toFixed(1)} mpg
- **R²**: Model explains ${(reg_metrics.r2 * 100).toFixed(1)}% of MPG variance

### Actual vs Predicted Plot

```js
Plot.plot({
  width: 600,
  height: 600,
  grid: true,

  marks: [
    // Perfect prediction line (y = x)
    Plot.line([{x: 0, y: 0}, {x: 50, y: 50}], {
      x: "x",
      y: "y",
      stroke: "red",
      strokeDasharray: "4,4",
      strokeWidth: 2
    }),

    // Actual vs predicted
    Plot.dot(y_cars_test.map((actual, i) => ({
      actual,
      predicted: y_cars_pred[i]
    })), {
      x: "actual",
      y: "predicted",
      fill: "steelblue",
      fillOpacity: 0.6,
      r: 5,
      tip: true
    })
  ],

  x: {label: "Actual MPG →", domain: [0, 50]},
  y: {label: "↑ Predicted MPG", domain: [0, 50]},
  caption: "Points near red line = accurate predictions"
})
```

### Residual Analysis

```js
Plot.plot({
  width: 700,
  height: 400,
  grid: true,

  marks: [
    Plot.ruleY([0], {stroke: "red", strokeDasharray: "4,4"}),

    Plot.dot(y_cars_pred.map((pred, i) => ({
      predicted: pred,
      residual: y_cars_test[i] - pred
    })), {
      x: "predicted",
      y: "residual",
      fill: "steelblue",
      fillOpacity: 0.5,
      r: 4
    })
  ],

  x: {label: "Predicted MPG →"},
  y: {label: "↑ Residual (Actual - Predicted)"},
  caption: "Random scatter around zero indicates good fit"
})
```

---

## 🎨 Polynomial Regression

For **non-linear relationships**, add polynomial terms:

```js
// Use formula syntax for polynomial regression
poly_model = new ds.stats.GLM({family: "gaussian"})

// Create data with polynomial features
carsDataWithPoly = carsData.map(d => ({
  mpg: d.Miles_per_Gallon,
  hp: d.Horsepower,
  hp2: d.Horsepower ** 2,
  hp3: d.Horsepower ** 3
}))

poly_model.fit("mpg ~ hp + hp2 + hp3", carsDataWithPoly)
```

```js
// Visualize polynomial fit
{
  const hpRange = d3.range(
    d3.min(carsData, d => d.Horsepower),
    d3.max(carsData, d => d.Horsepower),
    1
  )

  const predictions = hpRange.map(hp => ({
    hp,
    mpg: poly_model.predict([[hp, hp**2, hp**3]])[0]
  }))

  return Plot.plot({
    width: 700,
    height: 400,

    marks: [
      Plot.dot(carsData, {
        x: "Horsepower",
        y: "Miles_per_Gallon",
        fill: "gray",
        fillOpacity: 0.3,
        r: 3
      }),

      Plot.line(predictions, {
        x: "hp",
        y: "mpg",
        stroke: "red",
        strokeWidth: 3
      })
    ],

    x: {label: "Horsepower →"},
    y: {label: "↑ Miles per Gallon"},
    caption: "Cubic polynomial captures non-linear relationship"
  })
}
```

---

## 🧪 Model Evaluation & Selection

### Cross-Validation

**K-fold cross-validation** gives a more robust estimate of model performance:

```js
// 5-fold CV for regression
cv_results_reg = ds.stats.crossValidate(
  () => new ds.stats.GLM({family: "gaussian"}),
  X_cars,
  y_cars,
  {k: 5, metric: "rmse"}
)
```

```js
md`
**5-Fold Cross-Validation**:
- Mean RMSE: ${cv_results_reg.mean.toFixed(2)} mpg
- Std Dev: ${cv_results_reg.std.toFixed(2)} mpg
- Min: ${Math.min(...cv_results_reg.scores).toFixed(2)}
- Max: ${Math.max(...cv_results_reg.scores).toFixed(2)}
`
```

```js
// Visualize CV scores
Plot.plot({
  width: 500,
  height: 300,

  marks: [
    Plot.barY(cv_results_reg.scores.map((score, i) => ({
      fold: i + 1,
      rmse: score
    })), {
      x: "fold",
      y: "rmse",
      fill: "steelblue"
    }),

    Plot.ruleY([cv_results_reg.mean], {
      stroke: "red",
      strokeWidth: 2,
      strokeDasharray: "4,4"
    })
  ],

  x: {label: "Fold", tickFormat: d => `Fold ${d}`},
  y: {label: "↑ RMSE"},
  caption: "Red line = mean across folds"
})
```

### Learning Curves

**Learning curves** show how performance changes with training set size:

```js
learning_curve_data = {
  const trainSizes = [20, 40, 60, 80, 100, 150, 200]

  return trainSizes.map(size => {
    const model = new ds.stats.GLM({family: "gaussian"})
    const X_subset = X_cars.slice(0, size)
    const y_subset = y_cars.slice(0, size)

    model.fit(X_subset, y_subset)

    // Train error
    const y_train_pred = model.predict(X_subset)
    const train_mse = d3.mean(y_subset.map((y, i) =>
      Math.pow(y - y_train_pred[i], 2)
    ))

    // Validation error (on remaining data)
    const X_val = X_cars.slice(size)
    const y_val = y_cars.slice(size)
    const y_val_pred = model.predict(X_val)
    const val_mse = d3.mean(y_val.map((y, i) =>
      Math.pow(y - y_val_pred[i], 2)
    ))

    return {
      size,
      train_rmse: Math.sqrt(train_mse),
      val_rmse: Math.sqrt(val_mse)
    }
  })
}
```

```js
ds.plot.plotLearningCurve(
  learning_curve_data.map(d => ({
    size: d.size,
    train: d.train_rmse,
    val: d.val_rmse
  })),
  {width: 700, height: 400}
)
```

**Interpretation**:
- **Gap between curves**: Large gap = overfitting
- **Both curves high**: Underfitting (need better features/model)
- **Both curves low & close**: Good fit!

---

## 🌟 Feature Importance

Which features matter most?

```js
feature_importance = {
  const coefs = reg_model.model.coefficients.slice(1)  // Skip intercept
  const names = ["Horsepower", "Weight (1000 lbs)", "Cylinders", "Acceleration"]

  return names.map((name, i) => ({
    feature: name,
    coefficient: coefs[i],
    absCoefficient: Math.abs(coefs[i])
  })).sort((a, b) => b.absCoefficient - a.absCoefficient)
}
```

```js
ds.plot.plotFeatureImportance(feature_importance.map(d => ({
  feature: d.feature,
  importance: d.absCoefficient
})), {
  width: 600,
  height: 300
})
```

**Note**: For fair comparison, features should be standardized (mean=0, sd=1) before fitting.

---

## 🎯 Multiclass Classification

Classify into **more than 2 categories**:

```js
// Full iris dataset: 3 species
iris_X = irisData.map(d => [d.petalLength, d.petalWidth])
iris_y = irisData.map(d => d.species)
```

```js
// One-vs-rest approach: train 3 binary classifiers
multiclass_models = {
  const species = ["setosa", "versicolor", "virginica"]

  return species.map(sp => {
    const y_binary = iris_y.map(s => s === sp ? 1 : 0)
    const model = new ds.stats.GLM({family: "binomial"})
    model.fit(iris_X, y_binary)
    return {species: sp, model}
  })
}
```

```js
// Predict: choose class with highest probability
iris_predictions = iris_X.map(x => {
  const probs = multiclass_models.map(m => ({
    species: m.species,
    prob: m.model.predict([x])[0]
  }))

  return probs.reduce((max, p) => p.prob > max.prob ? p : max).species
})
```

```js
// Accuracy
multiclass_accuracy = iris_predictions.filter((pred, i) =>
  pred === iris_y[i]
).length / iris_y.length
```

```js
md`**Multiclass Accuracy**: ${(multiclass_accuracy * 100).toFixed(1)}%`
```

```js
// Visualize decision regions
{
  const plRange = d3.extent(iris_X, d => d[0])
  const pwRange = d3.extent(iris_X, d => d[1])

  const grid = []
  for (let pl = plRange[0]; pl <= plRange[1]; pl += 0.05) {
    for (let pw = pwRange[0]; pw <= pwRange[1]; pw += 0.05) {
      const probs = multiclass_models.map(m => ({
        species: m.species,
        prob: m.model.predict([[pl, pw]])[0]
      }))
      const predicted = probs.reduce((max, p) => p.prob > max.prob ? p : max).species

      grid.push({pl, pw, predicted})
    }
  }

  return Plot.plot({
    width: 700,
    height: 500,

    marks: [
      // Decision regions
      Plot.dot(grid, {
        x: "pl",
        y: "pw",
        fill: "predicted",
        fillOpacity: 0.1,
        r: 2
      }),

      // Actual data
      Plot.dot(irisData, {
        x: "petalLength",
        y: "petalWidth",
        stroke: "species",
        fill: "species",
        fillOpacity: 0.7,
        strokeWidth: 2,
        r: 5,
        tip: true
      })
    ],

    color: {
      domain: ["setosa", "versicolor", "virginica"],
      range: ["#e15759", "#59a14f", "#4e79a7"]
    },

    x: {label: "Petal Length →"},
    y: {label: "↑ Petal Width"}
  })
}
```

---

## 🚀 Putting It All Together: Complete ML Pipeline

```js
// 1. Load data
pipelineData = vega.data.cars.json().filter(d =>
  d.Miles_per_Gallon != null && d.Horsepower != null
)

// 2. Feature engineering
pipeline_X = pipelineData.map(d => [
  d.Horsepower,
  d.Weight_in_lbs / 1000,
  d.Cylinders
])

pipeline_y = pipelineData.map(d => d.Miles_per_Gallon > 25 ? 1 : 0)

// 3. Train/test split
{
  const n = pipeline_X.length
  const split = Math.floor(n * 0.7)
  const idx = d3.shuffle(d3.range(n))

  pipeline_train_idx = idx.slice(0, split)
  pipeline_test_idx = idx.slice(split)
}

// 4. Train model
pipeline_model = new ds.stats.GLM({family: "binomial"})
pipeline_model.fit(
  pipeline_train_idx.map(i => pipeline_X[i]),
  pipeline_train_idx.map(i => pipeline_y[i])
)

// 5. Evaluate
pipeline_test_pred = pipeline_model.predict(
  pipeline_test_idx.map(i => pipeline_X[i])
).map(p => p > 0.5 ? 1 : 0)

pipeline_test_accuracy = pipeline_test_pred.filter((p, i) =>
  p === pipeline_y[pipeline_test_idx[i]]
).length / pipeline_test_pred.length

md`✅ **Pipeline Accuracy**: ${(pipeline_test_accuracy * 100).toFixed(1)}%`
```

---

## 💡 Best Practices

1. ✅ **Always split data**: Train on one set, test on another
2. ✅ **Standardize features**: Especially for models sensitive to scale
3. ✅ **Cross-validate**: Don't trust a single train/test split
4. ✅ **Check for overfitting**: Compare train vs test performance
5. ✅ **Visualize**: Plot predictions, residuals, decision boundaries
6. ✅ **Start simple**: Begin with baseline model, add complexity gradually
7. ✅ **Domain knowledge**: Features matter more than algorithms
8. ✅ **Iterate**: ML is an iterative process

---

## 🎓 Exercises

**Exercise 1**: Build a classifier for a different binary outcome in the cars dataset (e.g., American vs non-American). Report precision, recall, and F1-score.

**Exercise 2**: Create polynomial features for the horsepower variable (degree 2, 3, 4) and compare models using cross-validation. Which degree is optimal?

**Exercise 3**: Build a multiclass classifier for iris species using all 4 features. Create a confusion matrix showing performance for each class.

**Exercise 4**: Implement a complete pipeline: load data → engineer features → train model → evaluate → tune hyperparameters → final test.

---

## 📚 Resources

- [Introduction to Statistical Learning (free book)](https://www.statlearning.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [tangent-ds Documentation](https://github.com/tangent-to/tangent-ds)

---

## 🔮 Coming Soon in tangent-ds

- 🌲 **Random Forests**: Ensemble of decision trees
- 🎯 **Regularization**: Ridge, Lasso, Elastic Net
- 📊 **Naive Bayes**: Fast probabilistic classifier
- 🧠 **Neural Networks**: Deep learning basics
- 📉 **Dimensionality Reduction**: t-SNE, UMAP

---

*Created with [tangent-ds](https://github.com/tangent-to/tangent-ds) - Bringing machine learning to Observable*
