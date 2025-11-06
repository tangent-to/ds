# Ordination Methods with tangent-ds

*Visualizing and exploring multidimensional data*

---

## What is Ordination?

Ordination methods help us **visualize high-dimensional data** in 2D or 3D space. Think of it as creating a "map" of your data where similar observations are close together and different ones are far apart.

These methods are widely used in:
- 🧬 **Ecology**: comparing species across different sites
- 📊 **Marketing**: understanding customer segments
- 🔬 **Genomics**: visualizing gene expression patterns
- 📈 **Finance**: analyzing stock correlations

---

## Setup

```js
import * as ds from "@tangent.to/ds"
import * as Plot from "@observablehq/plot"
import * as vega from "vega-datasets"
```

---

## 📍 Principal Component Analysis (PCA)

PCA finds the **directions of maximum variance** in your data. It's like finding the best angle to take a photo that captures the most interesting features.

### The Math Behind PCA

PCA computes eigenvectors of the covariance matrix:

$$\text{Cov}(X) = \frac{1}{n-1}X^TX$$

The first principal component (PC1) points in the direction of maximum variance, PC2 is orthogonal to PC1 with the next highest variance, and so on.

### Example: Iris Dataset

Let's visualize the famous iris dataset - measurements of iris flowers from three different species.

```js
// Load the iris dataset
irisData = vega.data.iris.json()
```

```js
// Prepare data for PCA - extract numeric features
irisFeatures = irisData.map(d => [
  d.sepalLength,
  d.sepalWidth,
  d.petalLength,
  d.petalWidth
])

// Store species for coloring
irisSpecies = irisData.map(d => d.species)
```

```js
// Run PCA
pca = {
  const model = new ds.mva.PCA({
    center: true,
    scale: true  // Standardize features to mean=0, sd=1
  })
  model.fit(irisFeatures)
  return model
}
```

```js
// Get transformed coordinates
pcaScores = pca.transform(irisFeatures)
```

Now let's create a beautiful scatter plot showing the first two principal components:

```js
Plot.plot({
  width: 800,
  height: 500,
  marginLeft: 60,
  marginBottom: 60,
  grid: true,

  marks: [
    // Add reference lines at origin
    Plot.ruleX([0], {stroke: "#ddd", strokeDasharray: "4,4"}),
    Plot.ruleY([0], {stroke: "#ddd", strokeDasharray: "4,4"}),

    // Scatter plot with species colors
    Plot.dot(pcaScores.map((score, i) => ({
      pc1: score[0],
      pc2: score[1],
      species: irisSpecies[i]
    })), {
      x: "pc1",
      y: "pc2",
      fill: "species",
      r: 5,
      fillOpacity: 0.7,
      stroke: "white",
      strokeWidth: 1,
      tip: true
    }),

    // Add convex hulls around each species
    Plot.hull(pcaScores.map((score, i) => ({
      pc1: score[0],
      pc2: score[1],
      species: irisSpecies[i]
    })), {
      x: "pc1",
      y: "pc2",
      fill: "species",
      fillOpacity: 0.1,
      stroke: "species",
      strokeWidth: 2
    })
  ],

  color: {
    legend: true,
    scheme: "tableau10"
  },

  x: {
    label: `PC1 (${(pca.model.explainedVariance[0] * 100).toFixed(1)}% variance) →`,
    labelAnchor: "center"
  },

  y: {
    label: `↑ PC2 (${(pca.model.explainedVariance[1] * 100).toFixed(1)}% variance)`,
    labelAnchor: "center"
  },

  caption: "Iris flowers separated by species using PCA. Each point is a flower, colored by species."
})
```

**Interpretation**: The three iris species form distinct clusters! PCA successfully reduced 4 dimensions (sepal/petal length/width) to 2 dimensions while preserving the species groupings.

### Scree Plot: How Many Components?

The **scree plot** shows how much variance each component explains. Look for the "elbow" where variance drops off.

```js
ds.plot.plotScree(pca, {
  width: 600,
  height: 300,
  style: "variance"  // or "cumulative"
})
```

**Rule of thumb**: Keep components that explain significant variance (usually above the "elbow" point).

### Biplot: Variables + Observations

A **biplot** shows both observations (points) and variables (arrows) together. Arrow direction shows correlation, arrow length shows importance.

```js
ds.plot.plotPCA(pca, {
  data: pcaScores.map((score, i) => ({
    pc1: score[0],
    pc2: score[1],
    species: irisSpecies[i]
  })),
  colorBy: "species",
  biplot: true,
  variableNames: ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
  width: 800,
  height: 600
})
```

**Reading the biplot**:
- 🎯 **Arrows pointing together** = positively correlated variables
- 🎯 **Arrows pointing opposite** = negatively correlated variables
- 🎯 **Long arrows** = variables that contribute most to variation
- 🎯 **Short arrows** = less important variables

---

## 📊 Correspondence Analysis (CA)

Correspondence Analysis is like PCA but for **categorical data** or **count tables**. It's perfect for analyzing:
- Survey responses
- Species abundance tables
- Customer purchase patterns

### The Math

CA analyzes a contingency table by finding the singular value decomposition of standardized residuals:

$$Z = D_r^{-1/2}(P - rc^T)D_c^{-1/2}$$

where $P$ is the probability matrix, $r$ and $c$ are row and column masses.

### Example: Movie Preferences by Age Group

```js
// Simulate movie preference data
movieData = [
  {age: "18-25", action: 45, comedy: 30, drama: 10, scifi: 35},
  {age: "26-35", action: 40, comedy: 35, drama: 15, scifi: 30},
  {age: "36-45", action: 25, comedy: 40, drama: 25, scifi: 15},
  {age: "46-55", action: 15, comedy: 35, drama: 35, scifi: 10},
  {age: "56+", action: 10, comedy: 25, drama: 45, scifi: 5}
]
```

```js
// Convert to matrix format
movieMatrix = movieData.map(d => [d.action, d.comedy, d.drama, d.scifi])
movieAgeGroups = movieData.map(d => d.age)
movieGenres = ["Action", "Comedy", "Drama", "Sci-Fi"]
```

```js
// Run Correspondence Analysis
ca = {
  const model = new ds.mva.CA()
  model.fit(movieMatrix)
  return model
}
```

```js
// Create biplot showing both age groups and genres
caPlotData = {
  // Row scores (age groups)
  const rowScores = ca.rowScores.map((score, i) => ({
    dim1: score[0],
    dim2: score[1],
    label: movieAgeGroups[i],
    type: "Age Group"
  }))

  // Column scores (genres)
  const colScores = ca.columnScores.map((score, i) => ({
    dim1: score[0],
    dim2: score[1],
    label: movieGenres[i],
    type: "Genre"
  }))

  return {rows: rowScores, cols: colScores}
}
```

```js
Plot.plot({
  width: 800,
  height: 600,
  marginLeft: 80,
  marginBottom: 80,
  grid: true,

  marks: [
    // Reference lines
    Plot.ruleX([0], {stroke: "#ddd", strokeDasharray: "4,4"}),
    Plot.ruleY([0], {stroke: "#ddd", strokeDasharray: "4,4"}),

    // Age groups as circles
    Plot.dot(caPlotData.rows, {
      x: "dim1",
      y: "dim2",
      r: 10,
      fill: "steelblue",
      fillOpacity: 0.7,
      stroke: "white",
      strokeWidth: 2,
      tip: true
    }),

    Plot.text(caPlotData.rows, {
      x: "dim1",
      y: "dim2",
      text: "label",
      dy: -15,
      fontWeight: "bold",
      fill: "steelblue"
    }),

    // Genres as diamonds
    Plot.dot(caPlotData.cols, {
      x: "dim1",
      y: "dim2",
      r: 10,
      fill: "coral",
      fillOpacity: 0.7,
      stroke: "white",
      strokeWidth: 2,
      symbol: "diamond",
      tip: true
    }),

    Plot.text(caPlotData.cols, {
      x: "dim1",
      y: "dim2",
      text: "label",
      dy: 15,
      fontWeight: "bold",
      fill: "coral"
    })
  ],

  x: {
    label: `Dimension 1 (${(ca.model.explainedInertia[0] * 100).toFixed(1)}%) →`
  },

  y: {
    label: `↑ Dimension 2 (${(ca.model.explainedInertia[1] * 100).toFixed(1)}%)`
  },

  caption: "Correspondence Analysis: Age groups (blue circles) and movie genres (red diamonds). Proximity indicates association."
})
```

**Interpretation**: Age groups close to specific genres prefer those genres more than average!

---

## 🎯 Constrained Ordination (RDA)

**Redundancy Analysis (RDA)** is like PCA, but we can explain variation using **predictor variables**. It answers: "How much of the variation can be explained by these factors?"

### Example: Cars Performance

```js
// Load car data
carsData = vega.data.cars.json().filter(d =>
  d.Horsepower != null &&
  d.Weight_in_lbs != null &&
  d.Acceleration != null &&
  d.Miles_per_Gallon != null
)
```

```js
// Response: performance metrics
Y_cars = carsData.map(d => [
  d.Miles_per_Gallon,
  d.Acceleration,
  d.Horsepower
])

// Predictors: car characteristics
X_cars = carsData.map(d => [
  d.Weight_in_lbs / 1000,  // Weight in 1000s lbs
  d.Cylinders,
  d.Year - 70  // Years since 1970
])

carOrigins = carsData.map(d => d.Origin)
```

```js
// Run RDA
rda = {
  const model = new ds.mva.RDA({
    scale: true
  })
  model.fit(Y_cars, X_cars)
  return model
}
```

```js
// Variance partitioning
Inputs.table([
  {
    Component: "Explained by predictors",
    Variance: (rda.model.constrainedVariance * 100).toFixed(1) + "%"
  },
  {
    Component: "Unexplained (residual)",
    Variance: (rda.model.unconstrainedVariance * 100).toFixed(1) + "%"
  }
])
```

```js
ds.plot.plotRDA(rda, {
  data: rda.model.fitted.map((score, i) => ({
    rda1: score[0],
    rda2: score[1],
    origin: carOrigins[i]
  })),
  colorBy: "origin",
  width: 800,
  height: 600,
  constraintNames: ["Weight", "Cylinders", "Year"]
})
```

**Key insight**: RDA tells us how much car design factors (weight, cylinders, year) explain performance variation!

---

## 🌟 Unified Ordination Plot

tangent-ds provides a **unified plotting interface** for all ordination methods:

```js
ds.plot.ordiplot(pca, {
  colorBy: irisSpecies,
  title: "Iris PCA Ordination",
  showCentroid: true,  // Add group centroids
  showEllipse: true,   // Add confidence ellipses
  width: 700,
  height: 500
})
```

**Features**:
- ✅ Automatic axis labeling with variance explained
- ✅ Group centroids and confidence ellipses
- ✅ Works with PCA, CA, RDA, and more
- ✅ Interactive tooltips

---

## 💡 Choosing the Right Method

| Method | Best For | Data Type | Output |
|--------|----------|-----------|---------|
| **PCA** | Finding patterns in continuous data | Numeric matrix | Unconstrained axes |
| **CA** | Analyzing categorical/count data | Contingency table | Symmetric biplot |
| **RDA** | Explaining variation with predictors | Y matrix + X matrix | Constrained axes |
| **CCA** | Like RDA but for count data | Count matrix + predictors | Constrained, count data |

---

## 📚 Learn More

- [PCA Tutorial](https://observablehq.com/@d3/pca)
- [Multivariate Analysis in Ecology](https://www.springer.com/gp/book/9781402022883)
- [tangent-ds Documentation](https://github.com/tangent-to/tangent-ds)

---

## Try It Yourself! 🎓

**Exercise 1**: Use PCA on the `cars` dataset with numeric variables. Which variables drive the most variation?

**Exercise 2**: Create a CA plot for a custom contingency table (e.g., coffee preferences by profession).

**Exercise 3**: Use RDA to see how environmental factors explain species composition.

---

*Created with [tangent-ds](https://github.com/tangent-to/tangent-ds) - A minimalist, browser-friendly data science library*
