# Clustering Analysis with tangent-ds

*Finding natural groups in your data*

---

## What is Clustering?

Clustering is an **unsupervised learning** technique that groups similar observations together. Unlike classification, we don't know the groups beforehand - the algorithm discovers them!

**Real-world applications**:
- 🛍️ **Customer segmentation**: Group customers by behavior
- 🧬 **Gene analysis**: Find genes with similar expression patterns
- 📰 **Document clustering**: Organize articles by topic
- 🗺️ **Geographic patterns**: Identify regional similarities

---

## Setup

```js
import * as ds from "@tangent.to/ds"
import * as Plot from "@observablehq/plot"
import * as vega from "vega-datasets"
```

---

## 🌳 Hierarchical Clustering

Hierarchical clustering builds a **tree (dendrogram)** showing how observations group together at different levels. It's like a family tree for your data!

### How it Works

1. Start with each observation as its own cluster
2. Find the two closest clusters and merge them
3. Repeat until all observations are in one cluster
4. The dendrogram shows the merging history

**Distance metrics**:
- **Euclidean**: Straight-line distance (most common)
- **Manhattan**: City-block distance
- **Correlation**: Based on pattern similarity

**Linkage methods**:
- **Single**: Minimum distance between clusters
- **Complete**: Maximum distance (creates compact clusters)
- **Average**: Average distance (balanced approach)
- **Ward**: Minimizes within-cluster variance (most popular)

### Example: Iris Flowers

```js
// Load iris data
irisData = vega.data.iris.json()
```

```js
// Extract numeric features
irisFeatures = irisData.map(d => [
  d.sepalLength,
  d.sepalWidth,
  d.petalLength,
  d.petalWidth
])

irisSpecies = irisData.map(d => d.species)
```

```js
// Perform hierarchical clustering
hclust = {
  const model = new ds.ml.HCA({
    method: "ward",      // Ward's minimum variance method
    metric: "euclidean"
  })
  model.fit(irisFeatures)
  return model
}
```

### Dendrogram: The Clustering Tree

The dendrogram visualizes the clustering hierarchy. Height shows dissimilarity - the higher the merge, the more different the groups.

```js
ds.plot.plotHCA(hclust, {
  labels: irisData.map((d, i) => `${d.species[0]}${i}`),
  width: 900,
  height: 400,
  marginBottom: 100,
  labelSize: 8,
  orientation: "bottom"
})
```

**How to read it**:
- 📏 **Height**: Dissimilarity level where clusters merge
- ✂️ **Cut line**: Draw a horizontal line to define clusters
- 🌿 **Branches**: Connected observations are similar

### Cutting the Tree

Let's cut the dendrogram to create 3 clusters:

```js
// Cut tree to get cluster assignments
clusters = hclust.cut(3)
```

```js
// Visualize clusters in PC space
{
  // Run PCA for visualization
  const pca = new ds.mva.PCA({scale: true})
  pca.fit(irisFeatures)
  const scores = pca.transform(irisFeatures)

  // Create plot data
  const plotData = scores.map((s, i) => ({
    pc1: s[0],
    pc2: s[1],
    cluster: `Cluster ${clusters[i]}`,
    actualSpecies: irisSpecies[i]
  }))

  return Plot.plot({
    width: 800,
    height: 500,
    grid: true,

    marks: [
      Plot.dot(plotData, {
        x: "pc1",
        y: "pc2",
        fill: "cluster",
        stroke: "actualSpecies",
        strokeWidth: 2,
        r: 6,
        fillOpacity: 0.6,
        tip: true
      })
    ],

    color: {
      legend: true,
      scheme: "category10"
    },

    x: {label: "PC1 →"},
    y: {label: "↑ PC2"},

    caption: "Fill color = discovered cluster | Stroke = actual species"
  })
}
```

**Insight**: Compare fill (discovered clusters) with stroke (actual species) - hierarchical clustering successfully found the natural iris species groups!

---

## 🎯 K-Means Clustering

K-Means is faster than hierarchical clustering and works well with large datasets. You specify the number of clusters $k$, and the algorithm finds the best grouping.

### The Algorithm

1. **Initialize**: Randomly place $k$ centroids
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Move centroid to cluster mean
4. **Repeat**: Until convergence

The objective is to minimize **within-cluster sum of squares (WCSS)**:

$$\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

where $\mu_i$ is the centroid of cluster $C_i$.

### Example: Customer Segmentation

Let's create customer segments based on spending patterns:

```js
// Simulate customer data
customerData = {
  const n = 200
  const random = d3.randomNormal.source(d3.randomLcg(42))

  // Three customer types
  const segments = [
    {freq: 2, spend: 20},   // Casual shoppers
    {freq: 8, spend: 60},   // Regular customers
    {freq: 15, spend: 150}  // VIP customers
  ]

  return Array.from({length: n}, (_, i) => {
    const seg = segments[i % 3]
    return {
      frequency: Math.max(0, random(seg.freq, 2)()),
      avgSpend: Math.max(0, random(seg.spend, 10)())
    }
  })
}
```

```js
customerFeatures = customerData.map(d => [d.frequency, d.avgSpend])
```

```js
// Run K-Means with k=3
kmeans = {
  const model = new ds.ml.KMeans({
    k: 3,
    maxIter: 100,
    nInit: 10  // Run 10 times and pick best
  })
  model.fit(customerFeatures)
  return model
}
```

```js
customerClusters = kmeans.predict(customerFeatures)
```

```js
Plot.plot({
  width: 800,
  height: 500,
  marginLeft: 60,
  marginBottom: 60,
  grid: true,

  marks: [
    // Customer points
    Plot.dot(customerData.map((d, i) => ({
      ...d,
      cluster: `Segment ${customerClusters[i] + 1}`
    })), {
      x: "frequency",
      y: "avgSpend",
      fill: "cluster",
      r: 5,
      fillOpacity: 0.6,
      stroke: "white",
      strokeWidth: 1,
      tip: true
    }),

    // Cluster centroids as stars
    Plot.dot(kmeans.model.centroids.map((c, i) => ({
      frequency: c[0],
      avgSpend: c[1],
      label: `Centroid ${i+1}`
    })), {
      x: "frequency",
      y: "avgSpend",
      fill: "black",
      r: 12,
      symbol: "star",
      stroke: "yellow",
      strokeWidth: 2,
      tip: true
    })
  ],

  color: {
    legend: true,
    domain: ["Segment 1", "Segment 2", "Segment 3"],
    range: ["#66c2a5", "#fc8d62", "#8da0cb"]
  },

  x: {
    label: "Purchase Frequency (per month) →",
    nice: true
  },

  y: {
    label: "↑ Average Spending ($)",
    nice: true
  },

  caption: "Customer segments discovered by K-Means clustering. Stars show cluster centroids."
})
```

### Cluster Profiles

Let's describe each segment:

```js
{
  const profiles = kmeans.model.centroids.map((centroid, i) => {
    const members = customerData.filter((_, j) => customerClusters[j] === i)

    return {
      Segment: i + 1,
      "Avg Frequency": centroid[0].toFixed(1),
      "Avg Spend ($)": centroid[1].toFixed(0),
      "Size": members.length,
      "Profile": centroid[0] < 4 ? "Casual 🛒" :
                 centroid[0] < 10 ? "Regular 🛍️" : "VIP ⭐"
    }
  })

  return Inputs.table(profiles)
}
```

---

## 📊 Choosing K: The Elbow Method

How many clusters should we use? The **elbow method** helps decide!

```js
// Calculate WCSS for different k values
elbowData = {
  const kValues = [1, 2, 3, 4, 5, 6, 7, 8]

  return kValues.map(k => {
    const model = new ds.ml.KMeans({k, nInit: 5})
    model.fit(customerFeatures)

    return {
      k: k,
      wcss: model.model.inertia
    }
  })
}
```

```js
Plot.plot({
  width: 600,
  height: 400,
  marginLeft: 60,

  marks: [
    Plot.line(elbowData, {
      x: "k",
      y: "wcss",
      stroke: "steelblue",
      strokeWidth: 3,
      marker: true
    }),

    Plot.dot(elbowData, {
      x: "k",
      y: "wcss",
      fill: "steelblue",
      r: 6,
      stroke: "white",
      strokeWidth: 2
    })
  ],

  x: {
    label: "Number of clusters (k) →",
    domain: [1, 8],
    ticks: 8
  },

  y: {
    label: "↑ Within-Cluster Sum of Squares",
    grid: true
  },

  caption: "Look for the 'elbow' - where adding clusters stops helping much"
})
```

**How to find the elbow**: Look for the point where the curve "bends" - where adding more clusters gives diminishing returns. Here, k=3 looks optimal!

---

## 🎨 Silhouette Analysis

The **silhouette score** measures how well each point fits its cluster:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

**Score ranges**:
- **+1**: Perfect clustering (far from other clusters)
- **0**: On the border between clusters
- **-1**: Probably in wrong cluster

```js
ds.plot.plotSilhouette(kmeans, customerFeatures, {
  width: 700,
  height: 400
})
```

**Interpretation**:
- 📊 **Wide bars**: Well-separated clusters
- 📉 **Negative values**: Misclassified points
- 📏 **Average line**: Overall clustering quality

---

## 🌍 Real Example: Clustering Countries

Let's cluster countries by development indicators using real data:

```js
// Use world development indicators
countriesData = await fetch("https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv")
  .then(r => r.text())
  .then(d3.csvParse)
  .then(data => data.filter(d => d.year === "2007"))  // Most recent year
```

```js
// Extract features: life expectancy and GDP per capita
countryFeatures = countriesData.map(d => [
  +d.lifeExp,
  Math.log(+d.gdpPercap)  // Log transform GDP (it's skewed)
])

countryNames = countriesData.map(d => d.country)
continents = countriesData.map(d => d.continent)
```

```js
// Cluster countries
countryKmeans = {
  const model = new ds.ml.KMeans({k: 4, nInit: 20})
  model.fit(countryFeatures)
  return model
}
```

```js
countryClusterLabels = countryKmeans.predict(countryFeatures)
```

```js
Plot.plot({
  width: 900,
  height: 600,
  marginLeft: 60,
  marginBottom: 60,

  marks: [
    // Countries colored by discovered cluster
    Plot.dot(countriesData.map((d, i) => ({
      lifeExp: +d.lifeExp,
      gdpPercap: +d.gdpPercap,
      country: d.country,
      continent: d.continent,
      cluster: `Group ${countryClusterLabels[i] + 1}`,
      pop: +d.pop
    })), {
      x: "gdpPercap",
      y: "lifeExp",
      r: d => Math.sqrt(d.pop) / 3000,
      fill: "cluster",
      fillOpacity: 0.6,
      stroke: "white",
      strokeWidth: 1,
      tip: {
        format: {
          country: true,
          continent: true,
          cluster: true,
          lifeExp: d => d.toFixed(1),
          gdpPercap: d => "$" + d.toFixed(0)
        }
      }
    }),

    // Centroids
    Plot.dot(countryKmeans.model.centroids.map((c, i) => ({
      gdpPercap: Math.exp(c[1]),
      lifeExp: c[0]
    })), {
      x: "gdpPercap",
      y: "lifeExp",
      fill: "black",
      r: 10,
      symbol: "star",
      stroke: "yellow",
      strokeWidth: 2
    })
  ],

  x: {
    type: "log",
    label: "GDP per Capita ($, log scale) →",
    grid: true
  },

  y: {
    label: "↑ Life Expectancy (years)",
    domain: [35, 85],
    grid: true
  },

  color: {
    legend: true
  },

  caption: "Countries grouped by development (bubble size = population)"
})
```

**Discoveries**:
- 💰 **Group 1**: High GDP, high life expectancy (developed)
- 📈 **Group 2**: Medium GDP, improving health
- 🌱 **Group 3**: Low GDP, lower life expectancy (developing)
- 🌍 **Group 4**: Special cases (outliers)

---

## 🔬 Comparing Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hierarchical** | • No need to specify k<br>• Shows structure<br>• Deterministic | • Slow on large data<br>• Can't "undo" merges | Exploring data structure, small-medium datasets |
| **K-Means** | • Fast<br>• Scales well<br>• Simple | • Must specify k<br>• Random initialization<br>• Assumes spherical clusters | Large datasets, known number of clusters |

---

## 🎓 Best Practices

1. **Scale your data**: Use standardized features (mean=0, sd=1)
2. **Handle outliers**: They can distort clusters
3. **Try multiple k values**: Use elbow method and silhouette
4. **Visualize**: Always plot your clusters (use PCA if needed)
5. **Interpret**: Give meaningful names to clusters
6. **Validate**: Do clusters make domain sense?

---

## 💡 Try It Yourself!

**Exercise 1**: Cluster the `cars` dataset by performance metrics. How many natural groups exist?

**Exercise 2**: Use hierarchical clustering on the `iris` dataset with different linkage methods (single, complete, average, ward). How do results differ?

**Exercise 3**: Create a customer segmentation with your own simulated data. Add a third dimension (e.g., age) and see how clusters change.

---

## Advanced: DBSCAN

For non-spherical clusters, try **DBSCAN** (Density-Based Spatial Clustering):

```js
// Coming soon in tangent-ds!
// DBSCAN finds clusters of arbitrary shape and identifies outliers
```

---

## 📚 Resources

- [K-Means Visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- [Hierarchical Clustering Guide](https://www.stat.berkeley.edu/~spector/s133/Clus.html)
- [tangent-ds Documentation](https://github.com/tangent-to/tangent-ds)

---

*Created with [tangent-ds](https://github.com/tangent-to/tangent-ds) - Making clustering accessible in the browser*
